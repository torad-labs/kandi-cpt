//! The hydration engine: graph drives coverage, model writes the prose, gates keep it faithful.
//!
//! For each grounded cell (relation × direction × surface_form, anchored to specific triples) we ask
//! the model for diverse lines, then FAITHFULNESS-FILTER: a line is rejected if it names any festival
//! entity (artist/stage) that is NOT in the cell's source facts. That is the anti-confabulation gate —
//! it would have killed the v2 "Prodigy → neonGARDEN" line, because neonGARDEN isn't in Prodigy's
//! source fact. Output is tagged JSONL (spec Part 4) + a coverage tally.

use crate::config::FestivalConfig;
use crate::openrouter::OpenRouter;
use crate::surface::{system_contract, SurfaceForm, POSITIVE_FORMS};
use crate::triples::TripleStore;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Serialize, Clone)]
pub struct Record {
    pub text: String,
    pub relation_type: String,
    pub direction: String,
    pub query_order: String,
    pub surface_form: String,
    pub source_triple_ids: Vec<String>,
    pub entity_ids: Vec<String>,
    pub split: String,
}

/// Festival entity vocabulary, for the faithfulness gate: every artist + every stage surface form.
struct Vocab {
    artists: Vec<String>,
    stage_terms: Vec<String>, // camel/spaced/aliases across all stages
}

impl Vocab {
    fn build(cfg: &FestivalConfig, store: &TripleStore) -> Self {
        let mut artists: Vec<String> = store.plays_at().into_iter().map(|p| p.artist).collect();
        artists.sort();
        artists.dedup();
        let mut stage_terms = Vec::new();
        for s in cfg.stages.values() {
            stage_terms.push(s.camel.clone());
            stage_terms.push(s.spaced.clone());
            stage_terms.extend(s.aliases.clone());
        }
        Self { artists, stage_terms }
    }
}

/// A unit of generation: a prompt grounded on specific facts, with the allowed entities pinned.
struct Cell {
    relation: &'static str,
    direction: &'static str,
    query_order: String,
    form: SurfaceForm,
    facts: String,                 // the GIVEN FACTS block (grounding)
    source_triple_ids: Vec<String>,
    entity_ids: Vec<String>,       // for tagging
    allow_artists: BTreeSet<String>, // faithfulness: artists this cell may mention
    allow_stages: BTreeSet<String>,  // stage slugs this cell may mention
    n: u32,
}

pub struct Engine {
    cfg: Arc<FestivalConfig>,
    or: Arc<OpenRouter>,
    system: Arc<String>,
    vocab: Arc<Vocab>,
    temperature: f32,
}

impl Engine {
    pub fn new(cfg: FestivalConfig, store: &TripleStore, model: &str, temperature: f32) -> anyhow::Result<Self> {
        let system = system_contract(&cfg);
        let vocab = Vocab::build(&cfg, store);
        Ok(Self {
            cfg: Arc::new(cfg),
            or: Arc::new(OpenRouter::new(model)?),
            system: Arc::new(system),
            vocab: Arc::new(vocab),
            temperature,
        })
    }

    /// Build the grounded cells for the SUPPORTED lineup + stage-attribute + derived relations.
    fn build_cells(&self, store: &TripleStore, base_n: u32) -> Vec<Cell> {
        let cfg = &self.cfg;
        let lineup = store.plays_at();
        let mut cells = Vec::new();

        // ── coverage rebalancing inputs: stage/day densities from the lineup (port of
        //    fact_expand.reps_for). Under-represented stages AND days get up to 3× the base so every
        //    stage and every day receives roughly equal total airtime (kills the v2 interference where
        //    neon=16 swamped circuit=2 and the modal Saturday=22 swamped Friday). ──
        let mut stage_counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut day_counts: BTreeMap<String, u32> = BTreeMap::new();
        for p in &lineup {
            *stage_counts.entry(p.stage_slug.clone()).or_default() += 1;
            *day_counts.entry(p.day.clone()).or_default() += 1; // includes "" (no-day) bucket, as the Python does
        }

        // ── per-fact lineup cells: artist↔stage↔day, both directions, order-matched, REBALANCED ──
        for p in &lineup {
            let camel = cfg.camel(&p.stage_slug);
            let spaced = cfg.spaced(&p.stage_slug);
            let dd = if p.day.is_empty() { String::new() } else { cfg.day_dated(&p.day) };
            let fact = if p.day.is_empty() {
                format!("- {} plays the {camel} ({spaced}) stage at {}.", p.artist, cfg.name)
            } else {
                format!("- {} plays the {camel} ({spaced}) stage on {dd} at {}.", p.artist, cfg.name)
            };
            let allow_a: BTreeSet<String> = [p.artist.clone()].into();
            let allow_s: BTreeSet<String> = [p.stage_slug.clone()].into();
            // rebalance: under-represented stages/days get more (parity with the densest), capped 3×.
            let n = reps_for(&p.stage_slug, &p.day, &stage_counts, &day_counts, base_n, 3);
            // ~22 DISTINCT structural forms per relationship (the diversity lever). The fact carries
            // stage+day, so every form covers the full artist↔stage↔day relationship; day-leading and
            // stage-leading directions are encoded by the ReverseDay/ReverseStage forms themselves.
            // All positive + single-entity-faithful — NO contrastive co-occurrence (v3 proved it
            // backfires: the model keeps the wrong-entity co-occurrence and drops the negation).
            for form in POSITIVE_FORMS.iter().copied() {
                let dir = if matches!(form, SurfaceForm::ReverseStage | SurfaceForm::ReverseDay) {
                    "reverse"
                } else {
                    "forward"
                };
                cells.push(Cell {
                    relation: "artist_to_stage", direction: dir, query_order: form.query_order().into(),
                    form, facts: fact.clone(),
                    source_triple_ids: vec![p.triple_id.clone()],
                    entity_ids: vec![p.artist.clone(), p.stage_slug.clone(), p.day.clone()],
                    allow_artists: allow_a.clone(), allow_stages: allow_s.clone(), n,
                });
            }
        }

        // ── per-stage attribute cells from config: genre / brand / rank-capacity / production-vibe.
        //    Each is its OWN relation so the coverage tally reads cleanly; all are stage-only
        //    (allow_artists empty, allow_stages = {this stage}). ──
        for (slug, st) in &cfg.stages {
            let allow_s: BTreeSet<String> = [slug.clone()].into();
            let meta_id = format!("stage_meta:{slug}");

            // genre (+ reverse genre→stage is implied by the leads_with_stage order + instruction)
            if !st.genre.is_empty() {
                let facts = format!("- The {} ({}) stage at {} is the {} stage.",
                                    st.camel, st.spaced, cfg.name, st.genre);
                for form in [SurfaceForm::Declarative, SurfaceForm::QaDirect] {
                    cells.push(Cell {
                        relation: "stage_to_genre", direction: "forward", query_order: "leads_with_stage".into(),
                        form, facts: facts.clone(),
                        source_triple_ids: vec![meta_id.clone()],
                        entity_ids: vec![slug.clone()],
                        allow_artists: BTreeSet::new(), allow_stages: allow_s.clone(), n: base_n,
                    });
                }
            }
            // brand / curator (e.g. Factory 93 runs Neon Garden) — reverse brand→stage query too
            if !st.brand.is_empty() {
                let facts = format!("- The {} ({}) stage at {} is presented/curated by {}.",
                                    st.camel, st.spaced, cfg.name, st.brand);
                for form in [SurfaceForm::Declarative, SurfaceForm::QaDirect] {
                    cells.push(Cell {
                        relation: "stage_brand", direction: "forward", query_order: "leads_with_stage".into(),
                        form, facts: facts.clone(),
                        source_triple_ids: vec![meta_id.clone()],
                        entity_ids: vec![slug.clone()],
                        allow_artists: BTreeSet::new(), allow_stages: allow_s.clone(), n: base_n,
                    });
                }
            }
            // rank / capacity (largest stage, second-largest, etc.)
            if !st.rank.is_empty() {
                let facts = format!("- At {}, {} ({}) is {}.", cfg.name, st.camel, st.spaced, st.rank);
                for form in [SurfaceForm::Declarative, SurfaceForm::QaDirect] {
                    cells.push(Cell {
                        relation: "stage_rank_capacity", direction: "forward", query_order: "leads_with_stage".into(),
                        form, facts: facts.clone(),
                        source_triple_ids: vec![meta_id.clone()],
                        entity_ids: vec![slug.clone()],
                        allow_artists: BTreeSet::new(), allow_stages: allow_s.clone(), n: base_n,
                    });
                }
            }
            // production character / vibe
            if !st.vibe.is_empty() {
                let facts = format!("- {} ({}) at {} is {}.", st.camel, st.spaced, cfg.name, st.vibe);
                for form in [SurfaceForm::Declarative, SurfaceForm::Conversational] {
                    cells.push(Cell {
                        relation: "stage_production_character", direction: "forward", query_order: "leads_with_stage".into(),
                        form, facts: facts.clone(),
                        source_triple_ids: vec![meta_id.clone()],
                        entity_ids: vec![slug.clone()],
                        allow_artists: BTreeSet::new(), allow_stages: allow_s.clone(), n: base_n,
                    });
                }
            }
        }

        // ── day_of_week_map: day↔date calendar cells (symmetric). No artist/stage is named, so
        //    allow_artists AND allow_stages are EMPTY — the gate must reject any artist/stage that
        //    leaks in. Facts = the calendar mapping (Friday = May 15, …). ──
        for (day, info) in &cfg.days {
            if info.date.is_empty() { continue; }
            let ord = if info.ordinal.is_empty() { String::new() } else { format!(" ({})", info.ordinal) };
            let facts = format!("- At {}, {} is {}{}.", cfg.name, day, info.date, ord);
            for (dir, order) in [("forward", "leads_with_day"), ("reverse", "leads_with_date")] {
                for form in [SurfaceForm::Declarative, SurfaceForm::QaDirect] {
                    cells.push(Cell {
                        relation: "day_of_week_map", direction: dir, query_order: order.into(),
                        form, facts: facts.clone(),
                        source_triple_ids: vec![format!("calendar:{day}")],
                        entity_ids: vec![day.clone(), info.date.clone()],
                        allow_artists: BTreeSet::new(), allow_stages: BTreeSet::new(), n: base_n,
                    });
                }
            }
        }

        // ── b2b_pairing: for each plays_at artist whose name contains " B2B ", emit symmetric
        //    partnership cells. The line names the joint act AND its individual members, so
        //    allow_artists = {full billed name} ∪ {each partner}; allow_stages = {that stage}. ──
        for p in &lineup {
            if !p.artist.contains(" B2B ") {
                continue;
            }
            let partners: Vec<String> = p.artist.split(" B2B ").map(|s| s.trim().to_string()).collect();
            let camel = cfg.camel(&p.stage_slug);
            let spaced = cfg.spaced(&p.stage_slug);
            let when = if p.day.is_empty() { String::new() } else { format!(" on {}", cfg.day_dated(&p.day)) };
            let joined = partners.join(" and ");
            let facts = format!(
                "- At {}, \"{}\" is a back-to-back (B2B) set — {} play together at the {} ({}) stage{}.",
                cfg.name, p.artist, joined, camel, spaced, when
            );
            // allow set: full name + every partner (partners are real co-billed entities of this fact)
            let mut allow_a: BTreeSet<String> = [p.artist.clone()].into();
            for part in &partners { allow_a.insert(part.clone()); }
            let allow_s: BTreeSet<String> = [p.stage_slug.clone()].into();
            let mut ents = vec![p.artist.clone()];
            ents.extend(partners.iter().cloned());
            ents.push(p.stage_slug.clone());
            for (dir, order, form) in [
                ("forward", "leads_with_artist", SurfaceForm::Declarative),
                ("forward", "leads_with_artist", SurfaceForm::QaDirect),
                ("reverse", "leads_with_partner", SurfaceForm::QaDirect),
            ] {
                cells.push(Cell {
                    relation: "b2b_pairing", direction: dir, query_order: order.into(),
                    form, facts: facts.clone(),
                    source_triple_ids: vec![p.triple_id.clone()],
                    entity_ids: ents.clone(),
                    allow_artists: allow_a.clone(), allow_stages: allow_s.clone(), n: base_n,
                });
            }
        }

        // ── multi_stage: an artist on >1 DISTINCT stage. Both facts are true, so the cell may name
        //    BOTH stages → allow_stages = ALL their stages (this is the ONLY place a single artist's
        //    cell legitimately spans multiple stages). Resolves "which stage does X play?" to both. ──
        let mut stages_of: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut days_of: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
        let mut triple_ids_of: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for p in &lineup {
            stages_of.entry(p.artist.clone()).or_default().insert(p.stage_slug.clone());
            triple_ids_of.entry(p.artist.clone()).or_default().push(p.triple_id.clone());
            if !p.day.is_empty() {
                days_of.entry((p.artist.clone(), p.stage_slug.clone())).or_default().insert(p.day.clone());
            }
        }
        for (artist, slugs) in &stages_of {
            if slugs.len() < 2 {
                continue;
            }
            let parts: Vec<String> = slugs.iter().map(|slug| {
                let days: Vec<String> = days_of.get(&(artist.clone(), slug.clone()))
                    .map(|d| d.iter().cloned().collect()).unwrap_or_default();
                if days.is_empty() {
                    cfg.camel(slug)
                } else {
                    format!("{} (on {})", cfg.camel(slug), days.join(", "))
                }
            }).collect();
            let facts = format!("- At {}, {} plays MORE THAN ONE stage: {}.", cfg.name, artist, parts.join(" and "));
            // partners too, if this multi-stage act is itself a B2B billing
            let mut allow_a: BTreeSet<String> = [artist.clone()].into();
            if artist.contains(" B2B ") {
                for part in artist.split(" B2B ") { allow_a.insert(part.trim().to_string()); }
            }
            let allow_s: BTreeSet<String> = slugs.clone(); // WIDEN to exactly this artist's stages
            let mut ents = vec![artist.clone()];
            ents.extend(slugs.iter().cloned());
            for form in [SurfaceForm::Declarative, SurfaceForm::QaDirect] {
                cells.push(Cell {
                    relation: "multi_stage", direction: "forward", query_order: "leads_with_artist".into(),
                    form, facts: facts.clone(),
                    source_triple_ids: triple_ids_of.get(artist).cloned().unwrap_or_default(),
                    entity_ids: ents.clone(),
                    allow_artists: allow_a.clone(), allow_stages: allow_s.clone(), n: base_n,
                });
            }
        }

        // ── per-stage roster (COMPOSITIONAL + MULTIHOP): one cell per stage that bundles the stage,
        //    its genre, and its full roster of artists. allow_* = the UNION of exactly those entities
        //    (this stage + every artist on it) — never broader. MultiHop lets the model chain
        //    genre→stage→artist; Compositional bundles several facts into one dense sentence. ──
        let mut artists_on: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut tids_on_stage: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for p in &lineup {
            artists_on.entry(p.stage_slug.clone()).or_default().insert(p.artist.clone());
            tids_on_stage.entry(p.stage_slug.clone()).or_default().push(p.triple_id.clone());
        }
        for (slug, arts) in &artists_on {
            let st = match cfg.stages.get(slug) { Some(s) => s, None => continue };
            let roster: Vec<String> = arts.iter().cloned().collect();
            let genre_bit = if st.genre.is_empty() { String::new() } else { format!(" ({} stage)", st.genre) };
            let facts = format!(
                "- The {} ({}) stage at {}{} hosts these artists: {}.",
                st.camel, st.spaced, cfg.name, genre_bit, roster.join(", ")
            );
            // allow: this stage only + exactly the artists on it (+ B2B partners among them)
            let allow_s: BTreeSet<String> = [slug.clone()].into();
            let mut allow_a: BTreeSet<String> = arts.clone();
            for a in arts {
                if a.contains(" B2B ") {
                    for part in a.split(" B2B ") { allow_a.insert(part.trim().to_string()); }
                }
            }
            let mut ents = vec![slug.clone()];
            ents.extend(roster.iter().cloned());
            let tids = tids_on_stage.get(slug).cloned().unwrap_or_default();
            for form in [SurfaceForm::Compositional, SurfaceForm::MultiHop] {
                cells.push(Cell {
                    relation: "stage_roster", direction: "reverse", query_order: "leads_with_stage".into(),
                    form, facts: facts.clone(),
                    source_triple_ids: tids.clone(),
                    entity_ids: ents.clone(),
                    allow_artists: allow_a.clone(), allow_stages: allow_s.clone(), n: base_n,
                });
            }
        }

        // ── recommendation (DERIVED, COMPOSITIONAL): worked-example "if you like <genre>, go to
        //    <stage>" cells, grounded on the stage's genre + brand. Stage-only: allow_stages =
        //    {this stage}, allow_artists EMPTY (names no artist — keeps the derivation faithful). ──
        for (slug, st) in &cfg.stages {
            if st.genre.is_empty() {
                continue;
            }
            let brand_bit = if st.brand.is_empty() { String::new() } else { format!(", curated by {}", st.brand) };
            let facts = format!(
                "- At {}, the {} ({}) stage{} is the place for {}.",
                cfg.name, st.camel, st.spaced, brand_bit, st.genre
            );
            for form in [SurfaceForm::Conversational, SurfaceForm::Compositional] {
                cells.push(Cell {
                    relation: "recommendation", direction: "forward", query_order: "leads_with_genre".into(),
                    form, facts: facts.clone(),
                    source_triple_ids: vec![format!("stage_meta:{slug}")],
                    entity_ids: vec![slug.clone()],
                    allow_artists: BTreeSet::new(), allow_stages: [slug.clone()].into(), n: base_n,
                });
            }
        }

        cells
    }

    /// Build cells + print a per-(relation,form) breakdown without calling the API. Returns cell count.
    pub fn dry_run_plan(&self, store: &TripleStore, base_n: u32, limit: Option<usize>) -> usize {
        let mut cells = self.build_cells(store, base_n);
        if let Some(l) = limit {
            cells.truncate(l);
        }
        let mut by: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for c in &cells {
            *by.entry((c.relation, c.form.name())).or_default() += 1;
        }
        for ((rel, form), n) in &by {
            println!("  {rel:24} {form:14} {n} cells");
        }
        cells.len()
    }

    /// Faithfulness gate: reject a line that names any artist/stage OUTSIDE the cell's allowed set.
    fn faithful(&self, line: &str, cell: &Cell) -> bool {
        let low = line.to_lowercase();
        // Reject LLM refusal / meta-commentary. These slip past the entity + day checks (they name a
        // valid artist and no wrong day) but are the model talking about its own constraints, not a
        // fact — e.g. "Since the provided facts do not specify a day for X, I cannot fulfill…". Found
        // 5 such lines contaminating the v4 corpus; this closes the gap.
        const REFUSAL: [&str; 15] = [
            "cannot fulfill", "i cannot", "without inventing", "the provided facts", "given facts",
            "as an ai", "i am unable", "i'm unable", "does not specify", "do not specify",
            "not specified", "strict faithfulness", "i can only", "cannot generate", "unable to comply",
        ];
        if REFUSAL.iter().any(|sig| low.contains(sig)) {
            return false;
        }
        for a in &self.vocab.artists {
            if !cell.allow_artists.contains(a) && low.contains(&a.to_lowercase()) {
                return false; // mentions a foreign artist
            }
        }
        for (slug, st) in &self.cfg.stages {
            if cell.allow_stages.contains(slug) {
                continue;
            }
            for term in [&st.camel, &st.spaced] {
                let t = term.to_lowercase();
                if t.len() > 3 && low.contains(&t) {
                    return false; // mentions a foreign stage
                }
            }
        }
        // day-faithfulness: reject any line naming a day NOT present in the grounding facts. The gate
        // derives the allowed day(s) from cell.facts (which states the true day), so a line that drifts
        // to a wrong day — or lists wrong days as options — is dropped. Closes the gap the stage gate
        // doesn't cover (all 3 days are valid tokens, so only this check catches wrong-day proximity).
        let facts_low = cell.facts.to_lowercase();
        for (day, date) in [("friday", "may 15"), ("saturday", "may 16"), ("sunday", "may 17")] {
            let allowed = facts_low.contains(day) || facts_low.contains(date);
            if !allowed && (low.contains(day) || low.contains(date)) {
                return false; // mentions a day the grounding fact does not assert
            }
        }
        let _ = &self.vocab.stage_terms;
        true
    }

    async fn run_cell(&self, cell: &Cell) -> Vec<Record> {
        let user = format!(
            "GIVEN FACTS:\n{}\n\nTASK: {}\nLead each line with the {} entity. Produce {} distinct lines.",
            cell.facts, cell.form.instruction(), cell.query_order.replace("leads_with_", ""), cell.n
        );
        let raw = match self.or.complete(&self.system, &user, self.temperature, 1200).await {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        for line in raw.lines() {
            let line = strip_bullet(line.trim());
            if line.len() < 8 {
                continue;
            }
            if !self.faithful(line, cell) {
                continue;
            }
            out.push(Record {
                text: line.to_string(),
                relation_type: cell.relation.to_string(),
                direction: cell.direction.to_string(),
                query_order: cell.query_order.clone(),
                surface_form: cell.form.name().to_string(),
                source_triple_ids: cell.source_triple_ids.clone(),
                entity_ids: cell.entity_ids.clone(),
                split: cell.form.split().to_string(),
            });
        }
        out
    }

    /// Generate the full corpus with bounded concurrency, STREAMING each faithful+deduped line to
    /// `out_path` as its cell completes (flushed per cell). A crash/kill therefore keeps everything
    /// generated so far on disk — no buffer-everything-then-write-once data-loss cliff (composes with the
    /// rehydrate_vc1.sh preserve-on-underperform wrapper). Returns the in-memory records for the tally.
    pub async fn hydrate(&self, store: &TripleStore, base_n: u32, concurrency: usize, limit: Option<usize>,
                         out_path: &std::path::Path) -> std::io::Result<Vec<Record>> {
        use std::io::Write as _;
        let mut cells = self.build_cells(store, base_n);
        if let Some(l) = limit {
            cells.truncate(l);
        }
        let cells = Arc::new(cells);
        let sem = Arc::new(Semaphore::new(concurrency));
        let mut set = tokio::task::JoinSet::new();
        for idx in 0..cells.len() {
            let sem = sem.clone();
            let cells = cells.clone();
            // SAFETY of borrows: clone the small bits the task needs via the engine's Arcs.
            let engine = self.clone_handle();
            set.spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                engine.run_cell(&cells[idx]).await
            });
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
        let mut seen = std::collections::BTreeSet::new();   // on-the-fly dedup (same key as the old dedup())
        let mut out = Vec::new();
        while let Some(res) = set.join_next().await {
            if let Ok(recs) = res {
                for r in recs {
                    let key = r.text.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
                    if seen.insert(key) {
                        writeln!(f, "{}", serde_json::to_string(&r).unwrap_or_default())?;
                        out.push(r);
                    }
                }
                f.flush()?;   // per-cell durability: a crash keeps every cell written so far
            }
        }
        Ok(out)
    }

    // Lightweight clonable handle (Arcs only) so spawned tasks own their refs.
    fn clone_handle(&self) -> Engine {
        Engine {
            cfg: self.cfg.clone(),
            or: self.or.clone(),
            system: self.system.clone(),
            vocab: self.vocab.clone(),
            temperature: self.temperature,
        }
    }
}

/// How many LLM variations a fact gets, scaled UP for under-represented stages/days so every stage
/// and every day receives roughly equal total airtime (port of fact_expand.reps_for). weight = the
/// worst under-representation across (stage, day), capped at `cap`, floored at `base`. Day weight is
/// 1.0 for no-day facts (the empty-day bucket is not balanced against — matches the Python).
fn reps_for(
    stage_slug: &str,
    day: &str,
    stage_counts: &BTreeMap<String, u32>,
    day_counts: &BTreeMap<String, u32>,
    base: u32,
    cap: u32,
) -> u32 {
    let s_tgt = stage_counts.values().copied().max().unwrap_or(1).max(1) as f64;
    let d_tgt = day_counts.values().copied().max().unwrap_or(1).max(1) as f64;
    let s_cnt = stage_counts.get(stage_slug).copied().unwrap_or(1).max(1) as f64;
    let w_stage = s_tgt / s_cnt;
    let w_day = if day.is_empty() {
        1.0
    } else {
        let d_cnt = day_counts.get(day).copied().unwrap_or(1).max(1) as f64;
        d_tgt / d_cnt
    };
    let weight = w_stage.max(w_day).min(cap as f64);
    let scaled = (base as f64 * weight).round() as u32;
    base.max(scaled)
}

fn strip_bullet(s: &str) -> &str {
    let s = s.trim_start_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_whitespace());
    // drop leading "1. " / "1) "
    if let Some(pos) = s.find(['.', ')']) {
        if pos <= 3 && s[..pos].chars().all(|c| c.is_ascii_digit()) {
            return s[pos + 1..].trim_start();
        }
    }
    s
}

// (dedup() was removed here — hydrate() now dedups on-the-fly while streaming each line to disk.)
