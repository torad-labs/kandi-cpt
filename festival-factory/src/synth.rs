//! Port of scripts/synth_corpus.py — the OpenRouter synthetic-corpus generator (run-once per refresh).
//!
//! Three flows: (1) chunk-strategy prose over 10_clean docs, (2) cultural primitives → many conversation
//! variations, (3) set-time artist-LAST variations. Reuses OpenRouter (parallel, bounded by a semaphore)
//! and the disjoint audit guards. Output is LLM-non-deterministic, so validation is STRUCTURAL (dry-run
//! plan + audit-clean, correctly-shaped output) — not byte-parity. Writes to a caller-chosen dir so it
//! never clobbers the committed 20_synthetic.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::Semaphore;

use crate::config::FestivalConfig;
use crate::disjoint;
use crate::openrouter::OpenRouter;
use crate::triples::TripleStore;

const SYSTEM_PREAMBLE: &str = "You are generating synthetic training data for the continued pre-training (CPT) of Kandi, an on-device AI festival companion. The objective is to install accurate, specific, durable knowledge about Electric Daisy Carnival (EDC) Las Vegas 2026 (May 15-17, 2026, Las Vegas Motor Speedway) into the model's weights, through many DIVERSE representations of the same underlying facts. Every output MUST be: (1) faithful to the source — invent no artists, stages, days, times, prices, or policies not present; (2) concrete and specific — name real artists, stages (in Insomniac camelCase: kineticFIELD, cosmicMEADOW, circuitGROUNDS, neonGARDEN, bassPOD, wasteLAND, quantumVALLEY, stereoBLOOM, bionicJUNGLE), days, and times; (3) self-contained and free of meta-commentary about the source. Your specific writing task for this passage:\n\n";

// (name, prompt) — the 7 chunk-strategy system prompts (verbatim from synth_corpus.STRATEGIES).
const STRATEGIES: &[(&str, &str)] = &[
    ("qa_pairs", "You are writing high-quality factual question-answer pairs for a knowledge corpus about Electric Daisy Carnival 2026 (Las Vegas Motor Speedway, May 15-17 2026). For the source content below, produce 5-10 specific Q&A pairs that capture every concrete fact. Each question should be specific (not 'tell me about X'), each answer should commit to a definite claim. Use the exact artist names, stage names (with Insomniac's camelCase: kineticFIELD, cosmicMEADOW, etc.), days, times, and venue details from the source. No hedging, no clarifying questions. Format:\n\nQ: <specific question>\nA: <definite answer>\n\nSource content follows."),
    ("extract_knowledge", "You are writing a textbook-style passage that distills the facts in the source text about Electric Daisy Carnival 2026 into a dense, factual prose. Use the structure of a reference work: define terms, name entities concretely, cite specific artists/stages/days/times. Multiple short paragraphs. Avoid filler. Avoid second-person 'you' framing. Avoid promotional language. Just facts in clean prose."),
    ("distill_summary", "Write a tight, fact-dense summary of the source content about EDC Las Vegas 2026. Preserve every artist name, stage assignment, day, time, location, and policy. Compress everything else. ~200 words. Maintain Insomniac's stage camelCase (kineticFIELD, cosmicMEADOW, circuitGROUNDS, neonGARDEN, bassPOD, wasteLAND, quantumVALLEY, stereoBLOOM, bionicJUNGLE)."),
    ("wikipedia", "Rewrite the source content in the formal encyclopedic style of Wikipedia — neutral tone, third person, factual claims with specific entity names. Cover the same facts but reorganize into thematic sections (Overview, Stages, Lineup, Logistics, Safety) as appropriate. Use Insomniac's stage camelCase. No 'in this article we will…' meta-language; just write the entry directly."),
    ("knowledge_list", "Convert the source content into a flat list of single-fact bullet points about EDC 2026. Each bullet: one concrete fact, one short sentence. Cover artists (with stage and day), stage descriptions, ops policies, safety knowledge, wayfinding tips. Use Insomniac's stage camelCase. 25-50 bullets total."),
    ("friend_chat", "Rewrite the source content as a casual but informative two-person chat where one friend is asking specific questions about EDC 2026 and a festival-veteran friend is answering with concrete details. Cover the same facts. Use natural conversation flow. The veteran uses Insomniac's stage camelCase (kineticFIELD, etc.) and commits to specific artist/stage/day claims. Format:\n\nFriend: <question>\nVet: <answer with specifics>\n\nGenerate 8-15 turns."),
    ("wayfinding_pov", "Rewrite the source content as a first-person wayfinding narrative from a festival veteran walking through EDC 2026. Describe what's where. Use concrete spatial relationships. Name stages with Insomniac camelCase. Name specific artists playing the stages you pass. Cover the lineup facts the source mentions. Like a walking-tour transcript."),
];

const AUDIENCES: &[&str] = &["a first-time festivalgoer", "a veteran raver", "a parent helping their kid", "a music journalist"];

const CULTURAL_DOCS: &[&str] = &[
    "official/festival_tips_aggregated.md", "official/insomniac_rookie_mistakes.md",
    "official/insomniac_voice_essay.md", "official/edc_guides_aggregated.md",
    "reddit/r-aves-2026.md", "reddit/r-electricdaisycarnival-2026.md",
];

const CULTURAL_PRIMITIVE_PROMPT: &str = "Below is veteran festival-culture writing about EDC Las Vegas 2026 (the Las Vegas Motor Speedway). Extract the DISTINCT cultural talking points — the concrete things experienced ravers tell newcomers. Each primitive: ONE self-contained piece of lived advice or ethos, in one or two sentences, faithful to the source (invent nothing). Cover the range actually present: footwear/feet, pacing across the three nights, hydration/electrolytes, the PLUR ethos, kandi trading, buddy/group safety, meet-up plans when cell service dies, sunrise sets as emotional peaks, sleep/recovery, what to bring, crowd navigation, harm-reduction. Assert NOTHING about which specific artist plays which day, stage, or time. Output ONE primitive per line, no numbering, no preamble.";

const CONVO_VARIATION_PROMPT: &str = "Here is ONE piece of EDC Las Vegas 2026 festival-culture wisdom:\n\n    {PRIM}\n\nWrite {N} DIFFERENT short two-person conversations that each teach EXACTLY this one point. Vary who is asking (a first-timer, a nervous parent, a seasoned raver, a curious journalist), the scenario, and the wording — but every conversation must land this same concrete advice, faithfully. Keep them natural and casual, 3-6 turns each, in the format 'Friend: …' / 'Vet: …'. Do NOT claim which artist plays which day, stage, or time. Separate each conversation with a line containing only '---'.";

const SETTIME_VARIATION_PROMPT: &str = "Here is ONE verified SET-TIME fact about EDC Las Vegas 2026 (May 15-17, 2026, at the Las Vegas Motor Speedway):\n\n    {FACT}\n\nWrite {N} DIVERSE ways to state this one set time. They must ALL follow the SAME token order so the model learns to predict the PERFORMER from the context (not the reverse): lead with the OCCASION (the {SLOT}), then the DAY, then the TIME and stage, and name the ARTIST LAST — as the final words of the sentence. Everything before the artist must narrow to exactly one performer. NEVER begin a line with the artist's name. Use BOTH the camelCase stage {CAMEL} and the spaced form {SPACED}. Name the day as {DAY} and as {DATE}. Assert NOTHING about any other artist, stage, day, or time. One variation per line, no numbering.";

fn chunk_text(text: &str, target_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if cur.len() + para.len() > target_chars && !cur.is_empty() {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(para);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn is_noise(t: &str) -> bool {
    let low = t.trim().to_lowercase();
    const SUB: &[&str] = &["no information regarding", "cannot be produced", "no factual", "consists entirely of",
        "does not contain", "no main article", "unable to generate", "i cannot create", "no relevant", "contains no information"];
    SUB.iter().any(|s| low.contains(s))
}

fn clean_lines(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|l| l.trim_start_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_numeric() || c == '.' || c == ')' || c == ' ').trim().to_string())
        .filter(|l| l.len() > 10 && !is_noise(l))
        .collect()
}

/// Plan (no API) — counts what each flow WOULD generate, for structural validation.
pub fn plan(cfg: &FestivalConfig, repo_root: &Path, settime_vars: usize, cultural_vars: usize) -> anyhow::Result<()> {
    let clean = cfg.clean_dir();
    let mut n_docs = 0;
    let mut n_chunks = 0;
    for d in crate::tokenize::collect_docs(cfg, repo_root)?.iter().filter(|d| d.tag == "clean") {
        n_docs += 1;
        n_chunks += chunk_text(&d.text, 3200).len();
    }
    let n_doc_gen = n_chunks * (STRATEGIES.len() + 2); // +2: friend_chat/extract_knowledge × audience
    let n_settimes = std::fs::read_to_string(cfg.edition_dir.join("set_times.jsonl"))
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0);
    eprintln!("[synth plan] docs: {n_docs} clean docs → {n_chunks} chunks → ~{n_doc_gen} strategy generations");
    eprintln!("[synth plan] cultural: {} source docs → ~30 primitives × {cultural_vars} convos", CULTURAL_DOCS.len());
    eprintln!("[synth plan] set-times: {n_settimes} sets × {settime_vars} artist-last variations");
    Ok(())
}

async fn complete(or: &OpenRouter, sem: &Arc<Semaphore>, system: &str, user: &str, max_tokens: u32) -> Option<String> {
    let _p = sem.acquire().await.ok()?;
    match or.complete(system, user, 0.85, max_tokens).await {
        Ok(t) if !t.trim().is_empty() => Some(t),
        Ok(_) => {
            eprintln!("[synth] empty completion");
            None
        }
        Err(e) => {
            eprintln!("[synth] openrouter error: {e}");
            None
        }
    }
}

/// Flow 3: set-time artist-last variations (the most validatable LLM flow — audit-guarded by violates).
pub async fn expand_settimes(cfg: &FestivalConfig, store: &TripleStore, or: Arc<OpenRouter>, n: usize,
                             concurrency: usize, limit: Option<usize>, out: &Path) -> anyhow::Result<usize> {
    let st_path = cfg.edition_dir.join("set_times.jsonl");
    if !st_path.exists() {
        return Ok(0);
    }
    let by_lower: std::collections::BTreeMap<String, String> = store.entities.values()
        .filter(|e| !e.name_canonical.is_empty())
        .map(|e| (e.name_canonical.to_lowercase(), e.name_canonical.clone())).collect();
    let rows: Vec<Value> = std::fs::read_to_string(&st_path)?.lines()
        .filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect();
    std::fs::create_dir_all(out)?;
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut set = tokio::task::JoinSet::new();
    for (i, r) in rows.into_iter().enumerate() {
        if limit.map_or(false, |l| i >= l) {
            break;
        }
        let artist_raw = r.get("artist").and_then(Value::as_str).unwrap_or("").to_string();
        let artist = by_lower.get(&artist_raw.to_lowercase()).cloned().unwrap_or_else(|| artist_raw.clone());
        let stage_slug = r.get("stage").and_then(Value::as_str).unwrap_or("").to_string();
        let camel = r.get("stage_raw").and_then(Value::as_str).map(String::from)
            .unwrap_or_else(|| cfg.stages.get(&stage_slug).map(|s| s.camel.clone()).unwrap_or_default());
        let spaced = cfg.stages.get(&stage_slug).map(|s| s.spaced.clone()).unwrap_or_else(|| camel.clone());
        let day = r.get("day").and_then(Value::as_str).unwrap_or("").to_string();
        let date = cfg.days.get(&day).map(|d| d.date.clone()).unwrap_or_default();
        let start = r.get("start").and_then(Value::as_str).unwrap_or("").to_string();
        let end = r.get("end").and_then(Value::as_str).unwrap_or("").to_string();
        let sunrise = r.get("tags").and_then(Value::as_array).map_or(false, |t| t.iter().any(|x| x.as_str() == Some("SUNRISE SET")));
        if artist.is_empty() || camel.is_empty() || day.is_empty() || start.is_empty() {
            continue;
        }
        let slot = if sunrise { "sunrise set (the dawn closer)" } else { "set" };
        let fact = format!("On {day} ({date}), the {start}\u{2013}{end} {slot} at the {camel} stage is played by {artist} at EDC Las Vegas 2026{}.",
            if sunrise { " — the sunrise set (the dawn closer)" } else { "" });
        let prompt = format!("{SYSTEM_PREAMBLE}{}", SETTIME_VARIATION_PROMPT
            .replace("{FACT}", &fact).replace("{N}", &n.to_string()).replace("{SLOT}", slot)
            .replace("{CAMEL}", &camel).replace("{SPACED}", &spaced).replace("{DAY}", &day).replace("{DATE}", &date));
        let (or, sem, cfg2) = (or.clone(), sem.clone(), cfg.clone());
        let out2 = out.to_path_buf();
        set.spawn(async move {
            let raw = complete(&or, &sem, &prompt, "Write the set-time variations now.", 1800).await?;
            let kept: Vec<String> = clean_lines(&raw).into_iter().filter(|l| disjoint::violates(&cfg2, l).is_none()).collect();
            if kept.is_empty() {
                return None;
            }
            let fname = out2.join(format!("{i:03}__{}.md", slugify(&artist)));
            let body = format!("# Set-time: {artist} -> {day}\n# Synthetic: true\n\n{}\n", kept.join("\n"));
            std::fs::write(fname, body).ok()?;
            Some(kept.len())
        });
    }
    let mut total = 0;
    while let Some(res) = set.join_next().await {
        if let Ok(Some(k)) = res {
            total += k;
        }
    }
    Ok(total)
}

/// Flow 2: cultural primitives → many conversation variations (audit-guarded by violates + cultural_conflict).
pub async fn expand_cultural(cfg: &FestivalConfig, or: Arc<OpenRouter>, n: usize, concurrency: usize, out: &Path) -> anyhow::Result<usize> {
    let clean = cfg.clean_dir();
    let source: String = CULTURAL_DOCS.iter()
        .filter_map(|p| std::fs::read_to_string(clean.join(p)).ok())
        .collect::<Vec<_>>().join("\n\n");
    if source.is_empty() {
        return Ok(0);
    }
    // 1) extract primitives (one call)
    let sys = format!("{SYSTEM_PREAMBLE}{CULTURAL_PRIMITIVE_PROMPT}");
    let raw = or.complete(&sys, &source.chars().take(24000).collect::<String>(), 0.85, 2200).await?;
    let mut primitives: Vec<String> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for l in clean_lines(&raw) {
        if l.chars().count() > 25 && seen.insert(slugify(&l)) {
            primitives.push(l);
        }
    }
    std::fs::create_dir_all(out)?;
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut set = tokio::task::JoinSet::new();
    for (i, prim) in primitives.into_iter().enumerate() {
        let prompt = format!("{SYSTEM_PREAMBLE}{}", CONVO_VARIATION_PROMPT.replace("{PRIM}", &prim).replace("{N}", &n.to_string()));
        let (or, sem, cfg2, out2) = (or.clone(), sem.clone(), cfg.clone(), out.to_path_buf());
        set.spawn(async move {
            let raw = complete(&or, &sem, &prompt, "Write the conversations now.", 3500).await?;
            let kept: Vec<String> = raw.split("\n---\n").map(|c| c.trim().to_string())
                .filter(|c| c.len() > 40 && !is_noise(c)
                    && disjoint::violates(&cfg2, c).is_none() && disjoint::cultural_conflict(&cfg2, c).is_none())
                .collect();
            if kept.is_empty() {
                return None;
            }
            let body = format!("# Cultural primitive: {prim}\n# Synthetic: true\n\nPRIMITIVE: {prim}\n\n{}\n", kept.join("\n\n---\n\n"));
            std::fs::write(out2.join(format!("{i:03}__{}.md", slugify(&prim))), body).ok()?;
            Some(kept.len())
        });
    }
    let mut total = 0;
    while let Some(r) = set.join_next().await {
        if let Ok(Some(k)) = r {
            total += k;
        }
    }
    Ok(total)
}

/// Flow 1: chunk-strategy prose over the 10_clean docs (the 7 STRATEGIES + audience-varied extract/friend_chat).
pub async fn synth_docs(cfg: &FestivalConfig, repo_root: &Path, or: Arc<OpenRouter>, concurrency: usize, limit: Option<usize>, out: &Path) -> anyhow::Result<usize> {
    let docs: Vec<_> = crate::tokenize::collect_docs(cfg, repo_root)?.into_iter().filter(|d| d.tag == "clean").collect();
    std::fs::create_dir_all(out)?;
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut set = tokio::task::JoinSet::new();
    let mut launched = 0;
    'outer: for d in &docs {
        let stem = Path::new(&d.path).file_stem().and_then(|s| s.to_str()).unwrap_or("doc").to_string();
        for (ci, chunk) in chunk_text(&d.text, 3200).into_iter().enumerate() {
            // the 7 strategies + 2 audience-varied (extract_knowledge, friend_chat), cycling audiences deterministically.
            let mut jobs: Vec<(String, String, Option<&str>)> =
                STRATEGIES.iter().map(|(n, p)| (n.to_string(), p.to_string(), None)).collect();
            for nm in ["extract_knowledge", "friend_chat"] {
                if let Some((_, p)) = STRATEGIES.iter().find(|(n, _)| *n == nm) {
                    jobs.push((nm.to_string(), p.to_string(), Some(AUDIENCES[(ci) % AUDIENCES.len()])));
                }
            }
            for (name, sp, aud) in jobs {
                if limit.map_or(false, |l| launched >= l) {
                    break 'outer;
                }
                launched += 1;
                let mut system = format!("{SYSTEM_PREAMBLE}{sp}");
                if let Some(a) = aud {
                    system.push_str(&format!("\n\nWrite this for the audience: {a}."));
                }
                let suffix = aud.map(|a| slugify(a)).unwrap_or_else(|| format!("chunk{ci}"));
                let (or, sem, out2) = (or.clone(), sem.clone(), out.to_path_buf());
                let (chunk, stem2) = (chunk.clone(), stem.clone());
                set.spawn(async move {
                    let t = complete(&or, &sem, &system, &chunk, 1300).await?;
                    if is_noise(&t) {
                        return None; // model declined (header/thin chunk) — same is_noise filter as the Python
                    }
                    let fname = out2.join(format!("{stem2}__{name}__{suffix}__{ci:03}.md"));
                    if let Err(e) = std::fs::write(&fname, format!("# Strategy: {name}\n# Synthetic: true\n\n{t}\n")) {
                        eprintln!("[synth docs] write {} failed: {e}", fname.display());
                        return None;
                    }
                    Some(1usize)
                });
            }
        }
    }
    eprintln!("[synth docs] launched {launched} jobs across {} clean docs", docs.len());
    let mut total = 0;
    while let Some(r) = set.join_next().await {
        match r {
            Ok(Some(k)) => total += k,
            Ok(None) => {}
            Err(e) => eprintln!("[synth docs] task error: {e}"),
        }
    }
    Ok(total)
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut prev_us = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }
    out.trim_matches('_').chars().take(40).collect()
}
