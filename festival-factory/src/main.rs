//! festival-factory CLI — raw festival data → relationally-complete, faithfulness-gated CPT corpus.
//! Training stays in Python; this owns everything up to trainable data. See README.
//!
//!   festival-factory plan    --festival edc --edition las_vegas/2026
//!   festival-factory hydrate --festival edc --edition las_vegas/2026 [--dry-run] [--limit N]

mod build_corpus;
mod config;
mod disjoint;
mod hydrate;
mod openrouter;
mod settimes;
mod surface;
mod synth;
mod taxonomy;
mod tokenize;
mod triples;

use clap::{Parser, Subcommand};
use config::FestivalConfig;
use std::collections::BTreeMap;
use std::path::PathBuf;
use taxonomy::DataStatus;
use triples::TripleStore;

#[derive(Parser)]
#[command(name = "festival-factory", version, about = "Festival CPT data factory (raw → trainable)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Load a festival edition + print the relation coverage plan (no API).
    Plan(Target),
    /// Generate the grounded, faithfulness-gated relational corpus (tagged JSONL).
    Hydrate(HydrateArgs),
    /// Emit the artist-last set-time floor lines (one per line) — for parity-diff vs the Python port.
    SetTimes(Target),
    /// Assemble the V6 relational corpus (base+disjoint+set-times); print texts (stdout) + counts (stderr).
    BuildCorpus(Target),
    /// Tokenize + pack the corpus with lever metadata; print stats (parity-check vs corpus_v1_stats.json).
    Tokenize(TokenizeArgs),
    /// Synthetic-corpus generation (OpenRouter). --plan = dry-run counts (no API); --settimes = generate.
    Synth(SynthArgs),
}

#[derive(Parser)]
struct SynthArgs {
    #[command(flatten)]
    target: Target,
    #[arg(long, default_value = "google/gemma-4-31b-it")]
    model: String,
    #[arg(long, default_value_t = 6)]
    concurrency: usize,
    #[arg(long, default_value_t = 10)]
    variations: usize,
    /// Dry-run: print the generation plan, no API calls.
    #[arg(long)]
    plan: bool,
    /// Generate the set-time variations flow (the audit-validatable flow).
    #[arg(long)]
    settimes: bool,
    /// Generate the cultural primitives → conversation-variations flow.
    #[arg(long)]
    cultural: bool,
    /// Generate the chunk-strategy prose flow over 10_clean docs.
    #[arg(long)]
    docs: bool,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, default_value = "data/edc/las_vegas/2026/20_synthetic/_settimes_rust")]
    out: PathBuf,
}

#[derive(Parser)]
struct TokenizeArgs {
    #[command(flatten)]
    target: Target,
    #[arg(long, default_value = "/home/marcos/Documents/dev/models/gemma-4-E2B-it/tokenizer.json")]
    tokenizer: PathBuf,
    #[arg(long, default_value_t = 512)]
    seq_len: usize,
    /// Write the packed corpus as a plain Arrow IPC file (omit to just print stats).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Parser)]
struct Target {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long)]
    festival: String,
    #[arg(long)]
    edition: String,
}

#[derive(Parser)]
struct HydrateArgs {
    #[command(flatten)]
    target: Target,
    #[arg(long, default_value = "google/gemma-4-31b-it")]
    model: String,
    /// Low temperature keeps generation tied to the given facts (faithfulness).
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,
    /// Base lines requested per cell (coverage top-up refines per relation).
    #[arg(long, default_value_t = 12)]
    base_n: u32,
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
    /// Cap number of cells (smoke test).
    #[arg(long)]
    limit: Option<usize>,
    /// Build + print the cell plan WITHOUT calling the API.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Plan(t) => plan(t),
        Cmd::Hydrate(a) => hydrate_cmd(a),
        Cmd::SetTimes(t) => set_times_cmd(t),
        Cmd::BuildCorpus(t) => build_corpus_cmd(t),
        Cmd::Tokenize(a) => tokenize_cmd(a),
        Cmd::Synth(a) => synth_cmd(a),
    }
}

fn synth_cmd(a: SynthArgs) -> anyhow::Result<()> {
    let root = std::fs::canonicalize(&a.target.root).unwrap_or_else(|_| a.target.root.clone());
    let cfg = FestivalConfig::load_edition(&root, &a.target.festival, &a.target.edition)?;
    let store = TripleStore::load(cfg.structured_dir())?;
    if a.plan {
        return synth::plan(&cfg, &root, a.variations, a.variations);
    }
    if a.settimes || a.cultural || a.docs {
        let rt = tokio::runtime::Runtime::new()?;
        let or = std::sync::Arc::new(openrouter::OpenRouter::new(&a.model)?);
        if a.settimes {
            let n = rt.block_on(synth::expand_settimes(&cfg, &store, or.clone(), a.variations, a.concurrency, a.limit, &a.out))?;
            eprintln!("set-time synth: {n} lines → {}", a.out.display());
        }
        if a.cultural {
            let n = rt.block_on(synth::expand_cultural(&cfg, or.clone(), a.variations, a.concurrency, &a.out))?;
            eprintln!("cultural synth: {n} conversations → {}", a.out.display());
        }
        if a.docs {
            let n = rt.block_on(synth::synth_docs(&cfg, &root, or.clone(), a.concurrency, a.limit, &a.out))?;
            eprintln!("doc-strategy synth: {n} generations → {}", a.out.display());
        }
        return Ok(());
    }
    eprintln!("nothing to do — pass --plan, --settimes, --cultural, or --docs");
    Ok(())
}

fn tokenize_cmd(a: TokenizeArgs) -> anyhow::Result<()> {
    // canonicalize root FIRST so cfg.edition_dir is absolute → strip_prefix(root) yields keep-list-shaped
    // relative paths ("data/edc/.../X.md"), matching keep_list.json.
    let root = std::fs::canonicalize(&a.target.root).unwrap_or_else(|_| a.target.root.clone());
    let cfg = FestivalConfig::load_edition(&root, &a.target.festival, &a.target.edition)?;
    let store = TripleStore::load(cfg.structured_dir())?;
    let r = tokenize::tokenize(&cfg, &store, &root, a.seq_len, &a.tokenizer)?;
    let mask_frac = r.entity_masked as f64 / r.total_tokens.max(1) as f64;
    eprintln!("total_tokens={} num_chunks={} entity_masked={} ({:.1}%)",
        r.total_tokens, r.blocks.len(), r.entity_masked, mask_frac * 100.0);
    eprintln!("tokens_by_tag={:?}", r.tokens_by_tag);
    let src_named: BTreeMap<&str, usize> = r.src_dist.iter()
        .map(|(k, v)| (match k { 1 => "qa", 2 => "conversation", 4 => "replay", _ => "raw_knowledge" }, *v))
        .collect();
    eprintln!("src_dist={src_named:?}");
    if let Some(out) = &a.out {
        tokenize::write_ipc(&r, out)?;
        eprintln!("wrote {} ({} blocks)", out.display(), r.blocks.len());
    }
    Ok(())
}

fn set_times_cmd(t: Target) -> anyhow::Result<()> {
    let cfg = FestivalConfig::load_edition(&t.root, &t.festival, &t.edition)?;
    let store = TripleStore::load(cfg.structured_dir())?;
    for l in settimes::set_time_lines(&cfg, &store)? {
        println!("{}", l.text);
    }
    Ok(())
}

fn build_corpus_cmd(t: Target) -> anyhow::Result<()> {
    let cfg = FestivalConfig::load_edition(&t.root, &t.festival, &t.edition)?;
    let store = TripleStore::load(cfg.structured_dir())?;
    let r = build_corpus::build(&cfg, &store)?;
    eprintln!(
        "base {} (dropped {} refusal, stripped {} co-mention) + disjoint {} + set-time {} = {} lines",
        r.n_base, r.dropped_refusal, r.stripped_comention, r.n_disjoint, r.n_settime, r.texts.len()
    );
    for t in &r.texts {
        println!("{t}");
    }
    Ok(())
}

fn plan(t: Target) -> anyhow::Result<()> {
    let cfg = FestivalConfig::load_edition(&t.root, &t.festival, &t.edition)?;
    let store = TripleStore::load(cfg.structured_dir())?;
    println!("Festival : {} ({})", cfg.name, cfg.festival_slug);
    println!("Venue    : {}   Dates: {}", cfg.venue, cfg.dates);
    println!("Stages   : {} configured, {} with lineup", cfg.stages.len(), store.stages_present().len());
    println!("Lineup   : {} plays_at facts ({} entities)\n", store.plays_at().len(), store.entities.len());
    let (mut sup, mut part, mut blk) = (0, 0, 0);
    println!("RELATION COVERAGE PLAN (▶ build / ◐ partial / ✗ blocked):");
    for r in taxonomy::relations() {
        match r.status {
            DataStatus::Supported => { sup += 1; println!("  ▶ {:26} {:?}", r.key, r.tier); }
            DataStatus::Partial => { part += 1; println!("  ◐ {:26} {:?}", r.key, r.tier); }
            DataStatus::Blocked => blk += 1,
        }
    }
    println!("\n  {sup} supported + {part} partial = {} active, {blk} blocked", sup + part);
    println!("\nBLOCKED — needs data:");
    for (file, rels) in taxonomy::needs_summary() {
        let have = matches!(file, "venue_map.json") && cfg.has_venue_map()
            || matches!(file, "set_times.jsonl") && cfg.has_set_times();
        println!("  {} {file:18} → {}", if have { "✓" } else { "✗" }, rels.join(", "));
    }
    Ok(())
}

fn hydrate_cmd(a: HydrateArgs) -> anyhow::Result<()> {
    let cfg = FestivalConfig::load_edition(&a.target.root, &a.target.festival, &a.target.edition)?;
    let store = TripleStore::load(cfg.structured_dir())?;
    let out_dir = cfg.hydrated_dir();
    std::fs::create_dir_all(&out_dir)?;

    let engine = hydrate::Engine::new(cfg, &store, &a.model, a.temperature)?;
    if a.dry_run {
        // build cells only (no API) — report the plan
        let recs = engine.dry_run_plan(&store, a.base_n, a.limit);
        println!("DRY RUN — {} cells planned (model={}, temp={})", recs, a.model, a.temperature);
        return Ok(());
    }

    // hydrate STREAMS each faithful+deduped line to relations.jsonl as it is produced (crash-safe);
    // it returns the in-memory records only for the coverage tally below.
    let path = out_dir.join("relations.jsonl");
    let rt = tokio::runtime::Runtime::new()?;
    let records = rt.block_on(engine.hydrate(&store, a.base_n, a.concurrency, a.limit, &path))?;

    // coverage tally (Part 5 gate 1)
    let mut by: BTreeMap<(String, String), usize> = BTreeMap::new();
    for r in &records {
        *by.entry((r.relation_type.clone(), r.direction.clone())).or_default() += 1;
    }
    println!("hydrated {} faithful lines → {}", records.len(), path.display());
    println!("\nCOVERAGE (relation × direction → lines):");
    for ((rel, dir), n) in &by {
        println!("  {rel:24} {dir:8} {n}");
    }
    Ok(())
}
