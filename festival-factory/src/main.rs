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
mod taxonomy;
mod triples;

use clap::{Parser, Subcommand};
use config::FestivalConfig;
use std::collections::BTreeMap;
use std::io::Write;
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
    }
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

    let rt = tokio::runtime::Runtime::new()?;
    let records = rt.block_on(engine.hydrate(&store, a.base_n, a.concurrency, a.limit));

    // write tagged JSONL
    let path = out_dir.join("relations.jsonl");
    let mut f = std::fs::File::create(&path)?;
    for r in &records {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }

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
