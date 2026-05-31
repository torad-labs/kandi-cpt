//! Per-festival config — the single source every stage reads, deserialized from
//! `data/<festival>/<city>/<year>/festival_meta.json`. This is what makes the pipeline a factory:
//! all festival-specific knowledge (stages, genres, brands, days↔dates) lives in data, not code.
//!
//! Optional drop-in files beside it (`venue_map.json`, `set_times.jsonl`) unblock the spatial /
//! set-time relation tiers with no code change — see taxonomy.rs.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct DayInfo {
    pub date: String,
    #[serde(default)]
    pub ordinal: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stage {
    pub camel: String,
    pub spaced: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub genre: String,
    #[serde(default)]
    pub brand: String,
    #[serde(default)]
    pub rank: String,
    #[serde(default)]
    pub vibe: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FestivalConfig {
    pub festival_slug: String,
    pub name: String,
    #[serde(default)]
    pub short_names: Vec<String>,
    pub venue: String,
    #[serde(default)]
    pub venue_aliases: Vec<String>,
    #[serde(default)]
    pub city: String,
    pub dates: String,
    #[serde(default)]
    pub n_main_stages: u32,
    /// Insertion order is NOT preserved by JSON objects; BTreeMap gives a stable, deterministic
    /// iteration order (alphabetical by slug) so generation is reproducible run-to-run.
    pub days: BTreeMap<String, DayInfo>,
    pub stages: BTreeMap<String, Stage>,

    /// Filled in by `load` — the edition directory the config was read from (root for all stage dirs).
    #[serde(skip)]
    pub edition_dir: PathBuf,
}

impl FestivalConfig {
    /// Load from an explicit edition directory containing `festival_meta.json`.
    pub fn load(edition_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = edition_dir.as_ref().to_path_buf();
        let meta = dir.join("festival_meta.json");
        let raw = std::fs::read_to_string(&meta)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", meta.display()))?;
        let mut cfg: FestivalConfig = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse {}: {e}", meta.display()))?;
        cfg.edition_dir = dir;
        Ok(cfg)
    }

    /// Resolve `data/<festival>/<edition>/` from the repo root (e.g. festival="edc",
    /// edition="las_vegas/2026"), then load.
    pub fn load_edition(repo_root: impl AsRef<Path>, festival: &str, edition: &str) -> anyhow::Result<Self> {
        let dir = repo_root.as_ref().join("data").join(festival).join(edition);
        Self::load(dir)
    }

    // ── stage paths (the canonical pipeline layout) ───────────────────────────────────────────
    pub fn clean_dir(&self) -> PathBuf { self.edition_dir.join("10_clean") }
    pub fn structured_dir(&self) -> PathBuf { self.edition_dir.join("15_structured") }
    pub fn synth_dir(&self) -> PathBuf { self.edition_dir.join("20_synthetic") }
    pub fn hydrated_dir(&self) -> PathBuf { self.edition_dir.join("25_hydrated") }

    // ── optional drop-in data that unblocks relation tiers ────────────────────────────────────
    pub fn venue_map_path(&self) -> PathBuf { self.edition_dir.join("venue_map.json") }
    pub fn set_times_path(&self) -> PathBuf { self.edition_dir.join("set_times.jsonl") }
    pub fn has_venue_map(&self) -> bool { self.venue_map_path().exists() }
    pub fn has_set_times(&self) -> bool { self.set_times_path().exists() }

    // ── stage helpers ─────────────────────────────────────────────────────────────────────────
    pub fn camel(&self, slug: &str) -> String {
        self.stages.get(slug).map(|s| s.camel.clone())
            .unwrap_or_else(|| title_case(slug))
    }
    pub fn spaced(&self, slug: &str) -> String {
        self.stages.get(slug).map(|s| s.spaced.clone())
            .unwrap_or_else(|| title_case(slug))
    }
    pub fn day_date(&self, day: &str) -> Option<&str> {
        self.days.get(day).map(|d| d.date.as_str())
    }
    /// "Friday (May 15)" or just the day if no date is known.
    pub fn day_dated(&self, day: &str) -> String {
        match self.day_date(day) {
            Some(d) => format!("{day} ({d})"),
            None => day.to_string(),
        }
    }
}

fn title_case(slug: &str) -> String {
    slug.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
