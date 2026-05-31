//! The typed triple store — loaded from `15_structured/{entities,triples}.jsonl`. This is the
//! grounding source ("RAG") for all generation: every synthetic line must trace to triples held here,
//! and the faithfulness gate checks generated text against them. Nothing is generated that isn't here.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Entity {
    pub id: String,
    #[serde(default)]
    pub entity_type: String,
    #[serde(default)]
    pub name_canonical: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Triple {
    pub subject_id: String,
    pub predicate: String,
    /// Entity id OR a literal (string/number).
    pub object: serde_json::Value,
    #[serde(default)]
    pub attrs: BTreeMap<String, serde_json::Value>,
}

/// A resolved lineup fact: artist plays a stage on a day (day may be empty when the source states
/// the stage but no day, e.g. Basspod).
#[derive(Debug, Clone)]
pub struct PlaysAt {
    pub artist: String,
    pub stage_slug: String,
    pub day: String,
    pub triple_id: String, // synthetic id for provenance tagging
}

#[derive(Debug, Default)]
pub struct TripleStore {
    pub entities: BTreeMap<String, Entity>,
    pub triples: Vec<Triple>,
}

fn slug_tail(id: &str) -> &str {
    id.rsplit(':').next().unwrap_or(id)
}

impl TripleStore {
    pub fn load(structured_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = structured_dir.as_ref();
        let mut store = TripleStore::default();
        let ef = dir.join("entities.jsonl");
        if ef.exists() {
            for (i, line) in std::fs::read_to_string(&ef)?.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() { continue; }
                match serde_json::from_str::<Entity>(line) {
                    Ok(e) => { store.entities.insert(e.id.clone(), e); }
                    Err(e) => eprintln!("[triples] skip entities.jsonl:{}: {e}", i + 1),
                }
            }
        }
        let tf = dir.join("triples.jsonl");
        if tf.exists() {
            for (i, line) in std::fs::read_to_string(&tf)?.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() { continue; }
                match serde_json::from_str::<Triple>(line) {
                    Ok(t) => store.triples.push(t),
                    Err(e) => eprintln!("[triples] skip triples.jsonl:{}: {e}", i + 1),
                }
            }
        }
        Ok(store)
    }

    pub fn name_of(&self, id: &str) -> String {
        self.entities
            .get(id)
            .map(|e| e.name_canonical.clone())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| slug_tail(id).replace('_', " "))
    }

    /// Resolve all `plays_at` triples into (artist, stage_slug, day) facts — the lineup spine.
    pub fn plays_at(&self) -> Vec<PlaysAt> {
        let mut out = Vec::new();
        for (i, t) in self.triples.iter().enumerate() {
            if t.predicate != "plays_at" {
                continue;
            }
            let artist = self.name_of(&t.subject_id);
            let stage_slug = match &t.object {
                serde_json::Value::String(s) => slug_tail(s).to_string(),
                _ => continue,
            };
            let day = t
                .attrs
                .get("day")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if artist.is_empty() || stage_slug.is_empty() {
                continue;
            }
            out.push(PlaysAt { artist, stage_slug, day, triple_id: format!("plays_at:{i}") });
        }
        out
    }

    /// Distinct stage slugs present in the lineup.
    pub fn stages_present(&self) -> Vec<String> {
        let mut v: Vec<String> = self.plays_at().into_iter().map(|p| p.stage_slug).collect();
        v.sort();
        v.dedup();
        v
    }
}
