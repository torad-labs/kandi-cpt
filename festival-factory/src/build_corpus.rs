//! Port of scripts/build_v6_corpus.py — assemble the V6 relational corpus (relations.jsonl).
//!
//! V4 base (relations.jsonl.pre-equalize.bak) with refusal/meta + confusable CO-MENTION lines stripped,
//! plus heavy disjoint-context over-exposure per confusable group (150/group, cycling the skeletons) and
//! the authoritative set-time floor. The co-mention strip + disjoint gen are the v3 guardrail in action.
//! Validated by set-diffing the emitted texts against the committed Python relations.jsonl.
use serde_json::Value;

use crate::config::FestivalConfig;
use crate::triples::TripleStore;
use crate::{disjoint, settimes};

// Refusal/meta contamination (lowercased substrings; matches the Python REFUSAL alternation, case-insensitive).
const REFUSAL: &[&str] = &[
    "cannot fulfill", "i cannot", "without inventing", "the provided facts", "given facts",
    "as an ai", "i am unable", "i'm unable", "does not specify", "do not specify", "not specified in",
    "strict faithfulness", "i can only", "cannot generate", "no information", "unable to comply",
];

const CONFUSABLE_TARGET: usize = 150;

pub struct BuildResult {
    pub texts: Vec<String>, // full corpus texts (with disjoint repetition), order: base + disjoint + settime
    pub n_base: usize,
    pub n_disjoint: usize,
    pub n_settime: usize,
    pub dropped_refusal: usize,
    pub stripped_comention: usize,
}

pub fn build(cfg: &FestivalConfig, store: &TripleStore) -> anyhow::Result<BuildResult> {
    // 1) v4 base: skip refusal/meta, then strip confusable co-mention violations (the v3 guardrail).
    let base_path = cfg.hydrated_dir().join("relations.jsonl.pre-equalize.bak");
    let mut base: Vec<String> = Vec::new();
    let (mut dropped_refusal, mut stripped) = (0usize, 0usize);
    for line in std::fs::read_to_string(&base_path)?.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)?;
        let text = v.get("text").and_then(Value::as_str).unwrap_or("");
        let low = text.to_lowercase();
        if REFUSAL.iter().any(|p| low.contains(p)) {
            dropped_refusal += 1;
            continue;
        }
        if disjoint::violates(cfg, text).is_some() {
            stripped += 1;
            continue;
        }
        base.push(text.to_string());
    }

    // 2) disjoint-context over-exposure: 150/group, cycling the distinct skeleton variants.
    let mut disjoint_texts: Vec<String> = Vec::new();
    for g in &cfg.confusable_groups {
        let variants = disjoint::generate_for_group(cfg, g);
        if variants.is_empty() {
            continue;
        }
        let n = variants.len().max(CONFUSABLE_TARGET);
        for i in 0..n {
            disjoint_texts.push(variants[i % variants.len()].text.clone());
        }
    }

    // 3) authoritative set-time floor.
    let settime_texts: Vec<String> =
        settimes::set_time_lines(cfg, store)?.into_iter().map(|l| l.text).collect();

    let (n_base, n_disjoint, n_settime) = (base.len(), disjoint_texts.len(), settime_texts.len());
    let mut texts = base;
    texts.extend(disjoint_texts);
    texts.extend(settime_texts);
    Ok(BuildResult {
        texts,
        n_base,
        n_disjoint,
        n_settime,
        dropped_refusal,
        stripped_comention: stripped,
    })
}
