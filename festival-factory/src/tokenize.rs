//! Port of scripts/tokenize_corpus.py + pipeline/span_match.py — pack the corpus into seq_len blocks
//! with the lever metadata (entity_mask / src / fact_ids) the Python trainer consumes.
//!
//! Validation: the packing order (Python seeds random.shuffle) is NOT reproduced — block CONTENTS differ
//! by order — but the order-independent stats (total_tokens, num_chunks, entity_masked, src dist,
//! tokens_by_tag) are byte-identical to corpus_v1_stats.json, and per-doc input_ids/entity_mask match.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, Int8Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use regex::Regex;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::config::FestivalConfig;
use crate::triples::TripleStore;

// source-type ids (Lever 2), mirroring tokenize_corpus.py SRC.
fn src_of(path: &str) -> i8 {
    let p = path.to_lowercase();
    if p.contains("qa_pairs__") {
        1 // qa
    } else if p.contains("friend_chat__") || p.contains("/_cultural/") {
        2 // conversation
    } else {
        0 // raw_knowledge
    }
}

pub struct Doc {
    pub tag: &'static str,
    pub path: String,
    pub text: String,
}

/// Span-matcher vocabulary: artists (lineup + set-times + entities), stages (camel+spaced), days, dates.
pub struct SpanMatcher {
    re: Regex,
    time_re: Regex,
    fact_artists: Vec<(String, String)>, // (lowercased artist, fact_id "artist|stage|day")
}

impl SpanMatcher {
    pub fn build(cfg: &FestivalConfig, store: &TripleStore) -> anyhow::Result<Self> {
        let mut terms: Vec<String> = Vec::new();
        // artists: lineup triples + set-times schedule + entities
        for p in store.plays_at() {
            terms.push(p.artist);
        }
        let st_path = cfg.edition_dir.join("set_times.jsonl");
        if st_path.exists() {
            for l in std::fs::read_to_string(&st_path)?.lines() {
                if let Ok(v) = serde_json::from_str::<Value>(l) {
                    if let Some(a) = v.get("artist").and_then(Value::as_str) {
                        terms.push(a.to_string());
                        terms.push(title_case(a));
                    }
                }
            }
        }
        for e in store.entities.values() {
            if !e.name_canonical.is_empty() {
                terms.push(e.name_canonical.clone());
            }
        }
        // stages camel + spaced
        for s in cfg.stages.values() {
            terms.push(s.camel.clone());
            terms.push(s.spaced.clone());
        }
        // days + dates
        for (day, di) in &cfg.days {
            terms.push(day.clone());
            terms.push(di.date.clone());
        }
        // dedup, drop <4 chars, longest-first (so the alternation prefers the longest match)
        terms.retain(|t| t.chars().count() >= 4);
        terms.sort();
        terms.dedup();
        terms.sort_by(|a, b| b.len().cmp(&a.len()));
        let pat = terms.iter().map(|t| regex::escape(t)).collect::<Vec<_>>().join("|");
        // Rust regex has no lookbehind; \b approximates Python's (?<!\w)...(?!\w) for the alnum-edged vocab.
        let re = Regex::new(&format!(r"(?i)\b(?:{pat})\b"))?;
        let time_re = Regex::new(r"(?i)\b\d{1,2}:\d{2}\s*(?:AM|PM)\b")?;

        // fact index: (lowercased artist, fact_id) — a doc "touches" a fact if the artist appears in it.
        let mut fact_artists = Vec::new();
        for p in store.plays_at() {
            let day = if p.day.is_empty() { "na".to_string() } else { p.day.clone() };
            fact_artists.push((p.artist.to_lowercase(), format!("{}|{}|{}", p.artist, p.stage_slug, day)));
        }
        Ok(Self { re, time_re, fact_artists })
    }

    /// char/byte spans of matched entity surface forms (byte offsets into `text`, consistent with token offsets).
    fn spans(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self.re.find_iter(text).map(|m| (m.start(), m.end())).collect();
        out.extend(self.time_re.find_iter(text).map(|m| (m.start(), m.end())));
        out
    }

    /// fact_ids whose artist surface form appears in `text` (case-insensitive).
    pub fn fact_ids_in(&self, text: &str) -> Vec<String> {
        let low = text.to_lowercase();
        self.fact_artists.iter().filter(|(a, _)| low.contains(a.as_str())).map(|(_, f)| f.clone()).collect()
    }
}

fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut sow = true;
    for c in s.chars() {
        if c.is_alphabetic() {
            if sow { out.extend(c.to_uppercase()); } else { out.extend(c.to_lowercase()); }
            sow = false;
        } else {
            out.push(c);
            sow = true;
        }
    }
    out
}

/// Per-token entity_mask (0/1): 1 if a token's byte range overlaps any entity span.
fn entity_mask(offsets: &[(usize, usize)], spans: &[(usize, usize)]) -> Vec<i8> {
    offsets
        .iter()
        .map(|&(ts, te)| {
            if te <= ts {
                0
            } else if spans.iter().any(|&(ss, se)| ts < se && ss < te) {
                1
            } else {
                0
            }
        })
        .collect()
}

fn read_keep(repo_root: &Path) -> Option<std::collections::HashSet<String>> {
    let p = repo_root.join("data/_build/curation/keep_list.json");
    let raw = std::fs::read_to_string(p).ok()?;
    let v: Vec<String> = serde_json::from_str(&raw).ok()?;
    Some(v.into_iter().collect())
}

pub fn collect_docs(cfg: &FestivalConfig, repo_root: &Path) -> anyhow::Result<Vec<Doc>> {
    let keep = read_keep(repo_root);
    let mut docs = Vec::new();
    // 10_clean (keep-list filtered)
    for entry in walk_md(&cfg.clean_dir()) {
        if entry.file_name().and_then(|s| s.to_str()) == Some("README.md") {
            continue;
        }
        if let Some(k) = &keep {
            let rel = entry.strip_prefix(repo_root).unwrap_or(&entry).to_string_lossy().to_string();
            if !k.contains(&rel) {
                continue;
            }
        }
        let t = std::fs::read_to_string(&entry).unwrap_or_default().trim().to_string();
        if !t.is_empty() {
            docs.push(Doc { tag: "clean", path: entry.to_string_lossy().into(), text: t });
        }
    }
    // 20_synthetic (skip _facts/_rosters)
    for entry in walk_md(&cfg.synth_dir()) {
        let p = entry.to_string_lossy().to_string();
        if p.contains("/_facts/") || p.contains("/_rosters/") {
            continue;
        }
        let t = std::fs::read_to_string(&entry).unwrap_or_default().trim().to_string();
        if !t.is_empty() {
            docs.push(Doc { tag: "synthetic", path: p, text: t });
        }
    }
    // 25_hydrated/*.jsonl — one doc per row's text
    for entry in walk_ext(&cfg.hydrated_dir(), "jsonl") {
        let p = entry.to_string_lossy().to_string();
        for line in std::fs::read_to_string(&entry).unwrap_or_default().lines() {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if let Some(t) = v.get("text").and_then(Value::as_str) {
                    let t = t.trim();
                    if !t.is_empty() {
                        docs.push(Doc { tag: "hydrated", path: p.clone(), text: t.to_string() });
                    }
                }
            }
        }
    }
    // _shared/knowledge
    let shared = repo_root.join("data/_shared/knowledge");
    for entry in walk_md(&shared) {
        let t = std::fs::read_to_string(&entry).unwrap_or_default().trim().to_string();
        if !t.is_empty() {
            docs.push(Doc { tag: "knowledge", path: entry.to_string_lossy().into(), text: t });
        }
    }
    Ok(docs)
}

fn walk_md(dir: &Path) -> Vec<PathBuf> {
    walk_ext(dir, "md")
}

fn walk_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return out;
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

pub struct Block {
    pub input_ids: Vec<i32>,
    pub entity_mask: Vec<i8>,
    pub src: Vec<i8>,
    pub fact_ids: Vec<String>, // distinct fact_ids touching this block
}

pub struct TokenizeResult {
    pub blocks: Vec<Block>,
    pub total_tokens: usize,
    pub tokens_by_tag: BTreeMap<String, usize>,
    pub entity_masked: usize,
    pub src_dist: BTreeMap<i8, usize>,
}

pub fn tokenize(cfg: &FestivalConfig, store: &TripleStore, repo_root: &Path, seq_len: usize, tokenizer_path: &Path) -> anyhow::Result<TokenizeResult> {
    let tok = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let bos = tok.token_to_id("<bos>").unwrap_or(2) as i32;
    let eos = tok.token_to_id("<eos>").unwrap_or(1) as i32;
    let sm = SpanMatcher::build(cfg, store)?;
    let docs = collect_docs(cfg, repo_root)?;

    let mut all_ids: Vec<i32> = Vec::new();
    let mut all_mask: Vec<i8> = Vec::new();
    let mut all_src: Vec<i8> = Vec::new();
    let mut doc_ranges: Vec<(usize, usize, Vec<String>)> = Vec::new(); // (tok_start, tok_end, fact_ids)
    let mut tokens_by_tag: BTreeMap<String, usize> = BTreeMap::new();
    let mut entity_masked = 0usize;

    for d in &docs {
        let enc = tok.encode(d.text.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let offs: Vec<(usize, usize)> = enc.get_offsets().to_vec();
        let spans = sm.spans(&d.text);
        let mask = entity_mask(&offs, &spans);
        let src = src_of(&d.path);
        let fids = sm.fact_ids_in(&d.text);
        let start = all_ids.len();
        all_ids.push(bos);
        all_mask.push(0);
        all_src.push(src);
        all_ids.extend(&ids);
        all_mask.extend(&mask);
        all_src.extend(std::iter::repeat(src).take(ids.len()));
        all_ids.push(eos);
        all_mask.push(0);
        all_src.push(src);
        doc_ranges.push((start, all_ids.len(), fids));
        entity_masked += mask.iter().filter(|&&m| m == 1).count();
        *tokens_by_tag.entry(d.tag.to_string()).or_default() += ids.len();
    }

    let total_tokens = all_ids.len();
    let n_blocks = total_tokens / seq_len;
    // per-block fact_ids via two-pointer over doc_ranges (both ordered)
    let block_facts = block_fact_ids(&doc_ranges, n_blocks, seq_len);
    let mut blocks = Vec::with_capacity(n_blocks);
    let mut src_dist: BTreeMap<i8, usize> = BTreeMap::new();
    for s in &all_src {
        *src_dist.entry(*s).or_default() += 1;
    }
    for b in 0..n_blocks {
        let (lo, hi) = (b * seq_len, (b + 1) * seq_len);
        blocks.push(Block {
            input_ids: all_ids[lo..hi].to_vec(),
            entity_mask: all_mask[lo..hi].to_vec(),
            src: all_src[lo..hi].to_vec(),
            fact_ids: block_facts[b].clone(),
        });
    }
    Ok(TokenizeResult { blocks, total_tokens, tokens_by_tag, entity_masked, src_dist })
}

/// Write the packed blocks as a plain Arrow IPC file (columns input_ids/entity_mask/src/fact_ids).
/// The Python trainer reads this via a small loader change (datasets.Dataset.from_file) instead of
/// load_from_disk — avoids replicating the version-coupled HF datasets on-disk format.
pub fn write_ipc(res: &TokenizeResult, path: &Path) -> anyhow::Result<()> {
    // ListBuilder emits nullable inner items by default — the schema must agree (nullable: true).
    let schema = Arc::new(Schema::new(vec![
        Field::new("input_ids", DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), false),
        Field::new("entity_mask", DataType::List(Arc::new(Field::new("item", DataType::Int8, true))), false),
        Field::new("src", DataType::List(Arc::new(Field::new("item", DataType::Int8, true))), false),
        Field::new("fact_ids", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), false),
    ]));
    let mut ids_b = ListBuilder::new(Int32Builder::new());
    let mut mask_b = ListBuilder::new(Int8Builder::new());
    let mut src_b = ListBuilder::new(Int8Builder::new());
    let mut fact_b = ListBuilder::new(StringBuilder::new());
    for b in &res.blocks {
        ids_b.values().append_slice(&b.input_ids);
        ids_b.append(true);
        mask_b.values().append_slice(&b.entity_mask);
        mask_b.append(true);
        src_b.values().append_slice(&b.src);
        src_b.append(true);
        for f in &b.fact_ids {
            fact_b.values().append_value(f);
        }
        fact_b.append(true);
    }
    let cols: Vec<ArrayRef> = vec![
        Arc::new(ids_b.finish()),
        Arc::new(mask_b.finish()),
        Arc::new(src_b.finish()),
        Arc::new(fact_b.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), cols)?;
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    // STREAM format (not File) — this is what HF datasets' .arrow files use, so datasets.Dataset.from_file
    // reads the Rust output natively (the trainer's make_loader dual-format path).
    let mut w = StreamWriter::try_new(std::fs::File::create(path)?, &schema)?;
    w.write(&batch)?;
    w.finish()?;
    Ok(())
}

fn block_fact_ids(ranges: &[(usize, usize, Vec<String>)], n_blocks: usize, seq_len: usize) -> Vec<Vec<String>> {
    let mut out = vec![Vec::new(); n_blocks];
    let mut lo = 0;
    for b in 0..n_blocks {
        let (bs, be) = (b * seq_len, (b + 1) * seq_len);
        while lo < ranges.len() && ranges[lo].1 <= bs {
            lo += 1;
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut j = lo;
        while j < ranges.len() && ranges[j].0 < be {
            if ranges[j].1 > bs {
                for f in &ranges[j].2 {
                    seen.insert(f.clone());
                }
            }
            j += 1;
        }
        out[b] = seen.into_iter().collect();
    }
    out
}
