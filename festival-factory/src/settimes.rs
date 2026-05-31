//! Port of pipeline/set_times_lines.py — the authoritative set-time FLOOR from set_times.jsonl.
//!
//! Context-first / artist-LAST ordering: the same stage+slot hosts a different artist each day, so the
//! discriminating context (slot → day → time → stage) must precede the performer and the artist must be
//! the final token — engineering P(artist | context) for retrieval, not the reverse. Each row emits two
//! deterministic lines; the 4:14 AM closers are tagged SUNRISE SET. Text strings are byte-identical to the
//! Python port (validated by diffing the `set_time` lines in 25_hydrated/relations.jsonl).
use std::collections::BTreeMap;

use serde::Deserialize;

use crate::config::FestivalConfig;
use crate::triples::TripleStore;

#[derive(Deserialize)]
struct SetRow {
    artist: String,
    #[serde(default)]
    stage: String,
    #[serde(default)]
    stage_raw: String,
    #[serde(default)]
    day: String,
    #[serde(default)]
    start: String,
    #[serde(default)]
    end: String,
    #[serde(default)]
    tags: Vec<String>,
}

pub struct SetTimeLine {
    pub text: String,
    pub entity_ids: Vec<String>,
}

/// Python str.title(): capitalize the first alphabetic char of each run, lowercase the rest. Only a
/// fallback — known artists resolve via the entity canonical map first.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut start_of_word = true;
    for c in s.chars() {
        if c.is_alphabetic() {
            if start_of_word {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            start_of_word = false;
        } else {
            out.push(c);
            start_of_word = true;
        }
    }
    out
}

pub fn set_time_lines(cfg: &FestivalConfig, store: &TripleStore) -> anyhow::Result<Vec<SetTimeLine>> {
    let path = cfg.edition_dir.join("set_times.jsonl");
    if !path.exists() {
        return Ok(vec![]);
    }
    // canonical map: lowercased name_canonical -> name_canonical (set_times artist names are UPPERCASE)
    let by_lower: BTreeMap<String, String> = store
        .entities
        .values()
        .filter(|e| !e.name_canonical.is_empty())
        .map(|e| (e.name_canonical.to_lowercase(), e.name_canonical.clone()))
        .collect();
    let fest = &cfg.name;
    let mut out = Vec::new();
    for line in std::fs::read_to_string(&path)?.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: SetRow = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let a = by_lower
            .get(r.artist.trim().to_lowercase().as_str())
            .cloned()
            .unwrap_or_else(|| title_case(&r.artist));
        // Use stage_raw (camelCase as scheduled); fall back to the config stage's camel form.
        let c = if !r.stage_raw.is_empty() {
            r.stage_raw.clone()
        } else {
            cfg.stages.get(&r.stage).map(|s| s.camel.clone()).unwrap_or_else(|| r.stage.clone())
        };
        // Festival-day date (Saturday→May 16), NOT set_times' clock start_date (a 4:14 AM set clocks next morning).
        let date = cfg.days.get(&r.day).map(|d| d.date.clone()).unwrap_or_default();
        let (start, end, day) = (&r.start, &r.end, &r.day);
        if a.is_empty() || c.is_empty() || day.is_empty() || start.is_empty() {
            continue;
        }
        let dd = if !date.is_empty() { format!("{day} ({date})") } else { day.clone() };
        let eid = vec![a.clone(), r.stage.clone(), day.clone()];
        let sunrise = r.tags.iter().any(|t| t == "SUNRISE SET");
        if sunrise {
            out.push(SetTimeLine {
                text: format!("At {fest}, the sunrise set at the {c} stage on {dd}, from {start} to {end}, is played by {a}."),
                entity_ids: eid.clone(),
            });
            out.push(SetTimeLine {
                text: format!("The {dd} sunrise set at {c} ({start}\u{2013}{end}) belongs to {a}."),
                entity_ids: eid,
            });
        } else {
            out.push(SetTimeLine {
                text: format!("At {fest}, the {dd} set at the {c} stage, {start} to {end}, is played by {a}."),
                entity_ids: eid.clone(),
            });
            out.push(SetTimeLine {
                text: format!("The {dd} {start}\u{2013}{end} slot at the {c} stage belongs to {a}."),
                entity_ids: eid,
            });
        }
    }
    Ok(out)
}
