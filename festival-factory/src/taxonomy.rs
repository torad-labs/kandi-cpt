//! Typed relation taxonomy — the single source of truth for COVERAGE (build spec Part 1-2 / Part 5).
//! Ported from scripts/festival_factory/taxonomy.py. Each relation is tagged BASE/DERIVED and
//! supported/partial/blocked; the engine populates supported+partial, the coverage gate reports
//! density vs floor and lists blocked cells with the data file that would unblock them.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier { Base, Derived }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataStatus { Supported, Partial, Blocked }

#[derive(Debug, Clone)]
pub struct Relation {
    pub key: &'static str,
    pub group: &'static str,
    pub tier: Tier,
    pub status: DataStatus,
    pub directions: &'static [&'static str],
    pub query_orders: &'static [&'static str],
    pub needs: &'static [&'static str],
    pub note: &'static str,
}

use Tier::*;
use DataStatus::*;

const FB: &[&str] = &["forward", "reverse"];
const SYM: &[&str] = &["symmetric"];

macro_rules! rel {
    ($key:literal, $group:literal, $tier:expr, $status:expr, $dirs:expr, $orders:expr, $needs:expr, $note:literal) => {
        Relation { key: $key, group: $group, tier: $tier, status: $status,
                   directions: $dirs, query_orders: $orders, needs: $needs, note: $note }
    };
}

/// The full registry. data_status reflects the EDC-LV-2026 Step-0 inventory; other festivals
/// re-inventory via their own festival_meta + optional venue_map.json / set_times.jsonl.
pub fn relations() -> Vec<Relation> {
    vec![
        // ── Spatial — all blocked: no map/distances in source ──
        rel!("stage_to_stage", "spatial", Base, Blocked, SYM, &["leads_with_stage_a","leads_with_stage_b"], &["venue_map.json"], "distance/walking_time/route — ~45 pairs"),
        rel!("stage_to_amenity", "spatial", Base, Blocked, FB, &[], &["venue_map.json"], "nearest water/medical/bathroom/food/ATM/exit"),
        rel!("stage_to_landmark", "spatial", Base, Blocked, FB, &[], &["venue_map.json"], "ferris wheel, art cars, owl, meetup anchors"),
        rel!("venue_zone", "spatial", Base, Blocked, FB, &[], &["venue_map.json"], "which stages sit in which area + adjacency"),
        rel!("arrival_path", "spatial", Base, Blocked, FB, &[], &["venue_map.json"], "camping/parking/rideshare → entrance → stage"),
        rel!("viewing_geography", "spatial", Base, Blocked, FB, &[], &["venue_map.json"], "good spots per stage, crowd-crush zones"),
        // ── Temporal ──
        rel!("artist_to_day", "temporal", Base, Supported, FB, &["leads_with_artist","leads_with_day"], &[], "the v2 failure — now date-anchored + MC-framed + rebalanced"),
        rel!("day_of_week_map", "temporal", Base, Supported, SYM, &["leads_with_day","leads_with_date"], &[], "Friday=May 15, etc."),
        rel!("day_structure", "temporal", Base, Partial, FB, &[], &["set_times.jsonl"], "gates/peak/closing absent; sunrise + opening 5pm present"),
        rel!("special_slot", "temporal", Base, Partial, FB, &["leads_with_artist","leads_with_slot"], &["set_times.jsonl"], "sunrise sets + opening ceremony + b2b slots present"),
        rel!("set_time", "temporal", Base, Blocked, FB, &["leads_with_artist","leads_with_time"], &["set_times.jsonl"], "per-artist clock times — app-only; primitive for conflict+travel"),
        rel!("set_order", "temporal", Base, Blocked, FB, &[], &["set_times.jsonl"], "who plays before/after on a stage"),
        rel!("cross_stage_conflict", "temporal", Derived, Blocked, FB, &[], &["set_times.jsonl"], "most-queried; install set_time densely + teach comparison"),
        rel!("travel_feasibility", "temporal", Derived, Blocked, FB, &[], &["set_times.jsonl","venue_map.json"], "the specialist capability; both primitives absent"),
        rel!("year_over_year", "temporal", Base, Partial, FB, &[], &[], "historical 'have headlined' lists (qualified past, NOT 2026)"),
        // ── Artist ──
        rel!("artist_to_stage", "artist", Base, Supported, FB, &["leads_with_artist","leads_with_stage"], &[], "the lineup spine — dense, both directions"),
        rel!("in_lineup", "artist", Base, Supported, &["forward"], &["leads_with_artist"], &[], "stageless lineup membership (the plays_at→festival coarsening) — the lineup spine where the stage is unknown"),
        rel!("b2b_pairing", "artist", Base, Supported, SYM, &[], &[], "two artists/one slot — breaks naive retrieval if missing"),
        rel!("set_style", "artist", Base, Partial, FB, &[], &[], "DJ set/live/sunrise — sparse but present"),
        rel!("artist_tier", "artist", Base, Partial, FB, &[], &[], "headliner vs undercard — coarse signal"),
        rel!("artist_to_genre", "artist", Base, Blocked, FB, &[], &[], "not stated directly; only via artist→stage→genre bridge"),
        rel!("artist_similarity", "artist", Derived, Blocked, FB, &[], &[], "needs artist_to_genre (absent)"),
        rel!("artist_alias", "artist", Base, Blocked, SYM, &[], &[], "side projects — not in source"),
        rel!("artist_label", "artist", Base, Blocked, FB, &[], &[], "label/collaboration — not in source"),
        // ── Stage identity ──
        rel!("stage_to_genre", "stage", Base, Supported, FB, &["leads_with_stage","leads_with_genre"], &[], "the HUB relation — bridges artist↔stage↔recommendation"),
        rel!("stage_brand", "stage", Base, Supported, FB, &["leads_with_stage","leads_with_brand"], &[], "Factory93→neon, Dreamstate→quantum, Bassrush→bass, Basscon→waste"),
        rel!("stage_rank_capacity", "stage", Base, Supported, FB, &[], &[], "Kinetic largest (~70k), Circuit second-largest"),
        rel!("stage_production_character", "stage", Base, Supported, &["forward"], &[], &[], "visuals a stage is known for (LED, fire, owl)"),
        rel!("stage_artist_fit", "stage", Derived, Supported, FB, &[], &[], "which artists fit a stage — via genre bridge; teach, don't enumerate"),
        // ── Experiential ──
        rel!("culture", "experiential", Base, Supported, FB, &[], &[], "kandi/PLUR/etiquette/first-timer"),
        rel!("venue_rules", "experiential", Base, Supported, FB, &[], &[], "re-entry, allowed items, lockers"),
        rel!("installations", "experiential", Base, Supported, FB, &[], &[], "roaming art cars"),
        rel!("crowd_density", "experiential", Derived, Blocked, FB, &[], &["set_times.jsonl"], "by stage by time — needs set_time"),
        rel!("heat_shade", "experiential", Base, Blocked, FB, &[], &["venue_map.json"], "time-of-day shade — needs orientation"),
        // ── Planning / derived — install primitives, teach the op, NEVER enumerate ──
        rel!("conflict_resolution", "planning", Derived, Blocked, FB, &[], &["set_times.jsonl","venue_map.json"], "worked examples only, once primitives exist"),
        rel!("schedule_building", "planning", Derived, Blocked, FB, &[], &["set_times.jsonl","venue_map.json"], ""),
        rel!("onboarding_path", "planning", Derived, Blocked, FB, &[], &["venue_map.json"], ""),
        rel!("pacing", "planning", Derived, Blocked, FB, &[], &["set_times.jsonl"], ""),
        rel!("recommendation", "planning", Derived, Supported, FB, &[], &[], "genre+tier → suggestions; teachable from supported primitives"),
    ]
}

impl Relation {
    pub fn is_active(&self) -> bool {
        matches!(self.status, DataStatus::Supported | DataStatus::Partial)
    }
}

/// data-file → relations it would unblock (the acquisition shopping list).
pub fn needs_summary() -> std::collections::BTreeMap<&'static str, Vec<&'static str>> {
    let mut out: std::collections::BTreeMap<&'static str, Vec<&'static str>> = Default::default();
    for r in relations() {
        for f in r.needs {
            out.entry(f).or_default().push(r.key);
        }
    }
    out
}
