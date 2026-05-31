//! Port of pipeline/disjoint_context.py — V6 distributional-hypothesis disjoint-context generation.
//!
//! Confusable entities share a stage+slot (kineticFIELD 4:14 sunrise: Charlotte=Fri, A&B=Sat, Armin=Sun;
//! cosmicMEADOW: Underworld=Fri, the trio=Sun) and are separated ONLY by day + a DISTINCT vocabulary
//! world per group. THE HARD RULE (violates): never co-mention two confusable groups, nor a confusable
//! entity with another's day/stage. Skeletons are context-first (distinctive block leads, day last).
use std::collections::{BTreeMap, BTreeSet};

use crate::config::{ConfusableGroup, FestivalConfig};

// Distinctive vocabulary WORLDS per group key. {a}=entity, {c}=stage camel, {dd}=day(date), {FEST}=name.
const SKELETONS: &[(&str, &[&str])] = &[
    ("armin_sunday_finale", &[
        "The grand finale of the entire weekend, an uplifting trance send-off as the last sunrise breaks — {a} on the {c} stage, {dd}.",
        "Closing night, the festival's very last main-stage moment before it ends: {a} headlines {c}, {dd}.",
        "An uplifting close to three days, the dawn farewell of {FEST}: {a} takes {c}, {dd}.",
        "Last set of the whole weekend, a Sunday-into-dawn trance send-off: {a} on {c}, {dd}.",
        "The weekend's grand finale, uplifting and trance-driven as the final sunrise comes — {a} at {c}, {dd}.",
        "When the very last sunrise of the festival arrives, the closing finale belongs to {a}: {c}, {dd}.",
        "The festival's final breath, an uplifting closing-night farewell: {a} headlines {c}, {dd}.",
        "A grand uplifting send-off to end three days at {FEST}: {a} on the {c} stage, {dd}.",
    ]),
    ("aboveandbeyond_saturday_heart", &[
        "Saturday's sunrise set at the {c} stage — anthemic, hands-in-the-air group therapy: {a}, {dd}.",
        "The {c} sunrise on {dd} belongs to {a}: the weekend's emotional, tearful midpoint.",
        "At the {c} stage, the Saturday sunrise is {a} — anthemic singalongs, the heart of the three days, {dd}.",
        "{a} plays the {c} sunrise set on {dd}, the mid-festival emotional centerpiece — group therapy at dawn.",
        "The Saturday (May 16) sunrise: {a} on {c}, anthemic and full of singalongs, the weekend's peak.",
        "On {dd}, {a} delivers the {c} sunrise set — the tearful midpoint of {FEST}.",
        "{a}'s {c} sunrise is Saturday's centerpiece: anthemic, emotional, hands in the air, {dd}.",
        "The mid-festival {c} sunrise on {dd} is {a} — group therapy, singalongs, the emotional core.",
    ]),
    ("underworld_friday_opening", &[
        "Opening night, the launch of the entire weekend by a veteran live act — {a} on the {c} stage, {dd}.",
        "The festival's first night, the set that kicks everything off: {a} headlines {c}, {dd}.",
        "Before anything else happens at {FEST}, the opening live performance: {a} on {c}, {dd}.",
        "The weekend's launch, a veteran live show christening the festival: {a} takes {c}, {dd}.",
        "First night, first headline set, the very start of three days: {a} opens {c}, {dd}.",
        "Kicking off the whole weekend as the first sunrise nears: {a} on the {c} stage, {dd}.",
        "The opening-night launch of {FEST} by a live electronic veteran: {a} on {c}, {dd}.",
        "The very first headline moment of the festival, opening night: {a} at {c}, {dd}.",
    ]),
    ("cosmic_sunday_melodic", &[
        "Closing day, the final-day melodic-bass showcase of the weekend — {a} on the {c} stage, {dd}.",
        "The last day of the festival, a melodic lineup in full: {a} headlines {c}, {dd}.",
        "Final-day melodic bass, the closing session of three days: {a} on {c}, {dd}.",
        "On the festival's last day, the melodic-bass showcase: {a} takes {c}, {dd}.",
        "Closing-day melodic bass at {FEST}, the final session of the weekend: {a} on {c}, {dd}.",
        "The weekend's last day, a melodic showcase to close it out: {a} headlines {c}, {dd}.",
    ]),
];

// day -> lowercased surface tokens (name + date), for the co-occurrence audit.
const DAY_TOKENS: &[(&str, &[&str])] = &[
    ("Friday", &["friday", "may 15"]),
    ("Saturday", &["saturday", "may 16"]),
    ("Sunday", &["sunday", "may 17"]),
];

pub struct DisjointLine {
    pub text: String,
    pub entity_ids: Vec<String>,
    pub context_profile: String,
}

fn camel(cfg: &FestivalConfig, slug: &str) -> String {
    cfg.stages.get(slug).map(|s| s.camel.clone()).unwrap_or_else(|| slug.to_string())
}

fn daydate(cfg: &FestivalConfig, day: &str) -> String {
    match cfg.days.get(day) {
        Some(d) if !d.date.is_empty() => format!("{day} ({})", d.date),
        _ => day.to_string(),
    }
}

fn skeletons_for(key: &str) -> &'static [&'static str] {
    SKELETONS.iter().find(|(k, _)| *k == key).map(|(_, s)| *s).unwrap_or(&[])
}

/// Distinct lines for one group (each member × each skeleton), in the group's own vocabulary world.
pub fn generate_for_group(cfg: &FestivalConfig, g: &ConfusableGroup) -> Vec<DisjointLine> {
    let c = camel(cfg, &g.stage);
    let dd = daydate(cfg, &g.day);
    let fest = &cfg.name;
    let mut out = Vec::new();
    for a in &g.members {
        for sk in skeletons_for(&g.key) {
            let text = sk
                .replace("{a}", a)
                .replace("{c}", &c)
                .replace("{dd}", &dd)
                .replace("{FEST}", fest);
            out.push(DisjointLine {
                text,
                entity_ids: vec![a.clone(), g.stage.clone(), g.day.clone()],
                context_profile: g.key.clone(),
            });
        }
    }
    out
}

/// member -> (stage_slug, day) and member -> group_key, over all confusable groups.
fn confusable_maps(cfg: &FestivalConfig) -> (BTreeMap<String, (String, String)>, BTreeMap<String, String>) {
    let mut conf = BTreeMap::new();
    let mut group_of = BTreeMap::new();
    for g in &cfg.confusable_groups {
        for m in &g.members {
            conf.insert(m.clone(), (g.stage.clone(), g.day.clone()));
            group_of.insert(m.clone(), g.key.clone());
        }
    }
    (conf, group_of)
}

/// THE v3 guardrail: returns a reason if `text` co-mentions confusable groups, or a confusable entity
/// with the wrong day / wrong stage; None if clean. Iterates `present` in the SAME order as the Python
/// (membership order across groups) so present[0] matches.
pub fn violates(cfg: &FestivalConfig, text: &str) -> Option<String> {
    let low = text.to_lowercase();
    let (conf, group_of) = confusable_maps(cfg);
    // membership order: groups in config order, members in group order (matches Python's GROUPS iteration)
    let mut present: Vec<String> = Vec::new();
    for g in &cfg.confusable_groups {
        for m in &g.members {
            if low.contains(&m.to_lowercase()) {
                present.push(m.clone());
            }
        }
    }
    if present.is_empty() {
        return None;
    }
    let groups: BTreeSet<&String> = present.iter().map(|e| &group_of[e]).collect();
    if groups.len() >= 2 {
        return Some(format!("co-mentions different confusable groups: {:?}", present));
    }
    let (good_stage, good_day) = &conf[&present[0]];
    for (day, toks) in DAY_TOKENS {
        if day != good_day && toks.iter().any(|t| low.contains(t)) {
            return Some(format!("{} (day={good_day}) co-mentions wrong day {day}", present[0]));
        }
    }
    for (slug, st) in &cfg.stages {
        if slug != good_stage && st.camel.len() > 3 && low.contains(&st.camel.to_lowercase()) {
            return Some(format!("{} (stage={good_stage}) co-mentions wrong stage {}", present[0], st.camel));
        }
    }
    None
}
