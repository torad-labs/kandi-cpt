"""The typed relation taxonomy — single source of truth for COVERAGE (build spec Part 1-2, Part 5 gate 1).

Each relation is tagged:
  tier        BASE (memorize via dense multi-form CPT) | DERIVED (install primitives + teach the
              operation; NEVER enumerate — Part 2, prevents O(artists^2) corpus explosion).
  data_status SUPPORTED (the source contains the facts → generate now) |
              BLOCKED (facts absent from source → generating = fabrication, forbidden by Part 3.5;
              emit a coverage GAP citing the data file that would unblock it) |
              PARTIAL (some instances present, e.g. special-slot times but not full set_times).
  directions  every relation must appear from EVERY entity as both source and target (reversal-curse
              mitigation, Part 3.1 / Berglund 2023) — listed query_orders enforce order-matching (3.3).

The generator iterates SUPPORTED relations; the coverage gate reports SUPPORTED density vs the floor
and lists BLOCKED cells with their `needs`. Drop the named data file beside festival_meta.json and a
BLOCKED relation flips to SUPPORTED with no code change — that is the factory contract.
"""
from __future__ import annotations

from dataclasses import dataclass

BASE, DERIVED = "BASE", "DERIVED"
SUPPORTED, BLOCKED, PARTIAL = "supported", "blocked", "partial"


@dataclass(frozen=True)
class Relation:
    key: str
    group: str                       # spatial | temporal | artist | stage | experiential | planning
    tier: str                        # BASE | DERIVED
    data_status: str                 # supported | blocked | partial
    directions: tuple = ("forward", "reverse")
    query_orders: tuple = ()         # surface token-orders to mirror (order-matched augmentation, 3.3)
    needs: tuple = ()                # data files that would unblock a BLOCKED/PARTIAL relation
    derives_from: tuple = ()         # for DERIVED: the BASE primitives to install instead of enumerating
    note: str = ""


def R(*a, **k) -> Relation:
    return Relation(*a, **k)


# ── the registry ──────────────────────────────────────────────────────────────────────────────────
# data_status reflects the Step-0 inventory of the EDC-LV-2026 source (lineup + 9 stage descriptions +
# culture present; NO map/distances, NO per-set clock times). Other festivals re-inventory and may flip
# statuses via their own festival_meta.json + optional venue_map.json / set_times.jsonl.
RELATIONS: dict[str, Relation] = {r.key: r for r in [
    # ── Spatial — ALL blocked: source has no map, no distances, no coordinates ───────────────────────
    R("stage_to_stage", "spatial", BASE, BLOCKED, ("symmetric",),
      ("leads_with_stage_a", "leads_with_stage_b"), needs=("venue_map.json",),
      note="distance / walking_time / canonical route — ~45 pairs; needs stage coordinates"),
    R("stage_to_amenity", "spatial", BASE, BLOCKED, ("forward", "reverse"),
      needs=("venue_map.json",), note="nearest water/medical/bathroom/food/ATM/exit per stage"),
    R("stage_to_landmark", "spatial", BASE, BLOCKED, needs=("venue_map.json",),
      note="ferris wheel, art cars, owl, meetup/orientation anchors"),
    R("venue_zone", "spatial", BASE, BLOCKED, needs=("venue_map.json",),
      note="which stages sit in which area + adjacency"),
    R("arrival_path", "spatial", BASE, BLOCKED, needs=("venue_map.json",),
      note="camping/parking/rideshare → entrance → stage, per segment"),
    R("viewing_geography", "spatial", BASE, BLOCKED, needs=("venue_map.json",),
      note="good spots per stage, crowd-crush zones"),

    # ── Temporal ─────────────────────────────────────────────────────────────────────────────────
    R("artist_to_day", "temporal", BASE, SUPPORTED, ("forward", "reverse"),
      ("leads_with_artist", "leads_with_day"), note="THE v2 failure — now date-anchored + MC-framed + rebalanced"),
    R("day_of_week_map", "temporal", BASE, SUPPORTED, ("symmetric",),
      ("leads_with_day", "leads_with_date"), note="Friday=May 15, etc. (calendar anchor)"),
    R("day_structure", "temporal", BASE, PARTIAL, needs=("set_times.jsonl",),
      note="gates-open / peak / closing windows absent; sunrise window + opening-ceremony 5pm present"),
    R("special_slot", "temporal", BASE, PARTIAL, ("forward", "reverse"),
      ("leads_with_artist", "leads_with_slot"), needs=("set_times.jsonl",),
      note="sunrise sets (Kinetic, ~4:30-5:30am) + opening ceremony (Cosmic, Fri 5pm) present; b2b slots present"),
    R("set_time", "temporal", BASE, BLOCKED, ("forward", "reverse"),
      ("leads_with_artist", "leads_with_time"), needs=("set_times.jsonl",),
      note="per-artist clock times — Insomniac-app-only, never in web source; primitive for conflict+travel"),
    R("set_order", "temporal", BASE, BLOCKED, needs=("set_times.jsonl",),
      derives_from=("set_time",), note="who plays before/after on a stage"),
    R("cross_stage_conflict", "temporal", DERIVED, BLOCKED, derives_from=("set_time",),
      needs=("set_times.jsonl",), note="most-queried real question; install set_time densely + teach comparison"),
    R("travel_feasibility", "temporal", DERIVED, BLOCKED, derives_from=("set_time", "stage_to_stage"),
      needs=("set_times.jsonl", "venue_map.json"), note="the specialist capability; both primitives absent"),
    R("year_over_year", "temporal", BASE, PARTIAL, ("forward", "reverse"),
      note="historical 'have headlined' lists present (qualified as past, NOT 2026) — not structured per-year"),

    # ── Artist ───────────────────────────────────────────────────────────────────────────────────
    R("artist_to_stage", "artist", BASE, SUPPORTED, ("forward", "reverse"),
      ("leads_with_artist", "leads_with_stage"), note="the lineup spine — dense, both directions"),
    R("b2b_pairing", "artist", BASE, SUPPORTED, ("symmetric",),
      note="two artists / one slot — 72 source mentions; breaks naive retrieval if missing"),
    R("set_style", "artist", BASE, PARTIAL, ("forward", "reverse"),
      note="DJ set / live / sunrise — sparse but present (Porter Robinson, Tinlicker, San Holo)"),
    R("artist_tier", "artist", BASE, PARTIAL, ("forward", "reverse"),
      note="headliner (in 'Headliner Highlights') vs undercard — coarse signal only"),
    R("artist_to_genre", "artist", BASE, BLOCKED, ("forward", "reverse"),
      note="not stated directly; only reachable via artist→stage→genre bridge (kept as DERIVED chain)"),
    R("artist_similarity", "artist", DERIVED, BLOCKED, derives_from=("artist_to_genre", "artist_tier"),
      note="if-you-like-X-try-Y; needs artist_to_genre which is absent"),
    R("artist_alias", "artist", BASE, BLOCKED, ("symmetric",),
      note="side projects / one human, many entries — not in source"),
    R("artist_label", "artist", BASE, BLOCKED, note="label / collaboration — not in source"),

    # ── Stage identity (the OTHER half of 'which stage?') ──────────────────────────────────────────
    R("stage_to_genre", "stage", BASE, SUPPORTED, ("forward", "reverse"),
      ("leads_with_stage", "leads_with_genre"), note="the HUB relation; bridges artist↔stage↔recommendation"),
    R("stage_brand", "stage", BASE, SUPPORTED, ("forward", "reverse"),
      ("leads_with_stage", "leads_with_brand"), note="Factory 93→neon, Dreamstate→quantum, Bassrush→bass, Basscon→waste"),
    R("stage_rank_capacity", "stage", BASE, SUPPORTED, ("forward", "reverse"),
      note="Kinetic largest (~70k), Circuit second-largest"),
    R("stage_production_character", "stage", BASE, SUPPORTED, ("forward",),
      note="structure / visuals a stage is known for (LED walls, fire, owl)"),
    R("stage_artist_fit", "stage", DERIVED, SUPPORTED, derives_from=("stage_to_genre", "artist_to_stage"),
      note="which artists fit which stage — via the genre bridge; teach, don't enumerate"),

    # ── Experiential ───────────────────────────────────────────────────────────────────────────────
    R("culture", "experiential", BASE, SUPPORTED, note="kandi / PLUR / etiquette / first-timer guidance"),
    R("venue_rules", "experiential", BASE, SUPPORTED, note="re-entry, allowed items, lockers — present in FAQ/tips"),
    R("installations", "experiential", BASE, SUPPORTED, note="roaming art cars (BeatBox, Metaphoenix, …)"),
    R("crowd_density", "experiential", DERIVED, BLOCKED, derives_from=("set_time", "artist_tier"),
      needs=("set_times.jsonl",), note="by stage by time — needs set_time"),
    R("heat_shade", "experiential", BASE, BLOCKED, needs=("venue_map.json",),
      note="time-of-day shade for open-air stages — needs orientation/coords"),

    # ── Planning / derived — install primitives + teach the op; NEVER enumerate (Part 2) ────────────
    R("conflict_resolution", "planning", DERIVED, BLOCKED, derives_from=("set_time", "stage_to_stage"),
      needs=("set_times.jsonl", "venue_map.json"), note="worked examples only, once primitives exist"),
    R("schedule_building", "planning", DERIVED, BLOCKED, derives_from=("set_time", "stage_to_stage"),
      needs=("set_times.jsonl", "venue_map.json")),
    R("onboarding_path", "planning", DERIVED, BLOCKED, derives_from=("arrival_path", "stage_to_stage"),
      needs=("venue_map.json",)),
    R("pacing", "planning", DERIVED, BLOCKED, derives_from=("set_time",), needs=("set_times.jsonl",)),
    R("recommendation", "planning", DERIVED, SUPPORTED,
      derives_from=("stage_to_genre", "artist_to_stage", "artist_tier"),
      note="genre+tier → suggestions; teachable from supported primitives (genre bridge), a few worked examples"),
]}


def supported() -> list[Relation]:
    """Relations the generator will populate now (SUPPORTED or PARTIAL with present instances)."""
    return [r for r in RELATIONS.values() if r.data_status in (SUPPORTED, PARTIAL)]


def blocked() -> list[Relation]:
    return [r for r in RELATIONS.values() if r.data_status == BLOCKED]


def base_relations() -> list[Relation]:
    return [r for r in RELATIONS.values() if r.tier == BASE]


def derived_relations() -> list[Relation]:
    return [r for r in RELATIONS.values() if r.tier == DERIVED]


def needs_summary() -> dict[str, list[str]]:
    """data-file → relations it would unblock (the data-acquisition shopping list)."""
    out: dict[str, list[str]] = {}
    for r in RELATIONS.values():
        for f in r.needs:
            out.setdefault(f, []).append(r.key)
    return out


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(RELATIONS)} relations | tier {dict(Counter(r.tier for r in RELATIONS.values()))} | "
          f"data {dict(Counter(r.data_status for r in RELATIONS.values()))}")
    print("\nSUPPORTED (build now):")
    for r in supported():
        print(f"  · {r.key:26s} {r.tier:7s} {r.group:13s} dirs={'/'.join(r.directions)}")
    print("\nBLOCKED (coverage gaps — need data):")
    for f, rels in needs_summary().items():
        print(f"  ✗ {f:18s} unblocks: {', '.join(rels)}")
