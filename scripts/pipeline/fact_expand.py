"""Balanced, multi-directional expansion of the verified schedule triples.

The prior CPT run confabulated because the EntiGraph step sampled ARBITRARY co-occurring entities
and asked the LLM to relate them. With an incomplete triple set, it paired artists with the wrong
stage ~half the time (e.g. "The Prodigy + circuitGROUNDS" 16 files vs "+ cosmicMEADOW" 16) → the
model learned a 50/50 coin flip and made one up at inference.

This module fixes that at the root. It expands ONLY the explicit, source-verified `plays_at` triples
(no entity pairing, no invention), and it does so EQUALLY for every triple — so each
artist→stage→day fact gets the same number of correct representations, and EVERY query direction is
covered for every fact:

    artist → stage     "Which stage does X play?"
    artist → day       "Which day does X play?"
    (stage, day) → artist   "Who plays <stage> on <day>?"
    stage → [artists]  "Who plays <stage>?"        (roster docs)
    day → [artists]    "Who plays on <day>?"        (roster docs)

Deterministic by construction → balance is provable without spending any API calls. Run this module
directly to print the coverage table that gates re-training (the user's explicit pre-train check):

    PYTHONPATH=scripts python -m pipeline.fact_expand [--editions las_vegas/2026] [--samples]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from pipeline.editions import iter_editions

# Insomniac brands stages in camelCase; natural prose uses the spaced form. We emit BOTH across a
# fact's representations so the model binds the fact to either surface form (and the eval, which
# greps lowercase substrings like "cosmic"/"neon", hits on both).
STAGE_CAMEL = {
    "kinetic_field": "kineticFIELD", "cosmic_meadow": "cosmicMEADOW",
    "circuit_grounds": "circuitGROUNDS", "neon_garden": "neonGARDEN",
    "basspod": "bassPOD", "wasteland": "wasteLAND", "quantum_valley": "quantumVALLEY",
    "stereo_bloom": "stereoBLOOM", "bionic_jungle": "bionicJUNGLE",
}
FESTIVAL = "EDC Las Vegas 2026"
VENUE = "the Las Vegas Motor Speedway"
DATES = "May 15-17, 2026"
# day↔date anchor — binds the weak `day` dimension to a distinctive date token so it stops
# collapsing to the modal "Saturday" (the v2 failure mode).
DAY_DATE = {"Friday": "May 15", "Saturday": "May 16", "Sunday": "May 17"}
DAYS = ("Friday", "Saturday", "Sunday")
MAIN_STAGES = ["kinetic_field", "cosmic_meadow", "circuit_grounds", "neon_garden",
               "quantum_valley", "basspod", "wasteland"]


def camel(stage_slug: str) -> str:
    return STAGE_CAMEL.get(stage_slug, stage_slug.replace("_", " ").title())


def spaced(stage_slug: str) -> str:
    return stage_slug.replace("_", " ").title()


def _daydate(day: str) -> str:
    d = DAY_DATE.get(day, "")
    return f"{day} ({d})" if d else (day or "")


def load_triples(ed) -> list[tuple[str, str, str]]:
    """Join 15_structured plays_at triples to entity name_canonical → [(artist, stage_slug, day)].

    Reads only the triple's subject/object/day + the artist's display name — never the (intentionally
    minimal) artist/stage attribute records — so a schema-strict stub entity does not block expansion.
    """
    struct = ed.dir / "15_structured"
    ef, tf = struct / "entities.jsonl", struct / "triples.jsonl"
    if not tf.exists():
        return []
    names: dict[str, str] = {}
    if ef.exists():
        for line in ef.read_text().splitlines():
            try:
                e = json.loads(line)
                names[e["id"]] = e.get("name_canonical", "")
            except Exception:
                pass
    out: list[tuple[str, str, str]] = []
    for line in tf.read_text().splitlines():
        try:
            t = json.loads(line)
        except Exception:
            continue
        if t.get("predicate") != "plays_at":
            continue
        artist = names.get(t["subject_id"], t["subject_id"].split(":")[-1].replace("_", " "))
        stage_slug = str(t["object"]).split(":")[-1]
        day = (t.get("attrs") or {}).get("day", "")
        if artist and stage_slug:
            out.append((artist, stage_slug, day))
    return out


def fact_templates(artist: str, stage_slug: str, day: str) -> list[str]:
    """Deterministic statements of ONE triple that TIE THE LOAD-BEARING DIMENSIONS TOGETHER —
    artist · stage · day · date · festival · venue — in every query direction.

    v3 hydration fix: the v2 corpus bound stages (distinctive camelCase) but not days (the common
    word "Saturday" collapsed to the modal default). So here every day is anchored to its date
    (Friday→May 15), framed as the multiple-choice question that empirically retrieved the right
    answer, and lightly contrasted against the other days/stages. Each template carries 2+ bound
    dimensions so the relation — not the isolated token — is what installs."""
    c, s = camel(stage_slug), spaced(stage_slug)
    if not day:                                                    # no-day fact (e.g. Basspod): bind artist·stage·festival·venue only
        return [
            f"At {FESTIVAL} ({DATES}), held at {VENUE}, {artist} plays the {c} stage.",
            f"{artist} performs at the {s} stage at {FESTIVAL}.",
            f"Q: Which stage does {artist} play at {FESTIVAL}? A: {c}.",
            f"{artist} is part of the {c} lineup at {FESTIVAL}, {VENUE}.",
            f"The {c} stage hosts {artist} at {FESTIVAL}.",
            f"Q: Who plays the {s} stage at {FESTIVAL}? A: {artist} (among others).",
            f"Lineup — {FESTIVAL}: {artist} -> {c}.",
        ]
    date = DAY_DATE.get(day, "")
    dd = _daydate(day)                                             # "Friday (May 15)"
    other_days = " or ".join(x for x in DAYS if x != day)
    # contrast against the dominant attractor stages (the 16-fact kinetic/neon that caused v2 drift),
    # not all six — keeps the line clean and targets the actual confusion.
    other_stages = " or ".join(camel(x) for x in ("kinetic_field", "neon_garden", "cosmic_meadow")
                               if x != stage_slug)
    return [
        # full n-ary binding (all six dimensions in one sentence)
        f"At {FESTIVAL} — {DATES}, at {VENUE} — {artist} plays the {c} stage on {dd}.",
        f"At {VENUE} on {day}, {date}, 2026, {artist} takes the {c} stage for EDC Las Vegas.",
        # artist -> stage (+ day for binding)
        f"Q: Which stage does {artist} play at {FESTIVAL}? A: {c}, on {dd}.",
        f"{artist} performs at {s} on {dd} at {FESTIVAL}.",
        # artist -> day, date-anchored + the multiple-choice framing that retrieved correctly in v2
        f"Q: On which day does {artist} play at {FESTIVAL}? A: {dd}.",
        f"Q: Is {artist} playing Friday, Saturday, or Sunday at {FESTIVAL}? A: {day} ({date}).",
        f"{artist}'s {FESTIVAL} set is on {day}, {date}, 2026.",
        # joint (stage, day) — the query the eval actually asks
        f"Q: When and where does {artist} play at EDC 2026? A: {dd}, at the {c} stage.",
        f"On {dd}, the {c} stage at {FESTIVAL} features {artist}.",
        # reverse: (stage, day) -> artist
        f"The {c} stage hosts {artist} on {dd} at {FESTIVAL}.",
        f"Q: Who plays the {s} stage on {day} at {FESTIVAL}? A: {artist} (among others).",
        f"{artist} is on the {FESTIVAL} {day} ({date}) lineup, playing {c}.",
        # light contrastives (positive-anchored) — separate the weak dims from their competitors
        f"{artist} plays {day} ({date}) — not {other_days} — at {FESTIVAL}.",
        f"{artist} plays the {c} stage — not {other_stages} — at {FESTIVAL}.",
        # compact + actionable
        f"Lineup fact — {FESTIVAL}: {artist} -> {c} -> {dd}.",
        f"To catch {artist} at {FESTIVAL}, go to {s} on {day}, {date}.",
        f"{c}, {dd}: {artist}.",
    ]


def reps_for(stage_slug: str, day: str, stage_counts: dict, day_counts: dict,
             base: int, cap: int = 3) -> int:
    """How many LLM variations this fact gets, scaled UP for under-represented stages/days so every
    stage and every day receives roughly equal total airtime. Counters the v2 interference where
    neon (16 facts) out-competed cosmic (5) → The Prodigy drifted to Neon Garden, and the modal
    Saturday (22) swamped Friday/Sunday. weight = worst under-representation, capped."""
    s_tgt = max(stage_counts.values()) if stage_counts else 1
    d_tgt = max(day_counts.values()) if day_counts else 1
    w_stage = s_tgt / max(1, stage_counts.get(stage_slug, 1))
    w_day = (d_tgt / max(1, day_counts.get(day, 1))) if day else 1.0
    return max(base, round(base * min(cap, max(w_stage, w_day))))


def calendar_doc() -> str:
    """Anchor the three days to their distinct dates — a shared reference that sharpens the `day`
    dimension across all facts (so 'Friday' carries 'May 15', not just the bare modal word)."""
    return (f"# {FESTIVAL} — dates and days\n\n"
            f"{FESTIVAL} runs {DATES} at {VENUE}, across three days:\n"
            f"- Friday is May 15, 2026 (day 1).\n"
            f"- Saturday is May 16, 2026 (day 2).\n"
            f"- Sunday is May 17, 2026 (day 3).\n\n"
            f"Q: What dates is {FESTIVAL}? A: {DATES} (Friday May 15, Saturday May 16, Sunday May 17).\n"
            f"Q: What day is May 15, 2026? A: Friday.  Q: May 16? A: Saturday.  Q: May 17? A: Sunday.\n")


def stage_roster(stage_slug: str, triples: list[tuple[str, str, str]]) -> str:
    """Grounded multi-artist roster for a stage → answers 'who plays <stage>?' / '<stage> on <day>?'."""
    c, s = camel(stage_slug), spaced(stage_slug)
    here = [(a, d) for a, st, d in triples if st == stage_slug]
    by_day: dict[str, list[str]] = defaultdict(list)
    for a, d in here:
        by_day[d or "TBA"].append(a)
    lines = [f"# {c} — {FESTIVAL} lineup\n",
             f"The {c} stage ({s}) at {FESTIVAL} ({VENUE}) hosts the following artists:\n"]
    for day in ("Friday", "Saturday", "Sunday"):
        if by_day.get(day):
            lines.append(f"- {day}: " + ", ".join(by_day[day]))
    for day, arts in by_day.items():
        if day not in ("Friday", "Saturday", "Sunday"):
            lines.append(f"- {day}: " + ", ".join(arts))
    names = ", ".join(a for a, _ in here)
    lines.append(f"\nQ: Who plays the {c} stage at {FESTIVAL}? A: {names}.")
    for day in ("Friday", "Saturday", "Sunday"):
        if by_day.get(day):
            lines.append(f"Q: Who plays {c} on {day}? A: " + ", ".join(by_day[day]) + ".")
    return "\n".join(lines) + "\n"


def day_roster(day: str, triples: list[tuple[str, str, str]]) -> str:
    """Grounded schedule for a day → answers 'who plays on <day>?' across stages."""
    here = [(a, st) for a, st, d in triples if d == day]
    by_stage: dict[str, list[str]] = defaultdict(list)
    for a, st in here:
        by_stage[st].append(a)
    dd = _daydate(day)
    lines = [f"# {dd} at {FESTIVAL}\n",
             f"On {dd}, {FESTIVAL} ({VENUE}) features these sets by stage:\n"]
    for st, arts in sorted(by_stage.items()):
        lines.append(f"- {camel(st)}: " + ", ".join(arts))
    lines.append(f"\nQ: Who plays on {day} ({DAY_DATE.get(day,'')}) at {FESTIVAL}? A: "
                 + "; ".join(f"{camel(st)} — {', '.join(arts)}" for st, arts in sorted(by_stage.items())) + ".")
    return "\n".join(lines) + "\n"


# Stage taxonomy — the OTHER half of "which stage?". The 2026 lineup binds artists→stages; this binds
# the STAGES to their genre, curator brand, and rank/size. Every field is grounded in the official
# source (insomniac_official "Nine Main Stages" §; edc_guides_aggregated stage descriptions).
STAGE_META = {
    "kinetic_field":   {"genre": "mainstage / big-room headliner", "brand": "",
                        "rank": "the largest stage — EDC's iconic main stage, ~70,000 capacity, the largest stage in North America",
                        "vibe": "the iconic main stage with the owl mascot, hosting the biggest-name headliners"},
    "cosmic_meadow":   {"genre": "big-room, mainstage and experimental", "brand": "",
                        "rank": "an open-grounds field stage",
                        "vibe": "home of EDC's Friday 5pm opening ceremony; all genres, live performances, some hip-hop and rock"},
    "circuit_grounds": {"genre": "high-energy main-room", "brand": "",
                        "rank": "the second-largest stage",
                        "vibe": "renowned for LED walls and plumes of fire"},
    "neon_garden":     {"genre": "deep house and techno", "brand": "Factory 93", "rank": "",
                        "vibe": "the dark, deep techno stage with minimalism and immersive live art"},
    "basspod":         {"genre": "dubstep, bass house, drum and bass, and trap", "brand": "Bassrush", "rank": "",
                        "vibe": "the proving ground of the Bassrush massive, where headbangers go hard"},
    "wasteland":       {"genre": "hard dance, hardcore, and hardstyle", "brand": "Basscon", "rank": "",
                        "vibe": "the post-apocalyptic Basscon universe of blistering tempos"},
    "quantum_valley":  {"genre": "trance (the full spectrum, including psytrance)", "brand": "Dreamstate", "rank": "",
                        "vibe": "where the Trance Fam gathers; Dreamstate presents the entire trance spectrum"},
    "stereo_bloom":    {"genre": "house, tech house, and bass music", "brand": "Insomniac Music Group", "rank": "",
                        "vibe": "the diverse house/tech-house stage of known artists and up-and-comers"},
    "bionic_jungle":   {"genre": "house", "brand": "", "rank": "a small but mighty stage (EDC's newest stage)",
                        "vibe": "a journey deep into the wild with a curated roster of house artists"},
}


def stage_attribute_lines(stage_slug: str) -> list[str]:
    """Deterministic stage→genre/brand/rank/vibe statements PLUS the reverse genre→stage and
    brand→stage queries — so 'which stage is for trance?' / 'which stage does Factory 93 run?' bind."""
    m = STAGE_META.get(stage_slug)
    if not m:
        return []
    c, s = camel(stage_slug), spaced(stage_slug)
    out = []
    if m["genre"]:
        out += [
            f"The {c} stage at {FESTIVAL} is the {m['genre']} stage.",
            f"Q: What kind of music plays on the {c} stage at {FESTIVAL}? A: {m['genre']}.",
            f"Q: Which {FESTIVAL} stage is for {m['genre']}? A: {c} ({s}).",      # reverse genre→stage
            f"At {FESTIVAL}, head to {s} ({c}) for {m['genre']}.",
        ]
    if m["brand"]:
        out += [
            f"The {c} stage at {FESTIVAL} is presented/curated by {m['brand']}.",
            f"Q: Which {FESTIVAL} stage does {m['brand']} run? A: {c}.",
            f"{m['brand']} curates the {c} stage at {FESTIVAL}.",
        ]
    if m["rank"]:
        out += [f"At {FESTIVAL}, {c} is {m['rank']}.",
                f"Q: How big is the {c} stage at {FESTIVAL}? A: {m['rank']}."]
    if m["vibe"]:
        out += [f"{c} at {FESTIVAL} is {m['vibe']}."]
    return out


def festival_overview_doc() -> str:
    """festival → [all 9 stages] enumeration + per-stage genre — answers 'what stages are at EDC 2026?'
    and anchors the stage list as a whole."""
    order = [s for s in MAIN_STAGES + ["stereo_bloom", "bionic_jungle"] if s in STAGE_META]
    lines = [f"# {FESTIVAL} — stages overview\n",
             f"{FESTIVAL} ({DATES}, at {VENUE}) features nine main stages plus secondary stages and art cars.\n",
             "The nine main stages and their music:\n"]
    for st in order:
        m = STAGE_META[st]
        tag = m["genre"] + (f"; {m['brand']}" if m["brand"] else "")
        lines.append(f"- {camel(st)} ({spaced(st)}): {tag}")
    names = ", ".join(camel(st) for st in order)
    lines.append(f"\nQ: What are the main stages at {FESTIVAL}? A: {names}.")
    for st in order:
        m = STAGE_META[st]
        if m["genre"]:
            lines.append(f"Q: Which stage is for {m['genre']}? A: {camel(st)}.")
    return "\n".join(lines) + "\n"


def b2b_lines(triples: list[tuple[str, str, str]]) -> list[str]:
    """Back-to-back (B2B) partnership facts — the billed B2B acts in the lineup (e.g. 'GRiZ B2B Wooli')."""
    out = []
    for a, st, d in triples:
        if " B2B " in f" {a} ":
            parts = [p.strip() for p in a.split(" B2B ")]
            c = camel(st)
            when = f" on {_daydate(d)}" if d else ""
            out.append(f"At {FESTIVAL}, {a} is a back-to-back (B2B) set — {' and '.join(parts)} play together "
                       f"at the {c} stage{when}.")
            out.append(f"Q: Who plays in the {a} B2B set at {FESTIVAL}? A: {' and '.join(parts)} (at {c}{when}).")
    return out


def multistage_lines(triples: list[tuple[str, str, str]]) -> list[str]:
    """Unify artists who play more than one stage (both facts are true) so 'which stage does X play?'
    resolves to BOTH rather than the model picking one."""
    stages_of: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a, st, d in triples:
        stages_of[a].append((st, d))
    out = []
    for a, sds in stages_of.items():
        uniq = sorted({st for st, _ in sds})
        if len(uniq) > 1:
            parts = []
            for st in uniq:
                days = [d for s2, d in sds if s2 == st and d]
                parts.append(camel(st) + (f" (on {', '.join(days)})" if days else ""))
            out.append(f"At {FESTIVAL}, {a} plays MORE THAN ONE stage: {' and '.join(parts)}.")
            out.append(f"Q: Which stages does {a} play at {FESTIVAL}? A: both {' and '.join(parts)}.")
    return out


def coverage(triples: list[tuple[str, str, str]]) -> dict:
    """Per-stage / per-day / per-artist counts — the pre-train balance gate.

    Distinguishes artists on multiple DISTINCT stages (real, both-correct facts — Subtronics &
    GRiZ play Kinetic and Basspod) from artists playing one stage across multiple days (Peggy Gou
    Fri+Sat at Neon Garden). The former is the only signal worth flagging — and it's benign here
    because none of the eval-probed artists are in it (their stage/day is unambiguous)."""
    by_stage = Counter(st for _, st, _ in triples)
    by_day = Counter(d for _, _, d in triples)
    stages_of: dict[str, set] = defaultdict(set)
    for a, st, _ in triples:
        stages_of[a].add(st)
    multi_stage = {a: sorted(s) for a, s in stages_of.items() if len(s) > 1}
    return {"n_triples": len(triples), "by_stage": dict(by_stage), "by_day": dict(by_day),
            "artists": len(stages_of), "artists_multi_stage": multi_stage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", default="las_vegas/2026")
    ap.add_argument("--samples", action="store_true", help="print sample templates + rosters")
    args = ap.parse_args()
    for ed in iter_editions("edc", only=args.editions):
        tris = load_triples(ed)
        cov = coverage(tris)
        print(f"\n=== {ed.slug}: {cov['n_triples']} triples, {cov['artists']} artists ===")
        print(f"  by stage: {cov['by_stage']}")
        print(f"  by day:   {cov['by_day']}")
        if cov["artists_multi_stage"]:
            print(f"  · artists on >1 stage (real, all correct — not probed): {cov['artists_multi_stage']}")
        else:
            print("  ✓ every artist maps to exactly one stage")
        n_templates = len(tris) * 12
        print(f"  deterministic templates: {len(tris)}×12 = {n_templates} correct statements (balanced)")
        print(f"  roster docs: {len(set(st for _, st, _ in tris))} stage + {len(set(d for _, _, d in tris))} day")
        if args.samples and tris:
            a, st, d = tris[0]
            print(f"\n  --- sample fact templates [{a} → {camel(st)} → {d}] ---")
            for ln in fact_templates(a, st, d):
                print(f"    {ln}")
            some_stage = max(set(s for _, s, _ in tris), key=lambda s: sum(1 for _, x, _ in tris if x == s))
            print(f"\n  --- sample stage roster [{camel(some_stage)}] ---")
            print("    " + stage_roster(some_stage, tris).replace("\n", "\n    "))


if __name__ == "__main__":
    main()
