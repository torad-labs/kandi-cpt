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

The per-item SENTENCE-TEMPLATE SHAPES (fact / stage-attribute / B2B / multi-stage) are domain-authored —
they live in configs/templates/<domain>.yaml `hydrate_templates` and render against a per-unit `Ctx`
(festival/venue/dates/day-calendar/stage-taxonomy). EDC carries today's exact strings → byte-identical.
A NEW domain swaps the templates; the ALGORITHMS here (balance weighting, roster/calendar builders,
multi-stage/B2B unification) stay. The module constants below are the EDC defaults the ~11 importers use.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from pipeline.domain import Domain
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


def _dd(day: str, ctx: "Ctx") -> str:
    """ctx-aware day → 'Day (Date)' (mirrors _daydate but reads the unit's calendar from ctx)."""
    d = ctx.day_date.get(day, "")
    return f"{day} ({d})" if d else (day or "")


_HYDRATE_CACHE: dict = {}


def _hydrate_templates(domain: str = "edc") -> dict:
    """The domain's `hydrate_templates` (configs/templates/<domain>.yaml), cached. {} when a domain
    declares none → the generators emit nothing rather than bleeding EDC's templates into a new domain."""
    if domain not in _HYDRATE_CACHE:
        _HYDRATE_CACHE[domain] = (Domain(domain).templates or {}).get("hydrate_templates", {})
    return _HYDRATE_CACHE[domain]


class Ctx:
    """Per-unit render context for the hydrate generators: festival/venue/dates/day-calendar + the stage
    taxonomy + the domain-authored template strings. Defaults to the EDC LV-2026 module constants +
    configs/templates/edc.yaml, so existing callers (and the ~11 fact_expand importers) stay byte-identical;
    a domain-general caller builds one from a unit's festival_meta.json via context_from_meta()."""

    def __init__(self, *, festival=None, venue=None, dates=None, day_date=None, days=None,
                 stage_meta=None, templates=None):
        self.festival = FESTIVAL if festival is None else festival
        self.venue = VENUE if venue is None else venue
        self.dates = DATES if dates is None else dates
        self.day_date = DAY_DATE if day_date is None else day_date
        self.days = DAYS if days is None else tuple(days)
        self.stage_meta = STAGE_META if stage_meta is None else stage_meta
        self.t = _hydrate_templates("edc") if templates is None else templates


_DEFAULT_CTX: "Ctx | None" = None


def default_ctx() -> "Ctx":
    """The EDC LV-2026 render context (module constants + edc templates), built once and cached."""
    global _DEFAULT_CTX
    if _DEFAULT_CTX is None:
        _DEFAULT_CTX = Ctx()
    return _DEFAULT_CTX


def context_from_meta(meta: dict, domain: str = "edc") -> "Ctx":
    """Build a render context from a unit's festival_meta.json (the domain-general path). NOTE the venue is
    meta.venue verbatim (a unit's bare venue, no article) — EDC's gold "the Las Vegas Motor Speedway" is a
    property of the DEFAULT (constant) context, not of a meta-built one, so the gold stays on default_ctx()."""
    days_map = meta.get("days") or {}
    day_date = {d: (info or {}).get("date", "") for d, info in days_map.items()}
    stages = meta.get("stages") or {}
    stage_meta = {slug: {k: (s or {}).get(k, "") for k in ("genre", "brand", "rank", "vibe")}
                  for slug, s in stages.items()}
    return Ctx(festival=meta.get("name") or FESTIVAL, venue=meta.get("venue") or VENUE,
               dates=str(meta.get("dates") or DATES), day_date=day_date or DAY_DATE,
               days=tuple(days_map) or DAYS, stage_meta=stage_meta or STAGE_META,
               templates=_hydrate_templates(domain))


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
        for line in ef.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
                names[e["id"]] = e.get("name_canonical", "")
            except Exception:
                pass
    out: list[tuple[str, str, str]] = []
    for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
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


def fact_templates(artist: str, stage_slug: str, day: str, ctx: "Ctx | None" = None) -> list[str]:
    """Deterministic statements of ONE triple that TIE THE LOAD-BEARING DIMENSIONS TOGETHER —
    artist · stage · day · date · festival · venue — in every query direction, rendered from the domain's
    `fact_no_day`/`fact_with_day` templates against the unit's `Ctx`.

    v3 hydration fix: the v2 corpus bound stages (distinctive camelCase) but not days (the common
    word "Saturday" collapsed to the modal default). So here every day is anchored to its date
    (Friday→May 15), framed as the multiple-choice question that empirically retrieved the right
    answer, and lightly contrasted against the other days/stages. Each template carries 2+ bound
    dimensions so the relation — not the isolated token — is what installs."""
    ctx = ctx or default_ctx()
    c, s = camel(stage_slug), spaced(stage_slug)
    if not day:                                                    # no-day fact (e.g. Basspod): bind artist·stage·festival·venue only
        kw = dict(festival=ctx.festival, venue=ctx.venue, dates=ctx.dates, artist=artist, c=c, s=s)
        return [t.format(**kw) for t in ctx.t["fact_no_day"]]
    date = ctx.day_date.get(day, "")
    dd = _dd(day, ctx)                                             # "Friday (May 15)"
    other_days = " or ".join(x for x in ctx.days if x != day)
    # contrast against the dominant attractor stages (the 16-fact kinetic/neon that caused v2 drift),
    # not all six — keeps the line clean and targets the actual confusion.
    other_stages = " or ".join(camel(x) for x in ctx.t.get("attractors", []) if x != stage_slug)
    kw = dict(festival=ctx.festival, venue=ctx.venue, dates=ctx.dates, artist=artist, c=c, s=s,
              dd=dd, day=day, date=date, other_days=other_days, other_stages=other_stages)
    return [t.format(**kw) for t in ctx.t["fact_with_day"]]


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


def stage_attribute_lines(stage_slug: str, ctx: "Ctx | None" = None) -> list[str]:
    """Deterministic stage→genre/brand/rank/vibe statements PLUS the reverse genre→stage and brand→stage
    queries — rendered from the domain's `stage_attribute` templates against the unit's stage taxonomy."""
    ctx = ctx or default_ctx()
    m = ctx.stage_meta.get(stage_slug)
    if not m:
        return []
    c, s = camel(stage_slug), spaced(stage_slug)
    kw = dict(festival=ctx.festival, c=c, s=s, genre=m.get("genre", ""), brand=m.get("brand", ""),
              rank=m.get("rank", ""), vibe=m.get("vibe", ""))
    out = []
    for key in ("genre", "brand", "rank", "vibe"):
        if m.get(key):
            out += [t.format(**kw) for t in ctx.t["stage_attribute"][key]]
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


def b2b_lines(triples: list[tuple[str, str, str]], ctx: "Ctx | None" = None) -> list[str]:
    """Back-to-back (B2B) partnership facts — the billed B2B acts in the lineup (e.g. 'GRiZ B2B Wooli')."""
    ctx = ctx or default_ctx()
    out = []
    for a, st, d in triples:
        if " B2B " in f" {a} ":
            parts = " and ".join(p.strip() for p in a.split(" B2B "))
            c = camel(st)
            when = f" on {_dd(d, ctx)}" if d else ""
            kw = dict(festival=ctx.festival, a=a, parts=parts, c=c, when=when)
            out += [t.format(**kw) for t in ctx.t["b2b"]]
    return out


def multistage_lines(triples: list[tuple[str, str, str]], ctx: "Ctx | None" = None) -> list[str]:
    """Unify artists who play more than one stage (both facts are true) so 'which stage does X play?'
    resolves to BOTH rather than the model picking one."""
    ctx = ctx or default_ctx()
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
            kw = dict(festival=ctx.festival, a=a, parts=" and ".join(parts))
            out += [t.format(**kw) for t in ctx.t["multistage"]]
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
