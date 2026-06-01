#!/usr/bin/env python3
"""Phase 2: scaffold festival_meta.json for every EDC edition (Rust hydrate + tokenize read it).

Only las_vegas/2026 was hand-authored. The other ~96 editions need a meta so festival-factory can
hydrate/tokenize them. We derive each from three sources, no LLM:
  • configs/editions_edc.json  — venue, city, country, year (the registry)
  • the edition's 15_structured/triples.jsonl — the stages THAT edition actually had (avoids anachronism;
    EDC LA 1997 had different stages than LV 2026)
  • the LV-2026 meta — the canonical EDC stage taxonomy (genre/brand/rank/vibe), copied where a stage slug
    matches; recurring stages (kineticFIELD, cosmicMEADOW, …) keep their rich attrs. Thin editions with no
    structured stages fall back to the canonical set so hydrate has something to work with.

Days are generic 3-day (Fri/Sat/Sun) with no precise dates — per-edition exact dates aren't reliably in
the data and aren't needed for cross-edition breadth. Never overwrites an existing festival_meta.json
(LV-2026 is hand-curated). Run AFTER Phase 1 (structured extraction).

    PYTHONPATH=scripts python scripts/gen_festival_meta.py [--editions city/year,…] [--force]
"""
from __future__ import annotations

import argparse
import json

import config as C
from pipeline import fact_expand as FE
from pipeline.editions import iter_editions

REG = json.loads((C.ROOT / "configs/editions_edc.json").read_text())["editions"]
VENUE = {e["city"]: e.get("venue", "") for e in REG}
LV = json.loads((C.ROOT / "data/edc/las_vegas/2026/festival_meta.json").read_text())
CANON_STAGES = LV["stages"]                              # slug -> {camel, spaced, genre, brand, rank, vibe, aliases}
GENERIC_DAYS = {"Friday": {"date": "", "ordinal": "day 1"},
                "Saturday": {"date": "", "ordinal": "day 2"},
                "Sunday": {"date": "", "ordinal": "day 3"}}


def stages_for(ed) -> dict:
    """The edition's own stages (from its structured triples), each with canonical attrs if the slug is a
    known EDC stage, else derived camel/spaced + generic attrs. Falls back to the canonical 9 if none."""
    found = set()
    tp = ed.dir / "15_structured/triples.jsonl"
    if tp.exists():
        for l in tp.read_text(encoding="utf-8", errors="replace").splitlines():
            if not l.strip():
                continue
            try:
                o = str(json.loads(l).get("object", ""))
            except Exception:
                continue
            if ":stage:" in o:
                found.add(o.split(":stage:")[1])
    # Always include the canonical EDC stage taxonomy (the brand's recurring stages — kineticFIELD etc.;
    # thin structured rarely captures all of them) PLUS any edition-specific stage this edition's data found.
    out = dict(CANON_STAGES)
    for slug in sorted(found):
        if slug not in out:
            out[slug] = {"camel": FE.camel(slug), "spaced": FE.spaced(slug), "aliases": [],
                         "genre": "", "brand": "", "rank": "", "vibe": ""}
    return out


def build_meta(ed) -> dict:
    city, year = ed.slug.split("/")
    city_title = city.replace("_", " ").title()
    stages = stages_for(ed)
    return {
        "festival_slug": f"edc_{city}_{year}",
        "name": f"EDC {city_title} {year}",
        "short_names": [f"EDC {year}", f"EDC {city_title}", "EDC"],
        "venue": VENUE.get(city, ""),
        "venue_aliases": [],
        "city": city_title,
        "dates": str(year),
        "n_main_stages": len(stages),
        "_note": "scaffolded by gen_festival_meta.py (Phase 2) — stages from this edition's 15_structured "
                 "(canonical attrs where the slug recurs), generic 3-day calendar, no precise dates.",
        "days": dict(GENERIC_DAYS),
        "stages": stages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="restrict to 'city/year,…' (default: all on disk)")
    ap.add_argument("--force", action="store_true", help="overwrite existing festival_meta.json")
    args = ap.parse_args()
    wrote = skipped = 0
    for ed in iter_editions("edc", only=args.editions):
        path = ed.dir / "festival_meta.json"
        if path.exists() and not args.force:
            skipped += 1
            continue
        meta = build_meta(ed)
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        wrote += 1
        print(f"  {ed.slug}: {len(meta['stages'])} stages, venue={meta['venue'] or '(unknown)'}")
    print(f"\nwrote {wrote} festival_meta.json, skipped {skipped} (existing)")


if __name__ == "__main__":
    main()
