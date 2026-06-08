#!/usr/bin/env python3
"""Phase 2: scaffold festival_meta.json for every EDC edition (Rust hydrate + tokenize read it).

Only las_vegas/2026 was hand-authored. The other ~96 editions need a meta so festival-factory can
hydrate/tokenize them. We derive each from three sources, no LLM:
  • the domain's unit registry (configs/editions_edc.json via the manifest) — venue, city, country, year
  • the edition's 15_structured/triples.jsonl — the stages THAT edition actually had (avoids anachronism;
    EDC LA 1997 had different stages than LV 2026)
  • configs/templates/<domain>.yaml `meta_scaffold` (via Domain.templates) — the canonical stage taxonomy
    (genre/brand/rank/vibe) + the generic day calendar. This WAS a hardcoded read of the LV-2026 gold meta;
    it now lives in domain-authored config so a non-festival domain swaps it (or omits it → empty, no bleed).
    Recurring stages (kineticFIELD, cosmicMEADOW, …) keep their rich attrs, copied where a stage slug
    matches; thin editions with no structured stages fall back to the canonical set so hydrate has work.

Days are generic 3-day (Fri/Sat/Sun) with no precise dates — per-edition exact dates aren't reliably in
the data and aren't needed for cross-edition breadth. Never overwrites an existing festival_meta.json
(LV-2026 is hand-curated). Run AFTER Phase 1 (structured extraction).

    PYTHONPATH=scripts python scripts/gen_festival_meta.py [--domain edc] [--editions city/year,…] [--force]
"""
from __future__ import annotations

import argparse
import json

import config as C
from pipeline import fact_expand as FE
from pipeline.domain import Domain
from pipeline.editions import iter_editions


def load_scaffold(domain: str = "edc") -> tuple[dict, dict, dict]:
    """The per-domain meta scaffold — `(canonical_stages, default_days, venue_by_city)` — sourced from the
    domain's authored content (configs/templates/<domain>.yaml `meta_scaffold`) + its unit registry, NOT
    hardcoded to the LV-2026 gold meta. `canonical_stages`/`default_days` default to {} when a domain
    declares none (so no festival bleed); `venue_by_city` comes from the manifest's `registry`."""
    dom = Domain(domain)
    scaffold = (dom.templates or {}).get("meta_scaffold", {})
    canonical_stages = scaffold.get("canonical_stages", {})
    default_days = scaffold.get("default_days", {})
    reg = json.loads((C.ROOT / dom.registry).read_text())["editions"] if dom.registry else []
    venue_by_city = {e["city"]: e.get("venue", "") for e in reg}
    return canonical_stages, default_days, venue_by_city


def stages_for(ed, canonical_stages: dict) -> dict:
    """The edition's own stages (from its structured triples), each with canonical attrs if the slug is a
    known stage, else derived camel/spaced + generic attrs. Falls back to the canonical set if none."""
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
    # Always include the canonical stage taxonomy (the brand's recurring stages — kineticFIELD etc.;
    # thin structured rarely captures all of them) PLUS any edition-specific stage this edition's data found.
    out = dict(canonical_stages)
    for slug in sorted(found):
        if slug not in out:
            out[slug] = {"camel": FE.camel(slug), "spaced": FE.spaced(slug), "aliases": [],
                         "genre": "", "brand": "", "rank": "", "vibe": ""}
    return out


def build_meta(ed, canonical_stages: dict, default_days: dict, venue_by_city: dict) -> dict:
    city, year = ed.slug.split("/")
    city_title = city.replace("_", " ").title()
    stages = stages_for(ed, canonical_stages)
    return {
        "festival_slug": f"edc_{city}_{year}",
        "name": f"EDC {city_title} {year}",
        "short_names": [f"EDC {year}", f"EDC {city_title}", "EDC"],
        "venue": venue_by_city.get(city, ""),
        "venue_aliases": [],
        "city": city_title,
        "dates": str(year),
        "n_main_stages": len(stages),
        "_note": "scaffolded by gen_festival_meta.py (Phase 2) — stages from this edition's 15_structured "
                 "(canonical attrs where the slug recurs), generic 3-day calendar, no precise dates.",
        "days": dict(default_days),
        "stages": stages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc", help="domain manifest (sources meta_scaffold + registry)")
    ap.add_argument("--editions", help="restrict to 'city/year,…' (default: all on disk)")
    ap.add_argument("--force", action="store_true", help="overwrite existing festival_meta.json")
    args = ap.parse_args()
    canonical_stages, default_days, venue_by_city = load_scaffold(args.domain)
    wrote = skipped = 0
    for ed in iter_editions(args.domain, only=args.editions):
        path = ed.dir / "festival_meta.json"
        if path.exists() and not args.force:
            skipped += 1
            continue
        meta = build_meta(ed, canonical_stages, default_days, venue_by_city)
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        wrote += 1
        print(f"  {ed.slug}: {len(meta['stages'])} stages, venue={meta['venue'] or '(unknown)'}")
    print(f"\nwrote {wrote} festival_meta.json, skipped {skipped} (existing)")


if __name__ == "__main__":
    main()
