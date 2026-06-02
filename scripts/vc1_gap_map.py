#!/usr/bin/env python3
"""VC1 Phase 0 — per-edition completeness gap map.

Triples are `subject_id`/`object` like 'edc_las_vegas_2013:artist:tommie_sunshine' — the entity TYPE is the
2nd ':'-segment (:artist:/:stage:/:day:/:set:/:time:). We count distinct entities per type per edition, plus
venue/dates/attendance from festival_meta.json, and emit a 'needs' list. Output: data/_build/vc1_gap_map.json
(drives the Phase-1 targeted scrape) + a console summary. Re-run after scraping to measure progress.

    PYTHONPATH=scripts python scripts/vc1_gap_map.py
"""
import json
from collections import Counter

import config as C
from pipeline.editions import iter_editions

# A "complete" edition target (heuristic floors for the needs-list).
MIN_ARTISTS, MIN_STAGES = 15, 3


def _etype(eid: str) -> str:
    parts = eid.split(":")
    return parts[1] if len(parts) >= 2 else "?"


def scan_edition(ed) -> dict:
    d = ed.dir
    m = json.loads((d / "festival_meta.json").read_text()) if (d / "festival_meta.json").exists() else {}
    venue = (m.get("venue") or "").strip()
    has_venue = bool(venue) and not venue.lower().startswith("various")
    by_type: dict[str, set] = {}
    n_tri = 0
    has_time = False
    tp = d / "15_structured/triples.jsonl"
    if tp.exists():
        for line in tp.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            n_tri += 1
            for fld in ("subject_id", "object"):
                eid = t.get(fld)
                if isinstance(eid, str) and ":" in eid:
                    by_type.setdefault(_etype(eid), set()).add(eid)
                    if _etype(eid) == "time":
                        has_time = True
            a = t.get("attrs") or {}
            if any(k for k in a if "time" in k.lower() or "slot" in k.lower()):
                has_time = True
    n_art = len(by_type.get("artist", set()))
    n_stage = len(by_type.get("stage", set()))
    n_day = len(by_type.get("day", set()))
    ndoc = sum(1 for _ in (d / "10_clean").rglob("*.md")) if (d / "10_clean").is_dir() else 0
    needs = []
    if not has_venue:
        needs.append("venue")
    if not (m.get("dates") or m.get("date") or m.get("start_date")):
        needs.append("dates")
    if n_art < MIN_ARTISTS:
        needs.append("lineup")            # partial lineup (news subset, not full official)
    if n_stage < MIN_STAGES:
        needs.append("stages")
    if not has_time:
        needs.append("set_times")
    return {"edition": ed.slug, "venue": has_venue, "n_artists": n_art, "n_stages": n_stage,
            "n_days": n_day, "has_set_times": has_time, "n_triples": n_tri, "n_clean_docs": ndoc,
            "attendance": bool(m.get("attendance")), "needs": needs}


def main():
    rows = [scan_edition(ed) for ed in iter_editions("edc")]
    agg = Counter()
    for r in rows:
        for need in r["needs"]:
            agg[f"needs_{need}"] += 1
    art = sorted(r["n_artists"] for r in rows)
    out = {"n_editions": len(rows), "min_artists": MIN_ARTISTS, "min_stages": MIN_STAGES,
           "aggregate": dict(agg), "editions": rows}
    p = C.ROOT / "data" / "_build" / "vc1_gap_map.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print(f"VC1 gap map — {len(rows)} editions  →  {p.relative_to(C.ROOT)}\n")
    print("NEEDS (editions requiring each):")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:3d} / {len(rows)}")
    mid = art[len(art) // 2]
    print(f"\nlineup depth (distinct artists/edition): min={art[0]} median={mid} max={art[-1]}")
    print(f"  fully-installed (>={MIN_ARTISTS} artists): {sum(1 for a in art if a >= MIN_ARTISTS)} / {len(rows)}")
    print(f"  set-times present: {sum(1 for r in rows if r['has_set_times'])} / {len(rows)}")
    worst = sorted(rows, key=lambda r: (r['n_artists'], r['n_triples']))[:12]
    print("\nthinnest editions (lineup-first):")
    for r in worst:
        print(f"  {r['edition']:26s} artists={r['n_artists']:3d} stages={r['n_stages']:2d} "
              f"triples={r['n_triples']:3d} needs={','.join(r['needs'])}")


if __name__ == "__main__":
    main()
