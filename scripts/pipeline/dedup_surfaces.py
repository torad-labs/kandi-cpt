#!/usr/bin/env python3
"""Semantic dedup of the hydrated layer (docs/cpt-overhaul/03).

After hydrate/synth, before audit: per fact, cluster surfaces and keep the most-diverse
representatives up to the exposure budget. Non-destructive — writes an exclusion list the
tokenizer honors; never deletes from 25_hydrated/.

Floor protection is structural: a fact's surviving count never drops below its exposure
requirement (default min(cap, max(2, n//2)) when no per-fact floor is set). A fact that
would fall below floor is a diversity FAILURE (generate more), not a successful dedup.

    PYTHONPATH=scripts python scripts/pipeline/dedup_surfaces.py \
        --domain <d> [--editions …] [--budget 40] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import config as C
from pipeline import diversity_lib as Div
from pipeline.editions import iter_editions
from qgre import spec as S


HYDRATED_NAMES = (
    "relations.jsonl",
    "relations_llm.jsonl",
    "relations_contrast.jsonl",
    "relations_qa.jsonl",
)


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def load_unit_surfaces(ed) -> list[dict]:
    """Load hydrated surfaces with stable provenance keys for exclusion."""
    rows = []
    hyd = ed.dir / "25_hydrated"
    if not hyd.is_dir():
        return rows
    for name in HYDRATED_NAMES:
        path = hyd / name
        for i, rec in enumerate(_iter_jsonl(path)):
            text = str(rec.get("text") or "").strip()
            if not text:
                continue
            rows.append({
                "text": text,
                "relation_type": rec.get("relation_type") or "?",
                "source_triple_ids": rec.get("source_triple_ids") or [],
                "surface_form": rec.get("surface_form") or name,
                "path": str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else path.as_posix(),
                "line": i,
                "key": f"{path.name}:{i}:{hash(text) & 0xFFFFFFFF:08x}",
                "fact": Div.fact_key(rec),
            })
    return rows


def exposure_floor(n_surfaces: int, cap: int, min_floor: int = 2) -> int:
    """Never cull below this many surfaces for a fact."""
    if n_surfaces <= min_floor:
        return n_surfaces
    # Keep at least half (rounded up) or min_floor, never above cap or n.
    half = (n_surfaces + 1) // 2
    return max(min_floor, min(cap, half, n_surfaces))


def dedup_unit(
    rows: list[dict],
    *,
    budget: int,
    cap: int,
    jaccard_threshold: float = 0.55,
    rare_threshold: float = 0.40,
) -> dict:
    """Return keep keys, exclude keys, per-fact before/after, floor violations."""
    by_fact: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fact[row["fact"]].append(row)

    keep_keys: set[str] = set()
    exclude_keys: set[str] = set()
    per_fact = []
    floor_violations = []

    for fact, members in sorted(by_fact.items()):
        texts = [m["text"] for m in members]
        floor = exposure_floor(len(members), cap=cap)
        # budget is a MAX; floor is a MIN — never cull below floor even when budget is tighter
        target = min(len(members), max(floor, min(budget, len(members))))
        picked_idx = Div.pick_diverse_representatives(
            texts,
            budget=target,
            jaccard_threshold=jaccard_threshold,
            rare_fact_threshold=rare_threshold if len(members) <= 4 else None,
        )
        if len(picked_idx) < floor:
            # Should not happen given pick_diverse floor; record if it does.
            floor_violations.append({"fact": fact, "before": len(members), "after": len(picked_idx), "floor": floor})
        picked = set(picked_idx)
        for i, m in enumerate(members):
            if i in picked:
                keep_keys.add(m["key"])
            else:
                exclude_keys.add(m["key"])
        per_fact.append({
            "fact": fact,
            "before": len(members),
            "after": len(picked),
            "floor": floor,
            "forms_before": Div.effective_forms(texts),
            "forms_after": Div.effective_forms([texts[i] for i in sorted(picked)]),
        })

    return {
        "keep_keys": sorted(keep_keys),
        "exclude_keys": sorted(exclude_keys),
        "per_fact": per_fact,
        "floor_violations": floor_violations,
        "n_before": len(rows),
        "n_after": len(keep_keys),
        "n_excluded": len(exclude_keys),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SemDeDup-style surface cull (non-destructive)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions", default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="max surfaces kept per fact (default = relations.exposure_cap_per_fact)")
    ap.add_argument("--jaccard-threshold", type=float, default=0.55)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = S.load_spec(args.domain)
    cap = int(S.relations_cfg(spec).get("exposure_cap_per_fact", 60))
    budget = int(args.budget if args.budget is not None else cap)

    out_root = C.BUILD / "dedup" / args.domain
    out_root.mkdir(parents=True, exist_ok=True)

    units = []
    total_ex = 0
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        rows = load_unit_surfaces(ed)
        if not rows:
            print(f"  {ed.slug}: no hydrated surfaces — skip")
            continue
        result = dedup_unit(
            rows, budget=budget, cap=cap, jaccard_threshold=args.jaccard_threshold,
        )
        unit_report = {
            "domain": args.domain,
            "unit": ed.slug,
            "budget": budget,
            "cap": cap,
            **{k: result[k] for k in ("n_before", "n_after", "n_excluded", "floor_violations", "per_fact")},
            "exclude_keys": result["exclude_keys"],
        }
        units.append(unit_report)
        total_ex += result["n_excluded"]
        print(f"  {ed.slug}: {result['n_before']} → {result['n_after']} "
              f"(excluded {result['n_excluded']}, floor_violations={len(result['floor_violations'])})")
        if args.dry_run:
            continue
        unit_dir = out_root / ed.slug.replace("/", "__")
        unit_dir.mkdir(parents=True, exist_ok=True)
        excl_path = unit_dir / "exclude_keys.json"
        tmp = excl_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result["exclude_keys"], indent=2), encoding="utf-8")
        os.replace(tmp, excl_path)
        rep_path = unit_dir / "report.json"
        tmp = rep_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(unit_report, indent=2), encoding="utf-8")
        os.replace(tmp, rep_path)

    summary = {
        "domain": args.domain,
        "n_units": len(units),
        "total_excluded": total_ex,
        "budget": budget,
        "units": [{"unit": u["unit"], "n_before": u["n_before"], "n_after": u["n_after"],
                   "n_excluded": u["n_excluded"], "floor_violations": len(u["floor_violations"])}
                  for u in units],
        "status": "FAIL" if any(u["floor_violations"] for u in units) else "PASS",
    }
    print(f"\ndedup {args.domain}: excluded {total_ex} surfaces across {len(units)} units "
          f"[{summary['status']}]")
    if not args.dry_run:
        sp = out_root / "summary.json"
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.replace(tmp, sp)
        print(f"  wrote {sp}")
    if summary["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
