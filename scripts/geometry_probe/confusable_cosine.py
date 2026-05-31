#!/usr/bin/env python3
"""Confusable-pair cosine — the V6 success metric.

Did disjoint context PUSH APART the entities that must stay distinct (Armin↔Above&Beyond same stage
different day; Underworld↔the Sunday-Cosmic trio same stage different day), while KEEPING TOGETHER the
ones that bind (the Sunday-Cosmic trio with each other)? Reads festival_meta.confusable_groups (the
single source of truth) + the saved cosine_<label>_cpt.npz from each geometry run, and reports:
  • CROSS-group confusable cosine  — want it LOWER in v6 than v4/v5 (separation worked)
  • WITHIN-group cosine            — want it HIGH (binding preserved)

    PYTHONPATH=scripts python -m geometry_probe.confusable_cosine \
        v4_camel:data/_build/geometry/v4_camel v5_camel:data/_build/geometry/v5_camel \
        v6_fp4_camel:data/_build/geometry/v6
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config as C


def load_run(spec: str):
    label, d = spec.split(":", 1)
    dd = C.ROOT / d
    meta = json.loads((dd / f"metrics_{label}.json").read_text())
    npz = np.load(dd / f"cosine_{label}_cpt.npz")
    idx = {e["name"]: i for i, e in enumerate(meta["entities"])}
    return label, npz, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label:dir specs (cosine_<label>_cpt.npz must exist in dir)")
    ap.add_argument("--layers", type=int, nargs="*", default=[9, 10, 11, 12, 27, 33])
    args = ap.parse_args()

    fm = json.loads((C.ROOT / "data/edc/las_vegas/2026/festival_meta.json").read_text())
    group_of = {m: g["key"] for g in fm["confusable_groups"] for m in g["members"]}
    members = list(group_of)
    cross = [(a, b) for i, a in enumerate(members) for b in members[i + 1:] if group_of[a] != group_of[b]]
    within = [(a, b) for i, a in enumerate(members) for b in members[i + 1:] if group_of[a] == group_of[b]]

    runs = [load_run(s) for s in args.runs]
    labels = [r[0] for r in runs]
    print(f"confusable entities ({len(members)}): {members}")
    print(f"CROSS-group pairs (want LOW / dropping): {len(cross)}   WITHIN-group (want HIGH): {len(within)}")

    def mean_cos(npz, idx, pairs, L):
        key = str(L)
        if key not in npz:
            return None
        M = npz[key]
        vals = [float(M[idx[a], idx[b]]) for a, b in pairs if a in idx and b in idx]
        return float(np.mean(vals)) if vals else None

    def table(title, pairs):
        print(f"\n=== {title} ===")
        print(f"{'layer':>5} | " + " | ".join(f"{l:>14}" for l in labels))
        for L in args.layers:
            cells = []
            for _, npz, idx in runs:
                v = mean_cos(npz, idx, pairs, L)
                cells.append("--" if v is None else f"{v:.4f}")
            print(f"{L:>5} | " + " | ".join(f"{c:>14}" for c in cells))

    table("CROSS-group confusable cosine  (LOWER = better separation)", cross)
    table("WITHIN-group cosine  (HIGHER = binding preserved)", within)
    print("\nSuccess (per the V6 plan): CROSS-group cosine DROPS in v6 vs v4/v5, WITHIN-group stays high.")


if __name__ == "__main__":
    main()
