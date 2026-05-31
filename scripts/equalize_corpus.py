#!/usr/bin/env python3
"""Equalize the hydrated relational corpus to an EXACT per-fact line distribution (V5 lever).

The hydration faithfulness gate + filters drop a VARIABLE number of lines per fact (v4 audit: CoV 0.29,
83-255 lines/fact — the model drifts more on stubborn pairings, so the gate cuts more of their lines,
silently re-introducing the coverage imbalance the rebalancing was meant to remove). This downsamples
each (artist, stage, day) fact to EXACTLY N lines so every relationship carries equal gradient mass —
the user's "equal distribution" requirement, tested cleanly.

Non-lineup relation types (stage attributes, b2b, calendar, roster) are kept as-is (different
cardinalities — 9 stages vs 64 artist-facts — so per-fact equalization applies to the artist→stage→day
lineup relationships, the ones the eval probes).

    PYTHONPATH=scripts python scripts/equalize_corpus.py [--in <jsonl>] [--target N] [--day-balance]

Default target = the global per-fact MIN (exact equality, no regeneration). Deterministic (seeded).
NOTE: a future step folds this into the Rust engine (emit equal at write-time); this is the fast
post-process so V5 is ready without a 30-min re-hydrate.
"""
from __future__ import annotations

import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="data/edc/las_vegas/2026/25_hydrated/relations.jsonl")
    ap.add_argument("--target", type=int, default=0, help="lines per fact (0 = global min → exact equal)")
    ap.add_argument("--day-balance", action="store_true",
                    help="also lift under-represented day tokens to parity (counters Saturday default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    perfact: dict[tuple, list] = collections.defaultdict(list)
    other = []
    for r in rows:
        if r.get("relation_type") == "artist_to_stage":
            e = r.get("entity_ids") or []
            perfact[tuple(e[:3])].append(r)
        else:
            other.append(r)

    counts = {k: len(v) for k, v in perfact.items()}
    base_n = args.target or min(counts.values())
    rng = random.Random(args.seed)

    # optional day-token balancing: per-day fact counts → over-sample (with light repetition) the
    # under-represented days so total day-token frequency is equal (Saturday=Friday=Sunday).
    day_target = {}
    if args.day_balance:
        day_facts = collections.Counter(k[2] for k in perfact if len(k) > 2 and k[2])
        if day_facts:
            max_day = max(day_facts.values())
            # to equalize day-token totals: N_for_day = base_n * (max_day_facts / this_day_facts)
            day_target = {d: round(base_n * max_day / n) for d, n in day_facts.items()}

    out = list(other)
    dup_log = []                               # facts whose target exceeds available lines (→ repetition)
    for k, v in perfact.items():
        rng.shuffle(v)
        day = k[2] if len(k) > 2 else ""
        n = day_target.get(day, base_n)
        if n <= len(v):
            out.extend(v[:n])
        else:                                  # day-balance requested more than exist → repeat (LOGGED, bounded)
            n = min(n, 2 * len(v))             # never duplicate a fact more than 1× (cap gradient over-weight)
            dup_log.append((k, len(v), n))
            out.extend(v + [rng.choice(v) for _ in range(n - len(v))])
    rng.shuffle(out)
    if dup_log:
        print(f"⚠ {len(dup_log)} fact(s) duplicated to hit day-balance target (capped at 2×). "
              f"Repeated lines re-weight gradient — review if this is large:")
        for k, have, n in sorted(dup_log, key=lambda x: -x[2])[:8]:
            print(f"    {' / '.join(map(str, k))}: had {have} → {n}")

    bak = args.inp + ".pre-equalize.bak"
    import shutil
    shutil.copy(args.inp, bak)
    with open(args.inp, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ats = sum(1 for r in out if r.get("relation_type") == "artist_to_stage")
    print(f"backup → {bak}")
    print(f"facts: {len(perfact)}  | was min={min(counts.values())} max={max(counts.values())} "
          f"→ now {'day-balanced' if args.day_balance else f'exactly {base_n}'}/fact")
    print(f"artist_to_stage lines: {ats}  | other relations (kept): {len(other)}  | total: {len(out)}")
    if day_target:
        print(f"day targets (lines/fact): {day_target}")


if __name__ == "__main__":
    main()
