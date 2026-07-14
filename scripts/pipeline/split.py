#!/usr/bin/env python3
"""Triple-id split — the seam that makes "held-out" TRUE instead of a label.

Before this module, `generic_probes.py` stamped every probe `split: held_out` while every triple
hydrated: deltas measured trained-fact recall + format transfer, never fact-holdout generalization
(the measured 2026-07-12 finding; the agent handbook's `eval/split-and-decontamination.md` carries
the design law this implements).

The contract:
  1. Run AFTER `structured`, BEFORE `hydrate`: writes a versioned, immutable split manifest per unit
     (`<unit>/splits/split_v1.json`) of content-addressed triple ids — sha1(subject|predicate|object).
  2. Holdout is carved by SUBJECT GROUP (all triples sharing a subject travel to one side — the
     cheap component grouping that keeps a held-out subject's other facts from training next to it),
     with a guard that no predicate loses its entire train side.
  3. `generic_expand` / `generic_synth` hydrate TRAIN ids only; `generic_probes` draws gold from
     HOLDOUT ids and stamps the split version. With no manifest present, everything behaves as
     before EXCEPT the probe stamp, which becomes the truthful `trained`.
  4. The manifest is immutable: re-running refuses to overwrite (a new split is a new version file,
     never a rewrite — re-cutting after a weak eval is goalpost-moving).

    PYTHONPATH=scripts python scripts/pipeline/split.py --domain <d> [--editions u1,u2]
        [--holdout-frac 0.10] [--seed 42] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict

SPLIT_VERSION = "v1"


def triple_id(t: dict) -> str:
    """Content-addressed id: stable across runs, derivable by every consumer from the record alone.
    Keyed on the semantic triple (subject id | predicate | object), not on file position."""
    key = f"{t.get('subject_id', '')}|{t.get('predicate', '')}|{t.get('object', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def split_path(ed_dir, version: str = SPLIT_VERSION):
    return ed_dir / "splits" / f"split_{version}.json"


def load_split(ed_dir, version: str = SPLIT_VERSION) -> dict | None:
    """The unit's split manifest, or None (no split carved — consumers fail open, probes stamp
    `trained`)."""
    p = split_path(ed_dir, version)
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    m["train_ids"] = set(m["train_ids"])
    m["holdout_ids"] = set(m["holdout_ids"])
    return m


def make_split(triples: list[dict], holdout_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Carve holdout by subject group. Greedy over shuffled groups until the holdout fraction is
    reached; a group is skipped when taking it would leave any of its predicates with zero TRAIN
    triples (an axis must never train on nothing)."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        by_subject[t.get("subject_id", "")].append(t)
    pred_train_counts: dict[str, int] = defaultdict(int)
    for t in triples:
        pred_train_counts[t.get("predicate", "")] += 1

    groups = sorted(by_subject.keys())
    random.Random(seed).shuffle(groups)
    target = int(len(triples) * holdout_frac)
    holdout: set[str] = set()
    taken = 0
    for g in groups:
        if taken >= target:
            break
        members = by_subject[g]
        preds = defaultdict(int)
        for t in members:
            preds[t.get("predicate", "")] += 1
        if any(pred_train_counts[p] - n <= 0 for p, n in preds.items()):
            continue                                     # would zero a predicate's train side — skip the group
        for t in members:
            holdout.add(triple_id(t))
            pred_train_counts[t.get("predicate", "")] -= 1
        taken += len(members)
    train = {triple_id(t) for t in triples} - holdout
    return train, holdout


def main():
    from pipeline.editions import iter_editions
    from pipeline.generic_expand import load_all_triples

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions")
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="print per-unit counts, write nothing")
    args = ap.parse_args()

    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        triples = load_all_triples(ed)
        if not triples:
            continue
        out = split_path(ed.dir)
        if out.exists() and not args.dry_run:
            print(f"  {ed.slug}: {out.name} EXISTS — a split is immutable; a new split is a new "
                  f"version file, never an overwrite. Skipping.", file=sys.stderr)
            continue
        train, holdout = make_split(triples, args.holdout_frac, args.seed)
        print(f"  {ed.slug}: {len(train)} train / {len(holdout)} holdout triple ids "
              f"(frac {args.holdout_frac}, seed {args.seed}, subject-grouped)")
        if args.dry_run:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "version": SPLIT_VERSION,
            "seed": args.seed,
            "holdout_frac": args.holdout_frac,
            "grouping": "subject",
            "n_triples": len(triples),
            "train_ids": sorted(train),
            "holdout_ids": sorted(holdout),
        }, indent=2), encoding="utf-8")
        print(f"    wrote {out}")


if __name__ == "__main__":
    main()
