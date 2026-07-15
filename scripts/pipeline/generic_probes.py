#!/usr/bin/env python3
"""Generic forced-choice probe generator — the domain-agnostic eval-items skeleton.

The festival `eval_cloze.py` builds forced-choice items over stages/days ("Which stage does X play?",
choices = the 7 stages). This is the generic analog: for each verified triple it builds a forced-choice
item — the question from `sft_templates.rel_q[predicate][0]`, the correct object, and N distractors drawn
from the SAME predicate's object pool (plausible wrong answers of the right type). Domain-agnostic: a new
domain authors NO probe code — the questions are its `rel_q`, the distractors are its own entities. The
probe *design* (held-out strategy, era tags, base-delta) stays the human-reviewed `domains/<d>.probes.md`;
this generates the *items* deterministically.

    PYTHONPATH=scripts python scripts/pipeline/generic_probes.py --domain <d> [--editions ...] [--distractors 3] [--dry-run]
Output: <edition>/probes.jsonl ({question, choices, answer, predicate, subject, source_triple_id, split}).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples
from pipeline.split import load_split, triple_id


def rel_q_for(domain: str) -> dict:
    """The domain's sft_templates.rel_q (predicate → [question, answer]) — the probe questions."""
    return ((Domain(domain).templates or {}).get("sft_templates") or {}).get("rel_q") or {}


def build_probes(gold_triples: list[dict], rel_q: dict, unit_name: str = "", n_distractors: int = 3,
                 seed: int = 0, pool_triples: list[dict] | None = None) -> list[dict]:
    """Forced-choice items over `gold_triples`, with distractors drawn from `pool_triples` (default:
    the same records) after excluding every object true for that subject+predicate. A gold triple is
    skipped if its predicate has no question template or no safe distractor pool (can't force a
    choice). Deterministic given `seed`."""
    rng = random.Random(seed)
    pool_triples = gold_triples if pool_triples is None else pool_triples
    by_pred: dict[str, set] = defaultdict(set)
    true_by_subject_pred: dict[tuple[str, str], set] = defaultdict(set)
    for t in pool_triples:
        if t["object"]:
            by_pred[t["predicate"]].add(t["object"])
            subject_id = str(t.get("subject_id", "") or t.get("subject", ""))
            true_by_subject_pred[(subject_id, t["predicate"])].add(t["object"])
    items: list[dict] = []
    seen_source_ids: set[str] = set()
    duplicate_sources = 0
    seen_q: dict = defaultdict(int)                        # predicate WITH a question template -> n triples seen
    made: dict = defaultdict(int)                          # predicate -> n probes produced
    drops: dict = defaultdict(lambda: defaultdict(int))    # predicate -> reason -> n
    for t in gold_triples:
        source_id = triple_id(t)
        if source_id in seen_source_ids:
            duplicate_sources += 1
            continue
        seen_source_ids.add(source_id)
        p, obj = t["predicate"], t["object"]
        q = rel_q.get(p)
        if not q:
            continue                                       # no question template → not a probe axis (no warn)
        seen_q[p] += 1
        if not obj:
            drops[p]["no_object"] += 1
            continue
        subject_id = str(t.get("subject_id", "") or t.get("subject", ""))
        pool = sorted(by_pred[p] - true_by_subject_pred[(subject_id, p)])
        if not pool:                                       # need ≥1 distractor of the right type
            drops[p]["no_safe_distractor"] += 1
            continue
        try:
            question = q[0].format(s=t["subject"], n=unit_name).strip()
        except (KeyError, IndexError):
            drops[p]["template_error"] += 1
            continue
        choices = rng.sample(pool, min(n_distractors, len(pool))) + [obj]
        rng.shuffle(choices)
        items.append({"question": question, "choices": choices, "answer": obj,
                      "predicate": p, "subject": t["subject"], "source_triple_id": source_id,
                      "split": "held_out"})
        made[p] += 1
    empty_pool_drops = sum(d.get("no_safe_distractor", 0) for d in drops.values())
    print(f"  source-id guard: dropped {duplicate_sources} duplicate semantic triple(s)")
    print(f"  multi-correct guard: dropped {empty_pool_drops} probe(s) with an empty distractor pool "
          f"after excluding all objects true for the same (subject_id, predicate)")
    # A predicate the operator authored a question for that yields ZERO probes is an UNTESTED fact axis — in an
    # eval set the absence of a test must not silently read as a passing test.
    for p, seen in seen_q.items():
        d = dict(drops[p])
        if not made[p]:
            print(f"  WARNING: predicate '{p}' has a question template but produced 0 probes "
                  f"({d} of {seen} triples) — that fact axis is UNTESTED", file=sys.stderr)
        elif d:                                            # partial drop — some instances silently left untested
            print(f"  WARNING: predicate '{p}': {made[p]}/{seen} triples → probes; {sum(d.values())} dropped "
                  f"({d}) — those fact instances are not in the eval set", file=sys.stderr)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions")
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="print per-unit counts, write nothing")
    args = ap.parse_args()
    rel_q = rel_q_for(args.domain)
    if not rel_q:
        raise SystemExit(f"domain '{args.domain}' has no sft_templates.rel_q — author it in "
                         f"configs/templates/{args.domain}.yaml (or check the key spelling/nesting if the file exists).")
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):   # consumes 15_structured, not 00_raw
        all_triples = load_all_triples(ed)
        if not all_triples:
            continue
        sp = load_split(ed.dir)
        if sp:
            gold_triples = [t for t in all_triples if triple_id(t) in sp["holdout_ids"]]
            split_label = "held_out"
            print(f"  {ed.slug}: split {sp['version']} held-out mode — {len(gold_triples)} gold / "
                  f"{len(all_triples)} distractor-pool triples")
        else:
            gold_triples = all_triples
            split_label = "trained"
            print(f"  {ed.slug}: trained mode — {len(gold_triples)} gold / {len(all_triples)} "
                  f"distractor-pool triples (no split manifest)")
        mp = ed.dir / "festival_meta.json"
        unit_name = (json.loads(mp.read_text()).get("name") if mp.exists() else None) or ed.slug
        items = build_probes(gold_triples, rel_q, unit_name, args.distractors,
                             pool_triples=all_triples)
        for item in items:
            item["split"] = split_label
            if sp:
                item["split_version"] = sp["version"]
        if args.dry_run:
            print(f"  {ed.slug}: {len(items)} forced-choice probes over {len({i['predicate'] for i in items})} predicates")
            continue
        out_path = ed.dir / "probes.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
        os.replace(tmp, out_path)
        print(f"  {ed.slug}: wrote {len(items)} {split_label} probes from {len(gold_triples)} gold triples "
              f"({len(all_triples)} distractor-pool triples)")


if __name__ == "__main__":
    main()
