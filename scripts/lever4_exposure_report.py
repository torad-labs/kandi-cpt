#!/usr/bin/env python3
"""Lever 4 — exposures-to-extraction curve, per fact_class.

Joins the {fact_id: cumulative_count} snapshot that train_cpt writes beside each checkpoint
(_save_exposure → <ckpt>/exposure.json) to the held-out cloze probe's per-fact pass/fail at that
checkpoint, then reports extraction rate as a function of cumulative exposures — bucketed, one curve per
fact_class {atomic, relational, confusable} — plus the exposure at which extraction crosses --threshold.

The payoff: which fact-classes are OVER-exposed (saturated well before max exposures → wasted tokens) and
which are UNDER-exposed (still climbing at max → add volume). Feeds back into corpus volume allocation.

Reuses scripts/eval_cloze.py (the shared probe) per convention #2 — it does NOT fork or reimplement it.

    PYTHONPATH=scripts python scripts/lever4_exposure_report.py \
        --checkpoints checkpoints/cpt/<run>/step_000100 checkpoints/cpt/<run>/step_000200 \
                      checkpoints/cpt/<run> [--threshold 0.8]
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import torch
from transformers import AutoTokenizer, Gemma4ForCausalLM

import config as C
import eval_cloze as EC
from pipeline import span_match as SM
from pipeline.fact_expand import camel


def per_fact_pass(model, tok, a_stage, stage_slugs) -> dict:
    """Per-artist stage extraction pass/fail: a true stage ranked #1 by a majority of the stage templates
    (mirrors eval_cloze.score's forced-choice; aggregated to a per-fact boolean for the join)."""
    stage_cands = [f" {camel(s)}" for s in stage_slugs]
    slug_of = {f" {camel(s)}": s for s in stage_slugs}
    out = {}
    for a, trueset in a_stage.items():
        hits = sum(int(slug_of[EC.ranked(model, tok, tpl(a), stage_cands)[0]] in trueset)
                   for _, tpl in EC.STAGE_TPLS)
        out[a] = hits >= 2                          # majority of the 3 stage templates
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="ckpt dirs that contain exposure.json")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
    a_stage, _a_day, stage_slugs = EC.load_facts()
    fclass = SM.fact_class_map()                     # fact_id -> {confusable, relational}
    art_facts = collections.defaultdict(list)
    for fid in fclass:
        art_facts[fid.split("|")[0]].append(fid)

    rows = []                                        # (fact_class, cumulative_exposure, passed)
    for ck in args.checkpoints:
        exp_path = os.path.join(ck, "exposure.json")
        if not os.path.exists(exp_path):
            print(f"[skip] {ck}: no exposure.json (was lever4 on during training?)")
            continue
        exp = json.load(open(exp_path)).get("exposure", {})
        from eval_probes import load_model
        _tok, m = load_model(ck, args.device)   # int8-master native (quartet fwd); never bf16-resident
        passed = per_fact_pass(m, tok, a_stage, stage_slugs)
        del m
        torch.cuda.empty_cache()
        for a, ok in passed.items():
            for fid in art_facts.get(a, []):
                rows.append((fclass.get(fid, "relational"), exp.get(fid, 0), int(ok)))

    if not rows:
        print("no (exposure, pass/fail) rows — need checkpoints with exposure.json")
        return

    # bucket exposures (log-ish edges) and report extraction rate per class + threshold crossing
    edges = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 10 ** 9]
    print(f"\n=== exposures → extraction (threshold {args.threshold}) ===")
    print(f"{'class':12} {'exposure bucket':>16} {'extraction':>11} {'n':>5}")
    for cls in ("confusable", "relational", "atomic"):
        crows = [(e, p) for c, e, p in rows if c == cls]
        if not crows:
            continue
        cross = None
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            b = [p for e, p in crows if lo <= e < hi]
            if not b:
                continue
            rate = sum(b) / len(b)
            lbl = f"{lo}-{hi if hi < 10**9 else '∞'}"
            print(f"{cls:12} {lbl:>16} {rate:>11.2f} {len(b):>5}")
            if cross is None and rate >= args.threshold:
                cross = lo
        print(f"{cls:12} → extraction first crosses {args.threshold} at ~{cross if cross is not None else '>max'} exposures\n")


if __name__ == "__main__":
    main()
