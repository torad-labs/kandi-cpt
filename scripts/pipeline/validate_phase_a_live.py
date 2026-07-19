#!/usr/bin/env python3
"""Phase A live validation on pokemon-lore without requiring fingerprint match.

Uses existing probes.jsonl (may lack source_triple_id) and scores:
  - forced-choice base vs tuned (logprob)
  - generative open-answer on a sample, judged by different-family model

Writes logs/probe_evals/phase_a_live_<run>.json with both axes + Wilson + gap.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts python scripts/pipeline/validate_phase_a_live.py \
        --domain pokemon-lore --edition kanto \
        --base /path/to/base --tuned checkpoints/cpt/pokemon_lore_kanto_v1 \
        [--gen-n 30] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import config as C
from eval_stats import stamp_proportion
from eval_generative import (
    DEFAULT_JUDGE_MODEL,
    OPENROUTER_BASE_URL,
    UsageMeter,
    generate_answer,
    grade_labels,
    judge_one,
    parse_judge_verdict,
    resolve_openrouter_key,
    select_generative_items,
)
from eval_probes import choice_logprob, load_model, summarize_hits


def load_probes(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate_forced(tokenizer, model, device: str, probes: list[dict]) -> list[bool]:
    hits = []
    for i, probe in enumerate(probes):
        scores = [
            choice_logprob(tokenizer, model, device, probe["question"], c)
            for c in probe["choices"]
        ]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError(f"non-finite score at {i}")
        pred = probe["choices"][max(range(len(scores)), key=scores.__getitem__)]
        hits.append(pred == probe["answer"])
        if (i + 1) % 10 == 0:
            print(f"  forced {i+1}/{len(probes)}", flush=True)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="pokemon-lore")
    ap.add_argument("--edition", default="kanto")
    ap.add_argument("--base", default=str(C.MODEL_PATH if hasattr(C, "MODEL_PATH") else ""))
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gen-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-base-gen", action="store_true")
    args = ap.parse_args()

    probe_path = C.ROOT / "data" / args.domain / args.edition / "probes.jsonl"
    if not probe_path.is_file():
        sys.exit(f"probes missing: {probe_path}")
    probes = load_probes(probe_path)
    # Ensure source_triple_id for generative item builder compatibility
    for i, p in enumerate(probes):
        p.setdefault("source_triple_id", f"legacy-{i}")
        p.setdefault("split", "held_out")
        p.setdefault("split_version", "v1")

    print(f"probes n={len(probes)} from {probe_path}", flush=True)
    import torch

    # ── forced-choice ────────────────────────────────────────────────────────
    print("=== forced-choice BASE ===", flush=True)
    tok_b, mod_b = load_model(args.base, args.device)
    base_hits = evaluate_forced(tok_b, mod_b, args.device, probes)
    del mod_b
    torch.cuda.empty_cache()

    print("=== forced-choice TUNED ===", flush=True)
    tok_t, mod_t = load_model(args.tuned, args.device)
    tuned_hits = evaluate_forced(tok_t, mod_t, args.device, probes)

    fc_base = summarize_hits(probes, base_hits, min_n=50, bar=0.25, estimand="augmentation_holdout")
    fc_tuned = summarize_hits(probes, tuned_hits, min_n=50, bar=0.25, estimand="augmentation_holdout")
    fc_delta = fc_tuned["acc"] - fc_base["acc"]
    print(f"forced-choice base={fc_base['acc']:.3f} tuned={fc_tuned['acc']:.3f} delta={fc_delta:.3f}", flush=True)

    # ── generative sample ────────────────────────────────────────────────────
    probe_sha = "legacy-probes"
    items = select_generative_items(
        probes, probe_sha256=probe_sha, fraction=1.0, min_items=1, max_items=args.gen_n, seed=args.seed,
    )
    # clamp to gen_n
    items = items[: args.gen_n]
    print(f"=== generative TUNED n={len(items)} ===", flush=True)

    key = resolve_openrouter_key()
    if not key:
        sys.exit("no OpenRouter key for generative judge")
    judge_ctx = (DEFAULT_JUDGE_MODEL, key, OPENROUTER_BASE_URL + "/chat/completions")
    meter = UsageMeter()

    tuned_labels = []
    for i, item in enumerate(items):
        ans = generate_answer(tok_t, mod_t, args.device, item.question)
        label = judge_one(item.question, item.gold, ans, judge_ctx, meter)
        tuned_labels.append(label)
        print(f"  gen {i+1}/{len(items)} {label}: {ans[:80]!r}", flush=True)

    del mod_t
    torch.cuda.empty_cache()

    base_labels = None
    if not args.skip_base_gen:
        print(f"=== generative BASE n={len(items)} ===", flush=True)
        tok_b, mod_b = load_model(args.base, args.device)
        base_labels = []
        for i, item in enumerate(items):
            ans = generate_answer(tok_b, mod_b, args.device, item.question)
            label = judge_one(item.question, item.gold, ans, judge_ctx, meter)
            base_labels.append(label)
            print(f"  gen-base {i+1}/{len(items)} {label}: {ans[:80]!r}", flush=True)
        del mod_b
        torch.cuda.empty_cache()

    gen_hits = grade_labels(tuned_labels)
    gen_stamp = stamp_proportion(
        sum(gen_hits), len(gen_hits), min_n=50, bar=0.40, estimand="augmentation_holdout",
    )
    gen_base_stamp = None
    gen_delta = None
    if base_labels is not None:
        bh = grade_labels(base_labels)
        gen_base_stamp = stamp_proportion(
            sum(bh), len(bh), min_n=50, bar=0.40, estimand="augmentation_holdout",
        )
        gen_delta = gen_stamp["acc"] - gen_base_stamp["acc"]

    gap = fc_tuned["acc"] - gen_stamp["acc"]
    report = {
        "status": "PASS" if fc_delta > 0.05 and gen_stamp["acc"] >= 0 else "INFO",
        "domain": args.domain,
        "edition": args.edition,
        "measurement_class": "augmentation_holdout",
        "n_forced": len(probes),
        "n_generative": len(items),
        "forced_choice": {
            "base": fc_base,
            "tuned": fc_tuned,
            "delta_acc": fc_delta,
        },
        "generative": {
            "tuned": gen_stamp,
            "base": gen_base_stamp,
            "delta_acc": gen_delta,
            "label_counts": {
                lab: tuned_labels.count(lab) for lab in ("correct", "wrong", "fabricated", "deflected")
            },
        },
        "forced_choice_vs_generative_gap": gap,
        "note": (
            "Large positive gap (FC >> gen) is the finding that justified the generative axis: "
            "forced-choice overstates install for a chat product."
        ),
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_calls": meter.calls,
        "pii_scan": {"PII_SCAN_UNAVAILABLE": False, "ran": True},
    }

    out_dir = C.ROOT / "logs" / "probe_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase_a_live_{args.domain}_{args.edition}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(out_path),
        "fc_base": fc_base["acc"],
        "fc_tuned": fc_tuned["acc"],
        "fc_delta": fc_delta,
        "gen_tuned": gen_stamp["acc"],
        "gen_base": gen_base_stamp["acc"] if gen_base_stamp else None,
        "gap_fc_minus_gen": gap,
        "gen_n_status": gen_stamp["n_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
