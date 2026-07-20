#!/usr/bin/env python3
"""Phase F Task 3 — generative axis + dual-axis merge (split_v2 series baseline).

Uses the Task-1 κ-calibration receipt (already κ=1.0 usable) instead of silently
re-running 28 judge calls with no progress. Model generate + live judge still run.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch
from cpt_config import CptConfig
from eval_generative import (
    DEFAULT_JUDGE_MODEL,
    KAPPA_USABLE,
    OPENROUTER_BASE_URL,
    UsageMeter,
    build_generative_report,
    generate_answer,
    judge_one,
    merge_dual_axis,
    resolve_openrouter_key,
    select_generative_items,
)
from eval_probes import load_model, load_probe_corpus


def main() -> None:
    cfg = CptConfig.from_yaml("configs/train_pokemon_lore_kanto_anneal.yaml", "cpt")
    ckpt = "checkpoints/cpt/pokemon_lore_kanto_anneal"
    device = cfg.eval.device  # cuda:0 under CUDA_VISIBLE_DEVICES=1 → physical GPU1
    print(f"device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    print("load probe corpus…", flush=True)
    corpus = load_probe_corpus(cfg.eval.probe_domain, cfg.eval.probe_editions, cfg.data.tokenized)
    items = select_generative_items(corpus.probes, probe_sha256=corpus.probe_sha256)
    print(f"generative items n={len(items)} split={corpus.split_version} class={corpus.measurement_class}", flush=True)

    # ── calibration from Task 1 receipt (κ already measured on the same set+model) ──
    cal_receipt = json.loads(Path("logs/phase_f_judge_calibration.json").read_text(encoding="utf-8"))
    final = cal_receipt["final"]
    calibration = {
        "n": cal_receipt["calibration_set"]["n"],
        "kappa": final["binary_kappa"],
        "accuracy": final["binary_accuracy"],
        "usable": bool(final["usable"]),
        "provisional": not bool(final["usable"]),
        "skipped": False,
        "source": "logs/phase_f_judge_calibration.json",
        "multiclass_kappa": final["multiclass_kappa"],
        "multiclass_accuracy": final["multiclass_accuracy"],
        "kappa_usable_threshold": KAPPA_USABLE,
        "judge_model": cal_receipt["judge_model"],
        "calibration_set_sha256": cal_receipt["calibration_set"]["sha256"],
        "note": "reused Task-1 live calibration (same judge model + calibration json); not re-spent",
    }
    print(
        f"calibration reused: binary_κ={calibration['kappa']:.3f} usable={calibration['usable']}",
        flush=True,
    )

    key = resolve_openrouter_key()
    if not key:
        raise SystemExit("no OpenRouter key")
    judge_model = DEFAULT_JUDGE_MODEL
    judge_ctx = (judge_model, key, OPENROUTER_BASE_URL + "/chat/completions")
    meter = UsageMeter()

    def grade(answers: list[str], tag: str) -> list[str]:
        labels = []
        for i, (item, ans) in enumerate(zip(items, answers, strict=True), 1):
            lab = judge_one(item.question, item.gold, ans, judge_ctx, meter)
            labels.append(lab)
            if i % 5 == 0 or i == len(items):
                print(f"  judge {tag} {i}/{len(items)} last={lab}", flush=True)
        return labels

    # ── tuned generate ──
    print(f"=== load TUNED {ckpt} ===", flush=True)
    tok, model = load_model(ckpt, device)
    print("tuned loaded; generating…", flush=True)
    tuned_answers = []
    for i, item in enumerate(items, 1):
        tuned_answers.append(generate_answer(tok, model, device, item.question))
        if i % 5 == 0 or i == len(items):
            print(f"  gen tuned {i}/{len(items)}", flush=True)
    del model
    torch.cuda.empty_cache()
    print("judge tuned…", flush=True)
    tuned_labels = grade(tuned_answers, "tuned")

    # ── base generate ──
    base_path = cfg.model.model_path
    print(f"=== load BASE {base_path} ===", flush=True)
    tok, model = load_model(base_path, device)
    print("base loaded; generating…", flush=True)
    base_answers = []
    for i, item in enumerate(items, 1):
        base_answers.append(generate_answer(tok, model, device, item.question))
        if i % 5 == 0 or i == len(items):
            print(f"  gen base {i}/{len(items)}", flush=True)
    del model
    torch.cuda.empty_cache()
    print("judge base…", flush=True)
    base_labels = grade(base_answers, "base")

    generative_bar = float(getattr(cfg.eval, "generative_bar", 0.40))
    report = build_generative_report(
        run_name=cfg.run.run_name,
        config_path=Path("configs/train_pokemon_lore_kanto_anneal.yaml"),
        base_path=base_path,
        tuned_path=ckpt,
        corpus=corpus,
        items=items,
        base_labels=base_labels,
        tuned_labels=tuned_labels,
        judge_model=judge_model,
        calibration=calibration,
        min_n=cfg.eval.probe_min_n,
        min_delta_acc=float(getattr(cfg.eval, "generative_min_delta_acc", 0.05)),
        generative_bar=generative_bar,
        config_fingerprint=cfg.fingerprint(),
        tokenized=cfg.data.tokenized,
    )
    report["judge"]["usage"] = {
        "calls": meter.calls,
        "prompt_tokens": meter.prompt_tokens,
        "completion_tokens": meter.completion_tokens,
    }
    report["answers"] = {
        "tuned": tuned_answers,
        "base": base_answers,
        "tuned_labels": tuned_labels,
        "base_labels": base_labels,
    }

    out_dir = Path("logs/probe_evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "pokemon_lore_kanto_anneal_generative_splitv2.json"
    # strip bulky answers from the durable report copy? keep them — diagnosis value.
    gen_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "acc_base": (report.get("base") or {}).get("acc"),
                "acc_tuned": (report.get("tuned") or {}).get("acc"),
                "kappa": calibration["kappa"],
                "usable": calibration["usable"],
                "label_counts": (report.get("tuned") or {}).get("label_counts"),
                "n": (report.get("tuned") or {}).get("n"),
                "path": str(gen_path),
            },
            indent=2,
        ),
        flush=True,
    )

    fc = json.loads(Path("logs/probe_evals/pokemon_lore_kanto_anneal_base_vs_tuned.json").read_text())
    dual = merge_dual_axis(fc, report)
    dual_path = out_dir / "pokemon_lore_kanto_anneal_install_strength.json"
    dual_path.write_text(json.dumps(dual, indent=2) + "\n")
    print("DUAL_DONE", dual["status"], dual_path, flush=True)
    print(
        json.dumps(
            {
                "status": dual["status"],
                "forced_choice": dual.get("forced_choice"),
                "generative": dual.get("generative"),
                "rule": dual.get("rule"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
        Path("logs/phase_f_split_v2_generative.ok").write_text("ok\n")
    except Exception:
        traceback.print_exc()
        Path("logs/phase_f_split_v2_generative.ok").write_text("fail\n")
        raise
