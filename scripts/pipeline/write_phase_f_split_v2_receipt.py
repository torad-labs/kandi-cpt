#!/usr/bin/env python3
"""Assemble logs/phase_f_split_v2_dual_axis.json from FC + generative + dual receipts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    fc = json.loads(Path("logs/probe_evals/pokemon_lore_kanto_anneal_base_vs_tuned.json").read_text())
    gen_path = Path("logs/probe_evals/pokemon_lore_kanto_anneal_generative_splitv2.json")
    dual_path = Path("logs/probe_evals/pokemon_lore_kanto_anneal_install_strength.json")
    if not gen_path.is_file() or not dual_path.is_file():
        raise SystemExit(f"missing generative/dual receipts: {gen_path} {dual_path}")
    gen = json.loads(gen_path.read_text())
    dual = json.loads(dual_path.read_text())
    cal = json.loads(Path("logs/phase_f_judge_calibration.json").read_text())
    probes = Path("data/pokemon-lore/kanto/probes.jsonl").read_bytes()
    probe_file_sha = hashlib.sha256(probes).hexdigest()
    n_probes = sum(1 for line in probes.splitlines() if line.strip())

    fc_base = fc["base"]["acc"]
    fc_tuned = fc["tuned"]["acc"]
    gen_base = (gen.get("base") or {}).get("acc")
    gen_tuned = (gen.get("tuned") or {}).get("acc")
    gen_labels = (gen.get("tuned") or {}).get("label_counts")

    receipt = {
        "title": "Phase F Task 3 — split_v2 dual-axis series baseline (pokemon-lore/kanto)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "series": "split_v2",
        "series_note": (
            "NEW eval series (alias+inverse closed). Do NOT merge with v1-era numbers "
            "(phase_a / phase_e). This receipt is the v2 baseline."
        ),
        "estimand": "augmentation_holdout",
        "estimand_note": (
            "Shard includes clean prose + hydrated/replay; holdout triples are graph-held-out "
            "under split_v2, but the training mix is not pure fact-holdout. Label every number "
            "augmentation_holdout."
        ),
        "checkpoint": {
            "tuned": "checkpoints/cpt/pokemon_lore_kanto_anneal",
            "scored": "anneal-end",
            "base": "/home/marcos/Documents/dev/infra/models/gemma-4-E2B-it",
            "config": "configs/train_pokemon_lore_kanto_anneal.yaml",
            "gpu": "physical GPU1 via CUDA_VISIBLE_DEVICES=1",
            "precision": "fp4 + int8 master (unchanged)",
        },
        "probes": {
            "path": "data/pokemon-lore/kanto/probes.jsonl",
            "n": n_probes,
            "split": "held_out",
            "split_version": "v2",
            "probe_sha256": fc["inputs"]["probe_sha256"],
            "probe_file_sha256": probe_file_sha,
            "split_sha256": fc["inputs"]["split_sha256"],
            "regenerated_from": (
                "generic_probes over split_v2 holdout_ids on post-Task-2 graph "
                "(triples.resolved 828 after second-pass extract; split_v2 immutable — no re-cut)"
            ),
            "legacy_v1_probes_backup": "data/pokemon-lore/kanto/probes_v1_legacy.jsonl",
            "graph_note": (
                "Task 2 second-pass changed the graph (+107 triples). Probes regenerated from "
                "the same split_v2 holdout over the corrected graph, as required."
            ),
        },
        "judge": {
            "model": "qwen/qwen3-32b",
            "calibration_receipt": "logs/phase_f_judge_calibration.json",
            "binary_kappa": cal["final"]["binary_kappa"],
            "usable": cal["final"]["usable"],
            "kappa_threshold": 0.6,
            "note": "Task-1 calibration reused into generative report (same set hash + model)",
        },
        "forced_choice": {
            "n": fc["base"]["n"],
            "base_acc": fc_base,
            "tuned_acc": fc_tuned,
            "delta_acc": fc["delta_acc"],
            "base_hits": fc["base"]["hits"],
            "tuned_hits": fc["tuned"]["hits"],
            "status": fc["status"],
            "n_status": fc["base"].get("n_status"),
            "estimand": "augmentation_holdout",
            "report": "logs/probe_evals/pokemon_lore_kanto_anneal_base_vs_tuned.json",
        },
        "generative": {
            "n": (gen.get("tuned") or {}).get("n"),
            "base_acc": gen_base,
            "tuned_acc": gen_tuned,
            "delta_acc": None if gen_base is None or gen_tuned is None else gen_tuned - gen_base,
            "label_counts_tuned": gen_labels,
            "status": gen.get("status"),
            "estimand": "augmentation_holdout",
            "report": str(gen_path),
        },
        "dual_axis": {
            "status": dual.get("status"),
            "forced_choice": dual.get("forced_choice"),
            "generative": dual.get("generative"),
            "rule": dual.get("rule"),
            "report": str(dual_path),
        },
        "v1_era_reference_DO_NOT_MERGE": {
            "note": "phase_e anneal numbers on legacy v1 probes — recall-class history only",
            "receipt": "logs/phase_e_anneal_dual_axis.json",
            "fc_tuned": 0.6389,
            "gen_tuned": 0.5667,
            "n_fc": 72,
        },
        "verdicts": {
            "run_gate": "probes regenerated + dual-axis scored on anneal-end vs Gemma base",
            "install_strength": dual.get("status"),
            "series_baseline": "THIS receipt is the split_v2 series baseline",
            "summary": (
                f"FC base={fc_base:.4f} tuned={fc_tuned:.4f} Δ={fc['delta_acc']:.4f} (n={fc['base']['n']}); "
                f"GEN base={gen_base} tuned={gen_tuned} (n={(gen.get('tuned') or {}).get('n')}); "
                f"dual={dual.get('status')}; estimand=augmentation_holdout; split_version=v2; "
                f"judge usable κ={cal['final']['binary_kappa']:.3f}"
            ),
        },
        "receipts": {
            "fc": "logs/probe_evals/pokemon_lore_kanto_anneal_base_vs_tuned.json",
            "generative": str(gen_path),
            "dual": str(dual_path),
            "generative_log": "logs/phase_f_split_v2_generative.log",
            "this_file": "logs/phase_f_split_v2_dual_axis.json",
            "judge_calibration": "logs/phase_f_judge_calibration.json",
            "audit_recall": "logs/phase_f_audit_recall_kanto.json",
        },
    }
    out = Path("logs/phase_f_split_v2_dual_axis.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "dual_status": dual.get("status"), "summary": receipt["verdicts"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
