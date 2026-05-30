#!/usr/bin/env python3
"""Autonomous overnight CPT exploration loop.

Iterates through a hyperparameter sweep, training + evaluating each config,
scoring by a weighted metric, and keeping the best checkpoint. Stops early
if a config hits the success criterion. All other checkpoints are deleted
to conserve disk (~1 GB per ckpt × 8 configs would eat ~8 GB).

Stop criteria:
  SUCCESS: era=2026 ≥ 3 AND voice == 20  → loop stops, declares winner
  BUDGET:  all configs tried              → loop stops, reports best
  ABORT:   3 consecutive train failures   → loop stops with error

Marcos can read live progress at:
  logs/loop_progress.md     — append-only log, tail -f to watch
  logs/best_ckpt.json       — current best config + metrics
  logs/loop_summary.md      — final summary (written when loop ends)

Run:
  CUDA_VISIBLE_DEVICES=1 python scripts/autonomous_loop.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "loop_progress.md"
SUMMARY_FILE = ROOT / "logs" / "loop_summary.md"
BEST_FILE = ROOT / "logs" / "best_ckpt.json"
CHECKPOINTS = ROOT / "checkpoints"

# Hyperparameter sweep — ordered by hypothesis strength.
# Each ~7 min train+eval × 8 configs = ~1 hour total, plenty of margin.
CONFIGS = [
    # Phase 1: LR sweep with embed_tokens in LoRA targets, 1 epoch.
    # ckpt 3 confirmed 5e-6 is too low (no loss descent). ckpt 2 at 2e-5/3ep
    # was too aggressive (catastrophic forgetting). Sweet spot is between.
    {"name": "iter_lr1e-5_e1",          "lr": 1e-5, "epochs": 1, "rank": 16, "embed_in_lora": True,  "embed_save": False},
    {"name": "iter_lr2e-5_e1",          "lr": 2e-5, "epochs": 1, "rank": 16, "embed_in_lora": True,  "embed_save": False},
    {"name": "iter_lr5e-5_e1",          "lr": 5e-5, "epochs": 1, "rank": 16, "embed_in_lora": True,  "embed_save": False},

    # Phase 2: more epochs at moderate LR
    {"name": "iter_lr1e-5_e2",          "lr": 1e-5, "epochs": 2, "rank": 16, "embed_in_lora": True,  "embed_save": False},
    {"name": "iter_lr5e-6_e3",          "lr": 5e-6, "epochs": 3, "rank": 16, "embed_in_lora": True,  "embed_save": False},

    # Phase 3: rank + control variations
    {"name": "iter_lr1e-5_e1_r32",      "lr": 1e-5, "epochs": 1, "rank": 32, "embed_in_lora": True,  "embed_save": False},
    {"name": "iter_lr1e-5_e1_no_embed", "lr": 1e-5, "epochs": 1, "rank": 16, "embed_in_lora": False, "embed_save": False},

    # Phase 4: try modules_to_save embed_tokens with paged optimizer
    # (will OOM if paged_adamw_8bit doesn't offload enough Adam state to CPU
    # — but worth one shot since this is the canonical Unsloth CPT path)
    {"name": "iter_lr1e-5_e1_embed_ft", "lr": 1e-5, "epochs": 1, "rank": 16, "embed_in_lora": False, "embed_save": True, "optim": "paged_adamw_8bit"},
]

SUCCESS_2026 = 3        # success: era=2026 pass count ≥ this
SUCCESS_VOICE = 20      # success: voice pass count == this
ABORT_FAILS = 3         # abort if this many consecutive failures
TRAIN_TIMEOUT = 1800    # 30 min per train
EVAL_TIMEOUT = 1200     # 20 min per eval

def log(msg, also_print=True):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    if also_print:
        print(msg, flush=True)

def run_subprocess(cmd, env, timeout, label):
    """Run subprocess, capture output, return (returncode, stdout_tail)."""
    try:
        res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=timeout)
        if res.returncode != 0:
            log(f"  {label} FAILED rc={res.returncode}")
            log(f"  stderr tail:\n```\n{res.stderr[-800:]}\n```")
        return res.returncode, res.stdout
    except subprocess.TimeoutExpired:
        log(f"  {label} TIMED OUT after {timeout}s")
        return -1, ""

def run_config(config):
    name = config["name"]
    env = {
        **os.environ,
        "KANDI_CKPT_NAME": name,
        "KANDI_LR": str(config["lr"]),
        "KANDI_EPOCHS": str(config["epochs"]),
        "KANDI_RANK": str(config["rank"]),
        "KANDI_EMBED_IN_LORA": "1" if config["embed_in_lora"] else "0",
        "KANDI_EMBED_SAVE": "1" if config["embed_save"] else "0",
        "KANDI_OPTIM": config.get("optim", "adamw_8bit"),
        "KANDI_SEQ_LEN": str(config.get("seq_len", 512)),
    }

    # Train
    rc, _ = run_subprocess(
        [sys.executable, "scripts/train_loop_iter.py"],
        env, TRAIN_TIMEOUT, f"train[{name}]",
    )
    if rc != 0:
        return None

    # Eval
    ckpt_path = CHECKPOINTS / name / "final"
    rc, _ = run_subprocess(
        [sys.executable, "scripts/eval.py", "--checkpoint", str(ckpt_path), "--max_probes", "30"],
        env, EVAL_TIMEOUT, f"eval[{name}]",
    )
    if rc != 0:
        return None

    # Move eval output to a named dir
    eval_dir = ROOT / "logs" / "eval_final"
    new_eval_dir = ROOT / "logs" / f"eval_{name}"
    if new_eval_dir.exists():
        shutil.rmtree(new_eval_dir)
    if eval_dir.exists():
        shutil.move(str(eval_dir), str(new_eval_dir))

    # Parse results
    try:
        probes = json.loads((new_eval_dir / "probes.json").read_text())
        voice = json.loads((new_eval_dir / "voice.json").read_text())
        train_metrics = json.loads((CHECKPOINTS / name / "metrics.json").read_text())
    except Exception as e:
        log(f"  parse-failed: {e}")
        return None

    era_2026 = probes["by_era"].get("2026", {"pass_count": 0, "total": 17})
    era_stable = probes["by_era"].get("stable", {"pass_count": 0, "total": 13})
    return {
        **train_metrics,
        "voice_pass": voice["pass_count"],
        "voice_total": voice["total"],
        "probes_pass": probes["pass_count"],
        "probes_total": probes["total"],
        "era_2026_pass": era_2026["pass_count"],
        "era_2026_total": era_2026["total"],
        "era_stable_pass": era_stable["pass_count"],
        "era_stable_total": era_stable["total"],
        "eval_dir": str(new_eval_dir),
    }

def score(m):
    """Higher = better. Heavy weight on era=2026 (the goal); penalize voice
    loss and stable regression. Tuned so any era=2026 gain dominates."""
    return (
        m["era_2026_pass"] * 5
        + m["era_stable_pass"]
        - max(0, 20 - m["voice_pass"]) * 3
    )

def is_success(m):
    return m["era_2026_pass"] >= SUCCESS_2026 and m["voice_pass"] >= SUCCESS_VOICE

def cleanup_ckpt(name, keep):
    """Delete ckpt dir unless name is in keep."""
    if name in keep:
        return
    p = CHECKPOINTS / name
    if p.exists():
        shutil.rmtree(p)

def write_summary(best, history, exit_reason):
    lines = [f"# Autonomous loop summary",
             f"Finished: {datetime.now()}",
             f"Exit: {exit_reason}",
             "",
             f"## Best config",
             "```json",
             json.dumps(best, indent=2) if best else "(none)",
             "```",
             "",
             f"## All iterations ({len(history)})",
             "",
             f"| name | lr | epochs | rank | embed_in | embed_ft | voice | 2026 | stable | score |",
             f"|---|---|---|---|---|---|---|---|---|---|"]
    for m in history:
        lines.append(
            f"| {m['name']} | {m['lr']} | {m['epochs']} | {m['rank']} | "
            f"{m['embed_in_lora']} | {m['embed_save']} | "
            f"{m['voice_pass']}/{m['voice_total']} | "
            f"{m['era_2026_pass']}/{m['era_2026_total']} | "
            f"{m['era_stable_pass']}/{m['era_stable_total']} | "
            f"{score(m)} |"
        )
    SUMMARY_FILE.write_text("\n".join(lines) + "\n")

def main():
    log(f"\n# Autonomous loop started at {datetime.now()}")
    log(f"  configs to try: {len(CONFIGS)}")
    log(f"  success criterion: era=2026 ≥ {SUCCESS_2026} AND voice == {SUCCESS_VOICE}")
    log(f"  best ckpt kept at: checkpoints/<best_name>; others auto-deleted")
    log("")

    best = None
    best_score = -1e9
    history = []
    consec_fails = 0
    exit_reason = "completed all configs"

    for i, config in enumerate(CONFIGS, 1):
        name = config["name"]
        log(f"\n## [{i}/{len(CONFIGS)}] {name}")
        log(f"  config: {json.dumps({k:v for k,v in config.items() if k != 'name'})}")
        t0 = time.time()
        metrics = run_config(config)
        dt = time.time() - t0
        if metrics is None:
            consec_fails += 1
            log(f"  failed after {dt:.0f}s (consec fails: {consec_fails}/{ABORT_FAILS})")
            cleanup_ckpt(name, keep={best["name"] if best else None})
            if consec_fails >= ABORT_FAILS:
                exit_reason = f"aborted after {consec_fails} consecutive failures"
                break
            continue
        consec_fails = 0
        history.append(metrics)

        s = score(metrics)
        ftl = metrics.get("final_train_loss")
        loss_str = f"{ftl:.3f}" if isinstance(ftl, (int, float)) else str(ftl)
        log(f"  voice: {metrics['voice_pass']}/{metrics['voice_total']}  "
            f"2026: {metrics['era_2026_pass']}/{metrics['era_2026_total']}  "
            f"stable: {metrics['era_stable_pass']}/{metrics['era_stable_total']}  "
            f"loss: {loss_str}  "
            f"score: {s}  ({dt:.0f}s)")

        if s > best_score:
            best_score = s
            # Clean up the previous best
            if best is not None:
                cleanup_ckpt(best["name"], keep=set())
            best = metrics
            BEST_FILE.write_text(json.dumps(best, indent=2))
            log(f"  *** NEW BEST (score {s}) ***")
        else:
            cleanup_ckpt(name, keep=set())

        if is_success(metrics):
            exit_reason = f"success criterion hit at {name} (era=2026={metrics['era_2026_pass']}, voice={metrics['voice_pass']})"
            log(f"\n  *** SUCCESS *** {exit_reason}")
            break

    log(f"\n# Loop done: {exit_reason}")
    if best:
        log(f"# Best: {best['name']} score={best_score}")
        log(f"# Checkpoint at: checkpoints/{best['name']}/final")
    write_summary(best, history, exit_reason)
    print(f"\nSummary written to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
