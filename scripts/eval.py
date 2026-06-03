#!/usr/bin/env python3
"""THE ONE EVAL ENTRYPOINT. Accepts only --config + --mode + --checkpoints (+ optional --labels).

╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║  One eval entrypoint, mirror of the one training entrypoint. Reads the gate thresholds + probe  ║
║  knobs + device from the config's `eval` section (no behavior flags). It DISPATCHES to the      ║
║  existing eval libraries via subprocess — it does NOT reimplement them, so their invariants     ║
║  (gate determinism, era-tagged probes, deflection-aware grading, voice-vs-memory prompt         ║
║  separation, NFKD gold-fold, forced-choice surface) are preserved untouched.                    ║
║  Full rule: AGENTS.md + docs/CONFIG_AUDIT.md.                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

  PYTHONPATH=scripts python scripts/eval.py --config configs/sft_edc.yaml --mode sft \\
      --checkpoints checkpoints/sft/edc_sft_v11/step_000200 checkpoints/sft/edc_sft_v11 \\
      --labels v11_s200 v11_final

  mode=sft → eval_sft_calibration.py (THE GATE: fab≤tau_fab ∧ acc≥tau_acc ∧ cov≥tau_cov), thresholds
             + n_absent/seed/scale/batch/device taken from config.eval.
  mode=cpt → eval_cpt.py (era-tagged install probes, per checkpoint) + eval_cloze.py (forced-choice),
             device from config.eval.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpt_config import CptConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")


def _run(argv: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": "scripts"}
    print(f"\n$ {' '.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=str(ROOT), env=env).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="kandi-cpt eval — the ONE eval entrypoint (config-driven).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cpt", "sft"], default=None, help="defaults to the config's mode")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint dir(s) to evaluate")
    ap.add_argument("--labels", nargs="*", default=None, help="display labels (sft gate); defaults to basenames")
    args = ap.parse_args()

    cfg = CptConfig.from_yaml(args.config, args.mode)
    e = cfg.eval

    if cfg.mode == "sft":
        labels = args.labels or [Path(c).name for c in args.checkpoints]
        if len(labels) != len(args.checkpoints):
            sys.exit(f"--labels ({len(labels)}) must match --checkpoints ({len(args.checkpoints)})")
        rc = _run([PY, "scripts/eval_sft_calibration.py",
                   "--checkpoints", *args.checkpoints, "--labels", *labels,
                   "--tau-fab", str(e.tau_fab), "--tau-acc", str(e.tau_acc), "--tau-cov", str(e.tau_cov),
                   "--n-absent", str(e.n_absent), "--seed", str(e.seed),
                   "--scale", str(e.scale), "--batch", str(e.batch), "--device", e.device])
        sys.exit(rc)

    # mode == cpt: install probes (per checkpoint) + forced-choice cloze (all at once).
    rc = 0
    for ckpt in args.checkpoints:
        rc |= _run([PY, "scripts/eval_cpt.py", "--checkpoint", ckpt])
    rc |= _run([PY, "scripts/eval_cloze.py", "--checkpoints", *args.checkpoints, "--device", e.device])
    sys.exit(rc)


if __name__ == "__main__":
    main()
