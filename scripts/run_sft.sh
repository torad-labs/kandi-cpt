#!/usr/bin/env bash
# Canonical SFT run = TRAIN → GATE, atomically. PROJECT RULE: no SFT run is ever left ungated.
# Use this instead of a bare train_fp4.sh for any SFT run, so the gate can't be forgotten/skipped.
#
#   scripts/run_sft.sh <label> [extra train_fp4 args...]      e.g.  scripts/run_sft.sh v12
#   CONFIG=configs/_sweep_x.yaml scripts/run_sft.sh sweep_x_t3   # any SFT config (sweeps / the launch action)
#
# Trains via $CONFIG (default configs/sft_edc.yaml: mb2×8 · fp4 · int8_master storage), then gates the
# final + last few step_* checkpoints with the comprehensive eval_sft_calibration → scorecard_<label>_*.json,
# which the dashboard (serve.py :8800) shows. The int8 checkpoints load via the int8-aware loader.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:?usage: run_sft.sh <label>  (e.g. v12)}"; shift || true
CFG="${CONFIG:-configs/sft_edc.yaml}"
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
# Source save_dir from the config CONTRACT (handles nested YAML; awk on '^save_dir:' silently broke when
# the config-drift refactor nested it under run:). One source of truth, no parser drift.
SAVE=$(PYTHONPATH=scripts .venv/bin/python -c "from cpt_config import CptConfig; print(CptConfig.from_yaml('$CFG', None).run.save_dir)")
[ -n "$SAVE" ] || { echo "no run.save_dir in $CFG"; exit 1; }

echo "===== [1/2] SFT train  (label=$LABEL · config=$CFG → $SAVE) ====="
bash scripts/train_fp4.sh --config "$CFG" "$@"
[ -f "$SAVE/model.safetensors" ] || { echo "✗ training produced no final checkpoint at $SAVE — NOT gating a non-existent run"; exit 1; }

echo "===== [2/2] GATE (mandatory) ====="
CKPTS="$SAVE"; LABELS="${LABEL}_final"
for s in $(ls -d "$SAVE"/step_* 2>/dev/null | sort | tail -2); do
  st=$(basename "$s" | sed -E 's/step_0*([0-9]+)/s\1/')
  CKPTS="$s $CKPTS"; LABELS="${LABEL}_${st} $LABELS"
done
# Gate through the ONE eval entrypoint (thresholds/device come from the config's eval: section).
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts .venv/bin/python scripts/eval.py \
  --config "$CFG" --mode sft --checkpoints $CKPTS --labels $LABELS
echo "===== $LABEL: trained + GATED → scorecard_${LABEL}_final.json (visible in the dashboard). ====="
