#!/usr/bin/env bash
# V9 — coverage-max SFT. Goal: push coverage toward 100% while keeping fabrication ≤ 0.25.
# Levers vs v8: (1) answer-aug 6→8 (max positive phrasings → more willing to answer),
# (2) DROP --abstention-frac — the campaign proved global abstention is flat against fabrication;
# it only inflates the refusal fraction and suppresses coverage. The only refusals are the
# targeted absent-negatives (the real fabrication lever), kept modest (125) so refusal_frac ≈ 0.11
# — coverage-leaning but still inside the safe band. Single GPU (GPU0). build → train → gate.
# Writes data/_build/train_status.json each phase so the console can follow the whole chain live.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
RUN=edc_sft_v9
LOG=/tmp/v9_chain.log
# total_steps for the live console heartbeat — read from the config so it can't drift (was hardcoded 450).
TOTAL_STEPS=$(PYTHONPATH=scripts .venv/bin/python scripts/cpt_config.py --config configs/sft_edc.yaml --mode sft --get train.max_steps)

setstatus () {  # $1=status $2=phase
  .venv/bin/python - "$1" "$2" "$RUN" "$LOG" "$$" "$TOTAL_STEPS" <<'PY'
import json, sys, time, pathlib
status, phase, run, log, pid, total = sys.argv[1:7]
p = pathlib.Path("data/_build/train_status.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"run": run, "log_path": log, "pid": int(pid),
    "total_steps": int(total), "started": time.time(), "status": status, "phase": phase}))
PY
}
fail () { setstatus stopped "failed: $1"; echo "V9 FAILED: $1"; exit 1; }

if [ -d data/_build/30_tokenized/sft_v9.arrow ]; then
  echo "===== [V9 1/3] dataset already built (sft_v9.arrow) — reusing ====="
else
  setstatus running "building dataset"
  echo "===== [V9 1/3] build SFT dataset (coverage-max: answer-aug 8, absent-neg 125, no global abstention) ====="
  PYTHONPATH=scripts .venv/bin/python scripts/build_sft.py \
    --score-model checkpoints/cpt/edc_all97_v1 --knowledge-filter --self-gen --drop-generic \
    --val-frac 0.1 --answer-aug 8 --absent-neg 125 \
    --out sft_v9
  test -d data/_build/30_tokenized/sft_v9.arrow || fail "no sft_v9.arrow dataset"
fi

setstatus running "training"
echo "===== [V9 2/3] train edc_sft_v9 (from edc_all97_v1) ====="
# Config-generator pattern: derive a resolved config from the canonical SFT base + this run's overrides,
# then launch the ONE entrypoint with --config only (no behavior flags — see docs/CONFIG_AUDIT.md).
# train_fp4.sh sets the FP4/CUDA toolchain env the flashinfer kernel JIT needs; the trainer preserves $LOG.
GEN=/tmp/v9_sft.yaml
PYTHONPATH=scripts .venv/bin/python scripts/gen_config.py --base configs/sft_edc.yaml --out "$GEN" --mode sft \
  --set data.tokenized=data/_build/30_tokenized/sft_v9.arrow \
        run.save_dir=checkpoints/sft/edc_sft_v9 \
        data.init_from=checkpoints/cpt/edc_all97_v1 || fail "config generation failed"
scripts/train_fp4.sh --config "$GEN"
test -f checkpoints/sft/edc_sft_v9/model.safetensors || fail "no trained checkpoint"

setstatus running "gating checkpoints"
echo "===== [V9 3/3] gate the three checkpoints ====="
.venv/bin/python scripts/eval_sft_calibration.py \
  --checkpoints checkpoints/sft/edc_sft_v9/step_000200 checkpoints/sft/edc_sft_v9/step_000400 checkpoints/sft/edc_sft_v9 \
  --labels levered_v9_s200 levered_v9_s400 levered_v9_final --device cuda:0 || fail "gate failed"

setstatus done "complete"
echo "===== V9 chain done. Pick the highest-coverage passing checkpoint as canonical levered_v9. ====="
