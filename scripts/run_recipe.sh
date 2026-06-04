#!/usr/bin/env bash
# Run a Coach RECIPE as a gated campaign: build the SFT dataset with the recipe's build_sft flags →
# train (FP4 · int8-master) → gate. A parameterized run_v9.sh — the ONE recipe-runner the "Run this
# recipe" button and the torad_apply_recipe action invoke. Writes data/_build/train_status.json each
# phase so the console (and the GPU mutex) follow the whole chain live.
#
#   scripts/run_recipe.sh <label> <init_from_ckpt> [build_sft flags...]
#   e.g. scripts/run_recipe.sh coach_v12 checkpoints/cpt/edc_all97_v2 --drop-generic --answer-aug 8 --absent-neg 110
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
LABEL="${1:?usage: run_recipe.sh <label> <init_from> [build_sft flags...]}"; shift
INIT="${1:?need init_from checkpoint}"; shift
RUN="edc_sft_${LABEL}"
SAVE="checkpoints/sft/${RUN}"
ARROW="data/_build/30_tokenized/sft_${LABEL}.arrow"
mkdir -p "$SAVE"
LOG="${SAVE}/train.log"   # same file the launcher tees stdout into, so the console tails the whole chain
TOTAL=$(PYTHONPATH=scripts .venv/bin/python scripts/cpt_config.py --config configs/sft_edc.yaml --mode sft --get train.max_steps 2>/dev/null || echo 400)

setstatus () {  # $1=status $2=phase
  .venv/bin/python - "$1" "$2" "$RUN" "$LOG" "$$" "$TOTAL" <<'PY'
import json, sys, time, pathlib
status, phase, run, log, pid, total = sys.argv[1:7]
p = pathlib.Path("data/_build/train_status.json"); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"run": run, "log_path": log, "pid": int(pid),
    "total_steps": int(total), "started": time.time(), "status": status, "phase": phase}))
PY
}
fail () { setstatus stopped "failed: $1"; echo "RECIPE FAILED: $1"; exit 1; }

echo "===== [recipe 1/3] build SFT dataset sft_${LABEL}  (flags: $*) ====="
setstatus running "building dataset"
PYTHONPATH=scripts .venv/bin/python scripts/build_sft.py --score-model "$INIT" --out "sft_${LABEL}" "$@" || fail "build_sft"
test -d "$ARROW" || fail "no dataset $ARROW"

echo "===== [recipe 2/3] train ${RUN} (from ${INIT}) ====="
setstatus running "training"
GEN="/tmp/recipe_${LABEL}.yaml"
PYTHONPATH=scripts .venv/bin/python scripts/gen_config.py --base configs/sft_edc.yaml --out "$GEN" --mode sft \
  --set data.tokenized="$ARROW" run.save_dir="$SAVE" run.run_name="$RUN" data.init_from="$INIT" || fail "config gen"
scripts/train_fp4.sh --config "$GEN"
test -f "$SAVE/model.safetensors" || fail "no trained checkpoint at $SAVE"

echo "===== [recipe 3/3] GATE ${RUN} ====="
setstatus running "gating"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts .venv/bin/python scripts/eval.py \
  --config "$GEN" --mode sft --checkpoints "$SAVE" --labels "${LABEL}_final" || fail "gate"
setstatus done "complete"
echo "===== recipe ${LABEL}: built + trained + GATED → scorecard_${LABEL}_final.json (visible in the console). ====="
