#!/usr/bin/env bash
# Lever 1 (entity-targeted loss) A/B sweep on the rebuilt corpus.
#
# Sequential, GPU0-solo — NEVER two GPUs at once (1200W PSU OCP trips → hard crash). Each run is native
# FP4 + 4-bit optimizer + Lever 1 at the given entity_loss_weight + Lever 4 exposure tracking, on the
# identical WSD schedule. weight=1.0 is the in-lever baseline (== unweighted mean CE). After each run we
# eval (batched cloze) and append to the summary, so partial results are visible as the sweep proceeds.
#
#   ./scripts/run_lever1_sweep.sh "1 2 3 5"   (default weights)
set -uo pipefail
cd "$(dirname "$0")/.."
WEIGHTS="${1:-1 2 3 5}"
SUMMARY=logs/lever1_sweep_results.log
echo "=== Lever 1 sweep started $(date '+%H:%M:%S')  weights={$WEIGHTS} ===" | tee -a "$SUMMARY"
CKS=()
for W in $WEIGHTS; do
  RUN="edc_lv_2026_L1w${W/./p}"
  GEN="/tmp/l1w${W/./p}.yaml"
  echo "=== [sweep] training $RUN (entity_loss_weight=$W) $(date '+%H:%M:%S') ==="
  # Config-generator pattern: vary the run by WRITING a config, not by passing flags (which no longer exist —
  # see docs/CONFIG_AUDIT.md). Lever1 at weight W + Lever4 + this run's save_dir, on the canonical base.
  PYTHONPATH=scripts .venv/bin/python scripts/gen_config.py --base configs/train_edc_lv_2026.yaml --out "$GEN" \
     --set train.optimizer_lowbit=4 \
           train.levers.lever1_entity_loss.enabled=true \
           train.levers.lever1_entity_loss.entity_loss_weight="$W" \
           train.levers.lever4_exposure_tracking.enabled=true \
           "run.save_dir=checkpoints/cpt/$RUN" \
     || { echo "=== [sweep] $RUN config-gen FAILED ===" | tee -a "$SUMMARY"; continue; }
  # success marker read FROM the generated config so it can't drift (was a hardcoded 'step 240/240').
  MAXS=$(PYTHONPATH=scripts .venv/bin/python scripts/cpt_config.py --config "$GEN" --get train.max_steps)
  ./scripts/train_fp4.sh --config "$GEN" > "logs/train_$RUN.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || ! grep -q "step ${MAXS}/${MAXS}" "logs/train_$RUN.log"; then
    echo "=== [sweep] $RUN FAILED (rc=$rc, expected step ${MAXS}/${MAXS}) — see logs/train_$RUN.log ===" | tee -a "$SUMMARY"
    tail -6 "logs/train_$RUN.log"
    continue
  fi
  echo "=== [sweep] $RUN done: $(grep "step ${MAXS}/${MAXS}" logs/train_$RUN.log | tail -1) ===" | tee -a "$SUMMARY"
  CKS+=("checkpoints/cpt/$RUN")
done

echo "=== [sweep] batched eval — base vs ${CKS[*]} $(date '+%H:%M:%S') ===" | tee -a "$SUMMARY"
PYTHONPATH=scripts CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_cloze.py \
  --checkpoints "${CKS[@]}" --device cuda:0 2>&1 | grep -ivE "Warning|warn|KernelPref|register_constant" | tee -a "$SUMMARY"

# Lever-4 exposure→extraction curve (uses the per-checkpoint exposure.json snapshots from the w=1 run)
if [ -f "checkpoints/cpt/edc_lv_2026_L1w1/exposure.json" ]; then
  echo "=== [sweep] Lever-4 exposure→extraction (w=1 run) ===" | tee -a "$SUMMARY"
  PYTHONPATH=scripts CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/lever4_exposure_report.py \
    --checkpoints checkpoints/cpt/edc_lv_2026_L1w1/step_000100 \
                  checkpoints/cpt/edc_lv_2026_L1w1/step_000200 \
                  checkpoints/cpt/edc_lv_2026_L1w1 --device cuda:0 2>&1 \
    | grep -ivE "Warning|warn|KernelPref|register_constant" | tee -a "$SUMMARY"
fi
echo "=== Lever 1 sweep complete $(date '+%H:%M:%S') ===" | tee -a "$SUMMARY"
