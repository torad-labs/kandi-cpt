#!/usr/bin/env bash
# 2-GPU FSDP2 launch for Gemma 4 E2B full-param CPT.
# Usage: scripts/launch.sh [configs/train_edc_lv_2026.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/train_edc_lv_2026.yaml}"
# cu13 toolchain + memory-safe compile caps (MAX_JOBS) — see scripts/fp4_env.sh
source scripts/fp4_env.sh
exec .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/train_cpt.py --config "$CFG"
