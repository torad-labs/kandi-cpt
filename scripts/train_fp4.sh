#!/usr/bin/env bash
# Canonical native-FP4 (Quartet-II / flashinfer cutlass sm120) training launcher.
#
# The flashinfer cutlass-FP4 sm120 kernel JIT-compiles with nvcc on first use. On THIS host that
# requires a specific toolchain — without it the JIT dies with cryptic errors:
#   • CUDA >= 12.9 toolkit (sm120/Blackwell is unsupported on 12.8; nvcc 12.8 -> "No supported CUDA
#     architectures found for major versions [12]")  =>  CUDA_HOME=/usr/local/cuda-12.9 + its nvcc on PATH
#   • host compiler gcc <= 14 (cuda-12.9 rejects the system gcc-15 -> "unsupported GNU version")
#     =>  NVCC_PREPEND_FLAGS="-ccbin g++-14"
#   • CCCL toolkit-compat check disabled (torch is cu130, toolkit is 12.9)
#     =>  NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
#
# These params are easy to lose (e.g. across a context compaction) and the failure mode looks like a
# code bug, not an env bug — hence this script is the single source of truth. All args pass straight
# through to scripts/train_cpt.py.  GPU defaults to 0 (the 5080); override via CUDA_VISIBLE_DEVICES.
#
# NOTE: never run two FP4 trainers at once — 1200W PSU OCP trips with both GPUs at full (hard crash).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin g++-14"
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=scripts
exec .venv/bin/python scripts/train_cpt.py "$@"
