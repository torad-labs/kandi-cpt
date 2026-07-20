#!/usr/bin/env bash
# Canonical native-FP4 (Quartet-II / flashinfer cutlass sm120) training launcher.
#
# The flashinfer cutlass-FP4 sm120 kernel JIT-compiles with nvcc on first use. On THIS host that
# requires a specific toolchain — without it the JIT dies with cryptic errors:
#   • CUDA >= 12.9 toolkit (sm120/Blackwell is unsupported on 12.8; nvcc 12.8 -> "No supported CUDA
#     architectures found for major versions [12]")  =>  CUDA_HOME points at a toolkit with nvcc on PATH
#   • host compiler gcc <= 14 (CUDA rejects the system gcc-15 -> "unsupported GNU version")
#     =>  NVCC_PREPEND_FLAGS="-ccbin g++-14"
#   • CCCL toolkit-compat check disabled (torch/toolkit minor skew can trip CCCL's guard)
#     =>  NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
#
# These params are easy to lose (e.g. across a context compaction) and the failure mode looks like a
# code bug, not an env bug — hence this script is THE single source of truth for the launch environment.
# (This is THE training launcher — the only one. scripts/fp4_env.sh still exists but only for fp4_smoke.py;
# it is NOT a training path. The old 2-GPU launch.sh was deleted — it violated the single-GPU 1200W-PSU rule.)
#
# ZERO-FLAG LAUNCH:  bash scripts/train_fp4.sh
#   → runs the canonical CPT defined entirely in configs/train_edc_lv_2026.yaml (corpus, save_dir,
#     max_steps, batch, precision, int8-master mode — all baked in; nothing to remember or re-derive).
#   Override only when A/B testing:  bash scripts/train_fp4.sh --config <other.yaml>   (or any train_cpt flag).
#   All args pass straight through to scripts/train_cpt.py. GPU defaults to 0 (the 5080).
#
# NOTE: never run two FP4 trainers at once — 1200W PSU OCP trips with both GPUs at full (hard crash).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
_KANDI_ROOT="$PWD"

find_cuda_home() {
  local roots=(
    "${QGRE_CUDA_HOME:-}"
    /usr/local/cuda-12.9
    /usr/local/cuda-13.3
    /usr/local/cuda-13
    "$_KANDI_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13"
    /usr/local/cuda
  )
  local root
  for root in "${roots[@]}"; do
    [[ -n "$root" && -x "$root/bin/nvcc" ]] || continue
    printf '%s\n' "$root"
    return 0
  done
  return 1
}

CUDA_HOME="$(find_cuda_home)" || {
  echo "train_fp4.sh: no CUDA >=12.9 toolkit with bin/nvcc found." >&2
  echo "Set QGRE_CUDA_HOME=/path/to/cuda or install nvcc under /usr/local/cuda-12.9 or /usr/local/cuda-13.3." >&2
  exit 127
}
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
for libdir in "$CUDA_HOME/lib64" "$CUDA_HOME/lib"; do
  [[ -d "$libdir" ]] || continue
  export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$libdir:${LIBRARY_PATH:-}"
  export CUDA_LIB_PATH="${CUDA_LIB_PATH:-$libdir}"
done
export NVCC_PREPEND_FLAGS="-ccbin g++-14"
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# flashinfer JIT-compiles the CUTLASS sm120 FP4 kernel with ninja (one job/core by default). On this
# 32-core/59GB box each cicc compile is 2-4GB → unconstrained fan-out exhausts RAM and earlyoom SIGKILLs
# the build (looks like a mystery crash). Cap it. Only matters on first compile; harmless once cached.
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=scripts
# Default to the canonical CPT config when none is passed → bare `bash scripts/train_fp4.sh` just works.
case " $* " in
  *" --config "*) launch_args=("$@") ;;
  *)              launch_args=(--config configs/train_edc_lv_2026.yaml "$@") ;;
esac
.venv/bin/python scripts/train_cpt.py "${launch_args[@]}"
# Training has exited, so its GPU allocations are gone. The same typed config now owns the complete
# trust seam; disabled historical configs return cleanly, enabled installs fail closed on missing evidence.
.venv/bin/python scripts/post_install_chain.py "${launch_args[@]}"
