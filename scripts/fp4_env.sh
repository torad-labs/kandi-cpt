#!/usr/bin/env bash
# FP4 / CUDA-13 toolchain environment for kandi-cpt.
# Source this before any FP4 JIT compile, kernel build, or training launch:
#   source scripts/fp4_env.sh
#
# ── MEMORY SAFETY (the important part) ───────────────────────────────────────
# flashinfer JIT-compiles its CUTLASS sm120 FP4 kernel with `ninja`, which by
# default runs one compile job PER CORE. This box has 32 cores, and each CUDA
# device compile (`cicc`) on CUTLASS template code uses 2-4 GB. 32 jobs in
# parallel => ~96 GB, which exhausts the 59 GB of RAM and FREEZES the machine
# before the OOM killer can recover it. MAX_JOBS caps that fan-out. Keep it low.
export MAX_JOBS="${MAX_JOBS:-4}"                       # torch & flashinfer ninja -j
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"  # cmake builds

# ── cu13 toolkit discovery (steer away from the system cuda-12.x) ────────────
_KANDI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_CU="$_KANDI_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13"
export CUDA_HOME="$_CU"
export CUDA_PATH="$_CU"
# CUDA_LIB_PATH points at the cu13 lib, which has libcudart.so.13 but NOT .so.12,
# so flashinfer/jit/__init__.py skips its hardcoded cu12 RTLD_GLOBAL load.
export CUDA_LIB_PATH="$_CU/lib"
export PATH="$_CU/bin:$PATH"
export LD_LIBRARY_PATH="$_CU/lib:${LD_LIBRARY_PATH:-}"
# LIBRARY_PATH is what the LINKER searches for -l flags. The pip toolkit ships
# libs under lib/, but flashinfer's link step hardcodes -L.../lib64 -> add lib/
# so `-lcudart` resolves (libcuda is found on the system default path).
export LIBRARY_PATH="$_CU/lib:${LIBRARY_PATH:-}"

# ── nvcc: force a supported host compiler + accept intra-CUDA-13 minor skew ──
# nvcc 13.2 rejects gcc-15 (system default) -> pin g++-14.
# nvcc 13.2 vs cudart-13.0 headers trips CCCL's CTK guard; the skew is patch-
# level within CUDA 13 and ABI-safe, so disable that specific check (sanctioned
# by CCCL itself via this macro).
export NVCC_PREPEND_FLAGS="-ccbin g++-14"
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# ── allocator ────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
