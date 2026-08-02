# conductor-trainer — the image the ALREADY-SHIPPED cloud path has always assumed and never had.
#
# Contract satisfied (conductor @ d54eff3d): /opt/qgre/bin/run-cloud-training <bundlePath> <runDir>
# (CloudRemoteLaunch.kt:6), bundle arrives at /workspace/qgre (CloudBundleTransfer.kt:33), telemetry
# reads <runDir>/metrics.jsonl + <runDir>/cpt_stages.json (CloudRemoteTelemetry.kt:134-148), and the
# instance is created with runtype "ssh_direct" (CloudProvisioner.kt:121) so sshd is mandatory.
#
# ── WHY THE BASE IS ubuntu:25.10, NOT A CUDA IMAGE AND NOT python:3.13-slim ────────────────────
# This image BAKES the engine's proven virtualenv rather than reinstalling from a manifest (see
# below). That venv is full of extensions compiled against the build host's glibc, so the
# container's glibc must be at least as new. 25.10 IS the host's own release (ubuntu 25.10 /
# glibc 2.42), which makes the match exact instead of hopeful — and the build now ASSERTS the
# pairing rather than trusting it, so a future host upgrade fails here instead of failing at
# import on a rented GPU. A CUDA base image is unnecessary: the venv already carries a complete
# CUDA 13 wheel stack INCLUDING a compiler at
# .venv/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc, which is candidate #6 of the engine's own
# toolkit resolver (cuda_home.py:29-38) and the only one that will exist here — so
# `cuda_home.resolve()` succeeds in-container with no apt CUDA toolkit at all.
#
# ── WHY THE VENV IS BAKED AND requirements.txt IS NOT USED ─────────────────────────────────────
# requirements.txt is STALE and cannot train. Measured on engine 9cd16822:
#   * it declares 7 packages and pins torch==2.10.0+cu130; the venv the engine actually runs on has
#     153 distributions and torch 2.12.0+cu130.
#   * it names NEITHER torchao NOR flashinfer, both of which train_cpt.py imports directly
#     (:226-228, :305, :347, :1319).
#   * decisively, the FP4 moat is `quartet2`, imported at train_cpt.py:257/:304/:325 on the
#     precision=="fp4" path (:480-481). quartet2 is NOT on any index: the installed distribution
#     records `quartet2 @ file:///home/marcos/Documents/dev/Quartet-II/kernels`, a path that NO
#     LONGER EXISTS (the source moved to /home/marcos/Documents/dev/projects/quartet-ii/repo/kernels),
#     and it is a scikit-build-core/CMake CUDA extension, not a pure wheel.
# So NO manifest derivable from either repo can reproduce a trainable environment. Baking the
# proven venv is not a shortcut here; it is the only route that reaches the fp4 path. The
# consequence — that the image's dependency set is pinned by a host directory rather than by a
# reviewable file — is real and is called out in the operator doc.
#
# NOTE ON BUILD SPEED: this build installs a handful of apt packages and NO pip wheels, so it
# finishes in minutes (measured 1m34s end-to-end from build-cloud-trainer.sh, of which most is the
# 7.8GB context copy), not the tens of minutes a torch+CUDA pip build takes. That is expected here and is NOT
# the "fast green build means you built the wrong thing" failure: the payload comes from the local
# context. What proves the image is right is the FP4 import chain, not the clock.

FROM ubuntu:25.10

# python3.13 : the venv's bin/python3 is an ABSOLUTE symlink to /usr/bin/python3
# unzip      : the host bundle is a zip (CloudBundleTransfer.kt:18)
# gawk       : the metrics bridge calls fflush(FILE); gawk supports the 1-arg form unambiguously
# g++-14     : train_fp4.sh:46 exports NVCC_PREPEND_FLAGS="-ccbin g++-14" for the flashinfer JIT
# gcc-14 + cc/gcc names : Triton JIT-compiles its CUDA driver shim (triton/runtime/build.py) with
#   whatever `cc`/`gcc` resolves at RUNTIME. The build host never exercised this (Blackwell takes
#   the real Quartet kernel; Triton pseudoquant only runs on capability < 10), so the gap surfaced
#   on the first rented Ampere card: paid smoke 2026-08-02, trainer dead in triton _build().
# python3.13-dev : Triton's cuda_utils.c does `#include <Python.h>`. gcc alone is not enough —
#   paid smoke attempt 14 (2026-08-02) got past "no C compiler" and died on missing Python.h.
# openssh-server : runtype "ssh_direct" (CloudProvisioner.kt:121) + the sshj transport
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.13 \
      python3.13-dev \
      unzip \
      gawk \
      g++-14 \
      gcc-14 \
      openssh-server \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.13 /usr/bin/python3 \
 && ln -sf /usr/bin/gcc-14 /usr/bin/gcc \
 && ln -sf /usr/bin/gcc /usr/bin/cc

# The baked venv's bin/python3 is an absolute symlink to /usr/bin/python3; if that is not 3.13 the
# whole site-packages tree is unloadable. Assert rather than discover it on a rented instance.
# Python.h is the next link in the Triton JIT chain (see python3.13-dev above).
RUN python3 --version 2>&1 | grep -q '^Python 3\.13' \
 || { echo "BUILD ASSERT FAILED: /usr/bin/python3 is $(python3 --version 2>&1), need 3.13" >&2; exit 1; } \
 && test -f /usr/include/python3.13/Python.h \
 || { echo "BUILD ASSERT FAILED: /usr/include/python3.13/Python.h missing (need python3.13-dev)" >&2; exit 1; }

# The venv is its OWN layer, baked BEFORE the engine tree: it is 6.8GB and changes ~never, while
# the tree changes with every engine commit. One combined COPY meant a one-line engine fix
# invalidated the whole ~8GB blob (measured 2026-08-02: a multi-hour ghcr re-upload for the gcc
# fix). Ordered this way, the venv layer and its two RUN fixups below stay cached across engine
# changes, and only the ~1GB tree layer re-ships. COPY merges into the existing directory, so the
# later tree COPY lands beside .venv without touching it.
COPY venv /opt/qgre/kandi-cpt/.venv

# The venv was built at a path that does not exist here, so every console script in .venv/bin still
# carries the build host's interpreter in its shebang. Repoint them at the baked interpreter.
# (Left unfixed this is a silent trap: `import torch` works because site-packages resolves by path,
# while `.venv/bin/pip` and friends die with "required file not found".)
RUN set -eu; \
    for f in /opt/qgre/kandi-cpt/.venv/bin/*; do \
      [ -f "$f" ] || continue; \
      head -c 2 "$f" | grep -q '#!' || continue; \
      sed -i '1s|^#!.*/\.venv/bin/python|#!/opt/qgre/kandi-cpt/.venv/bin/python|' "$f"; \
    done

# ── THE glibc WALL ─────────────────────────────────────────────────────────────────────────────
# The venv is a copy of extensions compiled on the build HOST. `FROM ubuntu:25.10` above was chosen
# by hand to match that host's glibc; nothing enforced the pairing, so a host upgrade (or a base
# image pinned by a different author) would silently produce an image that dies at `import torch`
# on a machine the operator is already paying for. Measure both sides and refuse.
#
# The requirement is read from the ARTIFACTS, not from the build host's `ldd --version`: every
# versioned glibc symbol an ELF needs appears verbatim in its dynamic string table, so the highest
# GLIBC_2.x string across the baked .so files IS the floor. Measured in this image: 555 shared
# objects, max requirement GLIBC_2.34, container glibc 2.42.
RUN set -eu; \
    have="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"; \
    need="$(find /opt/qgre/kandi-cpt/.venv -name '*.so*' \
              | xargs -r grep -aoh 'GLIBC_2\.[0-9][0-9]*' 2>/dev/null \
              | sort -u -V | tail -1 | sed 's/^GLIBC_//')"; \
    [ -n "$need" ] || { echo "BUILD ASSERT FAILED: found no GLIBC_2.x requirement in the baked venv — is .venv empty?" >&2; exit 1; }; \
    echo "build-assert: container glibc $have, baked venv needs at most GLIBC_$need ($(find /opt/qgre/kandi-cpt/.venv -name '*.so*' | wc -l) shared objects scanned)"; \
    lowest="$(printf '%s\n%s\n' "$have" "$need" | sort -V | head -1)"; \
    [ "$lowest" = "$need" ] || { \
      echo "BUILD ASSERT FAILED: this base image ships glibc $have but the baked venv needs GLIBC_$need." >&2; \
      echo "  The venv was compiled on a newer host than the FROM line. Raise the base image to a" >&2; \
      echo "  release whose glibc is >= $need, or rebuild the venv on a host matching this base." >&2; \
      exit 1; }

COPY engine /opt/qgre/kandi-cpt

COPY bin/ /opt/qgre/bin/
RUN chmod 0755 /opt/qgre/bin/run-cloud-training /opt/qgre/bin/qgre-sshd-entrypoint \
 && chmod 0644 /opt/qgre/bin/qgre-metrics-bridge.awk

ARG ENGINE_COMMIT=unknown
ARG ENGINE_DIRTY=unknown
RUN printf 'commit=%s\ndirty=%s\n' "$ENGINE_COMMIT" "$ENGINE_DIRTY" > /opt/qgre/ENGINE_COMMIT

# /workspace/qgre is where the host uploads the bundle (CloudBundleTransfer.kt:33) and
# /workspace/qgre/run is the runDir the launcher mkdir -p's (CloudRemoteLaunch.kt:10).
RUN mkdir -p /workspace/qgre/run

# ── BUILD-TIME WALLS ───────────────────────────────────────────────────────────────────────────
# Refuse to produce an image that cannot do its job. A broken venv copy or a typo in the entrypoint
# fails the BUILD, not the rented instance.
RUN set -eu; \
    bash -n /opt/qgre/bin/run-cloud-training; \
    sh -n /opt/qgre/bin/qgre-sshd-entrypoint; \
    gawk -f /opt/qgre/bin/qgre-metrics-bridge.awk /dev/null; \
    cd /opt/qgre/kandi-cpt; \
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python -c "\
import torch, torchao, flashinfer;\
from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig;\
from torchao.prototype.quantized_training import Int8QuantizedTrainingLinearWeight;\
from torchao.optim import AdamW4bit;\
import cpt_config;\
print('build-assert: fp4 import chain 7/7 (CPU-only), torch', torch.__version__)"; \
    test -f .venv/lib/python3.13/site-packages/quartet2/_quartet2.abi3.so; \
    test -x .venv/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc; \
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python -c "\
import cuda_home, sys;\
r = cuda_home.resolve();\
print('build-assert: cuda_home.resolve() ->', r);\
sys.exit(0 if r is not None else 1)"; \
    echo "build-assert: quartet2 unresolved links (measured, not assumed):"; \
    ldd .venv/lib/python3.13/site-packages/quartet2/_quartet2.abi3.so \
      | awk '/not found/ {print $1}' | sort -u | tee /opt/qgre/QUARTET2_UNRESOLVED; \
    unexpected="$(grep -vxE 'libcuda\.so\.1|libcudart\.so\.13' /opt/qgre/QUARTET2_UNRESOLVED || true)"; \
    [ -z "$unexpected" ] || { echo "BUILD ASSERT FAILED: quartet2 has unexpected unresolved libs: $unexpected" >&2; exit 1; }; \
    grep -qx 'libcuda\.so\.1' /opt/qgre/QUARTET2_UNRESOLVED \
      || { echo "BUILD ASSERT FAILED: expected libcuda.so.1 to be the driver-gated link" >&2; exit 1; }; \
    test -f .venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13 \
      || { echo "BUILD ASSERT FAILED: libcudart.so.13 is not in the venv, so it cannot resolve at runtime" >&2; exit 1; }; \
    echo "build-assert: libcudart.so.13 ships in the venv; libcuda.so.1 is the DRIVER and is supplied only by the GPU runtime"

# ninja: flashinfer JIT-builds cutlass fp4 gemm modules at RUNTIME on sm_120 (Blackwell).
# Paid smoke attempt 22 (2026-08-02) reached the real Quartet path on an RTX 5060 Ti and died
# FileNotFoundError: ninja. Kept as its OWN late layer so a one-package add does not invalidate
# the multi-GB venv/tree layers above (the python3.13-dev apt edit earlier forced that re-push).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ninja-build \
 && rm -rf /var/lib/apt/lists/* \
 && command -v ninja >/dev/null \
 || { echo "BUILD ASSERT FAILED: ninja not on PATH after ninja-build install" >&2; exit 1; }

LABEL org.opencontainers.image.title="conductor-trainer" \
      org.opencontainers.image.description="kandi-cpt engine + its proven virtualenv, for conductor cloud training" \
      org.opencontainers.image.revision="${ENGINE_COMMIT}" \
      ai.torad.qgre.engine.dirty="${ENGINE_DIRTY}"

ENV PYTHONPATH=/opt/qgre/kandi-cpt/scripts \
    PYTHONUNBUFFERED=1
WORKDIR /opt/qgre/kandi-cpt

EXPOSE 22
CMD ["/opt/qgre/bin/qgre-sshd-entrypoint"]
