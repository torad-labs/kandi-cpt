#!/usr/bin/env bash
# S3 replay-sweep probe evals — all four arms, SERIALLY, on CUDA 13.3.
#
# Serial on purpose: each arm loads an 8.5 GiB checkpoint and evalgate's box is 20G/22G;
# two concurrent loads do not fit one 16 GiB card either.
#
# CUDA 13.3 is pinned explicitly rather than inherited so the provenance is in the artifact,
# not in whoever's shell launched it. 12.8 is a debug-tools-only install (no compiler) and
# was the reason flashinfer's fp4 GEMM JIT failed with "nvcc: not found".
#
# Launch detached, never as a child of the terminal:
#   setsid nohup logs/run_s3_probe_evals.sh > logs/s3_driver.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-13.3
export CUDA_PATH=/usr/local/cuda-13.3
export CUDA_VISIBLE_DEVICES=1          # GPU 1 only; GPU 0 is sacred
export PYTHONPATH=scripts

# BOUND THE JIT. flashinfer/ninja default to one compile job per kernel variant: on a cold
# fp4 cache that is ~17 parallel `cicc` frontends at ~1.3G each = 20G, which starved the host
# (avail 29% -> 16%), and earlyoom SIGTERM'd the eval itself — correctly, since choom -n 800
# makes build-class processes die first. The eval's own scope peaked at only 3.9G; the compile
# was the whale, not the model load. flashinfer honours MAX_JOBS (jit/cpp_ext.py:345).
# Bounded, not configurable: an unbounded JIT fanned out to 16 cicc frontends at ~1.3G each
# (20G) and earlyoom SIGTERMd the eval. 4 is the value that fits; it is not a knob to forget.
export MAX_JOBS=4

STAMP="$(date +%Y%m%d-%H%M%S)"
DRIVER_LOG="logs/probe_eval_s3_driver_${STAMP}.log"

say() { echo "[driver $(date -Is)] $*" | tee -a "$DRIVER_LOG"; }

say "START — arms r005 r010 r020 r030, CUDA_HOME=$CUDA_HOME, GPU=$CUDA_VISIBLE_DEVICES, MAX_JOBS=$MAX_JOBS"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$DRIVER_LOG"
say "host mem avail: $(free -g | awk 'NR==2{print $7"G of "$2"G"}')"

# Refuse to start into a starved host rather than discovering it via a SIGTERM mid-load.
avail_g=$(free -g | awk 'NR==2{print $7}')
if [ "${avail_g:-0}" -lt 12 ]; then
    say "ABORT — only ${avail_g}G available; an arm needs headroom for the load plus the JIT."
    exit 2
fi

rc_all=0
for arm in r005 r010 r020 r030; do
    cfg="configs/evals/pokemon_lore_kanto_s3_${arm}.yaml"
    # v2 artifacts: bf16 storage purged (all linears + embeddings int8, scales fp8 e4m3).
    # Trained masters are byte-identical to v1 and the run_contract fingerprint is unchanged,
    # so the eval's fingerprint validation still passes without touching the configs.
    ckpt="checkpoints/cpt/pokemon_lore_kanto_s3_${arm}_v2"
    log="logs/probe_eval_s3_${arm}_cu133_${STAMP}.log"

    [ -f "$cfg" ]  || { say "SKIP $arm — missing $cfg";  rc_all=1; continue; }
    [ -d "$ckpt" ] || { say "SKIP $arm — missing $ckpt"; rc_all=1; continue; }

    say "$arm START -> $log"
    # evalgate GIVES THE MEMORY BOX (20G/22G) and sets choom -n 800 — die FIRST. oom-shield then
    # overrides the adj on itself to -800 (die LAST) and it inherits through exec. Both are
    # required: the box alone left this eval the first victim, and the whale daemon SIGTERMed it
    # at 13.7G on 2026-07-27 doing precisely what it is designed to do. hostshield's declared
    # policy is shield-or-die for paid/unrepeatable work; an unshielded run is the bug.
    evalgate oom-shield .venv/bin/python scripts/eval.py \
        --config "$cfg" --mode cpt --checkpoints "$ckpt" > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        say "$arm OK (exit 0)"
    else
        say "$arm FAILED exit=$rc — tail follows"
        tail -15 "$log" | sed 's/^/    /' | tee -a "$DRIVER_LOG"
        rc_all=1
    fi
done

say "ALL ARMS DONE (overall rc=$rc_all)"
exit "$rc_all"
