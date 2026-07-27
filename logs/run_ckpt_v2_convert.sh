#!/usr/bin/env bash
# v1 -> v2 re-store for the four s3 checkpoints. Trained masters pass BYTE-IDENTICAL;
# only storage changes (embeddings + formerly-exempt linears -> int8 rowwise, all scales fp8 e4m3).
# Serial: each is 8.1G in / ~4.4G out and the converter streams.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/convert_ckpt_v2_${STAMP}.log"
say() { echo "[convert $(date -Is)] $*" | tee -a "$LOG"; }
say "START — four s3 arms, v1 -> v2 re-store"
rc_all=0
for arm in r005 r010 r020 r030; do
    src="checkpoints/cpt/pokemon_lore_kanto_s3_${arm}"
    dst="checkpoints/cpt/pokemon_lore_kanto_s3_${arm}_v2"
    [ -d "$src" ] || { say "SKIP $arm — missing $src"; rc_all=1; continue; }
    [ -d "$dst" ] && { say "SKIP $arm — $dst already exists"; continue; }
    say "$arm START -> $dst"
    oom-shield .venv/bin/python scripts/convert_ckpt_v2.py --ckpt "$src" --out "$dst" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && say "$arm OK" || { say "$arm FAILED exit=$rc"; tail -8 "$LOG" | sed 's/^/    /'; rc_all=1; }
done
say "ALL DONE rc=$rc_all"
exit "$rc_all"
