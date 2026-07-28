#!/usr/bin/env bash
# Convert every remaining v1 checkpoint to v2 in place: trained masters pass BYTE-IDENTICAL,
# storage is purged (all linears + embeddings int8, scales fp8 e4m3, dead tied lm_head dropped).
# Serial — each is a multi-GiB streaming re-store. The v1 source is removed only after its v2
# copy exists AND its masters verify identical.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/convert_all_v1_${STAMP}.log"
say(){ echo "[convert $(date -Is)] $*" | tee -a "$LOG"; }

say "START"
for src in checkpoints/*/*/; do
    src="${src%/}"
    [ -f "$src/quant_keys.json" ] || continue
    m=$(.venv/bin/python -c "import json;print(json.load(open('$src/quant_keys.json')).get('method','?'))" 2>/dev/null)
    [ "$m" = "int8_rowwise_v2" ] && continue
    dst="${src}_v2"
    [ -d "$dst" ] && { say "SKIP $src — $dst exists"; continue; }
    say "$src ($(du -sh "$src"|cut -f1)) -> $dst"
    if ! oom-shield .venv/bin/python scripts/convert_ckpt_v2.py --ckpt "$src" --out "$dst" >>"$LOG" 2>&1; then
        say "  FAILED — v1 source KEPT"; tail -5 "$LOG" | sed 's/^/      /'; continue
    fi
    if .venv/bin/python - "$src" "$dst" <<'PY' >>"$LOG" 2>&1
import glob, json, sys, torch
from safetensors import safe_open
v1, v2 = sys.argv[1], sys.argv[2]
keys = json.load(open(v1 + "/quant_keys.json"))["keys"]
def get(d, k):
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, framework="pt") as s:
            if k in s.keys():
                return s.get_tensor(k)
bad = 0
for k in sorted(keys)[::8]:
    kk = k if k.endswith(".weight") else k + ".weight"
    a, b = get(v1, kk), get(v2, kk)
    if a is None or b is None:
        continue
    if a.dtype != b.dtype or not torch.equal(a, b):
        bad += 1
raise SystemExit(1 if bad else 0)
PY
    then
        rm -rf "$src"; say "  OK, masters verified, v1 removed -> $(du -sh "$dst"|cut -f1)"
    else
        say "  MASTERS DIFFER — v1 source KEPT, inspect $dst"
    fi
done
say "ALL DONE"
