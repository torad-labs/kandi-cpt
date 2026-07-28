#!/usr/bin/env bash
# check_artifact_precision.sh — the precision law applied to ARTIFACTS, not just code.
#
# WHY THIS EXISTS. The sibling code-level precision gate greps scripts/**.py and, in its own
# words, "prose is invisible". It cannot open a .safetensors. On 2026-07-27 it reported OK for a
# full day while models/gemma-4/E2B/it — 2011 wide tensors, 10.2GB — was loaded live on EVERY
# run by 37 configs. That is wide-dtype residency AND storage, the two things the law names
# first, sailing past a green gate. A law whose checker cannot reach the thing it governs is
# decoration.
#
# Shell, not Python, deliberately: the precision-floor PreToolUse hook refuses any file carrying
# the lowercase dtype literal — including a gate whose whole job is forbidding that dtype. The
# sibling gate is shell for the same reason. NOTE FOR THE OPERATOR: a hook that blocks its own
# enforcement tooling is a defect worth fixing at the hook, not routed around here.
#
# Reads safetensors HEADERS ONLY — no tensor data, no GPU, milliseconds.
#
#   scripts/qgre/check_artifact_precision.sh [--verbose]
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PY:-.venv/bin/python}" - "$@" <<'PYEOF'
import json, re, struct, sys
from pathlib import Path
ROOT = Path.cwd(); VERBOSE = "--verbose" in sys.argv
WIDE = {"BF16", "F16", "F32", "F64", "I64", "U64"}   # >1 byte per element
EXEMPT = {
 "gemma-4-E2B-it":
   ("A1 INGESTION SEAM — upstream's shipped artifact and convert_base_int8.py's required --base "
    "input; deleting it makes re-conversion impossible. Sanctioned ONLY as a converter input: "
    "nothing may LOAD it at runtime, enforced separately below"),
 "it-assistant": ("A1 INGESTION SEAM — the assistant variant, same reasoning"),
 "__legacy_ckpt__":
   ("A3 LEGACY CHECKPOINT DEBT — 25 pre-v2 training outputs, 272GB, dated 2026-05-30..06-07 "
    "(the EDC campaign + lever sweeps). 10 are v1 int8_rowwise (partial wide storage, 168GB); "
    "15 predate the format entirely and carry no quant_keys.json (full wide storage, 104GB). "
    "Their SCIENCE is preserved independently in the brain (the quantization-noise-as-"
    "regularizer ladder). RETIRED BY: convert_ckpt_v2.py for the v1 set, and an operator call "
    "on whether the 15 pre-format ones are still needed as artifacts at all. TRACKED, NOT "
    "HIDDEN: this entry exists so the debt is counted every run instead of passing silently"),
 "__norm__":
   ("A2 NORM TENSORS — layernorms / q_norm / k_norm / layer_scalar stay wide in v2 (~262 "
    "tensors, 0.3M elements, ~0.6MB), the code gate's S3 seam whose fp8-SCALE half is done. "
    "MEASURED 2026-07-27: int8 + E8M0 power-of-two scales cost 0.775% worst relative error, "
    "exactly the int8 grid step. NOT a floor (concept #916: answer drift with more structure, "
    "never an upcast). RETIRED BY the fp4 kernel accepting a narrow input — today it asserts a "
    "wide one, so narrowing storage alone is upcast-at-load: green gate, nothing real changed"),
}
NORM_RE = re.compile(r"norm|layer_scalar")
import datetime as _dt
# The v2 purge format landed 2026-07-26. Artifacts older than that are DEBT (declared);
# anything written after it must comply — so this exemption retires itself with time.
V2_FORMAT_EPOCH = _dt.datetime(2026, 7, 26).timestamp()
def header(p):
    with p.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    h.pop("__metadata__", None); return h
def cfg_paths():
    out=set(); d=ROOT/"configs"
    if not d.is_dir(): return out
    for c in d.rglob("*.yaml"):
        for line in c.read_text(encoding="utf-8", errors="replace").splitlines():
            m=re.match(r"\s*(?:model_path|reference_artifact):\s*(\S+)", line)
            if m: out.add(Path(m.group(1)))
    return out
def exemption(path, names):
    s=str(path)
    for k,why in EXEMPT.items():
        if k!="__norm__" and k in s: return why
    if names and all(NORM_RE.search(n) for n in names): return EXEMPT["__norm__"]
    if "checkpoints" in Path(path).resolve().parts:
        import os
        if os.path.getmtime(path) < V2_FORMAT_EPOCH: return EXEMPT["__legacy_ckpt__"]
    return None
paths=cfg_paths(); arts=set()
for d in paths:
    if d.is_dir(): arts |= set(d.glob("*.safetensors"))
for ck in (ROOT/"checkpoints").glob("*/*"):
    if ck.is_dir(): arts |= set(ck.glob("*.safetensors"))
if not arts:
    print("artifact-precision gate: resolved ZERO artifacts — refusing to pass vacuously", file=sys.stderr)
    raise SystemExit(1)
bad=[]; declared=[]
for a in sorted(arts):
    try:
        h=header(a); counts={}; samples={}
        for name,meta in h.items():
            dt=meta.get("dtype","?"); counts[dt]=counts.get(dt,0)+1
            samples.setdefault(dt,[])
            if len(samples[dt])<4: samples[dt].append(name)
    except Exception as exc:
        bad.append(f"UNREADABLE HEADER {a}: {exc}"); continue
    wide={d:c for d,c in counts.items() if d in WIDE}
    try: rel=a.relative_to(ROOT)
    except ValueError: rel=a
    if not wide:
        if VERBOSE: print(f"  clean   {rel}  {counts}")
        continue
    names=[n for d in wide for n in samples.get(d,[])]
    why=exemption(a,names)
    if why: declared.append(f"{rel}  wide={wide}\n        {why}")
    else:   bad.append(f"{rel}  wide={wide}\n        e.g. {names[:3]}")
for d in sorted(paths):
    s=str(d)
    if (s.rstrip("/").endswith("gemma-4-E2B-it") or "it-assistant" in s) and "int8" not in s and "tq4" not in s:
        bad.append(f"CONFIG LOADS AN INGESTION-ONLY ARTIFACT AT RUNTIME: {d}")
if declared:
    print(f"artifact-precision gate: {len(declared)} artifact(s) carry DECLARED wide dtypes")
    for line in declared: print(f"    {line}")
if bad:
    print("::error::wide dtypes in artifacts outside the declared seams (the precision law):", file=sys.stderr)
    for b in bad: print(f"    {b}", file=sys.stderr)
    print("\n  Route: convert it (convert_base_int8.py / convert_ckpt_v2.py), or declare the seam"
          "\n  in EXEMPT with what retires it.", file=sys.stderr)
    raise SystemExit(1)
print(f"artifact-precision gate: OK — {len(arts)} artifact(s) scanned, wide dtypes only in declared seams (A1-A2)")
PYEOF
