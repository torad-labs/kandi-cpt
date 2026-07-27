#!/usr/bin/env bash
# check_no_bf16.sh — the precision-law gate (operator law 2026-07-26: no bf16 residency,
# no bf16 storage, no new bf16 paths). Gates CODE-level bf16 usage; prose is invisible.
# Declared classes:
#   S1 one-time ingestion seams (Google's artifact): convert_base_int8.py, build_model fresh-base
#   S2 transient kernel-seam views (registers, never storage): quartet_bridge.py
#   S3 selective-precision design (engine-owned): MixedPrecisionPolicy (FSDP), construction dtype,
#      the bad_dtype policy assert, the save format (BF16 scales + non-quant tensors) and its
#      tests + the pread decoder map — all pending the operator's fp8-scale / TQ4-embed recipes
#   S4 LEGACY, scheduled for removal: causal_bf16_state_dict / iter_causal_bf16 /
#      load_causal_ckpt_bf16 / load_text_model_bf16 / _save_bf16_hf (incl. its body) / _sr_int8_master
#   S5 declared legacy/dev instruments (re-baseline or retire = operator decision):
#      geometry_probe, eval_voice.py, mem_breakdown.py, resolve_entities.py (roberta embedder)
set -euo pipefail
cd "$(dirname "$0")/../.."

PATTERN='torch\.bfloat16|torch_dtype\s*=\s*["'"'"']?bf16|\.to\(bf16\)|dtype\s*=\s*["'"'"']bf16["'"'"']'

hits=$(grep -rnE "$PATTERN" scripts/ --include="*.py" \
  | grep -vE "scripts/convert_base_int8\.py|scripts/convert_ckpt_v2\.py|scripts/quartet_bridge\.py" \
  | grep -vE "Int8ScaledEmbedding|Int8TiedLmHead|int8 masters from the shard|scale → bf16 at load" \
  | grep -vE "register_buffer\(\"weight_scale\"|register_buffer\(\"embed_scale\"|self\._emb\.weight\.to|f\.get_tensor\(qk \+ "\.weight_scale"\)\.to\(torch\.bfloat16\)" \
  | grep -vE "scripts/mem_breakdown\.py|scripts/pipeline/resolve_entities\.py|scripts/pipeline/curate_lib\.py|scripts/eval_cpt\.py" \
  | grep -vE "scripts/geometry_probe/|scripts/eval_voice\.py" \
  | grep -vE "scripts/fp4_smoke\.py|scripts/smoke_int8_subclass\.py|scripts/smoke_int8_master\.py" \
  | grep -vE "load_text_model_bf16" \
  | grep -vE "MixedPrecisionPolicy|model = model\.to\(torch\.bfloat16\)|bad_dtype" \
  | grep -vE "_tensor_nbytes\(shape|detach\(\)\.to\(dtype=torch\.bfloat16\)|out\.to\(torch\.bfloat16\)" \
  | grep -vE "_DT = |test_save_checkpoint_stream" \
  | grep -vE "device=device, dtype=torch\\.bfloat16\)" \
  | grep -vE "# bf16-sanctioned" \
  | grep -vE "check_no_bf16" \
  | grep -vE "^[^:]+:[0-9]+:\s*#" || true)

if [ -n "$hits" ]; then
  echo "::error::bf16 usage outside the declared seams (the precision law):"
  echo "$hits" | head -30
  echo ""
  echo "  Route: delete it, or declare the seam inline with '# bf16-sanctioned: <reason>'"
  echo "  and get the operator to approve the reason."
  exit 1
fi
echo "no-bf16 gate: OK — all bf16 usage confined to declared seams (S1-S5)"
