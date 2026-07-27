#!/usr/bin/env python3
"""convert_base_int8.py — ONE-TIME base → int8-master artifact, STREAMING. Format v2.

The law (operator, 2026-07-26): convert exactly once; everything downstream (trainer init,
evals) loads the converted artifact. Never ingest the bf16 base per-run again.

FORMAT V2 (operator decisions 2026-07-26):
  * ALL layers' linears quantized (the v1 first-2/last-5 bf16 exemption is gone)
  * embeddings (embed_tokens + embed_tokens_per_layer) quantized int8 rowwise
  * ALL scales stored fp8 e4m3 (#925: scales don't need >fp8)
  * lm_head stays tied to embed_tokens (no separate tensor)
V2 quantizes the trained-sensitivity design to evidence: if install craters on the v2
instrument, that is the run-evidence for re-widening — never the prior.

Bit-exactness: quantize_int8_rowwise is deterministic and is the same function the trainer's
from_float uses at init → the v1-key masters are IDENTICAL to what every fp4 run builds.
The base shard self-describes the linear set (KV-sharing layers lack k/v_proj by design).

    evalgate env PYTHONPATH=scripts .venv/bin/python scripts/convert_base_int8.py \
        --base /home/marcos/Documents/dev/infra/models/gemma-4-E2B-it \
        --like checkpoints/cpt/pokemon_lore_kanto_s2_reasoning \
        --out /home/marcos/Documents/dev/infra/models/gemma-4-E2B-it-int8-v2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torchao.prototype.quantized_training import quantize_int8_rowwise

from pipeline.model_load import _SafetensorsRead
from train_cpt import estimate_streaming_save_need_bytes, save_file_streaming, save_preflight

_PREFIX = "model.language_model."
_LINEAR_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
    r"|per_layer_input_gate|per_layer_projection)\.weight$")
_ROOT_LINEARS = {"model.per_layer_model_projection.weight"}
_EMBEDDINGS = {"model.embed_tokens.weight", "model.embed_tokens_per_layer.weight"}


def _entry(name, dtype_str, tensor):
    return {
        "name": name,
        "dtype_str": dtype_str,
        "shape": tuple(tensor.shape),
        "nbytes": tensor.numel() * tensor.element_size(),
        "factory": (lambda t=tensor: t.detach()),
    }


def _lazy_entry(f, src_key, name, dtype_str, transform):
    """Entry whose factory reads + transforms the tensor at write time (streaming)."""
    shape = tuple(f._meta[src_key]["shape"])
    numel = 1
    for d in shape:
        numel *= d
    nbytes = numel if dtype_str == "I8" else numel * 2
    if transform is quantize_int8_rowwise:
        transform = _quantize_chunked
    def factory():
        out = transform(f.get_tensor(src_key))
        return out[0] if isinstance(out, tuple) else out
    return {"name": name, "dtype_str": dtype_str, "shape": shape, "nbytes": nbytes, "factory": factory}


_GPU = None


def _quantize_chunked(w, rows_per_chunk=16384):
    """quantize_int8_rowwise ON GPU (operator 2026-07-26: the CPU pass was swap-thrash bait).
    The op is deterministic and device-independent (absmax/scale/round) — bit-identical to
    the CPU call. Host holds only the read buffer + the half-size int8 result, each briefly;
    GPU workspace is transient per tensor. Row-chunking retained for tensors >GPU budget.
    """
    global _GPU
    if _GPU is None:
        _GPU = "cuda:0" if torch.cuda.is_available() else "cpu"
    if _GPU == "cpu":
        return quantize_int8_rowwise(w)
    if w.shape[0] > rows_per_chunk and w.numel() * w.element_size() > 1500e6:
        datas, scales = [], []
        for i in range(0, w.shape[0], rows_per_chunk):
            d, s = _quantize_chunked(w[i:i + rows_per_chunk], 1 << 62)
            datas.append(d)
            scales.append(s)
        return torch.cat(datas), torch.cat(scales)
    wg = w.to(_GPU, non_blocking=True)
    del w
    d, s = quantize_int8_rowwise(wg)
    del wg
    torch.cuda.empty_cache()
    return d.cpu(), s.cpu()


def _lazy_scale_entry(f, src_key, name):
    def factory():
        _d, s = _quantize_chunked(f.get_tensor(src_key))
        return (s * 128.0).to(torch.float8_e4m3fn)   # fold ×128 into e4m3 range (absmax/127 ≈ 0.0013 is BELOW e4m3 floor; unfolded = +46% error)
    rows = f._meta[src_key]["shape"][0]
    return {"name": name + ".weight_scale", "dtype_str": "F8_E4M3", "shape": (rows,),
            "nbytes": rows, "factory": factory}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="bf16 base model dir (multimodal HF layout)")
    ap.add_argument("--like", required=True, help="an existing int8 checkpoint dir (v1 quant key check)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.base)
    like = Path(args.like)
    out = Path(args.out)
    v1_keys = set(json.loads((like / "quant_keys.json").read_text())["keys"])

    entries = []
    linear_keys = []
    embedding_keys = []
    f = _SafetensorsRead(str(base / "model.safetensors"))
    if True:
        available = set(f.keys())
        # mechanical, shard-described linear set: every linear-shaped layer weight with %16 dims
        for k in sorted(available):
            if not k.startswith(_PREFIX):
                continue
            mk = "model." + k[len(_PREFIX):]
            # LAZY quantization: factories read + quantize one tensor at a time during the
            # streaming write — the whole artifact is never resident (peak = one tensor).
            if mk in _EMBEDDINGS:
                embedding_keys.append(mk)
                entries.append(_lazy_entry(f, k, mk, "I8", quantize_int8_rowwise))
                entries.append(_lazy_scale_entry(f, k, mk))
            elif mk in _ROOT_LINEARS or _LINEAR_RE.match(mk):
                shape = f._meta[k]["shape"]
                if shape[0] % 16 or shape[1] % 16:
                    entries.append(_lazy_entry(f, k, mk, "BF16", lambda w: w.to(torch.bfloat16)))
                    continue
                linear_keys.append(mk)
                entries.append(_lazy_entry(f, k, mk, "I8", quantize_int8_rowwise))
                entries.append(_lazy_scale_entry(f, k, mk))
            else:
                entries.append(_lazy_entry(f, k, mk, "BF16", lambda w: w.to(torch.bfloat16)))
        if "lm_head.weight" in available:
            pass  # tied to embed_tokens — no separate tensor

    # structural proofs: v1 set must be exactly v2 minus the formerly-exempt layers
    v1_layers = {int(m.group(1)) for k in v1_keys for m in [re.search(r"layers\.(\d+)\.", k)] if m}
    v2_layers = {int(m.group(1)) for k in linear_keys for m in [re.search(r"layers\.(\d+)\.", k)] if m}
    missing_v1 = v1_keys - set(linear_keys)
    assert not missing_v1, f"v1 keys missing from v2: {sorted(missing_v1)[:5]}"
    print(f"v2 linear keys: {len(linear_keys)} (v1: {len(v1_keys)}) | layers v1={min(v1_layers)}-{max(v1_layers)} "
          f"v2={min(v2_layers)}-{max(v2_layers)} | embeddings: {embedding_keys}")

    need = estimate_streaming_save_need_bytes(entries)
    save_preflight(need)
    out.mkdir(parents=True, exist_ok=True)
    try:
        save_file_streaming(out / "model.safetensors", entries,
                            metadata={"format": "pt", "quant": "int8_rowwise_v2"})
    finally:
        f.close()
    (out / "quant_keys.json").write_text(json.dumps({
        "method": "int8_rowwise_v2",
        "keys": linear_keys,
        "embedding_keys": embedding_keys,
        "scale_dtype": "fp8_e4m3",
    }) + "\n")

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(str(base))
    text_cfg = getattr(cfg, "text_config", cfg)
    text_cfg._attn_implementation = "sdpa"
    text_cfg.architectures = ["Gemma4ForCausalLM"]
    text_cfg.save_pretrained(out)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(out)
    print(f"v2 artifact written -> {out}  ({len(entries)} tensors)")


if __name__ == "__main__":
    main()
