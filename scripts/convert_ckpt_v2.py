#!/usr/bin/env python3
"""convert_ckpt_v2.py — re-store a trained int8 checkpoint in FORMAT V2 (operator, 2026-07-26).

Trained masters (the int8 weights that hold the install) pass through BYTE-IDENTICAL. The v2
changes are re-storage only:
  * embeddings (bf16-stored under the old selective exemption) → int8 rowwise + fp8 scale
  * formerly-exempt first/last-layer linears (bf16) → int8 rowwise masters + fp8 scale
  * every scale (masters + new) re-rounded bf16 → fp8 e4m3
This IS the v2 re-baseline: same trained masters, fully purged storage — the measurement of
whether the purge holds install (never the prior).

    oom-shield env PYTHONPATH=scripts .venv/bin/python scripts/convert_ckpt_v2.py \
        --ckpt checkpoints/cpt/pokemon_lore_kanto_s3_r005 \
        --out  checkpoints/cpt/pokemon_lore_kanto_s3_r005_v2
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from torchao.prototype.quantized_training import quantize_int8_rowwise

from pipeline.model_load import _SafetensorsRead
from train_cpt import estimate_streaming_save_need_bytes, save_file_streaming, save_preflight

_GPU = "cuda:0" if torch.cuda.is_available() else "cpu"


def _quantize_gpu(w):
    """quantize_int8_rowwise on GPU — deterministic, device-independent math (absmax/scale/
    round), bit-identical to the CPU call, ~100× faster on big tensors (operator 2026-07-26).
    Row-chunked above ~1.5GB so the 4.7GB embeddings never OOM the card."""
    if _GPU == "cpu" or w.numel() * w.element_size() <= 1500e6:
        wg = w.to(_GPU, non_blocking=True) if _GPU != "cpu" else w
        d, s = quantize_int8_rowwise(wg)
        return (d.cpu(), s.cpu()) if _GPU != "cpu" else (d, s)
    datas, scales = [], []
    for i in range(0, w.shape[0], 16384):
        d, s = _quantize_gpu(w[i:i + 16384].contiguous())
        datas.append(d)
        scales.append(s)
    return torch.cat(datas), torch.cat(scales)

_LINEAR_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
    r"|per_layer_input_gate|per_layer_projection)\.weight$")
_ROOT_LINEARS = {"model.per_layer_model_projection.weight"}
_EMBEDDINGS = {"model.embed_tokens.weight", "model.embed_tokens_per_layer.weight"}


def _entry(name, dtype_str, shape, tensor):
    return {"name": name, "dtype_str": dtype_str, "shape": tuple(shape),
            "nbytes": tensor.numel() * tensor.element_size(),
            "factory": (lambda t=tensor: t.detach())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt, out = Path(args.ckpt), Path(args.out)
    qmeta = json.loads((ckpt / "quant_keys.json").read_text())
    v1_keys = set(qmeta["keys"])
    assert "embedding_keys" not in qmeta, f"{ckpt} is already v2"

    entries = []
    linear_keys = sorted(v1_keys)
    embedding_keys = []
    with _SafetensorsRead(str(ckpt / "model.safetensors")) as f:
        for k in sorted(f.keys()):
            if k.endswith(".weight_scale"):
                continue
            if k in v1_keys:                                   # trained master: byte-identical
                entries.append(_entry(k, "I8", f._meta[k]["shape"], f.get_tensor(k)))
                entries.append(_entry(k + ".weight_scale", "F8_E4M3", (f._meta[k]["shape"][0],),
                                      (f.get_tensor(k + ".weight_scale").to(torch.bfloat16) * 128.0).to(torch.float8_e4m3fn)))
            elif k in _EMBEDDINGS:
                int_data, scale = _quantize_gpu(f.get_tensor(k))
                embedding_keys.append(k)
                entries.append(_entry(k, "I8", f._meta[k]["shape"], int_data))
                entries.append(_entry(k + ".weight_scale", "F8_E4M3", (f._meta[k]["shape"][0],),
                                      (scale * 128.0).to(torch.float8_e4m3fn)))
            elif k in _ROOT_LINEARS or _LINEAR_RE.match(k):    # formerly-exempt linears → masters
                int_data, scale = _quantize_gpu(f.get_tensor(k))
                linear_keys.append(k)
                entries.append(_entry(k, "I8", f._meta[k]["shape"], int_data))
                entries.append(_entry(k + ".weight_scale", "F8_E4M3", (f._meta[k]["shape"][0],),
                                      (scale * 128.0).to(torch.float8_e4m3fn)))
            else:                                              # norms etc. — as stored
                entries.append(_entry(k, "BF16", f._meta[k]["shape"], f.get_tensor(k).to(torch.bfloat16)))

    need = estimate_streaming_save_need_bytes(entries)
    save_preflight(need)
    out.mkdir(parents=True, exist_ok=True)
    save_file_streaming(out / "model.safetensors", entries,
                        metadata={"format": "pt", "quant": "int8_rowwise_v2"})
    (out / "quant_keys.json").write_text(json.dumps({
        "method": "int8_rowwise_v2", "keys": sorted(linear_keys),
        "embedding_keys": embedding_keys, "scale_dtype": "fp8_e4m3",
        "source": str(ckpt),
    }) + "\n")
    for aux in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
                "run_contract.json"):
        if (ckpt / aux).is_file():
            shutil.copy2(ckpt / aux, out / aux)
    print(f"v2 checkpoint written -> {out}  (linears {len(linear_keys)}, embeddings {embedding_keys})")


if __name__ == "__main__":
    main()
