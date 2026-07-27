"""Gemma 4 E2B loaders — the precision-law forms.

* load_text_model_bf16 — base text-decoder load for CURATION perplexity scoring (S5, pending
  the operator's curation call): bf16-native residency by design of that path, streamed so the
  host never holds a second copy.
* load_causal_ckpt_quartet — THE eval/training-init loader: int8 masters straight into the
  torchao subclass, Quartet fp4 forward, no bf16 weights anywhere. v2 artifacts (int8
  embeddings + fp8 scales) install Int8ScaledEmbedding + a tied int8 head.
* _SafetensorsRead — pread-based shard access: mmap keeps shard pages in VmRSS (the
  whale-reaper's metric; fadvise can't evict mapped pages — proven no-op here 2026-07-26);
  pread keeps file data out of process RSS entirely.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, Gemma4ForCausalLM

import config as C

_PREFIX = "model.language_model."


class _SafetensorsRead:
    """read()-based safetensors access. safe_open mmaps the shard and the faulted-in pages
    sit in VmRSS — the whale-reaper's metric — making a ~10GB loader read as a ~19GB whale
    (rss=19.8GB kills, 2026-07-26). posix_fadvise(DONTNEED) cannot evict still-mapped pages
    (proven no-op on this box, same day). os.pread keeps file data out of process RSS
    entirely: page cache stays kernel-owned, MemAvailable counts it reclaimable, and the
    only process memory is the transient per-tensor buffer."""

    _DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
           "I8": torch.int8, "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64,
           "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
           "BOOL": torch.bool}

    def __init__(self, path):
        import json
        import os

        self._os = os
        self._fd = os.open(path, os.O_RDONLY)
        hdr_len = int.from_bytes(os.pread(self._fd, 8, 0), "little")
        self._meta = json.loads(os.pread(self._fd, hdr_len, 8))
        self._data_start = 8 + hdr_len

    def keys(self):
        return [k for k in self._meta if k != "__metadata__"]

    def get_tensor(self, key):
        info = self._meta[key]
        dtype = self._DT.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {info['dtype']!r} for {key!r}")
        begin, end = info["data_offsets"]
        # pread is capped at MAX_RW_COUNT (0x7ffff000) per call — loop for >2GB tensors.
        parts = []
        remaining = end - begin
        off = self._data_start + begin
        while remaining > 0:
            piece = self._os.pread(self._fd, remaining, off)
            if not piece:
                raise OSError(f"short read on {key!r}: {end - begin - remaining}/{end - begin} bytes")
            parts.append(piece)
            off += len(piece)
            remaining -= len(piece)
        return torch.frombuffer(bytearray(b"".join(parts)), dtype=dtype).reshape(info["shape"])

    def close(self):
        self._os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _rematerialize_buffers(model, text_cfg):
    """Re-materialize the persistent=False buffers exactly as __init__ computes them (fp32
    math, rounded to the storage dtype — identical to the old construct + cast path).
    Unguarded on purpose: a missing buffer = NaN outputs = silently wrong numbers; an
    AttributeError is the honest failure."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    rot = Gemma4TextRotaryEmbedding(text_cfg)
    for name, buf in model.model.rotary_emb.named_buffers():
        buf.copy_(getattr(rot, name))
    model.model.embed_tokens.embed_scale.copy_(torch.tensor(text_cfg.hidden_size ** 0.5))
    model.model.embed_tokens_per_layer.embed_scale.copy_(
        torch.tensor(text_cfg.hidden_size_per_layer_input ** 0.5))


def load_text_model_bf16(device: str = "cuda:0"):
    """Base text-decoder load for curation (S5). bf16-native via meta + to_empty (never the
    18.5GB fp32 phase), shard streamed per-tensor via _SafetensorsRead."""
    text_cfg = AutoConfig.from_pretrained(str(C.MODEL_PATH)).text_config
    text_cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Gemma4ForCausalLM(text_cfg)
    model = model.to(torch.bfloat16)
    model = model.to_empty(device="cpu")
    chunk: dict = {}
    with _SafetensorsRead(f"{C.MODEL_PATH}/model.safetensors") as f:
        for k in f.keys():
            if k.startswith(_PREFIX):
                nk = "model." + k[len(_PREFIX):]
            elif k == "lm_head.weight":
                nk = k
            else:
                continue                                    # vision tower etc. — not the text model
            chunk[nk] = f.get_tensor(k)
            if len(chunk) >= 64:
                model.load_state_dict(chunk, strict=False)
                chunk.clear()
        if chunk:
            model.load_state_dict(chunk, strict=False)
    del chunk
    model.tie_weights()
    _rematerialize_buffers(model, text_cfg)
    return model.to(device, dtype=torch.bfloat16).eval()  # bf16-sanctioned: S5 curation residency (operator-pending)


class Int8ScaledEmbedding(torch.nn.Module):
    """int8 vocab×dim embedding + per-row scale, dequantized per lookup — never resident wide.
    Replaces Gemma4TextScaledWordEmbedding (v2 artifacts): gather × row-scale × embed_scale.
    The fp8-stored scale is cast to bf16 AT LOAD (e4m3 ⊂ bf16 — exact); the bf16 product is
    a per-token transient at the embedding seam (the engine's sanctioned unpack-at-ALU seam)."""

    def __init__(self, weight, scale, embed_scale: float):
        super().__init__()
        self.register_buffer("weight", weight)                            # int8 [vocab, dim]
        self.register_buffer("weight_scale", scale.to(torch.bfloat16))    # [vocab]
        self.register_buffer("embed_scale", torch.tensor(float(embed_scale), dtype=torch.bfloat16))

    def forward(self, ids):
        return (self.weight[ids].to(torch.bfloat16)  # bf16-sanctioned: S2 transient dequant at the lookup seam
                * self.weight_scale[ids].unsqueeze(-1)
                * self.embed_scale)


class Int8TiedLmHead(torch.nn.Module):
    """lm_head tied to an Int8ScaledEmbedding: logits from the int8 weight, dequantized per
    matmul (transient). Replaces the tie — tie_weights cannot share an int8 buffer."""

    def __init__(self, emb: Int8ScaledEmbedding):
        super().__init__()
        self._emb = emb

    def forward(self, x):
        w = self._emb.weight.to(torch.bfloat16) * self._emb.weight_scale.unsqueeze(1)
        return torch.nn.functional.linear(x, w)


def load_causal_ckpt_quartet(path, device: str = "cuda:0", requires_grad: bool = False):
    """Load a saved int8-master checkpoint for eval/init THROUGH THE TRAINING FORWARD (Quartet
    FP4). The instrument law (operator, 2026-07-26): the eval asserts install success, so it
    runs the function that was actually trained — not a bf16 decode of it.

    Construction: meta skeleton → Quartet_II_linear at exactly the quant_keys linears → int8
    masters from the shard (no requantization) → everything else streamed → v2 artifacts get
    Int8ScaledEmbedding + tied int8 head → deterministic buffer rematerialization."""
    import json
    from pathlib import Path

    import torch.nn as nn
    from quartet2.linear import Quartet_II_linear
    from torchao.prototype.quantized_training import Int8QuantizedTrainingLinearWeight as _W
    from train_cpt import QuartetMasterMixin

    p = Path(path)
    qmap = p / "quant_keys.json"
    if not qmap.is_file():
        raise ValueError(
            f"{p}: no quant_keys.json — the precision law forbids bf16 weight residency. "
            f"Convert the artifact once (scripts/convert_base_int8.py) and load that.")
    qmeta = json.loads(qmap.read_text())
    int8_keys = set(qmeta.get("keys", []))
    embed_set = set(qmeta.get("embedding_keys", []))
    is_v2 = qmeta.get("method") == "int8_rowwise_v2" or bool(embed_set)
    cfg_obj = AutoConfig.from_pretrained(str(p))
    text_cfg = getattr(cfg_obj, "text_config", cfg_obj)
    text_cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Gemma4ForCausalLM(text_cfg)
    model = model.to(torch.bfloat16)
    model = model.to_empty(device=device)
    f = _SafetensorsRead(str(p / "model.safetensors"))
    try:
        # 1. Quartet-swap + master assignment at exactly the quant_keys linears.
        for qk in int8_keys:
            mod_path, _, _ = qk.rpartition(".weight")
            parent_path, _, attr = mod_path.rpartition(".")
            parent = model.get_submodule(parent_path)
            if not hasattr(parent, attr):
                continue    # shared-KV layers (15+): the shard carries unused k/v_proj the
                            # model class never instantiates (build_model drops them too)
            old = getattr(parent, attr)
            q = Quartet_II_linear(old.in_features, old.out_features,
                                  bias=old.bias is not None,
                                  device=device, dtype=torch.bfloat16)
            q.__class__ = type("QuartetMasterLinear", (QuartetMasterMixin, Quartet_II_linear), {})
            int_data = f.get_tensor(qk).to(device)
            # scale → bf16 at load: v1 stores bf16 (no-op), v2 stores fp8 e4m3 (⊂ bf16, exact).
            # The master weight's dequant dtype IS the scale's dtype and the fp4 kernel asserts
            # bf16 input — the in-memory scale stays bf16; fp8 is STORAGE only.
            scale = f.get_tensor(qk + ".weight_scale").to(torch.bfloat16).to(device)  # bf16-sanctioned: S2 fp8→bf16 exact cast at load (storage stays fp8)
            if is_v2:
                scale = scale / 128.0   # unfold the ×128 e4m3-range fold (exact power of 2)
            q.weight = nn.Parameter(_W(int_data, scale), requires_grad=requires_grad)
            setattr(parent, attr, q)
        # 2. Everything else streams as stored (norms etc.), skipping masters + v2 embeddings.
        chunk: dict = {}
        for k in f.keys():
            if k.endswith(".weight_scale") or k in int8_keys or k in embed_set:
                continue
            chunk[k] = f.get_tensor(k)
            if len(chunk) >= 64:
                model.load_state_dict(chunk, strict=False)
                chunk.clear()
        if chunk:
            model.load_state_dict(chunk, strict=False)
        del chunk
        # 3. v2: embeddings are int8 masters too — install Int8ScaledEmbedding + tied head.
        if is_v2:
            for ek in embed_set:
                mod_path, _, _ = ek.rpartition(".weight")
                parent_path, _, attr = mod_path.rpartition(".")
                parent = model.get_submodule(parent_path) if parent_path else model
                escale = (text_cfg.hidden_size ** 0.5) if attr == "embed_tokens" else (
                    text_cfg.hidden_size_per_layer_input ** 0.5)
                w = f.get_tensor(ek).to(device)
                s = f.get_tensor(ek + ".weight_scale").to(device) / 128.0   # unfold
                setattr(parent, attr, Int8ScaledEmbedding(w, s, escale))
            model.lm_head = Int8TiedLmHead(model.model.embed_tokens)
        else:
            model.tie_weights()
    finally:
        f.close()
    # 4. persistent=False buffers (v2 embeddings carry their own embed_scale — skip there).
    if is_v2:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
        rot = Gemma4TextRotaryEmbedding(text_cfg)
        for name, buf in model.model.rotary_emb.named_buffers():
            buf.copy_(getattr(rot, name))
    else:
        _rematerialize_buffers(model, text_cfg)
    model._quartet_eval = True    # scorer pads to the fp4 kernel's 128-token multiple
    return model.eval()
