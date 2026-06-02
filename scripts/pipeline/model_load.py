"""Load the Gemma 4 E2B text decoder (bf16) with the multimodal→text-only key remap.

The checkpoint is the multimodal Gemma4ForConditionalGeneration (text weights under
`model.language_model.*`); loading it into the text-only Gemma4ForCausalLM needs the remap or
every param silently random-inits. Shared by curation (perplexity scoring) and eval probes.
"""
from __future__ import annotations

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, Gemma4ForCausalLM

import config as C

_PREFIX = "model.language_model."


def load_text_model_bf16(device: str = "cuda:0"):
    text_cfg = AutoConfig.from_pretrained(str(C.MODEL_PATH)).text_config
    text_cfg._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(text_cfg)
    sd = load_file(f"{C.MODEL_PATH}/model.safetensors")
    remapped = {("model." + k[len(_PREFIX):]): v for k, v in sd.items() if k.startswith(_PREFIX)}
    if "lm_head.weight" in sd:
        remapped["lm_head.weight"] = sd["lm_head.weight"]
    model.load_state_dict(remapped, strict=False)
    model.tie_weights()
    return model.to(device, dtype=torch.bfloat16).eval()


def load_causal_ckpt_bf16(path, device: str = "cuda:0"):
    """Load a SAVED Gemma4ForCausalLM checkpoint (a CPT/SFT save_pretrained dir) for eval/generation.
    Centralizes the bf16 + sdpa incantation. Transparently handles int8-master checkpoints (a quant_keys.json
    sidecar): the weights are stored as int8 `_data` + per-row `.weight_scale`, so we dequantize them
    (int8 → bf16 × scale) on load instead of expecting plain bf16 weights."""
    import json
    from pathlib import Path
    p = Path(path)
    qmap = p / "quant_keys.json"
    if not qmap.is_file():                                  # standard bf16 checkpoint
        return Gemma4ForCausalLM.from_pretrained(
            str(p), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    # int8-master checkpoint → reconstruct bf16 weights from int8 + per-row scale
    int8_keys = set(json.loads(qmap.read_text()).get("keys", []))
    cfg_obj = AutoConfig.from_pretrained(str(p))
    text_cfg = getattr(cfg_obj, "text_config", cfg_obj)     # saved config is already the text config
    text_cfg._attn_implementation = "sdpa"
    raw = load_file(str(p / "model.safetensors"))
    sd = {}
    for k, v in raw.items():
        if k.endswith(".weight_scale"):
            continue
        if k in int8_keys:
            scale = raw[k + ".weight_scale"].to(torch.bfloat16)
            sd[k] = v.to(torch.bfloat16) * scale.unsqueeze(1)
        else:
            sd[k] = v.to(torch.bfloat16)
    model = Gemma4ForCausalLM(text_cfg)
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    return model.to(device, dtype=torch.bfloat16).eval()
