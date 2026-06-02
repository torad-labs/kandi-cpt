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


def causal_bf16_state_dict(path):
    """Read a saved Gemma4ForCausalLM checkpoint dir → a bf16 state dict. THE single source of truth for the
    int8-master round-trip: an int8 checkpoint (a quant_keys.json sidecar) is dequantized on the way out —
    int8 `_data` × per-row `.weight_scale` → bf16, and the `.weight_scale` tensors are dropped; a plain bf16
    checkpoint passes through as bf16. Every loader (eval, SFT init_from) goes through here so none can drift
    out of lock-step with train_cpt.save_checkpoint's write format (loading raw int8 without its scale = garbage)."""
    import json
    from pathlib import Path
    p = Path(path)
    raw = load_file(str(p / "model.safetensors"))
    qmap = p / "quant_keys.json"
    if not qmap.is_file():                                   # plain bf16 checkpoint
        return {k: v.to(torch.bfloat16) for k, v in raw.items()}
    int8_keys = set(json.loads(qmap.read_text()).get("keys", []))
    sd = {}
    for k, v in raw.items():
        if k.endswith(".weight_scale"):
            continue
        sd[k] = (v.to(torch.bfloat16) * raw[k + ".weight_scale"].to(torch.bfloat16).unsqueeze(1)
                 if k in int8_keys else v.to(torch.bfloat16))
    return sd


def load_causal_ckpt_bf16(path, device: str = "cuda:0"):
    """Load a SAVED Gemma4ForCausalLM checkpoint (a CPT/SFT save dir) for eval/generation — the bf16 + sdpa
    incantation. Dequantizes an int8-master checkpoint via causal_bf16_state_dict (the shared round-trip)."""
    from pathlib import Path
    p = Path(path)
    if not (p / "quant_keys.json").is_file():               # standard bf16 checkpoint → fast from_pretrained
        return Gemma4ForCausalLM.from_pretrained(
            str(p), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    cfg_obj = AutoConfig.from_pretrained(str(p))
    text_cfg = getattr(cfg_obj, "text_config", cfg_obj)     # saved config is already the text config
    text_cfg._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(text_cfg)
    model.load_state_dict(causal_bf16_state_dict(p), strict=False)
    model.tie_weights()
    return model.to(device, dtype=torch.bfloat16).eval()
