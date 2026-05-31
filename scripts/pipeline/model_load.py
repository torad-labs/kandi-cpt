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
