"""Hidden-state extraction (build spec Part 1a).

Reuses eval_cpt.py's model-loading path (Part 0.1): base via pipeline.model_load.load_text_model_bf16
(the multimodal→text remap), CPT via Gemma4ForCausalLM.from_pretrained — both bf16, so base-vs-CPT
isolates the CPT effect with no quantization confound (Part 4.4).

Extracts the RESIDUAL-STREAM hidden state (output_hidden_states=True), per layer, for each entity placed
in a FIXED neutral carrier context held identical across every entity and model version (Part 1a). Default
pooling = last token (aggregates preceding context in an autoregressive model); mean-pooling optional.
"""
from __future__ import annotations

import numpy as np
import torch

import config as C
from pipeline.model_load import load_text_model_bf16

# The carrier ENDS with the entity so the last-token hidden state has attended to the (constant) prefix.
CARRIER_DEFAULT = "At the Electric Daisy Carnival music festival, the entity is: {entity}"


def load_model(checkpoint: str | None, device: str = "cuda:0"):
    """checkpoint None/'base' → pre-CPT base (remap loader); else a CPT save_pretrained dir."""
    if checkpoint in (None, "", "base", "BASE"):
        return load_text_model_bf16(device), "base"
    from transformers import Gemma4ForCausalLM
    m = Gemma4ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to(device).eval()
    return m, checkpoint


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(C.MODEL_PATH))


@torch.no_grad()
def extract_all_layers(model, tok, entities: list[str], carrier: str = CARRIER_DEFAULT,
                       pooling: str = "last", device: str = "cuda:0") -> dict[int, np.ndarray]:
    """entities → {layer_index: float32 array [n_entities, hidden]}. Layer 0 = embeddings; 1..L = decoder
    residual stream. Identical carrier+pooling for every entity (the comparison is only valid if so)."""
    assert pooling in ("last", "mean")
    per_layer: list[list[np.ndarray]] | None = None
    for ent in entities:
        enc = tok(carrier.format(entity=ent), return_tensors="pt").to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple (L+1) of [1, seq, H]
        if per_layer is None:
            per_layer = [[] for _ in hs]
        for li, h in enumerate(hs):
            v = h[0].float().mean(0) if pooling == "mean" else h[0, -1].float()
            per_layer[li].append(v.cpu().numpy())
    return {li: np.stack(vs).astype(np.float32) for li, vs in enumerate(per_layer or [])}
