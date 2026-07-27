#!/usr/bin/env python3
"""One-prompt FP4/int8 inference for the Playground's LIVE mode — load a checkpoint in the int8-aware
loader and answer a single question through the Gemma chat template. Spawned as a subprocess by the
torad_playground_live action (so the heavy torch+model load is isolated and the GPU frees on exit).
Prints a single JSON line {"answer": ...} on the last line.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts python scripts/playground_infer.py \
      --checkpoint checkpoints/sft/edc_sft_v11 --prompt "Where is EDC Las Vegas 2026?" [--temp 1.2 --top-k 64]

memory mode = greedy (temp 0): what the weights actually encode.
voice mode  = sampled (temp>0): the on-device runtime sampling that gives Kandi its voice.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

import config as C
from pipeline.model_load import load_causal_ckpt_quartet


@torch.no_grad()
def answer(model, tok, q: str, temp: float, top_k: int, max_new: int) -> str:
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if temp and temp > 0:
        kw.update(do_sample=True, temperature=temp, top_k=top_k or 64)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    model = load_causal_ckpt_quartet(a.checkpoint, a.device)
    print(json.dumps({"answer": answer(model, tok, a.prompt, a.temp, a.top_k, a.max_new)}))


if __name__ == "__main__":
    main()
