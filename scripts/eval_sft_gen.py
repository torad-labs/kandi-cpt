#!/usr/bin/env python3
"""SFT confirmation: does the instruction-tuned model ANSWER EDC questions through the chat template?

The forced-choice evals prove the knowledge is in the weights; the SFT smoke proved the loss drops. This
is the actual product check — generate answers in the Gemma chat format and read them. Compares the
pre-SFT CPT brain (edc_all97_v1) against the SFT'd model (edc_sft_v1): the CPT model continues raw text but
doesn't follow the chat turn; the SFT'd one should answer conversationally + correctly.

    PYTHONPATH=scripts python scripts/eval_sft_gen.py \
        --checkpoints checkpoints/cpt/edc_all97_v1 checkpoints/sft/edc_sft_v1 [--device cuda:0]
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer, Gemma4ForCausalLM

import config as C

PROMPTS = [
    "Where is EDC Las Vegas 2026 held?",
    "Where is EDC Mexico City 2016 held?",
    "Which stage does deadmau5 play at EDC Orlando 2026?",
    "At which EDC edition did Skrillex perform?",
    "What is EDC?",
    "What is the capital of France?",                 # general-ability (forgetting) check, in chat form
]


@torch.no_grad()
def answer(model, tok, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
    for ck in args.checkpoints:
        print(f"\n========== {ck.split('/')[-1]} ==========")
        m = Gemma4ForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
        for q in PROMPTS:
            print(f"  Q: {q}\n     A: {answer(m, tok, q)!r}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
