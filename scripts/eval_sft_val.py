#!/usr/bin/env python3
"""Early-stop lever (#5): score held-out SFT val loss for each checkpoint → pick the pre-overfit one.

Gekhman EMNLP'24: SFT hallucination emerges as LATE-EPOCH overfitting — the train loss keeps dropping
while held-out factuality declines. We can't cheaply run generation-factuality per checkpoint, but masked
val CE on a held-out QA split (build_sft.py --val-frac) is a faithful overfitting signal: the step where
val loss bottoms out (before it climbs) is the early-stop point. Reports val loss per checkpoint; the
lowest is the one to keep / deploy.

    PYTHONPATH=scripts python scripts/eval_sft_val.py --val data/_build/30_tokenized/sft_v1_val.arrow \
        --checkpoints checkpoints/sft/edc_sft_v1/step_000200 checkpoints/sft/edc_sft_v1 [--device cuda:0]
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_from_disk
from transformers import Gemma4ForCausalLM


@torch.no_grad()
def val_loss(model, ds, device, bs=8) -> float:
    tot, ntok = 0.0, 0
    for i in range(0, len(ds), bs):
        rows = ds[i:i + bs]
        ids = torch.tensor(rows["input_ids"], dtype=torch.long, device=device)
        lab = torch.tensor(rows["labels"], dtype=torch.long, device=device)
        out = model(input_ids=ids, labels=lab)
        n = int((lab != -100).sum().item())
        tot += out.loss.item() * n; ntok += n          # loss is mean over supervised tokens → re-weight
    return tot / max(1, ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="val arrow from build_sft.py --val-frac")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ds = load_from_disk(args.val)
    print(f"val set: {len(ds)} examples\n")
    results = {}
    for ck in args.checkpoints:
        m = Gemma4ForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
        vl = val_loss(m, ds, args.device)
        results[ck] = vl
        print(f"  {ck.split('/')[-1]:>16}  val_loss={vl:.4f}")
        del m; torch.cuda.empty_cache()
    best = min(results, key=results.get)
    print(f"\nbest (early-stop): {best.split('/')[-1]}  val_loss={results[best]:.4f}")


if __name__ == "__main__":
    main()
