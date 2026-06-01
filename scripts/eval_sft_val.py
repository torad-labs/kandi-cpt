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
import math
import sys

import torch
from datasets import load_from_disk

from pipeline.model_load import load_causal_ckpt_bf16


@torch.no_grad()
def val_loss(model, ds, device, bs=2) -> float:   # bs small: labels= forward materializes [bs,seq,262K-vocab] logits
    tot, ntok = 0.0, 0
    for i in range(0, len(ds), bs):
        rows = ds[i:i + bs]
        ids = torch.tensor(rows["input_ids"], dtype=torch.long, device=device)
        lab = torch.tensor(rows["labels"], dtype=torch.long, device=device)
        out = model(input_ids=ids, labels=lab)
        n = int((lab != -100).sum().item())
        if n == 0 or not torch.isfinite(out.loss):     # all-pad batch → loss is nan; nan*0=nan would poison
            continue
        tot += out.loss.item() * n; ntok += n          # loss is mean over supervised tokens → re-weight
    return tot / ntok if ntok else float("nan")        # nan (not 0.0) so a fully-broken ckpt can't win min()


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
        m = load_causal_ckpt_bf16(ck, args.device)
        vl = val_loss(m, ds, args.device)
        results[ck] = vl
        print(f"  {ck.split('/')[-1]:>16}  val_loss={vl:.4f}")
        del m; torch.cuda.empty_cache()
    finite = {k: v for k, v in results.items() if math.isfinite(v)}
    if not finite:
        sys.exit("✗ all checkpoints scored nan/empty val loss — broken val set or diverged weights")
    best = min(finite, key=finite.get)
    print(f"\nbest (early-stop): {best.split('/')[-1]}  val_loss={finite[best]:.4f}")


if __name__ == "__main__":
    main()
