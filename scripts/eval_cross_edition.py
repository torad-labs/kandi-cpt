#!/usr/bin/env python3
"""Cross-edition install probe: does CPT install "EDC <city> <year> → <venue>" across all 97 editions?

eval_cloze.py measures the LV-2026 artist→stage/day install (the 0/17 wall). But the full-history campaign
hydrated 96 other editions, where the DENSE, source-supported fact is artist→edition + edition→venue (stage
data is sparse outside LV — only 363/2530 :stage: triples). This probe measures the venue install: a 24-way
forced choice (the distinct EDC venues) by continuation likelihood — base sits near chance (1/24≈0.042); a
CPT that installed the geography ranks the true venue #1.

Reuses eval_cloze's scoring (cont_logprob / ranked_batch + the batched-equivalence gate) verbatim.

    PYTHONPATH=scripts python scripts/eval_cross_edition.py \
        --checkpoints checkpoints/cpt/edc_all97_v1 [--device cuda:0]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import torch
from transformers import AutoTokenizer

import config as C
from pipeline.model_load import load_text_model_bf16, load_causal_ckpt_quartet
from eval_cloze import ranked, ranked_batch, _validate_batched

# Venue cloze templates — declarative + QA, mirroring the hydration surface ("held at", "venue for").
VENUE_TPLS = [
    ("held_at", lambda name: f"{name} is held at the"),
    ("venue_of", lambda name: f"The venue for {name} is the"),
    ("qa",       lambda name: f"Q: At what venue is {name} held? A: the"),
]


def load_editions():
    """Return [(festival_name, true_venue)] + the sorted distinct-venue candidate list, from festival_meta."""
    eds = []
    venues = set()
    for m in sorted(glob.glob(str(C.ROOT / "data/edc/*/*/festival_meta.json"))):
        d = json.loads(open(m).read())
        v = (d.get("venue") or "").strip()
        name = (d.get("name") or "").strip()
        if not v or not name or v.lower().startswith("various"):   # 'Various (San Juan)' isn't a single answer
            continue
        eds.append((name, v))
        venues.add(v)
    return eds, sorted(venues)


def score(model, tok, label, eds, venues, rank):
    cands = [f" {v}" for v in venues]
    truth = {f" {v}": v for v in venues}
    per_tpl = collections.defaultdict(lambda: [0, 0])
    hit = n = 0
    mrr = 0.0
    for name, true_v in eds:
        for tname, tpl in VENUE_TPLS:
            r = rank(model, tok, tpl(name), cands)
            ok = int(truth[r[0]] == true_v)
            hit += ok; n += 1
            per_tpl[tname][0] += ok; per_tpl[tname][1] += 1
            mrr += next((1.0 / (i + 1) for i, c in enumerate(r) if truth[c] == true_v), 0.0)
    tpl_str = "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in per_tpl.items())
    print(f"  [{label}]  venue@1 {hit}/{n}={hit / n:.3f}  venueMRR={mrr / n:.3f}")
    print(f"           per-template: {tpl_str}")
    return {"venue_acc": hit / n, "venue_mrr": mrr / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
    eds, venues = load_editions()
    print(f"cross-edition venue probe: {len(eds)} editions × {len(venues)} candidate venues  "
          f"(chance≈{1 / len(venues):.3f})\n")

    base = load_text_model_bf16(args.device)
    rank = ranked_batch if _validate_batched(base, tok) else ranked
    print(f"  scorer: {'batched' if rank is ranked_batch else 'sequential (fallback)'}\n")
    res = {"base": score(base, tok, "base", eds, venues, rank)}
    del base
    torch.cuda.empty_cache()

    for ck in args.checkpoints:
        m = load_causal_ckpt_quartet(ck, args.device)
        res[ck] = score(m, tok, ck.split("/")[-1], eds, venues, rank)
        del m
        torch.cuda.empty_cache()

    print("\n================ CROSS-EDITION INSTALL (venue forced-choice) ================")
    print(f"  {'model':28s} {'venue@1':>8} {'venueMRR':>9}")
    for k, v in res.items():
        print(f"  {k.split('/')[-1]:28s} {v['venue_acc']:8.3f} {v['venue_mrr']:9.3f}")
    print(f"  (base≈chance means EDC geography is NOT in the base model; high CPT venue@1 = the "
          f"edition→venue\n   association installed across all 97 editions — the breadth signal.)")


if __name__ == "__main__":
    main()
