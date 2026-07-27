#!/usr/bin/env python3
"""Cross-edition install probe #2: does CPT install "which EDC edition(s) did <artist> play"?

The densest cross-edition fact the campaign hydrated is artist→edition (1203 plays_at triples). This is
the breadth payload of "all EDC data": the model should associate an artist with the edition(s) they
actually played. Forced-choice by continuation likelihood over the edition labels ("<City> <Year>"); for
each artist we score every edition and check whether a TRUE edition ranks #1 (recall@1) / top-3, plus MRR
over the artist's true set. Base sits near chance (1/n_editions); a CPT that installed the lineups ranks
the artist's real edition(s) at the top.

Reuses eval_cloze's batched forced-choice scorer + its equivalence gate.

    PYTHONPATH=scripts python scripts/eval_artist_edition.py \
        --checkpoints checkpoints/cpt/edc_all97_v1 [--device cuda:0] [--max-artists 200]
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


@torch.no_grad()
def rank_chunked(model, tok, prefix, cands, chunk=8):
    """Forced-choice over many candidates, scored in memory-bounded chunks. Full-batch ranked_batch OOMs
    at ~98 candidates (the [N, len, 262K-vocab] log_softmax doesn't fit beside the resident model); chunking
    keeps each forward small while staying ~chunk× faster than per-candidate sequential. Recall@k is robust
    to the rare near-tie flip vs the exact sequential scorer."""
    pids = tok(prefix, return_tensors="pt").input_ids[0]
    p = pids.shape[0]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    sc = {}
    for i in range(0, len(cands), chunk):
        grp = cands[i:i + chunk]
        conts = [tok(c, add_special_tokens=False, return_tensors="pt").input_ids[0] for c in grp]
        seqs = [torch.cat([pids, c]) for c in conts]
        maxlen = max(s.shape[0] for s in seqs)
        batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            batch[j, : s.shape[0]] = s
            attn[j, : s.shape[0]] = 1
        logp = torch.log_softmax(
            model(batch.to(model.device), attention_mask=attn.to(model.device)).logits.float(), dim=-1)
        for j, (c, cids) in enumerate(zip(grp, conts)):
            n = cids.shape[0]
            sc[c] = sum(logp[j, p + k - 1, cids[k]].item() for k in range(n)) / n
        del logp
    return sorted(sc, key=sc.get, reverse=True)


ART_TPLS = [
    ("perf",  lambda a: f"{a} performed at EDC"),
    ("qa",    lambda a: f"Q: At which EDC did {a} perform? A: EDC"),
]


def load_artist_editions():
    """Return (artist -> {edition_label}), and the sorted distinct edition-label candidate list.
    edition_label = '<City> <Year>' from festival_meta; artists come from each edition's own slug-matched
    plays_at triples (so cross-festival noise — 'edma', 'buku' — is excluded)."""
    label_of = {}
    for m in glob.glob(str(C.ROOT / "data/edc/*/*/festival_meta.json")):
        parts = m.split("/"); ed = f"{parts[-3]}/{parts[-2]}"
        d = json.loads(open(m).read())
        city = (d.get("city") or parts[-3].replace("_", " ").title()).strip()
        label_of[ed] = f" {city} {parts[-2]}"
    art_eds: dict[str, set] = collections.defaultdict(set)
    for tp in glob.glob(str(C.ROOT / "data/edc/*/*/15_structured/triples.jsonl")):
        # .../<city>/<year>/15_structured/triples.jsonl → city=parts[-4], year=parts[-3]
        parts = tp.split("/"); ed = f"{parts[-4]}/{parts[-3]}"; exp = f"edc_{parts[-4]}_{parts[-3]}"
        if ed not in label_of:
            continue
        ents = {}
        ep = tp.replace("triples", "entities")
        try:
            for l in open(ep):
                o = json.loads(l); ents[o["id"]] = o.get("name_canonical", "")
        except FileNotFoundError:
            pass
        for l in open(tp):
            try: t = json.loads(l)
            except Exception: continue
            if t.get("predicate") == "plays_at" and str(t.get("subject_id", "")).startswith(exp + ":artist:"):
                a = ents.get(t["subject_id"], "").strip()
                if a and len(a) > 1:
                    art_eds[a].add(label_of[ed])
    cands = sorted(set(label_of.values()))
    return art_eds, cands


def score(model, tok, label, art_eds, cands, rank, max_artists):
    arts = sorted(art_eds)[:max_artists]
    r1 = r3 = n = 0
    mrr = 0.0
    per_tpl = collections.defaultdict(lambda: [0, 0])
    for a in arts:
        truth = art_eds[a]
        for tname, tpl in ART_TPLS:
            r = rank(model, tok, tpl(a), cands)
            hit1 = int(r[0] in truth)
            hit3 = int(any(c in truth for c in r[:3]))
            r1 += hit1; r3 += hit3; n += 1
            per_tpl[tname][0] += hit1; per_tpl[tname][1] += 1
            mrr += next((1.0 / (i + 1) for i, c in enumerate(r) if c in truth), 0.0)
    tpl_str = "  ".join(f"{k}@1={v[0]}/{v[1]}" for k, v in per_tpl.items())
    print(f"  [{label}]  edition@1 {r1}/{n}={r1 / n:.3f}  @3={r3 / n:.3f}  MRR={mrr / n:.3f}")
    print(f"           per-template: {tpl_str}")
    return {"ed_at1": r1 / n, "ed_at3": r3 / n, "ed_mrr": mrr / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-artists", type=int, default=200, help="cap probed artists (runtime bound)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
    art_eds, cands = load_artist_editions()
    print(f"artist→edition probe: {len(art_eds)} artists × {len(cands)} edition labels "
          f"(probing first {min(args.max_artists, len(art_eds))}; chance≈{1 / len(cands):.3f})\n")

    base = load_text_model_bf16(args.device)
    rank = rank_chunked      # memory-bounded chunked forced-choice (98 cands OOMs a full batch beside the 12GB model)
    print(f"  scorer: chunked-batch (memory-safe; recall@k robust to near-tie flips)\n")
    res = {"base": score(base, tok, "base", art_eds, cands, rank, args.max_artists)}
    del base
    torch.cuda.empty_cache()

    for ck in args.checkpoints:
        m = load_causal_ckpt_quartet(ck, args.device)
        res[ck] = score(m, tok, ck.split("/")[-1], art_eds, cands, rank, args.max_artists)
        del m
        torch.cuda.empty_cache()

    print("\n================ CROSS-EDITION INSTALL (artist→edition forced-choice) ================")
    print(f"  {'model':28s} {'ed@1':>7} {'ed@3':>7} {'MRR':>7}")
    for k, v in res.items():
        print(f"  {k.split('/')[-1]:28s} {v['ed_at1']:7.3f} {v['ed_at3']:7.3f} {v['ed_mrr']:7.3f}")


if __name__ == "__main__":
    main()
