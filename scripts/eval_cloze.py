#!/usr/bin/env python3
"""CPT-FAITHFUL install probe: declarative cloze, forced-choice by likelihood, in the TRAINING surface.

eval_cpt.py wraps every probe in a CHAT TEMPLATE (<start_of_turn>…), asks a QUESTION, lets the model
GENERATE, and keyword-matches a reworded answer ("cosmic"). For a pure-CPT model that has not been
instruction-tuned, that measures *retrieval under paraphrase* — an SFT capability — not whether the
fact is installed. A CPT model only knows how to CONTINUE text in its training distribution.

This probe removes every retrieval/format burden and reads the installed association directly:
  • NO chat template — raw declarative text, exactly as the corpus stated it.
  • TRAINING wording — prefixes mirror fact_expand.fact_templates verbatim.
  • camelCase stage tokens (cosmicMEADOW) — the surface the model actually trained on, not "Cosmic Meadow".
  • forced-choice by CONTINUATION LIKELIHOOD, not generation: is P(true_stage | prefix) the argmax over
    the 7 stages. Base sits near chance (1/7); a CPT that installed the fact ranks the true stage #1.

    PYTHONPATH=scripts python scripts/eval_cloze.py --checkpoints checkpoints/cpt/edc_lv_2026_v4 \
        checkpoints/cpt/edc_lv_2026_v5 [--device cuda:0]
"""
from __future__ import annotations

import argparse
import collections
import json

import torch
from transformers import AutoTokenizer

import config as C
from pipeline.fact_expand import DATES, FESTIVAL, VENUE, camel
from pipeline.model_load import load_text_model_bf16, load_causal_ckpt_quartet

TRIP = C.ROOT / "data/edc/las_vegas/2026/15_structured/triples.jsonl"
ENTS = C.ROOT / "data/edc/las_vegas/2026/15_structured/entities.jsonl"
DAYS3 = ["Friday", "Saturday", "Sunday"]

# Prefixes mirror fact_expand.fact_templates EXACTLY, truncated right before the cloze token. They end
# WITHOUT a trailing space; the candidate carries the leading space (correct for SentencePiece — the
# space binds to the first continuation token, matching how the corpus tokenizes "-> cosmicMEADOW").
STAGE_TPLS = [
    ("lineup",  lambda a: f"Lineup fact — {FESTIVAL}: {a} ->"),
    ("qa",      lambda a: f"Q: Which stage does {a} play at {FESTIVAL}? A:"),
    ("nary",    lambda a: f"At {FESTIVAL} ({DATES}), held at {VENUE}, {a} plays the"),
]
DAY_TPLS = [
    ("qa_day",  lambda a: f"Q: On which day does {a} play at {FESTIVAL}? A:"),
    ("qa_mc",   lambda a: f"Q: Is {a} playing Friday, Saturday, or Sunday at {FESTIVAL}? A:"),
]


@torch.no_grad()
def cont_logprob(model, tok, prefix: str, cont: str) -> float:
    """Mean per-token log-prob of `cont` continuing `prefix` (teacher-forced). Length-normalized so
    multi-subword stage names compete fairly."""
    pids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    cids = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    ids = torch.cat([pids, cids], dim=1)
    logp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
    s, n = pids.shape[1], cids.shape[1]
    return sum(logp[s + k - 1, cids[0, k]].item() for k in range(n)) / n


def ranked(model, tok, prefix: str, cands: list[str]) -> list[str]:
    sc = {c: cont_logprob(model, tok, prefix, c) for c in cands}
    return sorted(sc, key=sc.get, reverse=True)


@torch.no_grad()
def ranked_batch(model, tok, prefix: str, cands: list[str]) -> list[str]:
    """Same forced-choice as ranked(), but scores ALL candidates for `prefix` in ONE batched forward
    (parallel local inference). Right-padding + attention_mask: with causal attention, positions before
    the pad are identical to scoring each candidate alone — so this is numerically equivalent to ranked()
    (gated by _validate_batched on the base model), just ~Ncands× fewer forward passes."""
    pids = tok(prefix, return_tensors="pt").input_ids[0]
    p = pids.shape[0]
    conts = [tok(c, add_special_tokens=False, return_tensors="pt").input_ids[0] for c in cands]
    seqs = [torch.cat([pids, c]) for c in conts]
    maxlen = max(s.shape[0] for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : s.shape[0]] = s
        attn[i, : s.shape[0]] = 1
    logp = torch.log_softmax(
        model(batch.to(model.device), attention_mask=attn.to(model.device)).logits.float(), dim=-1)
    sc = {}
    for i, (c, cids) in enumerate(zip(cands, conts)):
        n = cids.shape[0]
        sc[c] = sum(logp[i, p + k - 1, cids[k]].item() for k in range(n)) / n
    return sorted(sc, key=sc.get, reverse=True)


def _validate_batched(model, tok):
    """Gate: confirm the batched scorer ranks identically to the sequential one before trusting it."""
    pfx, cands = f"Q: Which stage does Armin van Buuren play at {FESTIVAL}? A:", [" kineticFIELD", " cosmicMEADOW", " bassPOD"]
    return ranked(model, tok, pfx, cands) == ranked_batch(model, tok, pfx, cands)


def load_facts():
    by_id = {json.loads(l)["id"]: json.loads(l) for l in open(ENTS) if l.strip()}
    a_stage: dict[str, set] = collections.defaultdict(set)
    a_day: dict[str, set] = collections.defaultdict(set)
    stages: set[str] = set()
    for l in open(TRIP):
        if not l.strip():
            continue
        t = json.loads(l)
        if t.get("predicate") != "plays_at":
            continue
        if ":stage:" not in str(t.get("object", "")):
            continue          # the multi-edition campaign appended plays_at→festival triples; keep only →stage
        a = by_id.get(t["subject_id"], {}).get("name_canonical")
        if not a:
            continue
        s = t["object"].split(":stage:")[1]
        stages.add(s)
        a_stage[a].add(s)
        d = (t.get("attrs") or {}).get("day", "")
        if d:
            a_day[a].add(d)
    return a_stage, a_day, sorted(stages)


def score(model, tok, label, a_stage, a_day, stage_slugs, rank=ranked_batch):
    stage_cands = [f" {camel(s)}" for s in stage_slugs]
    slug_of = {f" {camel(s)}": s for s in stage_slugs}
    day_cands = [f" {d}" for d in DAYS3]

    per_tpl = collections.defaultdict(lambda: [0, 0])           # tpl_name -> [hit, n]
    st_hit = st_n = 0
    st_mrr = 0.0
    for a, trueset in a_stage.items():
        for name, tpl in STAGE_TPLS:
            r = rank(model, tok, tpl(a), stage_cands)
            hit = int(slug_of[r[0]] in trueset)
            st_hit += hit
            st_n += 1
            per_tpl[name][0] += hit
            per_tpl[name][1] += 1
            st_mrr += next((1.0 / (i + 1) for i, c in enumerate(r) if slug_of[c] in trueset), 0.0)

    dy_hit = dy_n = 0
    for a, trueset in a_day.items():
        for name, tpl in DAY_TPLS:
            r = rank(model, tok, tpl(a), day_cands)
            hit = int(r[0].strip() in trueset)
            dy_hit += hit
            dy_n += 1
            per_tpl[name][0] += hit
            per_tpl[name][1] += 1

    tpl_str = "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in per_tpl.items())
    print(f"  [{label}]  stage@1 {st_hit}/{st_n}={st_hit / st_n:.3f}  stageMRR={st_mrr / st_n:.3f}  "
          f"| day@1 {dy_hit}/{dy_n}={dy_hit / dy_n:.3f}")
    print(f"           per-template: {tpl_str}")
    return {"stage_acc": st_hit / st_n, "stage_mrr": st_mrr / st_n, "day_acc": dy_hit / dy_n,
            "stage_n": st_n, "day_n": dy_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[], help="CPT checkpoint dirs to compare vs base")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
    a_stage, a_day, stage_slugs = load_facts()
    print(f"facts: {len(a_stage)} artists × {len(stage_slugs)} stages | "
          f"stage chance≈{1 / len(stage_slugs):.3f}  day chance≈{1 / 3:.3f}")
    print("CPT-faithful cloze — no chat template, camelCase stages, forced-choice by continuation likelihood:\n")

    base = load_text_model_bf16(args.device)
    # Use the fast batched scorer ONLY if it is exactly equivalent to the canonical sequential probe on
    # this model; otherwise fall back to sequential so the numbers stay trustworthy + comparable to prior
    # runs. (Batched GEMM reduction order can flip near-tie rankings — correctness beats the speedup.)
    rank = ranked_batch if _validate_batched(base, tok) else ranked
    print(f"  scorer: {'batched (validated == sequential)' if rank is ranked_batch else 'sequential (batched failed equivalence — falling back for trustworthy numbers)'}\n")
    res = {"base": score(base, tok, "base", a_stage, a_day, stage_slugs, rank=rank)}
    del base
    torch.cuda.empty_cache()

    for ck in args.checkpoints:
        m = load_causal_ckpt_quartet(ck, args.device)
        res[ck] = score(m, tok, ck.split("/")[-1], a_stage, a_day, stage_slugs, rank=rank)
        del m
        torch.cuda.empty_cache()

    print("\n================ INSTALL (training-surface cloze) ================")
    print(f"  {'model':24s} {'stage@1':>8} {'stageMRR':>9} {'day@1':>7}")
    for k, v in res.items():
        print(f"  {k.split('/')[-1]:24s} {v['stage_acc']:8.3f} {v['stage_mrr']:9.3f} {v['day_acc']:7.3f}")
    print(f"  (base ≈ chance means the fact is NOT in the base model; a high CPT stage@1 = the "
          f"association IS installed,\n   independent of whether the model can yet RETRIEVE it under "
          f"question-form — that gap is the SFT job.)")


if __name__ == "__main__":
    main()
