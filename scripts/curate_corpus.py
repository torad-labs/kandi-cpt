#!/usr/bin/env python3
"""Curate the 10_clean source-of-truth across all editions (Llama-3-style), BEFORE synth.

Embeds every cleaned doc (roberta-large), drops near-duplicates (MinHash, keep highest-trust),
scores quality (trust+length+density+fact-signal) × difficulty (base-model perplexity), drops the
bottom fraction, and clusters for a topic-balance report. NEVER deletes a 10_clean .md — it writes
a keep-list (data/_build/curation/keep_list.json) that synth/tokenize honor, stamps each
.clean.json with the scores, and emits curation_report.json.

    PYTHONPATH=scripts python scripts/curate_corpus.py [--editions …] [--drop-frac 0.10]
        [--no-difficulty] [--clusters 20] [--carve-holdout las_vegas/2026:0.1]

Heavy step: install nothing new — uses transformers roberta-large (embeddings) + the base Gemma
model (perplexity); both load on cuda:0 sequentially.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as C
from pipeline.editions import iter_editions
from pipeline import provenance as P
from pipeline import curate_lib as L

SUBFOLDER_DOMAIN = {"official": "official-festival", "reference": "news-music-trade",
                    "youtube": "social-vetted", "news": "news-music-trade",
                    "reddit": "community-discussion"}


def body_of(md_text: str) -> str:
    """Strip the canonical header (everything up to and including the first '---')."""
    return md_text.split("\n---\n", 1)[-1].strip() if "\n---\n" in md_text else md_text.strip()


def collect(editions_only):
    """Return list of dicts: {md, sidecar, subfolder, domain_class, trust, text}."""
    docs = []
    for ed in iter_editions("edc", only=editions_only):
        clean_root = ed.dir / "10_clean"
        if not clean_root.is_dir():
            continue
        for md in sorted(clean_root.rglob("*.md")):
            if md.name == "README.md":
                continue
            sub = md.relative_to(clean_root).parts[0]
            sidecar = md.with_suffix(".clean.json")
            dc = ""
            if sidecar.exists():
                try:
                    dc = json.loads(sidecar.read_text()).get("domain_class", "")
                except Exception:
                    dc = ""
            dc = dc or SUBFOLDER_DOMAIN.get(sub, "user-contributed")
            text = body_of(md.read_text(encoding="utf-8", errors="replace"))
            docs.append({"md": md, "sidecar": sidecar, "subfolder": sub, "edition": ed.slug,
                         "domain_class": dc, "trust": P.trust_for(dc), "text": text})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="restrict to 'city/year,…'")
    ap.add_argument("--drop-frac", type=float, default=0.10, help="drop this bottom fraction by score")
    ap.add_argument("--min-words", type=int, default=60, help="hard floor: drop docs shorter than this (thin youtube/stubs)")
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--no-difficulty", action="store_true", help="skip base-model perplexity (faster)")
    ap.add_argument("--carve-holdout", help="'city/year:frac' — freeze a holdout sample from that edition")
    args = ap.parse_args()

    docs = collect(args.editions)
    n = len(docs)
    if not n:
        print("no 10_clean docs found"); return
    print(f"collected {n} cleaned docs across editions")
    texts = [d["text"] for d in docs]

    # 1) embeddings (roberta-large)
    emb = L.embed_docs(texts)

    # 2) near-dup → mark drops (keep highest-trust per group)
    dup_of = {}
    for grp in L.near_dup_groups(texts):
        keep = max(grp, key=lambda i: docs[i]["trust"])
        for i in grp:
            if i != keep:
                dup_of[i] = keep

    # 3) quality + difficulty
    qual = [L.quality_score(d["text"], d["trust"]) for d in docs]
    if args.no_difficulty:
        diff = [0.5] * n
    else:
        from pipeline.model_load import load_text_model_bf16
        from transformers import AutoTokenizer
        model = load_text_model_bf16("cuda:0")
        tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
        diff = L.normalize(L.perplexity(texts, model, tok))
        del model
        import torch; torch.cuda.empty_cache()
    combined = [q * (0.5 + 0.5 * d) for q, d in zip(qual, diff)]

    # 4) cluster (balance report)
    clusters = L.cluster(emb, k=args.clusters)

    # 5) decide keep/drop: drop near-dups + too-thin, then bottom drop_frac by combined among survivors
    too_thin = {i for i in range(n) if i not in dup_of and len(docs[i]["text"].split()) < args.min_words}
    survivors = [i for i in range(n) if i not in dup_of and i not in too_thin]
    survivors_sorted = sorted(survivors, key=lambda i: combined[i])
    n_drop = int(len(survivors) * args.drop_frac)
    dropped_low = set(survivors_sorted[:n_drop])

    # 6) holdout carve (exclude from keep-list, freeze in manifest) — BEFORE synth
    holdout = set()
    if args.carve_holdout:
        slug, frac = args.carve_holdout.split(":")
        cand = [i for i in survivors if i not in dropped_low and docs[i]["edition"] == slug]
        cand.sort(key=lambda i: combined[i], reverse=True)        # hold out the best (hardest eval)
        holdout = set(cand[: int(len(cand) * float(frac))])

    keep, report = [], []
    for i, d in enumerate(docs):
        if i in dup_of:
            status, reason = "drop", f"near-dup-of:{docs[dup_of[i]]['md'].name}"
        elif i in too_thin:
            status, reason = "drop", "too-thin"
        elif i in dropped_low:
            status, reason = "drop", "bottom-frac"
        elif i in holdout:
            status, reason = "holdout", "carved"
        else:
            status, reason = "keep", ""
            keep.append(str(d["md"].relative_to(C.ROOT)))
        rel = str(d["md"].relative_to(C.ROOT))
        report.append({"path": rel, "edition": d["edition"], "subfolder": d["subfolder"],
                       "domain_class": d["domain_class"], "quality": round(qual[i], 4),
                       "difficulty": round(diff[i], 4), "combined": round(combined[i], 4),
                       "cluster": clusters[i], "status": status, "reason": reason})
        # stamp the sidecar (only pipeline docs have one)
        if d["sidecar"].exists():
            try:
                sc = json.loads(d["sidecar"].read_text())
                sc.update({"quality_score": round(qual[i], 4), "difficulty_score": round(diff[i], 4),
                           "cluster_id": clusters[i], "curation_status": status})
                d["sidecar"].write_text(json.dumps(sc, indent=2))
            except Exception:
                pass

    out = C.BUILD / "curation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "keep_list.json").write_text(json.dumps(sorted(keep), indent=2))
    (out / "curation_report.json").write_text(json.dumps(report, indent=2))
    if holdout:
        hostems = sorted(docs[i]["md"].stem for i in holdout)
        man = C.edition("edc", *args.carve_holdout.split(":")[0].split("/")) / "holdout" / "manifest.json"
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({"stems": hostems, "carved_from": "10_clean (curated, pre-synth)"}, indent=2))

    # cluster-balance report
    from collections import Counter
    cc = Counter(clusters[i] for i in range(n) if report[i]["status"] == "keep")
    print(f"kept={len(keep)}  near-dup-dropped={len(dup_of)}  too-thin-dropped={len(too_thin)}  "
          f"bottom-dropped={len(dropped_low)}  holdout={len(holdout)}")
    print(f"cluster sizes (kept): {dict(sorted(cc.items()))}")
    print(f"wrote {out/'keep_list.json'} + curation_report.json")


if __name__ == "__main__":
    main()
