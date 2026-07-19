#!/usr/bin/env python3
"""Curate one domain's 10_clean source-of-truth, BEFORE synth.

Embeds every cleaned doc (roberta-large), drops near-duplicates (MinHash, keep highest-trust),
cross-dedups kept candidates against `_shared/replay` (a near-dup in both wastes the replay
budget), scores **educational density × trust** when the score stage ran (else legacy quality
heuristics with EDUCATIONAL_SCORE_UNAVAILABLE), reports base-model perplexity as a diagnostic
column only (not a drop axis), applies the rarity guard (bottom-decile mention entities are
drop-exempt), honors the pii_scan quarantine list, and clusters for a topic-balance report.

NEVER deletes a 10_clean .md — writes an EDC-compatible keep-list
(data/_build/curation/keep_list.json) or a domain-scoped keep-list
(data/_build/curation/keep_list_<domain>.json) that downstream consumers honor, stamps each
.clean.json with the scores, and emits the matching curation report.

    PYTHONPATH=scripts python scripts/curate_corpus.py [--domain edc] [--editions …] [--drop-frac 0.10]
        [--no-difficulty] [--clusters 20] [--carve-holdout las_vegas/2026:0.1]

Heavy step: transformers roberta-large (embeddings) + optional base Gemma (perplexity diagnostic);
both load on cuda:0 of the *visible* device — pin CUDA_VISIBLE_DEVICES=1.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import config as C
from pipeline.editions import iter_editions
from pipeline import provenance as P
from pipeline import curate_lib as L
from pipeline.score_educational import load_educational_scores
from pipeline.pii_scan import load_quarantine

SUBFOLDER_DOMAIN = {"official": "official-festival", "reference": "news-music-trade",
                    "youtube": "social-vetted", "news": "news-music-trade",
                    "reddit": "community-discussion"}

# Word-ish tokens for mention counts (rarity guard). Lowercase alnum runs ≥ 3 chars.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def body_of(md_text: str) -> str:
    """Strip the canonical header (everything up to and including the first '---')."""
    return md_text.split("\n---\n", 1)[-1].strip() if "\n---\n" in md_text else md_text.strip()


def collect(domain, editions_only):
    """Return list of dicts: {md, sidecar, subfolder, domain_class, trust, text, path}."""
    docs = []
    for ed in iter_editions(domain, only=editions_only):
        clean_root = ed.dir / "10_clean"
        if not clean_root.is_dir():
            continue
        for md in sorted(clean_root.rglob("*.md")):
            if md.name == "README.md":
                continue
            sub = md.relative_to(clean_root).parts[0] if md.relative_to(clean_root).parts else ""
            sidecar = md.with_suffix(".clean.json")
            dc = ""
            if sidecar.exists():
                try:
                    dc = json.loads(sidecar.read_text()).get("domain_class", "")
                except Exception:
                    dc = ""
            dc = dc or SUBFOLDER_DOMAIN.get(sub, "user-contributed")
            text = body_of(md.read_text(encoding="utf-8", errors="replace"))
            rel = str(md.relative_to(C.ROOT))
            docs.append({
                "md": md, "sidecar": sidecar, "subfolder": sub, "edition": ed.slug,
                "domain_class": dc, "trust": P.trust_for(dc), "text": text, "path": rel,
            })
    return docs


def mention_counts(texts: list[str]) -> Counter:
    """Corpus-wide token mention counts for the rarity guard."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(t.lower() for t in _TOKEN_RE.findall(text))
    return counts


def rare_entity_set(counts: Counter, decile: float = 0.10) -> set[str]:
    """Tokens at or below the decile of the mention-count distribution.

    A token with count ≤ the 10th-percentile count is rare. Long-tail install targets
    (count=1 on a corpus whose p10 is 1) are all rare — that is correct; the guard then
    protects every primary carrier of a hapax entity.
    """
    if not counts:
        return set()
    values = sorted(counts.values())
    # Nearest-rank percentile.
    idx = max(0, min(len(values) - 1, int(len(values) * decile) - 1 if decile > 0 else 0))
    # When decile>0, include at least the minimum count tier.
    threshold = values[idx] if decile > 0 else values[0]
    if decile > 0 and idx == 0:
        threshold = values[0]
    return {tok for tok, c in counts.items() if c <= threshold}


def doc_carries_rare(text: str, rare: set[str]) -> bool:
    if not rare:
        return False
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return bool(tokens & rare)


def derive_drop_frac(n_docs: int, override: float | None) -> tuple[float, str]:
    """Zero-knob drop-frac derivation (docs/cpt-overhaul/01). Override is recorded when set."""
    if override is not None:
        return float(override), f"operator-override={override}"
    if n_docs < 500:
        return 0.0, "corpus<500→0"
    if n_docs <= 5000:
        return 0.05, "corpus 500–5k→0.05"
    return 0.10, "corpus>5k→0.10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc", help="domain slug — selects data/<domain> and its registry")
    ap.add_argument("--editions", help="restrict to 'city/year,…'")
    ap.add_argument(
        "--drop-frac", type=float, default=None,
        help="override derived drop fraction (recorded). Default: derive from corpus size.",
    )
    ap.add_argument("--min-words", type=int, default=60, help="hard floor: drop docs shorter than this")
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument(
        "--no-difficulty", action="store_true",
        help="skip base-model perplexity diagnostic (faster). Perplexity is report-only either way.",
    )
    ap.add_argument("--carve-holdout", help="'city/year:frac' — freeze a holdout sample from that edition")
    args = ap.parse_args()

    docs = collect(args.domain, args.editions)
    n = len(docs)
    if not n:
        print("no 10_clean docs found"); return
    print(f"collected {n} cleaned docs across editions")
    texts = [d["text"] for d in docs]
    paths = [d["path"] for d in docs]

    # ── PII quarantine (non-destructive; scan stage writes the list) ──────────
    quarantine = load_quarantine(args.domain)
    quarantined: set[int] = set()
    if quarantine is None:
        pii_status = "PII_SCAN_NOT_RUN"
        print(f"pii: {pii_status} (no quarantine list — scan stage omitted or not yet run)")
    else:
        pii_status = "PII_SCAN_OK"
        for i, p in enumerate(paths):
            if p in quarantine:
                quarantined.add(i)
        print(f"pii: honoring quarantine list — {len(quarantined)} docs quarantined")

    # ── Educational scores (path→score) or heuristic fallback ─────────────────
    ed_scores = load_educational_scores(args.domain)
    if ed_scores is None:
        ed_status = "EDUCATIONAL_SCORE_UNAVAILABLE"
        print(f"educational: {ed_status} — falling back to quality heuristics")
        qual = [L.quality_score(d["text"], d["trust"]) for d in docs]
        # Normalize quality into a 1–5-ish product with trust for a comparable drop axis.
        edu = [q for q in qual]
    else:
        ed_status = "EDUCATIONAL_SCORE_OK"
        missing = sum(1 for p in paths if p not in ed_scores)
        print(f"educational: loaded {len(ed_scores)} scores ({missing} docs unscored → heuristic fill)")
        edu = []
        qual = []
        for d, p in zip(docs, paths):
            if p in ed_scores:
                # Teacher scores are 1–5; fold trust into the drop product.
                s = float(ed_scores[p])
                edu.append(s * (0.5 + 0.5 * float(d["trust"])))
                qual.append(s)
            else:
                q = L.quality_score(d["text"], d["trust"])
                edu.append(q)
                qual.append(q)

    drop_frac, drop_frac_src = derive_drop_frac(n, args.drop_frac)
    print(f"drop-frac={drop_frac} ({drop_frac_src})")

    # 1) embeddings (roberta-large) — still used for near-dup + clustering
    emb = L.embed_docs(texts)

    # 2) near-dup → mark drops (keep highest-trust per group)
    dup_of = {}
    for grp in L.near_dup_groups(texts):
        keep_i = max(grp, key=lambda i: docs[i]["trust"])
        for i in grp:
            if i != keep_i:
                dup_of[i] = keep_i

    # 2b) replay cross-dedup — domain docs near-dup of _shared/replay leave the keep-list
    # (non-destructive). A near-duplicate in both wastes the replay budget (docs/cpt-overhaul/01).
    replay_texts = L.load_replay_texts()
    replay_dups = L.indices_near_dup_of_reference(texts, replay_texts) - set(dup_of.keys()) - quarantined
    print(
        f"replay-cross-dedup: {len(replay_texts)} replay docs loaded; "
        f"{len(replay_dups)} domain docs near-dup of replay"
    )

    # 3) perplexity = DIAGNOSTIC ONLY (not a drop axis) — docs/cpt-overhaul/01
    if args.no_difficulty:
        diff = [0.5] * n
        diff_status = "skipped"
    else:
        from pipeline.model_load import load_text_model_bf16
        from transformers import AutoTokenizer
        model = load_text_model_bf16("cuda:0")
        tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
        diff = L.normalize(L.perplexity(texts, model, tok))
        del model
        import torch
        torch.cuda.empty_cache()
        diff_status = "diagnostic"
    # Drop product: educational × trust (already folded). Perplexity does NOT enter.
    combined = list(edu)

    # 4) cluster (balance report)
    clusters = L.cluster(emb, k=args.clusters)

    # 5) rarity guard — bottom-decile mention entities are drop-exempt
    counts = mention_counts(texts)
    rare = rare_entity_set(counts)
    protected = {
        i for i in range(n)
        if i not in dup_of and i not in quarantined and i not in replay_dups
        and doc_carries_rare(docs[i]["text"], rare)
    }
    print(f"rarity-guard: {len(rare)} rare tokens; {len(protected)} docs protected")

    # 6) decide keep/drop: quarantine + near-dups + replay-dups + too-thin, then bottom drop_frac
    #    among unprotected survivors. Protected docs are never bottom-dropped.
    too_thin = {
        i for i in range(n)
        if i not in dup_of and i not in quarantined and i not in replay_dups
        and len(docs[i]["text"].split()) < args.min_words
    }
    survivors = [
        i for i in range(n)
        if i not in dup_of and i not in too_thin and i not in quarantined and i not in replay_dups
    ]
    drop_candidates = [i for i in survivors if i not in protected]
    drop_candidates_sorted = sorted(drop_candidates, key=lambda i: combined[i])
    n_drop = int(len(survivors) * drop_frac) if drop_frac > 0 else 0
    # Never drop more than the unprotected pool.
    n_drop = min(n_drop, len(drop_candidates_sorted))
    dropped_low = set(drop_candidates_sorted[:n_drop])

    # 7) holdout carve (exclude from keep-list, freeze in manifest) — BEFORE synth
    holdout = set()
    if args.carve_holdout:
        slug, frac = args.carve_holdout.split(":")
        cand = [i for i in survivors if i not in dropped_low and docs[i]["edition"] == slug]
        cand.sort(key=lambda i: combined[i], reverse=True)        # hold out the best (hardest eval)
        holdout = set(cand[: int(len(cand) * float(frac))])

    keep, report = [], []
    for i, d in enumerate(docs):
        if i in quarantined:
            status, reason = "quarantine", "pii-dense"
        elif i in dup_of:
            status, reason = "drop", f"near-dup-of:{docs[dup_of[i]]['md'].name}"
        elif i in replay_dups:
            status, reason = "drop", "replay-near-dup"
        elif i in too_thin:
            status, reason = "drop", "too-thin"
        elif i in dropped_low:
            status, reason = "drop", "bottom-frac"
        elif i in holdout:
            status, reason = "holdout", "carved"
        else:
            status, reason = "keep", ("rarity-protected" if i in protected else "")
            keep.append(d["path"])
        rel = d["path"]
        report.append({
            "path": rel,
            "edition": d["edition"],
            "subfolder": d["subfolder"],
            "domain_class": d["domain_class"],
            "quality": round(float(qual[i]), 4),
            "educational": round(float(edu[i]), 4),
            "difficulty": round(float(diff[i]), 4),   # diagnostic only
            "difficulty_role": "diagnostic",
            "combined": round(float(combined[i]), 4),
            "cluster": clusters[i],
            "status": status,
            "reason": reason,
            "rarity_protected": i in protected,
        })
        # stamp the sidecar (only pipeline docs have one)
        if d["sidecar"].exists():
            try:
                sc = json.loads(d["sidecar"].read_text())
                sc.update({
                    "quality_score": round(float(qual[i]), 4),
                    "educational_score": round(float(edu[i]), 4),
                    "difficulty_score": round(float(diff[i]), 4),
                    "difficulty_role": "diagnostic",
                    "cluster_id": clusters[i],
                    "curation_status": status,
                    "rarity_protected": i in protected,
                })
                d["sidecar"].write_text(json.dumps(sc, indent=2))
            except Exception:
                pass

    out = C.BUILD / "curation"
    out.mkdir(parents=True, exist_ok=True)
    keep_name = "keep_list.json" if args.domain == "edc" else f"keep_list_{args.domain}.json"
    report_name = "curation_report.json" if args.domain == "edc" else f"curation_report_{args.domain}.json"
    meta = {
        "domain": args.domain,
        "n_docs": n,
        "drop_frac": drop_frac,
        "drop_frac_source": drop_frac_src,
        "educational_status": ed_status,
        "pii_status": pii_status,
        "difficulty_role": "diagnostic",
        "difficulty_status": diff_status,
        "rarity_protected_count": len(protected),
        "rarity_token_count": len(rare),
        "quarantined": len(quarantined),
        "kept": len(keep),
        "near_dup_dropped": len(dup_of),
        "replay_near_dup_dropped": len(replay_dups),
        "replay_docs_loaded": len(replay_texts),
        "too_thin_dropped": len(too_thin),
        "bottom_dropped": len(dropped_low),
        "holdout": len(holdout),
    }
    (out / keep_name).write_text(json.dumps(sorted(keep), indent=2))
    (out / report_name).write_text(json.dumps({"meta": meta, "docs": report}, indent=2))
    # Also write a small meta sidecar for machine consumers.
    meta_name = "curation_meta.json" if args.domain == "edc" else f"curation_meta_{args.domain}.json"
    (out / meta_name).write_text(json.dumps(meta, indent=2) + "\n")

    if holdout:
        hostems = sorted(docs[i]["md"].stem for i in holdout)
        unit_parts = args.carve_holdout.split(":")[0].split("/")
        man = C.DATA / args.domain / Path(*unit_parts) / "holdout" / "manifest.json"
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({"stems": hostems, "carved_from": "10_clean (curated, pre-synth)"}, indent=2))

    cc = Counter(clusters[i] for i in range(n) if report[i]["status"] == "keep")
    print(
        f"kept={len(keep)}  near-dup-dropped={len(dup_of)}  replay-near-dup={len(replay_dups)}  "
        f"too-thin-dropped={len(too_thin)}  bottom-dropped={len(dropped_low)}  "
        f"quarantined={len(quarantined)}  rarity-protected={len(protected)}  holdout={len(holdout)}"
    )
    print(f"cluster sizes (kept): {dict(sorted(cc.items()))}")
    print(f"wrote {out/keep_name} + {report_name} + {meta_name}")
    print(json.dumps({"educational_status": ed_status, "pii_status": pii_status,
                      "rarity_protected_count": len(protected), "drop_frac": drop_frac}))


if __name__ == "__main__":
    main()
