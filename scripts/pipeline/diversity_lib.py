#!/usr/bin/env python3
"""Diversity metrics for the hydrated synthetic layer (docs/cpt-overhaul/03).

CPU-first primitives used by:
  - gate_diversity in qgre/gates.py (audit stage)
  - dedup_surfaces.py (SemDeDup-style surface cull)

Embeddings are optional (roberta-large via curate_lib.embed_docs). Distinct-n and
token-jaccard clustering always run; a device-only path records EMBEDDINGS_UNAVAILABLE
and still fails a collapsed corpus on distinct-2 alone.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable


_WORD_RE = re.compile(r"[a-z0-9']+", re.I)


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Ratio of unique n-grams to total n-grams across texts. 0 if empty."""
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        grams = ngrams(tokenize(text), n)
        total += len(grams)
        unique.update(grams)
    if total == 0:
        return 0.0
    return len(unique) / total


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cluster_by_jaccard(texts: list[str], threshold: float = 0.55) -> list[list[int]]:
    """Greedy single-link clusters on word-set Jaccard. Order-stable; O(n²) ok for per-fact surfaces."""
    sets = [set(tokenize(t)) for t in texts]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard(sets[i], sets[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        groups[find(i)].append(i)
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: kv[1][0])]


def effective_forms(texts: list[str], threshold: float = 0.55) -> int:
    """Cluster count = effective exposure forms for #760 floor math."""
    if not texts:
        return 0
    return len(cluster_by_jaccard(texts, threshold=threshold))


def nn_cosine_p90(emb) -> float | None:
    """p90 of each row's max off-diagonal cosine. None if <2 rows."""
    if emb is None or getattr(emb, "numel", lambda: 0)() == 0 or emb.shape[0] < 2:
        return None
    # emb: [N, D] L2-normalized
    sim = emb @ emb.T  # [N,N]
    # mask diagonal
    sim = sim.clone()
    sim.fill_diagonal_(-1.0)
    nn_vals, _ = sim.max(dim=1)
    vals = sorted(float(x) for x in nn_vals.tolist())
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(math.ceil(0.9 * len(vals)) - 1)))
    return vals[idx]


def fact_key(rec: dict) -> str:
    ids = rec.get("source_triple_ids") or rec.get("entity_ids") or ["?"]
    return str(ids[0])


def group_by_predicate(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        out[str(rec.get("relation_type") or "?")].append(rec)
    return dict(out)


def group_by_fact(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        out[fact_key(rec)].append(rec)
    return dict(out)


def diversity_report(
    records: list[dict],
    *,
    min_distinct_2: float = 0.35,
    max_nn_cosine_p90: float = 0.90,
    jaccard_cluster_threshold: float = 0.55,
    embeddings=None,
    embeddings_unavailable: bool = False,
) -> dict:
    """Per-predicate distinct-n + optional NN cosine; per-fact effective forms; overall PASS/FAIL.

    FAIL names weak predicates. Collapsed corpus (one skeleton × many rewordings) fails distinct-2.
    """
    by_pred = group_by_predicate(records)
    by_fact = group_by_fact(records)
    pred_metrics: dict[str, dict] = {}
    failed: list[str] = []
    examples: list[dict] = []

    # Optional global embedding index for NN (map text → row); when provided, slice per predicate.
    text_to_idx: dict[str, int] = {}
    if embeddings is not None:
        # Build from records order used to embed — caller must pass embeddings aligned to `records`.
        for i, rec in enumerate(records):
            text_to_idx[str(rec.get("text") or "")] = i

    for pred, rows in sorted(by_pred.items()):
        texts = [str(r.get("text") or "") for r in rows]
        d1 = distinct_n(texts, 1)
        d2 = distinct_n(texts, 2)
        d3 = distinct_n(texts, 3)
        forms = effective_forms(texts, threshold=jaccard_cluster_threshold)
        metric = {
            "n_surfaces": len(texts),
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "distinct_3": round(d3, 4),
            "effective_forms": forms,
            "effective_forms_frac": round(forms / len(texts), 4) if texts else 0.0,
        }
        nn_p90 = None
        if embeddings is not None and len(texts) >= 2:
            idxs = [text_to_idx[t] for t in texts if t in text_to_idx]
            if len(idxs) >= 2:
                sub = embeddings[idxs]
                nn_p90 = nn_cosine_p90(sub)
                if nn_p90 is not None:
                    metric["nn_cosine_p90"] = round(nn_p90, 4)

        # Gate: distinct-2 is the collapse detector (always on).
        fail_reasons = []
        if d2 < min_distinct_2:
            fail_reasons.append(f"distinct_2 {d2:.3f} < min {min_distinct_2}")
        if nn_p90 is not None and nn_p90 > max_nn_cosine_p90:
            fail_reasons.append(f"nn_cosine_p90 {nn_p90:.3f} > max {max_nn_cosine_p90}")
        metric["fail_reasons"] = fail_reasons
        pred_metrics[pred] = metric
        if fail_reasons:
            failed.append(pred)
            examples.append({"text": pred, "reason": "; ".join(fail_reasons)})

    fact_forms = {
        fk: effective_forms([str(r.get("text") or "") for r in rows], threshold=jaccard_cluster_threshold)
        for fk, rows in by_fact.items()
    }
    total_surfaces = len(records)
    total_forms = sum(fact_forms.values())
    passed = not failed
    return {
        "passed": passed,
        "severity": "pass" if passed else "fail",
        "failed_predicates": failed,
        "min_distinct_2": min_distinct_2,
        "max_nn_cosine_p90": max_nn_cosine_p90,
        "n_surfaces": total_surfaces,
        "n_facts": len(by_fact),
        "n_predicates": len(by_pred),
        "total_effective_forms": total_forms,
        "mean_forms_per_fact": round(total_forms / len(by_fact), 3) if by_fact else 0.0,
        "by_predicate": pred_metrics,
        "EMBEDDINGS_UNAVAILABLE": embeddings_unavailable,
        "examples": examples,
    }


def pick_diverse_representatives(
    texts: list[str],
    budget: int,
    *,
    jaccard_threshold: float = 0.55,
    rare_fact_threshold: float | None = None,
) -> list[int]:
    """Keep up to `budget` diverse surface indices. Floor protection: never return empty if texts non-empty.

    rare_fact_threshold: when set and len(texts) is small, use a lower similarity threshold so two
    similar rare-fact surfaces beat one.
    """
    if not texts:
        return []
    budget = max(1, min(budget, len(texts)))
    thr = jaccard_threshold
    if rare_fact_threshold is not None and len(texts) <= 4:
        thr = rare_fact_threshold
    clusters = cluster_by_jaccard(texts, threshold=thr)
    # Prefer one rep per cluster first (most central = longest unique token count as cheap diversity proxy).
    sets = [set(tokenize(t)) for t in texts]
    picked: list[int] = []
    for cluster in sorted(clusters, key=lambda c: -len(c)):
        if len(picked) >= budget:
            break
        # pick member with max unique tokens vs already picked
        best_i, best_score = cluster[0], -1.0
        for i in cluster:
            novelty = len(sets[i] - set().union(*(sets[j] for j in picked))) if picked else len(sets[i])
            if novelty > best_score:
                best_i, best_score = i, float(novelty)
        picked.append(best_i)
    # Fill remaining budget with farthest from picked
    while len(picked) < budget:
        best_i, best_score = None, -1.0
        for i in range(len(texts)):
            if i in picked:
                continue
            # distance = 1 - max jaccard to picked
            dist = min(1.0 - jaccard(sets[i], sets[j]) for j in picked)
            if dist > best_score:
                best_i, best_score = i, dist
        if best_i is None:
            break
        picked.append(best_i)
    return sorted(picked)


def collapsed_corpus(skeleton: str, n: int = 40) -> list[dict]:
    """Synthetic collapsed set: one skeleton × n trivial rewordings — must FAIL diversity gate."""
    variants = []
    for i in range(n):
        # trivial rewordings that share almost all bigrams
        text = f"{skeleton} ({i})." if i % 3 == 0 else f"{skeleton}."
        if i % 5 == 0:
            text = f"Indeed, {skeleton.lower()}"
        variants.append(
            {
                "text": text,
                "relation_type": "collapsed_rel",
                "source_triple_ids": ["fact:collapsed:0"],
                "surface_form": "llm_synth",
            }
        )
    return variants
