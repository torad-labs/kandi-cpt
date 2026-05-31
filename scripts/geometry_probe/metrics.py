"""The measurement layer (build spec Part 1b/1c/1d). THE METRIC IS TRUTH (Part 4.1) — every conclusion
is drawn from these full-hidden-dimensional numbers, never from a 3D plot.

Pairs are labeled against the triple store: RELATED (artist plays_at stage), NEGATED (explicit "not"),
UNRELATED (everything else among artist×stage). Reported per layer for base and CPT so the delta is
explicit. Higher separation/AUC/MRR after CPT = correct geometry installed; NEGATED→low cosine = the
negation lever (V6/V7) worked.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Full-dimensional pairwise cosine [n, n]."""
    n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return n @ n.T


def _artist_stage_pairs(artists: list[str], stages: list[str], a_idx: dict, s_idx: dict,
                        related: set, negated: set):
    """Yield (i, j, label) for every artist×stage pair. label ∈ {related, negated, unrelated}."""
    out = []
    for a in artists:
        for s in stages:
            lab = "related" if (a, s) in related else ("negated" if (a, s) in negated else "unrelated")
            out.append((a_idx[a], s_idx[s], lab))
    return out


def separation_and_auc(cos: np.ndarray, pairs) -> dict:
    """mean cosine per class + separation (related−unrelated) + AUC(related vs unrelated). Part 1c."""
    by = {"related": [], "unrelated": [], "negated": []}
    for i, j, lab in pairs:
        by[lab].append(cos[i, j])
    mean = {k: (float(np.mean(v)) if v else float("nan")) for k, v in by.items()}
    rel, unr = by["related"], by["unrelated"]
    auc = float("nan")
    if rel and unr:
        y = [1] * len(rel) + [0] * len(unr)
        auc = float(roc_auc_score(y, rel + unr))
    return {
        "mean_cos_related": mean["related"], "mean_cos_unrelated": mean["unrelated"],
        "mean_cos_negated": mean["negated"],
        "separation": mean["related"] - mean["unrelated"] if rel and unr else float("nan"),
        "auc_related_vs_unrelated": auc,
        "n_related": len(rel), "n_unrelated": len(unr), "n_negated": len(by["negated"]),
    }


def retrieval(cos: np.ndarray, artists: list[str], stages: list[str], a_idx: dict, s_idx: dict,
              related: set) -> dict:
    """Per-artist: rank stages by cosine; MRR + precision@1 of the true stage(s). Part 1c, rigorous form
    of 'closer to Kinetic Field than to Cosmic Meadow'."""
    rr, p1, n = [], 0, 0
    sj = [s_idx[s] for s in stages]
    for a in artists:
        true = {s for s in stages if (a, s) in related}
        if not true:
            continue
        n += 1
        scores = sorted(((cos[a_idx[a], j], s) for s, j in zip(stages, sj)), reverse=True)
        ranks = {s: r for r, (_, s) in enumerate(scores, 1)}
        best = min(ranks[s] for s in true)
        rr.append(1.0 / best)
        p1 += int(scores[0][1] in true)
    return {"mrr": float(np.mean(rr)) if rr else float("nan"),
            "precision_at_1": p1 / n if n else float("nan"), "n_queries": n}


def linear_probe(X: np.ndarray, labels: list[str], seed: int = 0) -> dict:
    """Linear extractability (Part 1d / Allen-Zhu): can a linear classifier read `labels` off the hidden
    states? Stratified CV over ENTITIES (each entity is one sample → no entity leaks across the split,
    Part 4.5). Reports mean test accuracy + a majority-class baseline."""
    y = np.array(labels)
    classes, counts = np.unique(y, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    if len(classes) < 2:
        return {"probe_acc": float("nan"), "baseline": baseline, "n": len(y)}
    k = int(min(5, counts.min()))
    if k < 2:
        return {"probe_acc": float("nan"), "baseline": baseline, "n": len(y), "note": "a class has <2 samples"}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return {"probe_acc": float(np.mean(accs)), "probe_acc_std": float(np.std(accs)),
            "baseline": baseline, "n": len(y), "folds": k}


def build_relation_sets(entities_meta: list[dict], triples_path: str):
    """From the triple store → (artists, stages, related set, negated set). related/negated are sets of
    (artist_name, stage_slug). NEGATED comes from triples with predicate 'not_plays_at' (V6+; empty now)."""
    import json
    ents = {}
    for m in entities_meta:
        ents[m["id"]] = m
    related, negated = set(), set()
    for line in open(triples_path):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        subj = ents.get(t["subject_id"], {}).get("name_canonical")
        obj = str(t.get("object", ""))
        stage = obj.rsplit(":", 1)[-1] if ":" in obj else obj
        if not subj:
            continue
        if t["predicate"] == "plays_at":
            related.add((subj, stage))
        elif t["predicate"] in ("not_plays_at", "does_not_play_at"):
            negated.add((subj, stage))
    return related, negated
