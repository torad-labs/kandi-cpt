"""Curation primitives: RoBERTa embeddings, MinHash near-dup, quality + difficulty scoring, KMeans.

Llama-3-style data curation on the 10_clean source-of-truth: drop near-duplicates and the
bottom-quality tail, cluster for topic balance. Emits scores; the driver writes a keep-list
(never deletes 10_clean). Uses transformers `roberta-large` directly (no sentence-transformers →
no risk to the validated cu130 torch).
"""
from __future__ import annotations

import math
import re

import torch

# festival-salient tokens for the fact_signal quality term (mirrors eval.py's intent)
_STAGES = ("kineticfield", "cosmicmeadow", "circuitgrounds", "neongarden", "basspod",
           "wasteland", "quantumvalley", "stereobloom", "bionicjungle")
_FACT_RE = re.compile(r"\b(20\d{2}|may|june|set times?|lineup|headliner|stage|festival|edc|"
                      r"insomniac|camp|shuttle|gate|venue)\b", re.I)
_DATE_NUM_RE = re.compile(r"(\d{1,2}:\d{2}|\$\d+|\b\d{4}\b|\bday \d)", re.I)
_STOP = set("the a an and or of to in on for with at by is are was were be been it this that as "
            "from you your we our they their he she his her i".split())


# ---------- embeddings (roberta-large, mean-pooled) ----------

def embed_docs(texts: list[str], device: str = "cuda:0", batch: int = 16):
    """Mean-pooled roberta-large embeddings, L2-normalized. Returns a CPU float tensor [N,1024]."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModel.from_pretrained("roberta-large").to(device, dtype=torch.bfloat16).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state                      # [b,T,1024]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)    # mean over real tokens
            out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out) if out else torch.empty(0, 1024)


# ---------- near-duplicate (MinHash LSH) ----------

def near_dup_groups(texts: list[str], threshold: float = 0.85, num_perm: int = 128) -> list[list[int]]:
    """Group indices whose word-shingle Jaccard ≥ threshold (syndicated/near-identical docs)."""
    from datasketch import MinHash, MinHashLSH

    def shingles(t: str, k: int = 5):
        w = re.findall(r"\w+", t.lower())
        return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))} or {t.lower()[:50]}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh: list[MinHash] = []
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for s in shingles(t):
            m.update(s.encode())
        mh.append(m)
        lsh.insert(str(i), m)
    seen, groups = set(), []
    for i in range(len(texts)):
        if i in seen:
            continue
        dup = [int(j) for j in lsh.query(mh[i])]
        grp = sorted(set(dup) | {i})
        for j in grp:
            seen.add(j)
        if len(grp) > 1:
            groups.append(grp)
    return groups


# ---------- quality + difficulty ----------

def quality_score(text: str, trust: float) -> float:
    words = text.split()
    n = len(words)
    length_adequacy = 1 / (1 + math.exp(-(n - 150) / 80))            # sigmoid, plateau ≥~400w
    uniq = len({w.lower() for w in words if w.lower() not in _STOP})
    info_density = min(1.0, uniq / max(1, n) * 2.0)
    low = text.lower()
    fact = min(1.0, (len(_FACT_RE.findall(low)) + len(_DATE_NUM_RE.findall(low))
                     + sum(s in low for s in _STAGES)) / 12.0)
    return round(0.5 * trust + 0.2 * length_adequacy + 0.15 * info_density + 0.15 * fact, 4)


def perplexity(texts: list[str], model, tok, device: str = "cuda:0", max_len: int = 512) -> list[float]:
    """Mean per-token NLL of each doc under the base model (high = surprising = high-value)."""
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            if ids.numel() < 2:
                out.append(0.0); continue
            loss = model(input_ids=ids, labels=ids).loss
            out.append(float(loss))
    return out


def normalize(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.5] * len(xs) if hi <= lo else [(x - lo) / (hi - lo) for x in xs]


# ---------- clustering ----------

def cluster(embeddings, k: int = 20) -> list[int]:
    if len(embeddings) < k:
        return [0] * len(embeddings)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(embeddings.numpy()).tolist()
