#!/usr/bin/env python3
"""Three-tier knowledge routing — hot / stable / proven (docs/trust-loop/04-three-tier-routing.md).

Per-predicate (or predicate×unit) tier from the domain manifest. Default when undeclared: hot
(misroute-to-hot is cheap; misroute-to-weights is a governance incident).

  hot    — NEVER reaches hydrate/tokenize train shards; lives in the retrieval index over 10_clean
  stable — batch CPT + boosters (durable facts + ALL style/voice)
  proven — digest-approved boosters only (usage-evidenced)

    PYTHONPATH=scripts python scripts/pipeline/tier_routing.py \
        --domain pokemon-lore [--editions kanto] [--build-hot-index] [--audit-shard]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as C
from pipeline.domain import Domain
from pipeline.editions import iter_editions

TIERS = frozenset({"hot", "stable", "proven"})
DEFAULT_TIER = "hot"
STYLE_KINDS = frozenset({"style", "voice"})


class TierError(ValueError):
    """Tier routing refused."""


def load_predicate_tiers(domain: str) -> dict[str, str]:
    """predicate → tier. Undeclared predicates are absent (caller defaults to hot)."""
    spec = Domain(domain).spec
    preds = spec.get("predicates") or {}
    out: dict[str, str] = {}
    if not isinstance(preds, dict):
        return out
    for name, meta in preds.items():
        if not isinstance(meta, dict):
            continue
        tier = meta.get("tier")
        if tier is None:
            continue
        tier = str(tier).strip().lower()
        if tier not in TIERS:
            raise TierError(f"predicate {name!r} has illegal tier {tier!r} (want {sorted(TIERS)})")
        out[str(name)] = tier
    # Domain-level override block (optional): tiers: {pred: tier}
    block = spec.get("tiers") or {}
    if isinstance(block, dict):
        for name, tier in block.items():
            t = str(tier).strip().lower()
            if t not in TIERS:
                raise TierError(f"tiers block: {name!r} → illegal {t!r}")
            out[str(name)] = t
    return out


def tier_for(predicate: str, tiers: dict[str, str], *, default: str = DEFAULT_TIER) -> str:
    return tiers.get(str(predicate), default)


def assert_style_all_stable(domain: str) -> None:
    """Style/voice domains must be uniform-stable — a hot style predicate fails closed."""
    spec = Domain(domain).spec
    kind = str(spec.get("kind") or (spec.get("ontology") or {}).get("kind") or "").lower()
    if kind not in STYLE_KINDS and "style" not in kind and "voice" not in kind:
        return
    tiers = load_predicate_tiers(domain)
    preds = spec.get("predicates") or {}
    names = list(preds.keys()) if isinstance(preds, dict) else list(preds or [])
    bad = []
    for name in names:
        t = tier_for(str(name), tiers, default="stable")  # style default stable if undeclared
        if t != "stable":
            bad.append((str(name), t))
    # Also: any explicit non-stable is illegal
    for name, t in tiers.items():
        if t != "stable":
            bad.append((name, t))
    if bad:
        raise TierError(
            f"style domain {domain!r} must be uniform-stable; non-stable predicates: {bad}"
        )


def filter_triples_by_tier(
    triples: list[dict],
    tiers: dict[str, str],
    *,
    allow: frozenset[str] | set[str] = frozenset({"stable", "proven"}),
    default: str = DEFAULT_TIER,
) -> tuple[list[dict], dict[str, int]]:
    """Keep triples whose predicate tier is in `allow`. Returns (kept, drop_counts_by_tier)."""
    kept = []
    drops: dict[str, int] = {t: 0 for t in TIERS}
    for t in triples:
        pred = str(t.get("predicate") or "")
        tier = tier_for(pred, tiers, default=default)
        if tier in allow:
            kept.append(t)
        else:
            drops[tier] = drops.get(tier, 0) + 1
    return kept, drops


def derivation_rows(domain: str) -> list[dict[str, Any]]:
    """Planner-facing table: each predicate's tier + reason."""
    spec = Domain(domain).spec
    preds = spec.get("predicates") or {}
    tiers = load_predicate_tiers(domain)
    rows = []
    names = list(preds.keys()) if isinstance(preds, dict) else list(preds or [])
    for name in names:
        declared = name in tiers
        tier = tier_for(str(name), tiers)
        if declared:
            reason = f"manifest tier={tier}"
        else:
            reason = f"undeclared → default {DEFAULT_TIER} (uncertainty law: hot is cheap)"
        meta = preds.get(name) if isinstance(preds, dict) else {}
        rows.append({
            "predicate": str(name),
            "tier": tier,
            "declared": declared,
            "reason": reason,
            "object_types": list((meta or {}).get("object") or []) if isinstance(meta, dict) else [],
        })
    return rows


def build_hot_index(domain: str, editions: str | None = None) -> dict[str, Any]:
    """Governed retrieval index over unit 10_clean — same admission surface as training data."""
    docs = []
    for ed in iter_editions(domain, only=editions, with_raw=False):
        root = ed.dir / "10_clean"
        if not root.is_dir():
            continue
        for md in sorted(root.rglob("*.md")):
            if md.name == "README.md":
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
            body = text.split("\n---\n", 1)[-1] if "\n---\n" in text else text
            body = body.strip()
            if not body:
                continue
            rel = str(md.relative_to(ed.dir))
            docs.append({
                "unit": ed.slug,
                "path": rel,
                "doc_id": md.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(md.read_bytes()).hexdigest(),
                "title": text.splitlines()[0].lstrip("# ").strip() if text.splitlines() else md.name,
                "n_chars": len(body),
                # lightweight bag for keyword retrieval (no external embedder required)
                "terms": sorted(set(
                    w.lower() for w in body.replace("\n", " ").split()
                    if len(w) >= 3
                ))[:400],
            })
    index = {
        "schema": "kandi.hot_index.v1",
        "domain": domain,
        "tier": "hot",
        "n_docs": len(docs),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "docs": docs,
        "note": (
            "Retrieval lane over 10_clean. Same PII/admission surface as train data. "
            "Hot answers must cite path+sha; never merged into install-strength estimands."
        ),
    }
    out = C.BUILD / "hot_index" / domain / "index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    index["path"] = str(out)
    return index


def retrieve_hot(domain: str, query: str, *, k: int = 5) -> list[dict[str, Any]]:
    """Keyword overlap retrieval over the hot index. Returns ranked docs with lane label."""
    path = C.BUILD / "hot_index" / domain / "index.json"
    if not path.is_file():
        raise TierError(f"hot index missing at {path} — run tier_routing --build-hot-index")
    index = json.loads(path.read_text(encoding="utf-8"))
    qterms = {w.lower() for w in query.split() if len(w) >= 3}
    scored = []
    for doc in index.get("docs") or []:
        terms = set(doc.get("terms") or [])
        overlap = len(qterms & terms)
        if overlap <= 0:
            continue
        scored.append({
            "lane": "hot",
            "score": overlap,
            "unit": doc.get("unit"),
            "path": doc.get("path"),
            "doc_id": doc.get("doc_id"),
            "sha256": doc.get("sha256"),
            "title": doc.get("title"),
            "provenance": {
                "source_path": doc.get("path"),
                "sha256": doc.get("sha256"),
                "index_built_at": index.get("built_at"),
            },
        })
    scored.sort(key=lambda r: (-r["score"], r.get("path") or ""))
    return scored[:k]


def audit_shard_has_no_hot(
    domain: str,
    editions: str | None = None,
) -> dict[str, Any]:
    """Confirm hot-tier triples are absent from 25_hydrated train surfaces (and resolved train triples optional)."""
    tiers = load_predicate_tiers(domain)
    hot_preds = {p for p, t in tiers.items() if t == "hot"}
    # Also treat undeclared as hot only if we want full default — for audit of DECLARED hot
    violations = []
    n_surfaces = 0
    for ed in iter_editions(domain, only=editions, with_raw=False):
        hyd = ed.dir / "25_hydrated"
        if not hyd.is_dir():
            continue
        for jf in hyd.glob("*.jsonl"):
            for line in jf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                n_surfaces += 1
                rec = json.loads(line)
                pred = str(rec.get("predicate") or rec.get("relation_type") or "")
                if pred in hot_preds:
                    violations.append({
                        "unit": ed.slug,
                        "file": jf.name,
                        "predicate": pred,
                        "subject": rec.get("subject") or rec.get("subject_id"),
                    })
    return {
        "domain": domain,
        "hot_predicates_declared": sorted(hot_preds),
        "n_surfaces_scanned": n_surfaces,
        "n_violations": len(violations),
        "violations_sample": violations[:20],
        "pass": len(violations) == 0,
    }


def adaptive_probe_budget(
    per_predicate_n: dict[str, int],
    *,
    total_budget: int,
    wilson_widths: dict[str, float] | None = None,
    booster_touched: set[str] | None = None,
) -> dict[str, int]:
    """Widest-interval-first allocation across predicates. Fixed total budget."""
    if total_budget <= 0:
        return {p: 0 for p in per_predicate_n}
    preds = sorted(per_predicate_n)
    if not preds:
        return {}
    wilson_widths = wilson_widths or {}
    booster_touched = booster_touched or set()
    # score: width + bonus for booster-touched
    scores = {
        p: float(wilson_widths.get(p, 1.0)) + (0.5 if p in booster_touched else 0.0)
        for p in preds
    }
    total_score = sum(scores.values()) or float(len(preds))
    alloc = {p: 0 for p in preds}
    # floor 1 for any pred that had mass, then distribute remainder by score
    remaining = total_budget
    for p in preds:
        if remaining <= 0:
            break
        alloc[p] = 1
        remaining -= 1
    if remaining > 0 and total_score > 0:
        extras = sorted(preds, key=lambda p: -scores[p])
        i = 0
        while remaining > 0:
            alloc[extras[i % len(extras)]] += 1
            remaining -= 1
            i += 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser(description="three-tier routing")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--build-hot-index", action="store_true")
    ap.add_argument("--audit-shard", action="store_true")
    ap.add_argument("--derive", action="store_true", help="print derivation table JSON")
    ap.add_argument("--check-style", action="store_true")
    args = ap.parse_args()
    try:
        if args.check_style:
            assert_style_all_stable(args.domain)
            print(json.dumps({"domain": args.domain, "style_uniform_stable": True}))
        if args.derive or not any([args.build_hot_index, args.audit_shard, args.check_style]):
            rows = derivation_rows(args.domain)
            print(json.dumps({"domain": args.domain, "derivations": rows}, indent=2))
        if args.build_hot_index:
            idx = build_hot_index(args.domain, args.editions)
            print(json.dumps({"wrote": idx.get("path"), "n_docs": idx["n_docs"]}))
        if args.audit_shard:
            rep = audit_shard_has_no_hot(args.domain, args.editions)
            print(json.dumps(rep, indent=2))
            sys.exit(0 if rep["pass"] else 1)
    except TierError as error:
        sys.exit(f"tier_routing failed closed: {error}")


if __name__ == "__main__":
    main()
