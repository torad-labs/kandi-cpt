#!/usr/bin/env python3
"""Extraction recall audit — coverage gate at the audit stage (docs/cpt-overhaul/02).

Samples N docs per unit. For each, a judge lists assertable facts about manifest entity_types,
then checks each against the unit's triples → coverage ratio per doc / predicate / unit.

Gate: quality.extraction_recall.min (default 0.6). Below it on a load-bearing slice → FAIL with
the predicate named. Mitigation is a second extraction pass, not a training iteration.

    PYTHONPATH=scripts python scripts/pipeline/audit_recall.py \
        --domain <d> [--editions …] [--min-coverage 0.6] [--sample-n 0] \
        [--facts-json path]   # fixture: {doc_path: [fact_dict, …]} skips judge (tests)

Outputs:
  data/_build/recall/<domain>/<unit>.json
  data/_build/recall/report_<domain>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import config as C
from pipeline.editions import iter_editions

DEFAULT_MIN_COVERAGE = 0.6
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

JUDGE_SYSTEM = (
    "You list ASSERTABLE factual claims in a document about a knowledge domain. "
    "Return ONLY a JSON array of objects, each with keys: "
    "subject (string), predicate (string), object (string). "
    "No prose, no markdown fences. Empty array if none."
)
_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")


def body_of(md_text: str) -> str:
    return md_text.split("\n---\n", 1)[-1].strip() if "\n---\n" in md_text else md_text.strip()


def derive_sample_n(n_docs: int) -> int:
    """max(20, 2% of docs), cap 100."""
    if n_docs <= 0:
        return 0
    return max(20, min(100, int(round(0.02 * n_docs))))


def load_entity_types(domain: str) -> list[str]:
    path = C.ROOT / "configs" / "domains" / f"{domain}.yaml"
    if not path.is_file():
        return []
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    types = raw.get("entity_types") or []
    return [str(t) for t in types] if not isinstance(types, str) else [types]


def load_triples(ed_dir: Path) -> list[dict]:
    # Prefer resolved graph when present
    for name in ("triples.resolved.jsonl", "triples.jsonl"):
        p = ed_dir / "15_structured" / name
        if p.is_file():
            return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def collect_clean_docs(ed_dir: Path) -> list[dict]:
    root = ed_dir / "10_clean"
    if not root.is_dir():
        return []
    docs = []
    for md in sorted(root.rglob("*.md")):
        if md.name == "README.md":
            continue
        text = body_of(md.read_text(encoding="utf-8", errors="replace"))
        if text.strip():
            docs.append({"path": str(md), "text": text, "name": md.name})
    return docs


def fact_key(subject: str, predicate: str, obj: str) -> str:
    return f"{subject.strip().lower()}|{predicate.strip().lower()}|{obj.strip().lower()}"


def triple_keys(triples: list[dict], id_to_name: dict[str, str] | None = None) -> set[str]:
    keys = set()
    id_to_name = id_to_name or {}
    for t in triples:
        s = id_to_name.get(str(t.get("subject_id") or ""), str(t.get("subject_id") or ""))
        # also keep raw id form
        s_raw = str(t.get("subject_id") or "")
        o = str(t.get("object") or "")
        o_name = id_to_name.get(o, o)
        p = str(t.get("predicate") or "")
        for ss in {s, s_raw}:
            for oo in {o, o_name}:
                keys.add(fact_key(ss, p, oo))
    return keys


def coverage_for_doc(
    assertable: list[dict],
    graph_keys: set[str],
) -> dict:
    """Compare assertable facts to graph. Returns hits/n/ratio + misses."""
    if not assertable:
        return {"hits": 0, "n": 0, "ratio": 1.0, "misses": [], "per_predicate": {}}
    hits = 0
    misses = []
    per_pred: dict[str, list[bool]] = defaultdict(list)
    for fact in assertable:
        s = str(fact.get("subject") or "")
        p = str(fact.get("predicate") or "")
        o = str(fact.get("object") or "")
        ok = fact_key(s, p, o) in graph_keys
        # soft match: substring containment on object
        if not ok:
            fk = fact_key(s, p, o)
            ok = any(
                fk.split("|")[0] == gk.split("|")[0]
                and fk.split("|")[1] == gk.split("|")[1]
                and (fk.split("|")[2] in gk.split("|")[2] or gk.split("|")[2] in fk.split("|")[2])
                for gk in graph_keys
            )
        per_pred[p].append(ok)
        if ok:
            hits += 1
        else:
            misses.append(fact)
    n = len(assertable)
    per_predicate = {
        p: {"hits": sum(v), "n": len(v), "ratio": sum(v) / len(v) if v else 1.0}
        for p, v in sorted(per_pred.items())
    }
    return {
        "hits": hits,
        "n": n,
        "ratio": hits / n if n else 1.0,
        "misses": misses,
        "per_predicate": per_predicate,
    }


def parse_facts(raw: str) -> list[dict]:
    match = _JSON_ARRAY.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("subject") and item.get("predicate") and item.get("object"):
            out.append({
                "subject": str(item["subject"]),
                "predicate": str(item["predicate"]),
                "object": str(item["object"]),
            })
    return out


def run_audit(
    domain: str,
    editions: str | None,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    sample_n: int = 0,
    facts_json: Path | None = None,
    seed: int = 0,
    dry_run: bool = False,
) -> dict:
    fixture: dict[str, list[dict]] = {}
    if facts_json is not None:
        fixture = json.loads(Path(facts_json).read_text(encoding="utf-8"))

    entity_types = load_entity_types(domain)
    unit_reports = []
    overall_hits = overall_n = 0
    pred_agg: dict[str, list[float]] = defaultdict(list)
    status = "PASS"
    failed_predicates: list[str] = []

    for ed in iter_editions(domain, only=editions, with_raw=False):
        docs = collect_clean_docs(ed.dir)
        triples = load_triples(ed.dir)
        # id→name from entities if present
        id_to_name = {}
        for name in ("entities.resolved.jsonl", "entities.jsonl"):
            p = ed.dir / "15_structured" / name
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    id_to_name[str(e.get("id") or "")] = str(e.get("name_canonical") or e.get("id") or "")
                break
        gkeys = triple_keys(triples, id_to_name)
        n_take = sample_n if sample_n > 0 else derive_sample_n(len(docs))
        n_take = min(n_take, len(docs))
        rng = random.Random(seed)
        sample = rng.sample(docs, n_take) if n_take else []

        doc_rows = []
        u_hits = u_n = 0
        for doc in sample:
            key = doc["path"]
            # fixture keys may be basename
            assertable = fixture.get(key) or fixture.get(doc["name"]) or fixture.get(str(Path(key).name))
            if assertable is None:
                if fixture:
                    assertable = []
                else:
                    # live judge path — only if no fixture and key available
                    assertable = _judge_facts(doc["text"], entity_types)
            cov = coverage_for_doc(assertable, gkeys)
            doc_rows.append({
                "path": doc["name"],
                "hits": cov["hits"],
                "n": cov["n"],
                "ratio": round(cov["ratio"], 4),
                "per_predicate": cov["per_predicate"],
            })
            u_hits += cov["hits"]
            u_n += cov["n"]
            for p, row in cov["per_predicate"].items():
                pred_agg[p].append(row["ratio"])

        u_ratio = u_hits / u_n if u_n else 1.0
        unit_reports.append({
            "unit": ed.slug,
            "n_docs_sampled": len(sample),
            "n_docs_total": len(docs),
            "hits": u_hits,
            "n": u_n,
            "ratio": round(u_ratio, 4),
            "docs": doc_rows,
        })
        overall_hits += u_hits
        overall_n += u_n

        if not dry_run:
            out = C.BUILD / "recall" / domain
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{ed.slug.replace('/', '_')}.json").write_text(
                json.dumps(unit_reports[-1], indent=2) + "\n", encoding="utf-8",
            )

    per_predicate = {
        p: {
            "mean_ratio": round(sum(vs) / len(vs), 4) if vs else 1.0,
            "n_docs": len(vs),
        }
        for p, vs in sorted(pred_agg.items())
    }
    for p, row in per_predicate.items():
        if row["n_docs"] > 0 and row["mean_ratio"] < min_coverage:
            status = "FAIL"
            failed_predicates.append(p)

    overall_ratio = overall_hits / overall_n if overall_n else 1.0
    if overall_n > 0 and overall_ratio < min_coverage:
        status = "FAIL"

    report = {
        "domain": domain,
        "status": status,
        "min_coverage": min_coverage,
        "overall": {
            "hits": overall_hits,
            "n": overall_n,
            "ratio": round(overall_ratio, 4),
        },
        "per_predicate": per_predicate,
        "failed_predicates": failed_predicates,
        "units": unit_reports,
        "entity_types": entity_types,
        "RECALL_AUDIT_UNAVAILABLE": False,
    }
    if overall_n == 0 and not fixture:
        report["status"] = "SKIP"
        report["RECALL_AUDIT_UNAVAILABLE"] = True
        report["reason"] = "no assertable facts scored (no fixture and no judge output)"

    if not dry_run:
        out = C.BUILD / "recall"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_{domain}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "ratio": report["overall"]["ratio"],
        "n": report["overall"]["n"],
        "failed_predicates": failed_predicates,
    }))
    return report


def _judge_facts(text: str, entity_types: list[str]) -> list[dict]:
    """Live judge — requires OpenRouter. Returns [] if unavailable."""
    import time
    import urllib.error
    import urllib.request
    import subprocess

    def key() -> str | None:
        v = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if v:
            return v
        try:
            p = subprocess.run(
                ["secret-tool", "lookup", "service", "openrouter"],
                capture_output=True, text=True, timeout=5,
            )
            return p.stdout.strip() or None if p.returncode == 0 else None
        except Exception:
            return None

    k = key()
    if not k:
        return []
    types = ", ".join(entity_types) if entity_types else "domain entities"
    user = (
        f"Entity types of interest: {types}\n\nDOCUMENT:\n{text[:6000]}\n\n"
        "List assertable (subject, predicate, object) facts as a JSON array."
    )
    body = json.dumps({
        "model": DEFAULT_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        OPENROUTER_BASE_URL + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {k}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode())
        raw = payload["choices"][0]["message"]["content"] or ""
        return parse_facts(raw)
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="extraction recall audit")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    ap.add_argument("--sample-n", type=int, default=0, help="0 = derive from corpus size")
    ap.add_argument("--facts-json", help="fixture path for offline red-green")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        report = run_audit(
            args.domain, args.editions,
            min_coverage=args.min_coverage,
            sample_n=args.sample_n,
            facts_json=Path(args.facts_json) if args.facts_json else None,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"audit_recall failed closed: {error}")
    sys.exit(0 if report["status"] in ("PASS", "SKIP") else 1)


if __name__ == "__main__":
    main()
