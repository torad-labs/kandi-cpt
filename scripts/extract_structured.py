#!/usr/bin/env python3
"""LLM structured extraction: curated 10_clean → 15_structured/{entities,triples,discourse}.jsonl.

One LLM call per kept doc returns semantic content; the pipeline injects provenance and validates
EVERY record against the JSON Schemas before writing (rejects → data/_build/structured_rejects/).
Per-edition JSONL + a .extracted.json ledger (idempotent). Artists/venues also upsert into
data/_shared/entities_global/. Defaults to las_vegas/2026 (cost discipline).

    PYTHONPATH=scripts python scripts/extract_structured.py [--editions …] [--limit N]
        [--min-tier 3] [--yes]   # --yes required for >1 edition (OpenRouter cost)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

import config as C
from pipeline.editions import iter_editions
from pipeline import provenance as P
from pipeline import structured_prompts as SP
from pipeline.schema_validate import validate

REJECT_DIR = C.BUILD / "structured_rejects"
GLOBAL_DIR = C.SHARED / "entities_global"


def llm_json(system: str, user: str) -> dict | None:
    import httpx
    import synth_corpus as S
    body = {"model": S.MODEL, "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}],
            "temperature": 0.2, "max_tokens": 4000, "response_format": {"type": "json_object"}}
    try:
        r = httpx.Client(timeout=180).post(S.OPENROUTER_URL, json=body, headers=S.HEADERS)
        if r.status_code != 200:
            print(f"  ! HTTP {r.status_code}: {r.text[:160]}", file=sys.stderr)
            return None
        txt = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ! {type(e).__name__}: {e}", file=sys.stderr)
        return None
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def body_of(md: str) -> str:
    return md.split("\n---\n", 1)[-1].strip() if "\n---\n" in md else md.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", default="las_vegas/2026")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-tier", type=int, default=3, help="skip docs below this tier (1=best)")
    ap.add_argument("--yes", action="store_true", help="confirm multi-edition (OpenRouter cost)")
    args = ap.parse_args()

    eds = list(iter_editions("edc", only=args.editions))
    if len(eds) > 1 and not args.yes:
        sys.exit(f"{len(eds)} editions × LLM calls = real cost. Re-run with --yes, or --editions las_vegas/2026.")

    keep = set(json.loads((C.BUILD / "curation" / "keep_list.json").read_text())) \
        if (C.BUILD / "curation" / "keep_list.json").exists() else None
    REJECT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()

    for ed in eds:
        clean_root = ed.dir / "10_clean"
        if not clean_root.is_dir():
            continue
        struct = ed.dir / "15_structured"
        struct.mkdir(parents=True, exist_ok=True)
        ledger_path = struct / ".extracted.json"
        ledger = set(json.loads(ledger_path.read_text())) if ledger_path.exists() else set()
        ent_f, tri_f, dis_f = (struct / "entities.jsonl", struct / "triples.jsonl", struct / "discourse.jsonl")
        rej_f = REJECT_DIR / f"{ed.city}_{ed.year}.jsonl"

        docs = sorted(clean_root.rglob("*.md"))
        if args.limit:
            docs = docs[: args.limit]
        for md in docs:
            if md.name == "README.md":
                continue
            rel = str(md.relative_to(C.ROOT))
            if keep is not None and rel not in keep:
                continue
            sub = md.relative_to(clean_root).parts[0]
            tier, _ = P.tier_for(_domain_of(md, sub))
            if tier > args.min_tier:
                continue
            sha = _sha(md)
            if sha in ledger:
                stats["unchanged"] += 1
                continue

            title = md.read_text(encoding="utf-8", errors="replace").splitlines()[0].lstrip("# ").strip()
            meta = {"domain_class": _domain_of(md, sub), **_scrape_meta(md, sub)}
            out = llm_json(SP.SYSTEM, SP.user_prompt(ed.festival_slug, title, body_of(md.read_text())))
            if not out:
                stats["llm_fail"] += 1
                continue

            with ent_f.open("a") as ef, tri_f.open("a") as tf, dis_f.open("a") as df, rej_f.open("a") as rf:
                for kind, fh, recs in (("entity", ef, out.get("entities", [])),
                                       ("triple", tf, out.get("triples", [])),
                                       ("discourse", df, out.get("discourse", []))):
                    for rec in (recs or []):
                        if not isinstance(rec, dict):
                            continue
                        # normalize common model slips before validating (recover avoidable rejects)
                        if kind == "entity":
                            if "name" in rec and "name_canonical" not in rec:
                                rec["name_canonical"] = rec.pop("name")
                            rec.setdefault("attributes", {})
                        if kind == "discourse" and not rec.get("anchors"):
                            continue                       # empty-anchor discourse is unusable; skip silently
                        rec = SP.inject_provenance(rec, kind, meta, ed.festival_slug, {})
                        errs = validate(kind, rec)
                        if errs:
                            rf.write(json.dumps({"kind": kind, "errs": errs[:3], "rec": rec}) + "\n")
                            stats[f"reject_{kind}"] += 1
                            continue
                        fh.write(json.dumps(rec) + "\n")
                        stats[f"ok_{kind}"] += 1
                        if kind == "entity" and rec["entity_type"] in ("artist", "venue"):
                            _upsert_global(rec)
            ledger.add(sha)
            stats["docs"] += 1
        ledger_path.write_text(json.dumps(sorted(ledger)))

    print("\n=== structured extraction ===")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))


# ---- small helpers (sidecar/sha/global upsert) ----

def _sha(md):
    import hashlib
    return hashlib.sha256(md.read_bytes()).hexdigest()


def _sidecar(md):
    sc = md.with_suffix(".clean.json")
    return json.loads(sc.read_text()) if sc.exists() else {}


def _domain_of(md, sub):
    return _sidecar(md).get("domain_class") or {"official": "official-festival", "reference": "news-music-trade",
            "youtube": "social-vetted", "news": "news-music-trade", "reddit": "community-discussion"}.get(sub, "user-contributed")


def _scrape_meta(md, sub):
    sc = _sidecar(md)
    return {"url": sc.get("source_url", ""), "final_url": sc.get("source_url", ""), "fetched_at": ""}


def _upsert_global(rec):
    kind = rec["entity_type"] + "s"
    path = GLOBAL_DIR / f"{kind}.jsonl"
    name = rec.get("name_canonical", "")
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        return
    seen = set()
    if path.exists():
        seen = {json.loads(l).get("slug") for l in path.read_text().splitlines() if l.strip()}
    if slug not in seen:
        with path.open("a") as f:
            f.write(json.dumps({"slug": slug, "name": name, "global_id": f"global:{rec['entity_type']}:{slug}"}) + "\n")


if __name__ == "__main__":
    main()
