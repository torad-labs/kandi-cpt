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
        with httpx.Client(timeout=180) as c:                # context-managed → close conn (parallel-safe)
            r = c.post(S.OPENROUTER_URL, json=body, headers=S.HEADERS)
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


def _write_records(out, item, paths, stats):
    """Serial (main-thread) post-processing of one doc's LLM result: normalize → provenance → validate →
    write per-edition jsonl + upsert globals. No locks needed (single consumer thread)."""
    ent_f, tri_f, dis_f, rej_f = paths[item["slug"]]
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
                rec = SP.inject_provenance(rec, kind, item["meta"], item["festival_slug"], {})
                errs = validate(kind, rec)
                if errs:
                    rf.write(json.dumps({"kind": kind, "errs": errs[:3], "rec": rec}) + "\n")
                    stats[f"reject_{kind}"] += 1
                    continue
                fh.write(json.dumps(rec) + "\n")
                stats[f"ok_{kind}"] += 1
                if kind == "entity" and rec["entity_type"] in ("artist", "venue"):
                    _upsert_global(rec)


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", default="las_vegas/2026")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-tier", type=int, default=3, help="skip docs below this tier (1=best)")
    ap.add_argument("--concurrency", type=int, default=16, help="parallel OpenRouter calls (paid account)")
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

    # Build one flat work list across ALL editions (skipping ledger/keep/tier), so the thread pool stays
    # saturated end-to-end instead of draining between editions. The slow part — the LLM call — runs in
    # workers; all writes/upserts/ledger happen serially in the main thread (no locks, no global race).
    work: list[dict] = []
    ledgers: dict[str, list] = {}                    # slug -> [ledger_set, ledger_path]
    paths: dict[str, tuple] = {}                     # slug -> (ent_f, tri_f, dis_f, rej_f)
    for ed in eds:
        clean_root = ed.dir / "10_clean"
        if not clean_root.is_dir():
            continue
        struct = ed.dir / "15_structured"
        struct.mkdir(parents=True, exist_ok=True)
        ledger_path = struct / ".extracted.json"
        ledger = set(json.loads(ledger_path.read_text())) if ledger_path.exists() else set()
        ledgers[ed.slug] = [ledger, ledger_path]
        paths[ed.slug] = (struct / "entities.jsonl", struct / "triples.jsonl",
                          struct / "discourse.jsonl", REJECT_DIR / f"{ed.city}_{ed.year}.jsonl")
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
            text = md.read_text(encoding="utf-8", errors="replace")
            title = text.splitlines()[0].lstrip("# ").strip()
            meta = {"domain_class": _domain_of(md, sub), **_scrape_meta(md, sub)}
            work.append({"slug": ed.slug, "festival_slug": ed.festival_slug, "sha": sha,
                         "meta": meta, "user": SP.user_prompt(ed.festival_slug, title, body_of(text))})

    print(f"structured extraction: {len(work)} pending docs across {len(ledgers)} editions, "
          f"concurrency={args.concurrency} (skipped {stats['unchanged']} already-done)", flush=True)

    per_ed_total: Counter = Counter()
    per_ed_fail: Counter = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(llm_json, SP.SYSTEM, item["user"]): item for item in work}
            n = 0
            for fut in as_completed(futs):
                item = futs[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    print(f"  ! {type(e).__name__}: {e}", file=sys.stderr)
                    out = None
                per_ed_total[item["slug"]] += 1
                if not out:
                    stats["llm_fail"] += 1
                    per_ed_fail[item["slug"]] += 1
                else:
                    _write_records(out, item, paths, stats)
                    ledgers[item["slug"]][0].add(item["sha"])
                    stats["docs"] += 1
                n += 1
                if n % 25 == 0 or n == len(work):
                    print(f"  …{n}/{len(work)} docs  ok={stats['docs']} fail={stats['llm_fail']}", flush=True)
    finally:
        for ledger, ledger_path in ledgers.values():       # persist progress even if interrupted
            ledger_path.write_text(json.dumps(sorted(ledger)))

    print("\n=== structured extraction ===")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    # Fail loud on systematic LLM failure: a bad/expired key, no credit, or all-429 yields 0 docs but would
    # otherwise exit 0 and silently flow an empty corpus into hydrate/SFT/eval (looks like a clean success).
    if work and stats["docs"] == 0:
        sys.exit(f"\n✗ extraction produced 0 docs from {len(work)} pending — systematic LLM failure "
                 f"(llm_fail={stats['llm_fail']}). Check OPENROUTER_API_KEY / credit / rate limits.")
    if work and stats["llm_fail"] / len(work) > 0.25:
        print(f"  !! WARNING: {stats['llm_fail']}/{len(work)} docs failed LLM extraction "
              f"({stats['llm_fail'] / len(work):.0%}) — corpus likely incomplete", file=sys.stderr)
    # Per-edition guard: a single dead edition is invisible in the global rate but flows an EMPTY
    # 15_structured downstream (hydrate skips it silently). Surface it loudly.
    dead = sorted(s for s, t in per_ed_total.items() if t and per_ed_fail[s] == t)
    if dead:
        print(f"  !! WARNING: {len(dead)} edition(s) had 100% extraction failure → empty corpus "
              f"flows downstream: {dead[:8]}{' …' if len(dead) > 8 else ''}", file=sys.stderr)


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


_GLOBAL_SEEN: dict = {}   # kind -> set(slug), loaded once per process (avoids O(n²) re-read of the growing file)


def _upsert_global(rec):
    kind = rec["entity_type"] + "s"
    path = GLOBAL_DIR / f"{kind}.jsonl"
    name = rec.get("name_canonical", "")
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        return
    if kind not in _GLOBAL_SEEN:                         # first touch: read existing slugs once
        _GLOBAL_SEEN[kind] = {json.loads(l).get("slug") for l in path.read_text().splitlines()
                              if l.strip()} if path.exists() else set()
    if slug not in _GLOBAL_SEEN[kind]:
        with path.open("a") as f:
            f.write(json.dumps({"slug": slug, "name": name, "global_id": f"global:{rec['entity_type']}:{slug}"}) + "\n")
        _GLOBAL_SEEN[kind].add(slug)


if __name__ == "__main__":
    main()
