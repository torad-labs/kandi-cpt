#!/usr/bin/env python3
"""Faithfulness-gated LLM lineup enumeration over EXISTING 10_clean docs — the no-new-scraping tail lever.

The deterministic parsers (extract_lineups / extract_official_lineup) cover structured lists (MFW blocks,
official tabs). The residual thin editions are old/discontinued ones whose only lineup signal is PROSE in
news/reddit docs already on disk — which the original structured pass under-recalled (its prompt asked for
"key facts" across entities+triples+discourse under a token cap). Here we do ONE focused job per edition:
ask the LLM to enumerate every act named as performing, then KEEP ONLY names that appear verbatim in the
source text (case-insensitive). The verbatim gate makes it faithful-by-construction — the LLM is a recall
mechanism, the substring check is the precision gate, so it cannot fabricate an artist not in the docs.

    PYTHONPATH=scripts python scripts/extract_lineup_llm.py --editions los_angeles/2009,… [--min-artists 15] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.editions import iter_editions
from pipeline import structured_prompts as SP
from pipeline.schema_validate import validate
from extract_structured import _upsert_global, _domain_of, GLOBAL_DIR
from extract_lineups import _slug, _smart_title

SYSTEM = (
    "You extract festival lineups. The text is about ONE specific Electric Daisy Carnival event. List EVERY "
    "music artist, DJ, or act named as performing/playing at it — pull names from any lineup paragraph, "
    "stage list, schedule, or set-times block in the text, not just the headliners. Return ONLY JSON: "
    '{"artists": ["Name", ...]}. Names exactly as written. Skip anyone named only for comparison or as a '
    "different event/year. If the text lists no performers, return an empty list. Invent nothing."
)
# prose channels that may carry lineup mentions, in trust order (structured lists handled deterministically)
LINEUP_CHANNELS = ("news", "reference", "youtube", "reddit")


def _llm(user: str) -> list[str]:
    import httpx
    import synth_corpus as S
    body = {"model": S.MODEL, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}], "temperature": 0.0, "max_tokens": 4000,
            "response_format": {"type": "json_object"}}
    try:
        with httpx.Client(timeout=180) as c:
            r = c.post(S.OPENROUTER_URL, json=body, headers=S.HEADERS)
        if r.status_code != 200:
            return []
        txt = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return []
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return []
    try:
        return [a for a in json.loads(m.group(0)).get("artists", []) if isinstance(a, str)]
    except json.JSONDecodeError:
        return []


_KW = ("lineup", "line-up", "line up", "b2b", "presents", "stage", "set time", "headlin", "perform")


def candidate_docs(ed, top=6):
    """The most lineup-dense prose docs, news/reference first — processed individually so a single lineup
    article isn't truncated or buried by reddit chatter in a giant concatenation."""
    clean = ed.dir / "10_clean"
    scored = []
    for rank, ch in enumerate(LINEUP_CHANNELS):
        d = clean / ch
        if not d.is_dir():
            continue
        for md in d.glob("*.md"):
            t = md.read_text(encoding="utf-8", errors="replace")
            density = sum(t.lower().count(k) for k in _KW)
            scored.append(((rank, -density), ch, md, t))
    scored.sort(key=lambda x: x[0])
    return [(ch, md, t) for _, ch, md, t in scored[:top]]


def _url_of(md):
    sc = md.with_suffix(".clean.json")
    if sc.exists():
        try:
            return json.loads(sc.read_text()).get("source_url", "")
        except Exception:
            pass
    return ""


def process(ed):
    kept = {}                                  # slug -> (name, domain_class, url)
    for ch, md, text in candidate_docs(ed):
        user = (f"Event: EDC {ed.city.replace('_', ' ').title()} {ed.year}.\n\n"
                f"Text:\n{text[:14000]}\n\nReturn the JSON now.")
        low = text.lower()
        dc, url = _domain_of(md, ch), _url_of(md)
        for name in _llm(user):
            name = name.strip()
            s = _slug(name)
            if s and len(name) <= 40 and name.lower() in low and s not in kept:   # verbatim faithfulness gate
                kept[s] = (name, dc, url)
    return ed.slug, [(s, n, dc, url) for s, (n, dc, url) in kept.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="comma 'city/year' (default: all thin <min-artists)")
    ap.add_argument("--min-artists", type=int, default=15, help="only process editions below this many artists")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    registry: dict[str, str] = {}
    rp = GLOBAL_DIR / "artists.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                registry.setdefault(r.get("slug", ""), r.get("name", ""))

    # select editions: explicit list, or all whose current artist-count < min
    eds = list(iter_editions("edc", only=args.editions))
    def n_artists(ed):
        tp = ed.dir / "15_structured" / "triples.jsonl"
        if not tp.exists():
            return 0
        return len({ln.split('"subject_id": "', 1)[1].split('"')[0]
                    for ln in tp.read_text().splitlines() if ":artist:" in ln and '"subject_id"' in ln})
    if not args.editions:
        eds = [ed for ed in eds if n_artists(ed) < args.min_artists]
    print(f"LLM lineup enumeration over {len(eds)} editions (faithfulness-gated)…", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(process, ed): ed for ed in eds}
        for fut in as_completed(futs):
            slug, recs = fut.result()
            results[slug] = recs

    grand = 0
    by_slug = {ed.slug: ed for ed in eds}
    for slug in sorted(results):
        recs = results[slug]
        ed = by_slug[slug]
        have_art, have_ids = set(), set()
        ent_path = ed.dir / "15_structured" / "entities.jsonl"
        tri_path = ed.dir / "15_structured" / "triples.jsonl"
        if ent_path.exists():
            for line in ent_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    have_ids.add(r.get("id"))
                    if r.get("entity_type") == "artist":
                        have_art.add(r["id"].split(":")[-1])
        new = [(s, n, dc, url) for s, n, dc, url in recs if s not in have_art]
        if not new:
            continue
        grand += len(new)
        print(f"  {slug:24s} +{len(new):3d} artists (had {len(have_art)})", flush=True)
        if args.dry_run:
            continue
        fest_id = f"{ed.festival_slug}:festival:edc_{ed.city}"
        with ent_path.open("a") as ef, tri_path.open("a") as tf:
            if fest_id not in have_ids:
                fe = SP.inject_provenance({"id": fest_id, "entity_type": "festival",
                    "name_canonical": f"Electric Daisy Carnival {ed.city.replace('_', ' ').title()}",
                    "attributes": {"year": str(ed.year)}}, "entity",
                    {"domain_class": "news-music-trade"}, ed.festival_slug, {})
                if not validate("entity", fe):
                    ef.write(json.dumps(fe) + "\n")
            for s, n, dc, url in new:
                aid = f"{ed.festival_slug}:artist:{s}"
                meta = {"domain_class": dc, "url": url, "final_url": url, "fetched_at": ""}
                ent = SP.inject_provenance({"id": aid, "entity_type": "artist",
                    "name_canonical": _smart_title(n, registry), "attributes": {}}, "entity", meta, ed.festival_slug, {})
                tri = SP.inject_provenance({"subject_id": aid, "predicate": "plays_at", "object": fest_id,
                    "attrs": {}}, "triple", meta, ed.festival_slug, {})
                if validate("entity", ent) or validate("triple", tri):
                    continue
                ef.write(json.dumps(ent) + "\n")
                tf.write(json.dumps(tri) + "\n")
                _upsert_global(ent)

    if not eds:
        sys.exit("no editions selected")
    print(f"\n{'(dry-run) ' if args.dry_run else ''}LLM lineup enumeration: +{grand} verbatim-gated artist facts")


if __name__ == "__main__":
    main()
