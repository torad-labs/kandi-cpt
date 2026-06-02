#!/usr/bin/env python3
"""last.fm festival lineups → artist + plays_at, for the residual editions MFW/official/Wikipedia lack.

The deep tail (old LA raves, discontinued international, pre-MFW editions) has no clean lineup source except
last.fm's festival pages — which carry full paginated lineups (`.big-artist-list-title`, 15/page). last.fm
deprecated festival SEARCH, so the per-edition URLs can't be resolved programmatically; they're harvested by
hand into configs/lastfm_urls.json. This walks each page until exhausted and emits, schema-validated, via
the same path as the other lineup extractors. Idempotent (dedups against existing artists).

    PYTHONPATH=scripts python scripts/extract_lastfm_lineup.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import time

from selectolax.parser import HTMLParser

import config as C
from pipeline.editions import iter_editions
from pipeline import structured_prompts as SP
from pipeline.schema_validate import validate
from extract_structured import _upsert_global, GLOBAL_DIR
from extract_lineups import _slug, _smart_title, B2B

MAP = C.ROOT / "configs" / "lastfm_urls.json"


def _get_retry(http, url: str, tries: int = 4):
    """last.fm throttles bursts with 406/429 — back off and retry rather than giving up the edition."""
    for attempt in range(tries):
        d = http.get(url)
        if d and d.status == 200:
            return d
        if d and d.status in (406, 429, 503):
            time.sleep(6 * (attempt + 1))
            continue
        return d
    return None


def walk_lineup(http, base_url: str, max_pages: int = 25) -> list[str]:
    """All artist names across the paginated /lineup view (stop when a page adds nothing new)."""
    names, seen = [], set()
    for page in range(1, max_pages + 1):
        d = _get_retry(http, f"{base_url.rstrip('/')}/lineup?page={page}")
        if not d or d.status != 200:
            break
        page_names = [n.text(strip=True) for n in
                      HTMLParser(d.content.decode("utf-8", "replace")).css(".big-artist-list-title")]
        fresh = [n for n in page_names if n and n not in seen]
        if not fresh:
            break
        for n in fresh:
            seen.add(n)
            names.append(n)
        time.sleep(1.5)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    urls = {k: v for k, v in json.loads(MAP.read_text()).items() if not k.startswith("_")}
    registry: dict[str, str] = {}
    rp = GLOBAL_DIR / "artists.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                registry.setdefault(r.get("slug", ""), r.get("name", ""))
    eds = {ed.slug: ed for ed in iter_editions("edc")}

    # last.fm rate-blocks the plain HttpFetcher (406); a real headed browser (run under xvfb) is not blocked
    from scrape.fetcher import BrowserFetcher
    http = BrowserFetcher(headless=False, ignore_robots=True)
    grand = 0
    try:
        for slug, url in urls.items():
            ed = eds.get(slug)
            if ed is None:
                print(f"  {slug}: not in registry, skipping", flush=True)
                continue
            raw = walk_lineup(http, url)
            # split B2B + clean
            names = {}
            for r in raw:
                for part in (p.strip() for p in B2B.split(r)):
                    s = _slug(part)
                    if s and len(part) <= 40 and s not in names:
                        names[s] = part
            # dedup against existing artists
            ent_path = ed.dir / "15_structured" / "entities.jsonl"
            tri_path = ed.dir / "15_structured" / "triples.jsonl"
            have_art, have_ids = set(), set()
            if ent_path.exists():
                for line in ent_path.read_text().splitlines():
                    if line.strip():
                        rr = json.loads(line)
                        have_ids.add(rr.get("id"))
                        if rr.get("entity_type") == "artist":
                            have_art.add(rr["id"].split(":")[-1])
            new = {s: n for s, n in names.items() if s not in have_art}
            print(f"  {slug:22s} last.fm={len(names):3d}  +{len(new):3d} new (had {len(have_art)})", flush=True)
            grand += len(new)
            if args.dry_run or not new:
                continue

            ed.dir.joinpath("15_structured").mkdir(parents=True, exist_ok=True)
            fest_id = f"{ed.festival_slug}:festival:edc_{ed.city}"
            meta = {"domain_class": "community-vetted", "url": url, "final_url": url, "fetched_at": ""}
            with ent_path.open("a") as ef, tri_path.open("a") as tf:
                if fest_id not in have_ids:
                    fe = SP.inject_provenance({"id": fest_id, "entity_type": "festival",
                        "name_canonical": f"Electric Daisy Carnival {ed.city.replace('_', ' ').title()}",
                        "attributes": {"year": str(ed.year)}}, "entity", meta, ed.festival_slug, {})
                    if not validate("entity", fe):
                        ef.write(json.dumps(fe) + "\n")
                for s, n in new.items():
                    aid = f"{ed.festival_slug}:artist:{s}"
                    ent = SP.inject_provenance({"id": aid, "entity_type": "artist",
                        "name_canonical": _smart_title(n, registry), "attributes": {}}, "entity", meta, ed.festival_slug, {})
                    tri = SP.inject_provenance({"subject_id": aid, "predicate": "plays_at", "object": fest_id,
                        "attrs": {}}, "triple", meta, ed.festival_slug, {})
                    if validate("entity", ent) or validate("triple", tri):
                        continue
                    ef.write(json.dumps(ent) + "\n")
                    tf.write(json.dumps(tri) + "\n")
                    _upsert_global(ent)
    finally:
        http.close()

    print(f"\n{'(dry-run) ' if args.dry_run else ''}last.fm lineups: +{grand} artist facts across {len(urls)} editions")


if __name__ == "__main__":
    main()
