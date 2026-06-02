#!/usr/bin/env python3
"""Official EDC Las Vegas past-highlights archive → artist + STAGE + day, at official-festival confidence.

lasvegas.electricdaisycarnival.com/past-highlights/<year>-lineup/ is the authoritative source the user
pointed at. Each page carries three tabbed views of the same lineup: Alphabetical (full), By Day
(Fri/Sat/Sun), By Stage (Kinetic Field, Circuit Grounds, Neon Garden, Basspod, Wasteland, Quantum Valley,
Stereo Bloom, art cars, …). We parse all three → for every act we learn its stage AND its day, emitting
stage entities + plays_at(artist→stage, attrs={day}) — the deepest completeness win and the one source that
fills the *stages* gap. Trafilatura would flatten the tabs, so we parse the DOM directly and also drop a
clean markdown rendering into 10_clean/official/ for the data browser + provenance.

Idempotent (dedups against existing entities). LV-only (other cities expose no past archive).

    PYTHONPATH=scripts python scripts/extract_official_lineup.py [--years 2014-2025] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re

from selectolax.parser import HTMLParser

from pipeline.editions import iter_editions
from pipeline import structured_prompts as SP
from pipeline.schema_validate import validate
from extract_structured import _upsert_global, GLOBAL_DIR
from extract_lineups import _slug, _smart_title

BASE = "https://lasvegas.electricdaisycarnival.com/past-highlights/{year}-lineup/"
# active-city subdomains whose current /lineup/ page is the upcoming/most-recent edition
CITY_SUB = {"las_vegas": "lasvegas", "orlando": "orlando", "mexico_city": "mexico",
            "incheon": "korea", "seoul": "korea", "shanghai": "china", "zhuhai": "china",
            "suzhou": "china", "chiba": "japan", "new_delhi": "india", "phuket": "thailand",
            "medellin": "colombia"}
DAY_RE = re.compile(r"^(mon|tues|wednes|thurs|fri|satur|sun)day", re.I)
JUNK = {"sign up for edc updates", "thank you!", "related", "sponsors", "alphabetical", "by day", "by stage"}
B2B = re.compile(r"\s*[Bb]2[Bb]\s*")              # space-less in the DOM: "PlanetB2BKill" → Planet | Kill


def _classify(hdr: str) -> str:
    h = hdr.strip().lower()
    if DAY_RE.match(hdr.strip()):
        return "day"
    if h in JUNK or "headliner" in h:
        return "junk"
    return "stage"


def _items(lst) -> list[str]:
    out = []
    for li in (lst.css("li") or lst.css(".song__artist")):
        t = li.text(strip=True)
        if not t or "headliner" in t.lower():
            continue
        out.extend(p.strip() for p in B2B.split(t) if p.strip())
    return out


def parse_page(html: str):
    """→ (full_names, stage_of{slug:stage}, day_of{slug:day}). Zip h2↔list positionally within each tab."""
    tree = HTMLParser(html)
    full, stage_of, day_of = [], {}, {}
    for tab in tree.css(".tabs__content"):
        hdrs = [h for h in tab.css("h2") if _classify(h.text(strip=True)) != "junk"]
        lists = tab.css(".lineup__list")
        if not hdrs:                                  # Alphabetical tab — one flat list, no sections
            for lst in lists:
                full.extend(_items(lst))
            continue
        for hdr, lst in zip(hdrs, lists):
            kind = _classify(hdr.text(strip=True))
            label = hdr.text(strip=True).split(",")[0].strip()    # "Friday, May 18" → "Friday"
            for name in _items(lst):
                full.append(name)
                s = _slug(name)
                if not s:
                    continue
                if kind == "day":
                    day_of.setdefault(s, label)
                else:
                    stage_of.setdefault(s, label)
    return full, stage_of, day_of


def emit_edition(ed, names, stage_of, day_of, url, registry) -> tuple[int, int]:
    """Write artist/stage entities + plays_at(→stage, attrs={day}) for one edition. Idempotent."""
    fslug = ed.festival_slug
    fest_base = f"edc_{ed.city}"
    fest_id = f"{fslug}:festival:{fest_base}"
    fest_name = f"Electric Daisy Carnival {ed.city.replace('_', ' ').title()}"
    struct = ed.dir / "15_structured"
    struct.mkdir(parents=True, exist_ok=True)
    ent_path, tri_path = struct / "entities.jsonl", struct / "triples.jsonl"
    have_art, have_ids = set(), set()
    if ent_path.exists():
        for line in ent_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                have_ids.add(r.get("id"))
                if r.get("entity_type") == "artist":
                    have_art.add(r["id"].split(":")[-1])
    meta = {"domain_class": "official-festival", "url": url, "final_url": url, "fetched_at": ""}
    stages = sorted(set(stage_of.values()))
    n_new_a = 0
    with ent_path.open("a") as ef, tri_path.open("a") as tf:
        if fest_id not in have_ids:
            fe = SP.inject_provenance({"id": fest_id, "entity_type": "festival",
                "name_canonical": fest_name, "attributes": {"year": str(ed.year)}}, "entity", meta, fslug, {})
            if not validate("entity", fe):
                ef.write(json.dumps(fe) + "\n")
        stage_ids = {}
        for st in stages:
            sid = f"{fslug}:stage:{_slug(st)}"
            stage_ids[st] = sid
            if sid in have_ids:
                continue
            se = SP.inject_provenance({"id": sid, "entity_type": "stage", "name_canonical": st,
                "attributes": {}}, "entity", meta, fslug, {})
            if not validate("entity", se):
                ef.write(json.dumps(se) + "\n")
                have_ids.add(sid)
        for s, raw in names.items():
            aid = f"{fslug}:artist:{s}"
            if s not in have_art:
                ent = SP.inject_provenance({"id": aid, "entity_type": "artist",
                    "name_canonical": _smart_title(raw, registry), "attributes": {}}, "entity", meta, fslug, {})
                if validate("entity", ent):
                    continue
                ef.write(json.dumps(ent) + "\n")
                _upsert_global(ent)
                have_art.add(s)
                n_new_a += 1
            obj = stage_ids.get(stage_of.get(s), fest_id)
            attrs = {"day": day_of[s]} if s in day_of else {}
            tri = SP.inject_provenance({"subject_id": aid, "predicate": "plays_at", "object": obj,
                "attrs": attrs}, "triple", meta, fslug, {})
            if not validate("triple", tri):
                tf.write(json.dumps(tri) + "\n")
    # clean markdown rendering for the data browser + provenance
    clean_dir = ed.dir / "10_clean" / "official"
    clean_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# {fest_name} {ed.year} — Official Lineup\n", f"Source: {url}\n", "## Lineup\n"]
    md += [f"- {names[s]}" for s in sorted(names)]
    for st in stages:
        acts = [names[s] for s in names if stage_of.get(s) == st]
        md += [f"\n## {st}\n"] + [f"- {a}" for a in sorted(acts)]
    (clean_dir / f"official-{ed.year}-lineup.md").write_text("\n".join(md))
    return n_new_a, len(stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2014-2025", help="LV past-highlights range, e.g. 2014-2025")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--browser", action="store_true",
                    help="render JS-gated /lineup/ pages via headless Chrome (orlando/china/japan/…)")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))

    registry: dict[str, str] = {}
    rp = GLOBAL_DIR / "artists.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                registry.setdefault(r.get("slug", ""), r.get("name", ""))

    # editions keyed by city/year + each city's max (current/upcoming) year
    eds = {e.slug: e for e in iter_editions("edc")}
    max_year = {}
    for e in eds.values():
        max_year[e.city] = max(max_year.get(e.city, 0), e.year)

    # work list: (edition, url). LV → per-year past-highlights archive.
    work = []
    for y in range(lo, hi + 1):
        ed = eds.get(f"las_vegas/{y}")
        if ed:
            work.append((ed, BASE.format(year=y)))
    # current /lineup/ is ONE page per subdomain (the upcoming edition) — several cities can share a
    # subdomain (seoul+incheon→korea, shanghai/zhuhai/suzhou→china), so attribute it only to that
    # subdomain's most-recent edition, never the older siblings (else we'd mis-stamp it on seoul/2019).
    # (LV's archive covers 2014-2025; its current /lineup/ adds the upcoming 2026 — the primary edition)
    by_sub: dict = {}
    for city, sub in CITY_SUB.items():
        ed = eds.get(f"{city}/{max_year.get(city, 0)}")
        if ed and (sub not in by_sub or ed.year > by_sub[sub].year):
            by_sub[sub] = ed
    for sub, ed in by_sub.items():
        work.append((ed, f"https://{sub}.electricdaisycarnival.com/lineup/"))

    def _parse(html):
        full, stage_of, day_of = parse_page(html)
        names = {}
        for n in full:
            s = _slug(n)
            if s:
                names.setdefault(s, n)
        return names, stage_of, day_of

    from scrape.fetcher import HttpFetcher
    http = HttpFetcher(ignore_robots=True)
    browser = None
    if args.browser:
        from scrape.fetcher import BrowserFetcher
        browser = BrowserFetcher(headless=True, ignore_robots=True)
    grand_a = grand_s = 0
    try:
        for ed, url in work:
            d = http.get(url)
            html = d.content.decode("utf-8", "replace") if (d and d.status == 200) else ""
            names, stage_of, day_of = _parse(html) if html else ({}, {}, {})
            if not names and browser is not None:                 # JS-gated → render it
                bd = browser.get(url)
                if bd and bd.content:
                    names, stage_of, day_of = _parse(bd.content.decode("utf-8", "replace"))
            if not names:
                tag = "JS-gated, browser failed" if browser else "JS-gated (retry --browser)"
                print(f"  {ed.slug}: 0 acts ({tag}) — {url}", flush=True)
                continue
            stages = sorted(set(stage_of.values()))
            print(f"  {ed.slug:22s}: {len(names)} acts, {len(stages)} stages, "
                  f"{len(set(day_of.values()))} days  {stages[:3]}…", flush=True)
            grand_a += len(names)
            grand_s += len(stages)
            if not args.dry_run:
                emit_edition(ed, names, stage_of, day_of, url, registry)
    finally:
        http.close()
        if browser is not None:
            browser.close()

    print(f"\n{'(dry-run) ' if args.dry_run else ''}official site: {grand_a} acts, {grand_s} stage-slots "
          f"across {len(work)} edition-pages")


if __name__ == "__main__":
    main()
