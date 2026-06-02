#!/usr/bin/env python3
"""EDM Wiki (edm.fandom.com) → lineup + stages + set-times, for the post-2016 residual editions.

The deepest tail source: each EDC edition the EDM wiki covers (≈2016+) has a "Lineup" section (every act) AND
a "Set Times" section grouped by day → stage → timed slots. One page fills lineup + stages + set-times at
once. Fandom 403s plain HTTP (bot-protected) but a real headed browser renders it (run under xvfb). We parse
the MediaWiki DOM: the Lineup <ul> for entities, the Set Times h3(day)/h4(stage)/<li> slots for
plays_at(artist→stage, attrs={day,set_time}). Idempotent; B2B sets split into individual artists.

    DISPLAY=:0 PYTHONPATH=scripts python scripts/extract_fandom_lineup.py [--editions …] [--dry-run]
    xvfb-run -a bash -c 'PYTHONPATH=scripts .venv/bin/python scripts/extract_fandom_lineup.py'   # off-screen
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

# registry city → EDC brand name used in the fandom page title
FANDOM_NAME = {
    "las_vegas": "Las_Vegas", "orlando": "Orlando", "mexico_city": "Mexico", "los_angeles": "Los_Angeles",
    "san_bernardino": "San_Bernardino", "denver": "Colorado", "dallas": "Dallas", "new_york_nj": "New_York",
    "chicago": "Chicago", "san_juan": "Puerto_Rico", "london": "London", "milton_keynes": "United_Kingdom",
    "sao_paulo": "Brazil", "new_delhi": "India", "shanghai": "China", "zhuhai": "China", "suzhou": "China",
    "chiba": "Japan", "seoul": "Korea", "incheon": "Korea", "portimao": "Portugal", "phuket": "Thailand",
    "medellin": "Colombia",
}
_DAY_RE = re.compile(r"^(mon|tues|wednes|thurs|fri|satur|sun)day", re.I)
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b.*$", re.I)   # trailing "2pm-2:58" etc.


def _names_from_li(li):
    """Artist name(s) from a fandom <li>: each linked act, splitting collabs; fall back to plain text."""
    out = [a.text(strip=True) for a in li.css("a") if a.text(strip=True)]
    if not out:
        txt = _TIME.sub("", li.text(strip=True)).strip()      # strip trailing set-time
        if txt:
            out = [txt]
    return [n for n in out if n and not n.lower().startswith(("surprise", "tba", "opening", "intermezzo",
            "closing", "kinetic experience"))]


def _slot_time(li) -> str:
    m = re.search(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?", li.text())
    return m.group(0) if m else ""


def _heading(node):
    """(level, text) if node is a heading or wraps one (modern MediaWiki nests h2 in div.mw-heading)."""
    if node.tag in ("h2", "h3", "h4"):
        return int(node.tag[1]), node.text(strip=True).split("[")[0].strip()
    cls = node.attributes.get("class") or ""
    if "mw-heading" in cls:
        h = node.css_first("h2, h3, h4")
        if h:
            return int(h.tag[1]), h.text(strip=True).split("[")[0].strip()
    return 0, ""


def parse_fandom(html: str):
    """→ (lineup_names[set], stage_of{slug:stage}, day_of{slug:day}, time_of{slug:set_time}).

    Walks the .mw-parser-output direct children in DOCUMENT ORDER (selectolax css() groups by selector,
    which would scramble the section state machine; this also skips the infobox's nested h3s)."""
    tree = HTMLParser(html)
    body = tree.css_first(".mw-parser-output")
    if not body:
        return set(), {}, {}, {}
    lineup, stage_of, day_of, time_of = set(), {}, {}, {}
    section = day = stage = None
    node = body.child
    while node is not None:
        level, txt = _heading(node)
        if level:
            low = txt.lower()
            if level == 2:
                section = "settimes" if "set time" in low else ("lineup" if "lineup" in low else None)
                day = stage = None
            elif level == 3:
                day = txt if _DAY_RE.match(txt) else day
            elif level == 4:
                stage = txt
            node = node.next
            continue
        if node.tag != "ul":
            node = node.next
            continue
        # a <ul> — interpret per current section
        if section == "lineup":
            for li in node.css("li"):
                for n in _names_from_li(li):
                    s = _slug(n)
                    if s:
                        lineup.add((s, n))
        elif section == "settimes" and stage:
            t = ""
            for li in node.css("li"):
                t = _slot_time(li) or t
                for n in _names_from_li(li):
                    s = _slug(n)
                    if not s:
                        continue
                    lineup.add((s, n))
                    stage_of.setdefault(s, stage)
                    if day:
                        day_of.setdefault(s, day)
                    if t:
                        time_of.setdefault(s, t)
        node = node.next
    return lineup, stage_of, day_of, time_of


def emit(ed, lineup, stage_of, day_of, time_of, url, registry) -> int:
    fslug = ed.festival_slug
    fest_id = f"{fslug}:festival:edc_{ed.city}"
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
    meta = {"domain_class": "community-vetted", "url": url, "final_url": url, "fetched_at": ""}
    stages = sorted(set(stage_of.values()))
    n_new = 0
    with ent_path.open("a") as ef, tri_path.open("a") as tf:
        if fest_id not in have_ids:
            fe = SP.inject_provenance({"id": fest_id, "entity_type": "festival",
                "name_canonical": f"Electric Daisy Carnival {ed.city.replace('_', ' ').title()}",
                "attributes": {"year": str(ed.year)}}, "entity", meta, fslug, {})
            if not validate("entity", fe):
                ef.write(json.dumps(fe) + "\n")
        stage_ids = {}
        for st in stages:
            sid = f"{fslug}:stage:{_slug(st)}"
            stage_ids[st] = sid
            if sid not in have_ids:
                se = SP.inject_provenance({"id": sid, "entity_type": "stage", "name_canonical": st,
                    "attributes": {}}, "entity", meta, fslug, {})
                if not validate("entity", se):
                    ef.write(json.dumps(se) + "\n")
                    have_ids.add(sid)
        for s, raw in sorted(lineup):
            aid = f"{fslug}:artist:{s}"
            if s not in have_art:
                ent = SP.inject_provenance({"id": aid, "entity_type": "artist",
                    "name_canonical": _smart_title(raw, registry), "attributes": {}}, "entity", meta, fslug, {})
                if validate("entity", ent):
                    continue
                ef.write(json.dumps(ent) + "\n")
                _upsert_global(ent)
                have_art.add(s)
                n_new += 1
            obj = stage_ids.get(stage_of.get(s), fest_id)
            attrs = {}
            if s in day_of:
                attrs["day"] = day_of[s]
            if s in time_of:
                attrs["set_time"] = time_of[s]
            tri = SP.inject_provenance({"subject_id": aid, "predicate": "plays_at", "object": obj,
                "attrs": attrs}, "triple", meta, fslug, {})
            if not validate("triple", tri):
                tf.write(json.dumps(tri) + "\n")
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="comma 'city/year' (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    registry: dict[str, str] = {}
    rp = GLOBAL_DIR / "artists.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                registry.setdefault(r.get("slug", ""), r.get("name", ""))

    from scrape.fetcher import BrowserFetcher
    browser = BrowserFetcher(headless=False, ignore_robots=True)
    grand_a = grand_s = grand_t = 0
    try:
        for ed in iter_editions("edc", only=args.editions):
            name = FANDOM_NAME.get(ed.city)
            if not name:
                continue
            urls = [f"https://edm.fandom.com/wiki/Electric_Daisy_Carnival_{name}_{ed.year}",
                    f"https://edm.fandom.com/wiki/EDC_{name}_{ed.year}"]
            lineup, stage_of, day_of, time_of = set(), {}, {}, {}
            used = ""
            for url in urls:
                d = browser.get(url)
                if d and d.status == 200 and b"noarticletext" not in d.content:   # skip empty/404 wiki pages
                    lp, so, do, to = parse_fandom(d.content.decode("utf-8", "replace"))
                    if lp:
                        lineup, stage_of, day_of, time_of, used = lp, so, do, to, url
                        break
            if not lineup:
                continue
            grand_a += len(lineup)
            grand_s += len(set(stage_of.values()))
            grand_t += len(time_of)
            print(f"  {ed.slug:22s}: {len(lineup)} acts, {len(set(stage_of.values()))} stages, "
                  f"{len(time_of)} set-times  {used.split('/')[-1]}", flush=True)
            if not args.dry_run:
                emit(ed, lineup, stage_of, day_of, time_of, used, registry)
    finally:
        browser.close()
    print(f"\n{'(dry-run) ' if args.dry_run else ''}fandom: {grand_a} acts, {grand_s} stage-slots, "
          f"{grand_t} set-times")


if __name__ == "__main__":
    main()
