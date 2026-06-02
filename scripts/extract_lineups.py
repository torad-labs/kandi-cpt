#!/usr/bin/env python3
"""Deterministic lineup extraction: bulleted artist lists in 10_clean → 15_structured artist entities + plays_at.

The LLM structured pass (extract_structured.py) under-recalls long lineups badly — a Music Festival Wizard
reference page lists 200+ acts as `- NAME` bullets, but the model samples a handful (max_tokens + summarize
bias). A bulleted list is the one case where an LLM is the WRONG tool: a parser is exhaustive, free, instant,
and 100% faithful. This recovers every act in the list as an `artist` entity + a `plays_at` triple
(object = the edition's festival entity), schema-validated through the same path as the LLM records.

Idempotent: skips artists already present in the edition's entities.jsonl, so it composes with / re-runs
after the LLM pass. Casing prefers the global registry's canonical name (consistency), else title-cases the
ALL-CAPS source. Back-to-back sets ("ACRAZE B2B KREAM") split into individual artists.

    PYTHONPATH=scripts python scripts/extract_lineups.py [--editions city/year,…] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re

from selectolax.parser import HTMLParser

from pipeline.editions import iter_editions
from pipeline import structured_prompts as SP
from pipeline.schema_validate import validate
# reuse the structured pass's provenance/meta/global-upsert helpers (no duplication)
from extract_structured import _domain_of, _scrape_meta, _upsert_global, GLOBAL_DIR

LINEUP_HDR = re.compile(r"^#{0,4}\s*(?:\d{4}\s+)?(?:artist\s+)?line[\s-]?up\b", re.I)
BULLET = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
B2B = re.compile(r"\s+[Bb]2[Bb]\s+")
_REJECT = {"lineup", "line-up", "line up", "more", "tickets", "tba", "tbd", "and more", "more tba",
           "special guest", "special guests", "surprise guest", "phase 1", "phase 2", "headliners"}
# channel → confidence rank (parse the most-trustworthy doc first; first occurrence of a slug wins provenance)
_RANK = {"official": 0, "reference": 1, "news": 2, "youtube": 3, "reddit": 4}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


def _smart_title(name: str, registry: dict) -> str:
    s = _slug(name)
    if s in registry:                      # established canonical casing (consistency + dedup)
        return registry[s]
    return name.title() if name.isupper() else name


def parse_lineup(md_text: str) -> list[str]:
    """Names from a contiguous bulleted list under a 'lineup' header. Splits B2B sets; drops non-name bullets."""
    out, in_block = [], False
    for ln in md_text.splitlines():
        if LINEUP_HDR.search(ln) and len(ln) < 48:
            in_block = True
            continue
        if not in_block:
            continue
        m = BULLET.match(ln)
        if m:
            raw = m.group(1).strip().lstrip("•").strip()
            if (not raw or len(raw) > 40 or raw.count(" ") > 6 or raw.endswith(".")
                    or any(c in raw for c in ":/|") or "http" in raw.lower()
                    or raw.lower() in _REJECT):
                continue
            out.extend(p.strip() for p in B2B.split(raw) if p.strip())
        elif ln.strip() and not ln.lstrip().startswith(("#", ">")):
            in_block = False               # a non-bullet content line ends the list
    return out


def mfw_raw_lineup(ed_dir) -> tuple[list[str], str]:
    """(names, url) from the raw rendered MFW page's `.lineupblock`/`.hublineup` <li> items. MFW renders
    lineups client-side and short (few-artist) pages fail trafilatura's 250-word prose gate, so they never
    reach 10_clean — but the artist list is right there in the rendered HTML. Parse it directly."""
    htmls = sorted(ed_dir.glob("00_raw/mfw/*.html"))
    if not htmls:
        return [], ""
    tree = HTMLParser(htmls[0].read_text(encoding="utf-8", errors="replace"))
    block = tree.css_first(".lineupblock") or tree.css_first(".hublineup")
    names = [li.text(strip=True) for li in block.css("li")] if block else []
    url = ""
    meta_p = htmls[0].with_name(htmls[0].name + ".meta.json")
    if meta_p.exists():
        try:
            m = json.loads(meta_p.read_text())
            url = m.get("final_url") or m.get("source_url") or m.get("url") or ""
        except Exception:
            pass
    return [n for n in names if n], url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="comma 'city/year' list (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report recoverable counts, write nothing")
    args = ap.parse_args()

    reg_path = GLOBAL_DIR / "artists.jsonl"
    registry: dict[str, str] = {}
    if reg_path.exists():
        for line in reg_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                registry.setdefault(r.get("slug", ""), r.get("name", ""))

    grand = 0
    for ed in iter_editions("edc", only=args.editions):
        clean = ed.dir / "10_clean"
        if not clean.is_dir():
            continue
        struct = ed.dir / "15_structured"
        ent_path, tri_path = struct / "entities.jsonl", struct / "triples.jsonl"
        # existing artist slugs + festival-entity ids already on disk (for dedup + object resolution)
        have_art, have_ids = set(), set()
        if ent_path.exists():
            for line in ent_path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                have_ids.add(r.get("id"))
                if r.get("entity_type") == "artist":
                    have_art.add(r["id"].split(":")[-1])

        fest_id = f"{ed.festival_slug}:festival:edc_{ed.city}"
        # walk docs most-trustworthy first; first slug wins provenance
        docs = sorted(clean.rglob("*.md"),
                      key=lambda p: _RANK.get(p.relative_to(clean).parts[0], 9))
        found: dict[str, dict] = {}          # slug -> {name, meta}
        for md in docs:
            if md.name == "README.md":
                continue
            sub = md.relative_to(clean).parts[0]
            for raw in parse_lineup(md.read_text(encoding="utf-8", errors="replace")):
                s = _slug(raw)
                if not s or s in have_art or s in found:
                    continue
                found[s] = {"name": _smart_title(raw, registry),
                            "meta": {"domain_class": _domain_of(md, sub), **_scrape_meta(md, sub)}}

        # also harvest the raw MFW lineup block (the short-lineup pages trafilatura's gate dropped)
        mfw_names, mfw_url = mfw_raw_lineup(ed.dir)
        mfw_meta = {"domain_class": "review-aggregator", "url": mfw_url, "final_url": mfw_url, "fetched_at": ""}
        for raw in mfw_names:
            for part in (p.strip() for p in B2B.split(raw)):
                s = _slug(part)
                if not part or len(part) > 40 or part.lower() in _REJECT or s in have_art or s in found:
                    continue
                found[s] = {"name": _smart_title(part, registry), "meta": mfw_meta}

        if not found:
            continue
        grand += len(found)
        print(f"  {ed.slug:24s} +{len(found):3d} artists (had {len(have_art)})", flush=True)
        if args.dry_run:
            continue

        struct.mkdir(parents=True, exist_ok=True)
        with ent_path.open("a") as ef, tri_path.open("a") as tf:
            # ensure the festival object exists so plays_at resolves
            if fest_id not in have_ids:
                fe = SP.inject_provenance(
                    {"id": fest_id, "entity_type": "festival",
                     "name_canonical": f"Electric Daisy Carnival {ed.city.replace('_', ' ').title()}",
                     "attributes": {"year": str(ed.year)}},
                    "entity", next(iter(found.values()))["meta"], ed.festival_slug, {})
                if not validate("entity", fe):
                    ef.write(json.dumps(fe) + "\n")
            for s, info in found.items():
                aid = f"{ed.festival_slug}:artist:{s}"
                ent = SP.inject_provenance(
                    {"id": aid, "entity_type": "artist", "name_canonical": info["name"], "attributes": {}},
                    "entity", info["meta"], ed.festival_slug, {})
                tri = SP.inject_provenance(
                    {"subject_id": aid, "predicate": "plays_at", "object": fest_id, "attrs": {}},
                    "triple", info["meta"], ed.festival_slug, {})
                e_err, t_err = validate("entity", ent), validate("triple", tri)
                if e_err or t_err:
                    print(f"    ! reject {aid}: {(e_err or t_err)[:2]}", flush=True)
                    continue
                ef.write(json.dumps(ent) + "\n")
                tf.write(json.dumps(tri) + "\n")
                _upsert_global(ent)

    print(f"\n{'(dry-run) ' if args.dry_run else ''}lineup extraction recovered {grand} artist plays_at facts")


if __name__ == "__main__":
    main()
