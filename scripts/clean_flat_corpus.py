#!/usr/bin/env python3
"""Flat corpus → 10_clean converter, for data-only domain onboarding (the pokemon-lore Kanto pilot's seam).

    PYTHONPATH=scripts .venv/bin/python scripts/clean_flat_corpus.py \
        --domain pokemon-lore --editions kanto --domain-class wiki-bulbapedia

A domain whose raw material arrives as a FLAT directory of already-fetched files (data/<domain>/<unit>/
00_raw/*.{txt,html}) has no scraper-source layout (00_raw/<source>/*.meta.json), so `-m extract` — which
walks that layout — finds nothing. This converter produces the SAME 10_clean contract the structured stage
reads (10_clean/**/*.md with the canonical clean_header + a .clean.json sidecar carrying domain_class):
.txt passes through (it was extracted at scrape time); .html goes through trafilatura. Idempotent by
raw-sha, like `-m extract`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from collections import Counter

from pipeline import provenance as P
from pipeline.editions import iter_editions

SUBFOLDER = "wiki"
MIN_BODY_CHARS = 200  # below this the page carried no prose worth structuring — skip, loudly counted


def _title_of(stem: str) -> str:
    return urllib.parse.unquote(stem).replace("_", " ").strip()


def _body_of(path) -> str | None:
    if path.suffix.lower() == ".html":
        import trafilatura
        return trafilatura.extract(path.read_text(encoding="utf-8", errors="replace"))
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions", help="restrict to unit slugs (CSV); default: all with 00_raw")
    ap.add_argument("--domain-class", required=True, help="provenance class for every doc (e.g. wiki-bulbapedia)")
    ap.add_argument("--force", action="store_true", help="re-convert even if the sidecar is current")
    args = ap.parse_args()

    stats: Counter = Counter()
    for ed in iter_editions(args.domain, only=args.editions):
        out_dir = ed.dir / "10_clean" / SUBFOLDER
        for raw in sorted((ed.dir / "00_raw").iterdir()):
            if not raw.is_file() or raw.suffix.lower() not in (".txt", ".html"):
                continue
            raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            stem = raw.stem.replace(" ", "_")
            out_md, sidecar = out_dir / f"{stem}.md", out_dir / f"{stem}.clean.json"
            if P.is_unchanged(sidecar, raw_sha) and not args.force:
                stats["unchanged"] += 1
                continue
            body = _body_of(raw)
            if not body or len(body.strip()) < MIN_BODY_CHARS:
                stats["skipped_thin"] += 1
                print(f"  ! {ed.slug}/{raw.name}: no/thin prose — skipped", file=sys.stderr)
                continue
            title = _title_of(raw.stem)
            text = P.clean_header(title, raw.name, args.domain_class) + body.strip() + "\n"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_md.write_text(text, encoding="utf-8")
            sidecar.write_text(json.dumps({
                "raw_sha256": raw_sha,
                "clean_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "extractor": "clean_flat_corpus",
                "llm_rescued": False,
                "quality_gate": "flat-passthrough",
                "domain_class": args.domain_class,
                "source_channel": SUBFOLDER,
                "source_url": raw.name,
                "title": title,
            }, indent=2), encoding="utf-8")
            stats["cleaned"] += 1
        print(f"  {ed.slug}: " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    print("--- totals: " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if not stats["cleaned"] and not stats["unchanged"]:
        sys.exit("no documents converted — wrong unit/layout?")


if __name__ == "__main__":
    main()
