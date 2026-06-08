"""EDC scraper CLI.  Run:  PYTHONPATH=scripts .venv/bin/python -m scrape [opts]

  python -m scrape --city las_vegas --year 2024 --sources mfw,official,youtube,reddit,news --limit 5
  python -m scrape                       # full sweep: every edition x year x source
  python -m scrape --no-browser          # HTTP-only sources (mfw, youtube, news) — no Chrome

Browser sources (official, reddit) need real headed Chrome → run with a display:
  DISPLAY=:0 PYTHONPATH=scripts .venv/bin/python -m scrape ...        (your screen)
  xvfb-run -a PYTHONPATH=scripts .venv/bin/python -m scrape ...       (off-screen, for bulk)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import config as C
from .fetcher import HttpFetcher, BrowserFetcher
from .sources import SOURCES

BROWSER_SOURCES = {"official", "reddit"}   # hard-require the patched browser


def _select(reg: dict, city: str | None, status: str | None) -> list[dict]:
    eds = reg["editions"]
    if city:
        eds = [e for e in eds if e["city"] == city]
    if status:
        eds = [e for e in eds if e.get("status") == status]
    return eds


def main():
    ap = argparse.ArgumentParser(prog="scrape", description="Scrape EDC raw data into 00_raw/<source>/.")
    ap.add_argument("--festival", "--domain", default="edc",
                    help="domain/festival slug (--domain is the master-plan alias; picks data/<slug> + its registry). "
                         "The city/year selection below is the festival unit shape; a domain with a different unit "
                         "shape authors its own dispatch alongside its source modules (sources/_TEMPLATE.py).")
    ap.add_argument("--city", help="one city slug (default: all)")
    ap.add_argument("--year", type=int, help="one year (default: all held years)")
    ap.add_argument("--status", choices=["active", "discontinued", "planned"], help="filter editions")
    ap.add_argument("--sources", default="mfw,official,youtube,reddit,news")
    ap.add_argument("--limit", type=int, default=10, help="max docs per source per edition")
    ap.add_argument("--no-browser", action="store_true", help="skip browser sources, HTTP-only")
    ap.add_argument("--headless", action="store_true", help="headless Chrome (weaker vs Cloudflare)")
    ap.add_argument("--ignore-robots", action="store_true")
    args = ap.parse_args()

    reg = C.load_editions(args.festival)
    eds = _select(reg, args.city, args.status)
    if not eds:
        sys.exit(f"no editions selected for '{args.festival}' "
                 f"(registry empty, or --city/--status filtered everything out) — nothing to scrape")
    sources = [s for s in args.sources.split(",") if s in SOURCES]
    if not sources:
        ap.error(f"no valid sources in {args.sources!r}; choose from {sorted(SOURCES)}")

    need_browser = (not args.no_browser) and bool(
        set(sources) & BROWSER_SOURCES or "news" in sources or "mfw" in sources)
    if args.no_browser:
        sources = [s for s in sources if s not in BROWSER_SOURCES]
    if need_browser and not args.headless and not os.environ.get("DISPLAY"):
        print("WARN: browser sources need a display — set DISPLAY=:0 or use xvfb-run. Continuing…", flush=True)

    http = HttpFetcher(ignore_robots=args.ignore_robots)
    browser = (BrowserFetcher(headless=args.headless, ignore_robots=args.ignore_robots)
               if need_browser else None)

    if any("city" not in ed for ed in eds):
        sys.exit(
            "festival dispatch is city/year-only; a domain with a different unit shape "
            "authors its own acquire dispatch (see scripts/scrape/sources/_TEMPLATE.py)"
        )

    coverage: dict[str, dict[str, int]] = {}
    t0 = time.time()
    try:
        for ed in eds:
            years = [args.year] if args.year else list(ed["years"])
            for year in years:
                if year not in ed["years"]:
                    continue
                key = f"{ed['city']}/{year}"
                coverage[key] = {}
                for s in sources:
                    try:
                        paths = SOURCES[s].scrape(args.festival, ed, year,
                                                  http=http, browser=browser, limit=args.limit)
                    except Exception as e:
                        print(f"  ! {key} {s}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                        paths = []
                    coverage[key][s] = len(paths)
                print(f"[{key}] " + "  ".join(f"{s}={coverage[key].get(s, 0)}" for s in sources), flush=True)
    finally:
        http.close()
        if browser:
            browser.close()

    # coverage table + JSON summary
    total = sum(sum(v.values()) for v in coverage.values())
    print(f"\n=== coverage ({total} docs in {time.time() - t0:.0f}s) ===")
    print("edition".ljust(22) + "".join(s[:7].ljust(9) for s in sources))
    for key, row in coverage.items():
        print(key.ljust(22) + "".join(str(row.get(s, 0)).ljust(9) for s in sources))
    C.LOGS.mkdir(parents=True, exist_ok=True)
    (C.LOGS / "scrape_summary.json").write_text(json.dumps(
        {"festival": args.festival, "sources": sources, "total_docs": total, "coverage": coverage}, indent=2))
    print(f"\nsummary -> {C.LOGS / 'scrape_summary.json'}")


if __name__ == "__main__":
    main()
