#!/usr/bin/env python3
"""Recover authoritative insomniac.com EDC official pages from the Wayback Machine.

The audit found official/ scraped for only 2/97 editions — the live insomniac.com hosts only current
editions; historical official pages (the best lineup/stage/date source) were taken down. But archive.org
preserved many. This queries the CDX API for insomniac.com/events/edc-<place>-<year> snapshots (status
200), maps each to our edition, fetches the raw archived page (the id_ snapshot = no Wayback chrome), and
saves it to 00_raw/official/ in the standard provenance format so `python -m extract` cleans it like any
official doc.

    PYTHONPATH=scripts python scripts/scrape_wayback.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import urllib.request

import config as C
from scrape.fetcher import HttpFetcher, save_doc

CDX = ("http://web.archive.org/cdx/search/cdx?url=insomniac.com/events/edc-*"
       "&output=text&collapse=urlkey&filter=statuscode:200&fl=original,timestamp&limit=600")

# festtimetable/insomniac place token -> our registry city slug. Ambiguous multi-city countries
# (china, uk) are omitted — handled per-edition only if unambiguous.
PLACE = {
    "las-vegas": "las_vegas", "orlando": "orlando", "mexico": "mexico_city", "korea": "incheon",
    "colombia": "medellin", "portugal": "portimao", "japan": "chiba", "india": "new_delhi",
    "brazil": "sao_paulo", "new-york": "new_york_nj", "puerto-rico": "san_juan", "china": None,
    "uk": None, "thailand": "phuket",
}
# non-edition event pages on insomniac that match edc-* but aren't festival editions
SKIP = ("secret-lineup", "afterhours", "recovery-party", "pre-party", "week-")


def edition_for(url: str):
    """edc-<place>-<year>... → (city_slug, year) or None."""
    m = re.search(r"/events/edc-([a-z-]+?)-(\d{4})\b", url)
    if not m:
        return None
    place, year = m.group(1), m.group(2)
    if any(s in url for s in SKIP):
        return None
    # longest place-token match (handles 'las-vegas', 'new-york', 'puerto-rico')
    for tok in sorted(PLACE, key=len, reverse=True):
        if place == tok or place.startswith(tok + "-"):
            city = PLACE[tok]
            return (city, year) if city else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    http = HttpFetcher()
    cdx = urllib.request.urlopen(CDX, timeout=30).read().decode("utf-8", "replace")

    # pick the latest snapshot per edition (most-complete archived page)
    best: dict = {}
    for line in cdx.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        url, ts = parts[0], parts[1]
        ed = edition_for(url)
        if not ed:
            continue
        if ed not in best or ts > best[ed][1]:
            best[ed] = (url, ts)

    print(f"Wayback EDC official pages mapped to our editions: {len(best)}")
    for (city, year), (url, ts) in sorted(best.items()):
        print(f"  {city}/{year}  <- {url}  @{ts}")
    if args.dry_run:
        print("\n[dry-run] nothing fetched.")
        return

    saved = 0
    for (city, year), (url, ts) in sorted(best.items()):
        raw_dir = C.ROOT / "data/edc" / city / year / "00_raw" / "official"
        wb = f"http://web.archive.org/web/{ts}id_/{url}"
        doc = http.get(wb)
        if not doc or doc.status >= 400:
            print(f"  ! {city}/{year}: fetch failed (status={getattr(doc,'status','none')})", flush=True)
            continue
        save_doc(doc, raw_dir, f"wayback-official-{ts}", source_channel="official",
                 domain_class="official-festival",
                 edition={"festival": "edc", "city": city, "year": int(year)},
                 extra={"wayback_ts": ts, "wayback_src": url})
        saved += 1
        print(f"  ✓ {city}/{year}: saved official from Wayback ({ts})", flush=True)
    http.close()
    print(f"\nsaved {saved} Wayback official docs across {len(best)} editions")


if __name__ == "__main__":
    main()
