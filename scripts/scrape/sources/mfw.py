"""Music Festival Wizard — the historical backbone (lineups, dates, venues per year).

MFW's per-festival slug is inconsistent across years (`edc-las-vegas-2024` vs
`electric-daisy-carnival-las-vegas-2013` vs others), so we try several slug forms and,
failing those, discover the real URL via MFW's search. One page per edition."""
import re
from pathlib import Path

from selectolax.parser import HTMLParser

import config as C
from scrape.fetcher import save_doc

DOMAIN_CLASS = "review-aggregator"
BASE = "https://www.musicfestivalwizard.com"


def _slug_candidates(ed: dict, year: int) -> list[str]:
    city = ed["city"].replace("_", "-")
    base = ed.get("mfw_slug") or f"edc-{city}"
    cands = [f"{base}-{year}",
             f"electric-daisy-carnival-{city}-{year}",
             f"edc-{city}-{year}",
             f"electric-daisy-carnival-{year}",
             f"edc-{year}"]
    seen: list[str] = []
    for c in cands:
        if c not in seen:
            seen.append(c)
    return seen


def _discover(http, ed: dict, year: int) -> str | None:
    """Find the festival URL via MFW search when slug guessing fails."""
    city_q = ed["city"].replace("_", "+")
    doc = http.get(f"{BASE}/?s=electric+daisy+carnival+{city_q}+{year}")
    if not doc or doc.status != 200:
        return None
    for a in HTMLParser(doc.content.decode("utf-8", "replace")).css("a[href]"):
        href = a.attributes.get("href") or ""
        if "/festivals/" in href and str(year) in href and re.search(r"edc|electric-daisy", href, re.I):
            return href if href.startswith("http") else BASE + href
    return None


def scrape(festival, ed, year, *, http, browser, limit=None) -> list[Path]:
    dest = C.raw_dir(festival, ed["city"], year, "mfw")
    tag = {"festival": festival, "city": ed["city"], "year": year}

    doc = name = None
    for slug in _slug_candidates(ed, year):
        d = http.get(f"{BASE}/festivals/{slug}/")
        if d and d.status == 200:
            doc, name = d, slug
            break
    if doc is None:
        href = _discover(http, ed, year)
        if href:
            doc = http.get(href)
            name = href.rstrip("/").split("/")[-1]
    if not doc or doc.status != 200:
        return []

    p = save_doc(doc, dest, name or f"edc-{ed['city']}-{year}", source_channel="mfw",
                 domain_class=DOMAIN_CLASS, edition=tag)
    return [p] if p else []
