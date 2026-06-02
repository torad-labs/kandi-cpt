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

    # resolve the festival URL via HTTP (a 200 shell is enough to confirm the slug exists, even though
    # MFW's lineup loads via JS behind Cloudflare and isn't in that shell)
    name = url = None
    for slug in _slug_candidates(ed, year):
        d = http.get(f"{BASE}/festivals/{slug}/")
        if d and d.status == 200:
            name, url = slug, f"{BASE}/festivals/{slug}/"
            break
    if url is None:
        href = _discover(http, ed, year)
        if href:
            name, url = href.rstrip("/").split("/")[-1], href
    if url is None:
        return []

    # MFW renders the full lineup client-side behind a Cloudflare JS challenge — a real (headed) browser
    # solves the challenge and executes the JS; plain HTTP only gets the lineup-less shell. Prefer the
    # browser when one is available, fall back to the HTTP shell otherwise.
    doc = (browser.get(url) if browser is not None else None) or http.get(url)
    if not doc or doc.status != 200:
        return []
    p = save_doc(doc, dest, name or f"edc-{ed['city']}-{year}", source_channel="mfw",
                 domain_class=DOMAIN_CLASS, edition=tag)
    return [p] if p else []
