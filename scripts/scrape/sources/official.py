"""insomniac.com — the official source (trust 1.0). Cloudflare-gated, so browser-only.
Insomniac's site reflects the CURRENT edition, so we scrape the city hub + discovered
sublinks once, into that city's latest-year partition; historical years come from MFW."""
import re
from pathlib import Path

from selectolax.parser import HTMLParser

import config as C
from scrape.fetcher import save_doc

DOMAIN_CLASS = "official-festival"
BASE = "https://www.insomniac.com"
SUBLINK_RE = re.compile(r"/(lineup|line-up|schedule|set-times|info|festival-guide|faq|news|"
                        r"experience|art|stages|tickets|guide)", re.I)


def _hub_slug(ed: dict) -> str:
    return ed.get("insomniac_slug") or "edc-" + ed["city"].replace("_", "-")


def scrape(festival, ed, year, *, http, browser, limit=8) -> list[Path]:
    if browser is None or year != max(ed["years"]):   # current-edition-centric
        return []
    slug = _hub_slug(ed)
    tag = {"festival": festival, "city": ed["city"], "year": year}
    dest = C.raw_dir(festival, ed["city"], year, "official")
    hub = browser.get(f"{BASE}/events/{slug}/")
    if not hub or hub.status >= 400:
        return []
    saved = []
    p = save_doc(hub, dest, f"{slug}-hub", source_channel="official", domain_class=DOMAIN_CLASS, edition=tag)
    if p:
        saved.append(p)
    # discover internal sublinks (lineup / schedule / guide / news …)
    seen, links = set(), []
    for a in HTMLParser(hub.content.decode("utf-8", "replace")).css("a[href]"):
        href = a.attributes.get("href") or ""
        if not SUBLINK_RE.search(href):
            continue
        if href.startswith("/"):
            href = BASE + href
        if href.startswith(BASE) and href not in seen:
            seen.add(href)
            links.append(href)
    for href in links[:limit]:
        d = browser.get(href)
        if d and d.status < 400:
            name = f"{slug}-" + (href.rstrip("/").split("/")[-1] or "page")
            sp = save_doc(d, dest, name, source_channel="official", domain_class=DOMAIN_CLASS, edition=tag)
            if sp:
                saved.append(sp)
    return saved
