"""EDM trade / mainstream news. WordPress-style ?s= search on each site, then fetch the
top discovered article pages. HTTP first, browser fallback on block/empty."""
import re
from pathlib import Path
from urllib.parse import urlparse

from selectolax.parser import HTMLParser

import config as C
from scrape.fetcher import save_doc, slugify

DOMAIN_CLASS = "news-music-trade"
SITES = ("dancingastronaut.com", "youredm.com", "edmidentity.com", "edm.com")
SKIP = re.compile(r"/(category|tag|author|page|wp-|feed|about|contact|privacy|advertise)/", re.I)


def _fetch(url, http, browser):
    doc = http.get(url)
    if (not doc or doc.status >= 400 or not doc.content) and browser is not None:
        doc = browser.get(url)
    return doc if (doc and doc.status < 400 and doc.content) else None


def scrape(festival, ed, year, *, http, browser, limit=8) -> list[Path]:
    q = f"EDC+{ed['city'].replace('_', '+')}+{year}"
    dest = C.raw_dir(festival, ed["city"], year, "news")
    tag = {"festival": festival, "city": ed["city"], "year": year}
    per_site = max(1, limit // len(SITES))
    saved = []
    for dom in SITES:
        listing = _fetch(f"https://{dom}/?s={q}", http, browser)
        if not listing:
            continue
        save_doc(listing, dest, f"search-{dom}-{year}", source_channel="news",
                 domain_class=DOMAIN_CLASS, edition=tag, extra={"query": q, "kind": "listing"})
        # discover same-domain article links
        seen, arts = set(), []
        for a in HTMLParser(listing.content.decode("utf-8", "replace")).css("a[href]"):
            href = (a.attributes.get("href") or "").split("?")[0]
            if not href.startswith(f"https://{dom}") and not href.startswith(f"https://www.{dom}"):
                continue
            if SKIP.search(href) or urlparse(href).path.strip("/").count("/") < 1:
                continue
            if href not in seen:
                seen.add(href)
                arts.append(href)
        for href in arts[:per_site]:
            d = _fetch(href, http, browser)
            if d:
                name = slugify(f"{dom}-" + urlparse(href).path.strip("/").replace("/", "-"))
                p = save_doc(d, dest, name, source_channel="news", domain_class=DOMAIN_CLASS,
                             edition=tag, extra={"kind": "article"})
                if p:
                    saved.append(p)
    return saved
