"""Reddit — community discussion.

Reddit's 2026 "Security Refresh" blocks unauthenticated automated access ("blocked by
network security"), so reads go through the *logged-in* Patchright browser profile: the
user logs in once (scripts/scrape login flow) and the session persists in the scraper's
Chrome profile (/tmp/kandi_scrape_chrome). With that session, old.reddit's `.json` returns
real structured results.

Reddit's robots.txt blanket-disallows bots, so we bypass it (check_robots=False) for this
authenticated, user-initiated research pull. Requires the browser fetcher (skipped under
--no-browser).
"""
import json
from pathlib import Path

import config as C
from scrape.fetcher import save_doc

DOMAIN_CLASS = "community-discussion"
SUBREDDITS = ("EDC", "aves", "electricdaisycarnival")


def scrape(festival, ed, year, *, http, browser, limit=25) -> list[Path]:
    if browser is None:
        return []
    q = f"{ed['city'].replace('_', ' ')} {year}".strip().replace(" ", "+")
    dest = C.raw_dir(festival, ed["city"], year, "reddit")
    tag = {"festival": festival, "city": ed["city"], "year": year}
    saved = []
    for sub in SUBREDDITS:
        url = (f"https://old.reddit.com/r/{sub}/search.json?q={q}"
               f"&restrict_sr=1&sort=top&t=all&limit={limit}")
        doc = browser.get(url, body_text=True, check_robots=False)
        if not doc or doc.status >= 400:
            continue
        try:
            children = json.loads(doc.content.decode("utf-8", "replace")).get("data", {}).get("children", [])
        except Exception:
            continue
        if not children:
            continue
        p = save_doc(doc, dest, f"r-{sub}-{year}", source_channel="reddit",
                     domain_class=DOMAIN_CLASS, edition=tag,
                     extra={"subreddit": sub, "query": q, "results": len(children)})
        if p:
            saved.append(p)
    return saved
