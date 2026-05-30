"""Source modules — each exposes scrape(festival, ed, year, *, http, browser, limit) -> [Path].

ed = one registry entry (city, country, venue, status, years[, mfw_slug/insomniac_slug]).
http = HttpFetcher, browser = BrowserFetcher (may be None if --no-browser).
"""
from . import mfw, official, youtube, reddit, news

SOURCES = {
    "mfw": mfw,            # Music Festival Wizard — historical backbone (HTTP)
    "official": official,  # insomniac.com — current edition (browser / Cloudflare)
    "youtube": youtube,    # yt-dlp metadata + descriptions
    "reddit": reddit,      # old.reddit search JSON (browser)
    "news": news,          # EDM trade / mainstream (HTTP + browser fallback)
}
