"""Source-module SKELETON — copy to scripts/scrape/sources/<your-domain>.py and register it in __init__.py.

A source module fetches raw documents for ONE acquisition channel and saves them under
data/<domain>/<unit>/00_raw/<source>/. The acquire stage is the MOST domain-specific code you write — the
master plan: "the scraper is the ONLY real code." PokeAPI, a wiki dump, a REST API, a file drop, an RSS
feed — whatever your sources are. Everything DOWNSTREAM (extract → structured → hydrate → synth → probes →
tokenize → sft) is the generic engine; only this is bespoke.

Worked examples to copy the exact `save_doc` call + the fetch patterns from:
  - scripts/scrape/sources/mfw.py      — plain HTTP (the `http` fetcher)
  - scripts/scrape/sources/reddit.py   — authenticated browser + JSON (the `browser` fetcher)
  - scripts/scrape/sources/official.py — browser + HTML behind Cloudflare

Interface the dispatcher (scripts/scrape/__main__.py) calls:

    DOMAIN_CLASS = "<provenance-class>"   # MUST be a key in your manifest's ontology.confidence_map
                                          # (provenance is pipeline-owned, never model-invented)

    def scrape(domain, unit, year, *, http, browser, limit) -> list[Path]:
        # domain : the domain slug (e.g. your <your-domain>).
        # unit   : one registry entry (configs/editions_<domain>.json) — ITS shape is yours. The festival
        #          sources expect {city, year, ...}; a region-partitioned domain might pass {region: "kanto"}
        #          and ignore `year`. Key off your own fields.
        # http   : HttpFetcher (always present).  browser : BrowserFetcher (None under --no-browser).
        # limit  : max docs to pull for this channel/unit.
        # Returns: the list of 00_raw paths written (via save_doc).
"""
from pathlib import Path

from scrape.fetcher import save_doc   # see reddit.py / mfw.py for the exact save_doc(...) call

DOMAIN_CLASS = "<your-provenance-class>"   # e.g. "wiki-bulbapedia", "api-official", "community-discussion"


def scrape(domain, unit, year=None, *, http, browser, limit: int = 10) -> list[Path]:
    """Fetch + save this channel's raw docs for one unit; return the written 00_raw paths.

    REPLACE the body: build your source URL(s) from `unit`, fetch with `http` (or `browser` for JS/auth),
    and save each document with `save_doc(...)`. Skeleton only — see the worked sources named in the module
    docstring for the precise fetch + save_doc signature.
    """
    raise NotImplementedError(
        "author this source for your domain — copy the fetch + save_doc pattern from "
        "scripts/scrape/sources/{mfw,reddit,official}.py")
