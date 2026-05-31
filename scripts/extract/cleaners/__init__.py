"""Per-source cleaners: each `clean(artifact_path, meta, *, llm_rescue) -> CleanDoc | None`.

Mirrors `scrape/sources/` — the driver dispatches on `meta['source_channel']`. Returns the
extracted title + body markdown (no header; the driver prepends the canonical 10_clean header)
or None to skip (listing pages, empty/too-thin docs, failed rescue).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CleanDoc:
    title: str
    body: str                       # markdown body, NO header (driver adds it)
    source: str                     # url/string for the "Scraped … from <source>" line
    extractor: str                  # "trafilatura" | "llm-rescue" | "json-reddit" | "json-youtube"
    llm_rescued: bool = False
    quality_gate: dict = field(default_factory=dict)


# source_channel → 10_clean subfolder (mfw is the historical reference backbone)
SUBFOLDER = {"news": "news", "reddit": "reddit", "youtube": "youtube",
             "official": "official", "mfw": "reference"}


def get_cleaners():
    """Lazy import (trafilatura is heavy) → {source_channel: clean_fn}."""
    from . import news, mfw, official, youtube, reddit
    return {"news": news.clean, "mfw": mfw.clean, "official": official.clean,
            "youtube": youtube.clean, "reddit": reddit.clean}
