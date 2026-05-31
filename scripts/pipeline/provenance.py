"""Provenance read + tier/trust mapping + the canonical 10_clean header.

`00_raw/<source>/<name>.meta.json` sidecars carry the scrape provenance (url, domain_class,
source_channel, edition, sha256, fetched_at). This module turns a `domain_class` into the
`Tier N (name)` label the 10_clean header uses and the numeric trust weight downstream stages
score with (`configs/trust_weights.json`), and stamps the header every cleaned doc starts with.
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C

_TRUST = json.loads((C.CONFIGS / "trust_weights.json").read_text())

# domain_class → (tier_int, tier_name) for the 10_clean header. Bands track trust + channel
# semantics; consistent with the curated set (official→T1 "load-bearing", harm-reduction→T2).
TIER: dict[str, tuple[int, str]] = {
    "official-festival": (1, "load-bearing"),
    "official-venue": (1, "load-bearing"),
    "news-mainstream": (2, "reference"),
    "news-music-trade": (2, "reference"),
    "review-aggregator": (2, "reference"),
    "social-vetted": (2, "community-reference"),
    "community-vetted": (2, "community-reference"),
    "community-discussion": (3, "community-texture"),
    "blog-influencer": (3, "community-texture"),
    "user-contributed": (3, "community-texture"),
    "social-personal": (3, "personal-anecdote"),
    "inferred-by-model": (4, "model-inferred"),
}


def tier_for(domain_class: str) -> tuple[int, str]:
    return TIER.get(domain_class, (3, "community-texture"))


def trust_for(domain_class: str, parent: float | None = None) -> float:
    """§7.2 trust weight. `inferred-by-model` inherits parent_record × factor."""
    w = _TRUST["weights"].get(domain_class)
    if w is not None:
        return float(w)
    rel = _TRUST.get("relative", {}).get(domain_class)
    if rel:
        return (parent if parent is not None else 0.7) * float(rel["factor"])
    return 0.5


def read_meta(meta_path) -> dict:
    return json.loads(Path(meta_path).read_text(encoding="utf-8"))


def clean_header(title: str, source: str, domain_class: str, scraped_iso: str | None = None) -> str:
    """The canonical 10_clean preamble: `# Title` / Scraped-line / Tier-line / `---`."""
    date = (scraped_iso or "")[:10] or "unknown"
    tier_n, tier_name = tier_for(domain_class)
    return (f"# {title}\n\n"
            f"Scraped {date} from {source}.\n"
            f"Tier {tier_n} ({tier_name}) corpus material.\n\n"
            f"---\n\n")


def is_unchanged(sidecar_path, raw_sha: str) -> bool:
    """True if a `.clean.json` sidecar already records this raw sha (idempotent re-extract)."""
    p = Path(sidecar_path)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("raw_sha256") == raw_sha
    except Exception:
        return False
