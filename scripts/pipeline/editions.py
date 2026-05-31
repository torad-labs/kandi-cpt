"""Iterate the worldwide edition registry, filtered to editions that actually have raw data.

`configs/editions_edc.json` lists 24 locations × their years; only the subset that was
successfully scraped has `00_raw/` on disk (the 97 city-year editions). Every pipeline stage
iterates these via `iter_editions()` and accepts the same `--editions city/year,…` / `--limit`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config as C


@dataclass(frozen=True)
class Ed:
    festival: str
    city: str
    year: int
    venue: str = ""
    country: str = ""
    status: str = ""

    @property
    def dir(self) -> Path:
        return C.edition(self.festival, self.city, self.year)

    @property
    def slug(self) -> str:
        return f"{self.city}/{self.year}"

    @property
    def festival_slug(self) -> str:
        """Prefix for §2 entity ids, e.g. `edc_las_vegas_2026:stage:kinetic_field`."""
        return f"{self.festival}_{self.city}_{self.year}"


def parse_only(only) -> set[str] | None:
    """Normalize a --editions value ('las_vegas/2026,orlando/2024' or list) to a slug set."""
    if not only:
        return None
    if isinstance(only, str):
        only = only.split(",")
    return {s.strip() for s in only if s.strip()} or None


def iter_editions(festival: str = "edc", only=None, with_raw: bool = True):
    """Yield `Ed` for each (city, year) in the registry.

    only      — restrict to these 'city/year' slugs (str CSV or iterable).
    with_raw  — keep only editions whose `00_raw/` exists on disk (the scraped 97).
    """
    only_set = parse_only(only)
    for e in C.load_editions(festival)["editions"]:
        for y in e["years"]:
            ed = Ed(festival, e["city"], int(y), e.get("venue", ""),
                    e.get("country", ""), e.get("status", ""))
            if only_set and ed.slug not in only_set:
                continue
            if with_raw and not (ed.dir / "00_raw").is_dir():
                continue
            yield ed
