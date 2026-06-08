"""Iterate a domain's unit registry, filtered to units that actually have raw data on disk.

EDC is festival→city→year (`configs/editions_edc.json` lists 24 cities × their years; the 97 with a
`00_raw/` landing zone are the scraped set). A NON-festival domain partitions differently — a flat
`[unit]`, `[product, version]`, … — so its registry lists units by their path `parts`. `iter_editions`
supports BOTH shapes and every stage iterates it the same way. The festival path is byte-identical: an EDC
unit still yields `Ed` with `slug="city/year"`, `city=…`, `year=int`, `festival_slug="edc_city_year"`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import config as C


@dataclass(frozen=True)
class Ed:
    """One unit of a domain's corpus — a festival city/year, or any domain's partition. Constructed ONLY by
    `iter_editions`; every consumer reads the PROPERTIES below (no call site builds an `Ed` directly). `parts`
    is the unit path under `data/<festival>/`: `("las_vegas","2026")` for EDC, `("kanto",)` for a flat domain."""

    festival: str                       # the DOMAIN slug (name kept — ~28 call sites read `ed.festival`)
    parts: tuple[str, ...]              # the unit path under data/<festival>/
    venue: str = ""
    country: str = ""
    status: str = ""

    @property
    def dir(self) -> Path:
        return C.DATA / self.festival / Path(*self.parts)

    @property
    def slug(self) -> str:
        return "/".join(self.parts)

    @property
    def city(self) -> str:
        """Back-compat: the first unit level (festival code reads `ed.city`)."""
        return self.parts[0] if self.parts else ""

    @property
    def year(self):
        """Back-compat: a festival unit is (city, year) — the last level as an `int` when numeric, else the
        raw token. A non-festival unit has no year; the festival year-arithmetic code is never reached there."""
        last = self.parts[-1] if self.parts else ""
        return int(last) if last.isdigit() else last

    @property
    def festival_slug(self) -> str:
        """The §2 entity-id prefix — `edc_las_vegas_2026` for EDC, `<domain>_<parts>` for any domain."""
        return "_".join([self.festival, *self.parts])


def parse_only(only) -> set[str] | None:
    """Normalize a --editions value ('las_vegas/2026,orlando/2024' or list) to a slug set."""
    if not only:
        return None
    if isinstance(only, str):
        only = only.split(",")
    return {s.strip() for s in only if s.strip()} or None


def _registry_units(reg: dict):
    """Yield (parts, meta) per registry unit, supporting BOTH the festival (city × years) and the generic
    (a unit declares its own path via `parts`/`unit`/`slug`/`id`) registry shapes. A unit that declares
    NEITHER is malformed — it is WARNED and skipped, never silently dropped: a dropped unit is a whole
    city/topic silently absent from the trained model (the project's cardinal silent-failure)."""
    dropped = 0
    for e in reg["editions"]:
        if "city" in e and "years" in e:                       # festival shape: city × years → (city, year)
            for y in e["years"]:
                yield (e["city"], str(int(y))), e
        elif "city" in e:                                      # a festival unit that LOST its `years` — loud
            dropped += 1
            print(f"  WARNING: festival unit '{e['city']}' has no 'years' — DROPPED (a registry typo would "
                  f"silently remove a whole city from the corpus)", file=sys.stderr)
        else:                                                  # generic shape: an explicit unit path
            raw = next((e[k] for k in ("parts", "unit", "slug", "id") if e.get(k) not in (None, "", [])), None)
            if raw is None:
                dropped += 1
                print(f"  WARNING: registry unit declares no path (parts/unit/slug/id) — DROPPED: {e}",
                      file=sys.stderr)
                continue
            parts = tuple(str(p) for p in (raw if isinstance(raw, (list, tuple)) else [raw]))
            yield parts, e
    if dropped:
        print(f"  WARNING: editions registry: DROPPED {dropped} malformed unit(s) — see above", file=sys.stderr)


def iter_editions(festival: str = "edc", only=None, with_raw: bool = True):
    """Yield `Ed` for each unit in the registry.

    only      — restrict to these slugs (str CSV or iterable).
    with_raw  — keep only units whose `00_raw/` exists on disk (the scraped set).
    """
    only_set = parse_only(only)
    for parts, e in _registry_units(C.load_editions(festival)):
        ed = Ed(festival, parts, e.get("venue", ""), e.get("country", ""), e.get("status", ""))
        if only_set and ed.slug not in only_set:
            continue
        if with_raw and not (ed.dir / "00_raw").is_dir():
            continue
        yield ed
