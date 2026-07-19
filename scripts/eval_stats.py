#!/usr/bin/env python3
"""Shared eval statistics — Wilson intervals, n floors, within-noise vs bar.

Closes the handbook's [needs-engine-field] hole: every reported proportion carries a
Wilson score interval; n < min_n is labeled insufficient-n mechanically; a point estimate
inside the interval of its bar is within-noise / not-cleared without hand arithmetic.
"""
from __future__ import annotations

import math
from typing import Any


# Handbook floor (eval/estimands-and-statistics.md): no PASS/FAIL on fewer than ~50 items
# per reported number. Callers may override for pilot-class reports, but the label is always
# mechanical once min_n is known.
DEFAULT_MIN_N = 50
# Normal quantile for a two-sided 95% Wilson interval.
_Z_95 = 1.959963984540054


def wilson_interval(hits: int, n: int, z: float = _Z_95) -> dict[str, Any] | None:
    """Wilson score interval for a binomial proportion. None when n == 0."""
    if n <= 0:
        return None
    if hits < 0 or hits > n:
        raise ValueError(f"hits must be in [0, n]; got hits={hits} n={n}")
    phat = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return {
        "low": low,
        "high": high,
        "center": center,
        "level": 0.95,
        "method": "wilson",
    }


def rough_bound(n: int) -> float | None:
    """Handbook quick check: SE of a proportion ≤ 0.5/√n. None when n == 0."""
    if n <= 0:
        return None
    return 0.5 / math.sqrt(n)


def stamp_proportion(
    hits: int,
    n: int,
    *,
    min_n: int = DEFAULT_MIN_N,
    bar: float | None = None,
    estimand: str | None = None,
    slice_name: str | None = None,
) -> dict[str, Any]:
    """One reported number with the fields the estimand law requires.

    Always carries: hits, n, acc, wilson (or null), rough_bound, n_status.
    When `bar` is set: bar, clears_bar, within_noise_of_bar, bar_status
    (`cleared` | `within-noise` | `below-bar` | `insufficient-n`).
    """
    if n < 0 or hits < 0 or hits > n:
        raise ValueError(f"invalid hits/n: hits={hits} n={n}")
    acc = (hits / n) if n else 0.0
    interval = wilson_interval(hits, n)
    bound = rough_bound(n)
    n_status = "ok" if n >= min_n else "insufficient-n"
    out: dict[str, Any] = {
        "hits": hits,
        "n": n,
        "acc": acc,
        "wilson": interval,
        "rough_bound_half_over_sqrt_n": bound,
        "n_status": n_status,
        "min_n": min_n,
    }
    if estimand is not None:
        out["estimand"] = estimand
    if slice_name is not None:
        out["slice"] = slice_name
    if bar is not None:
        out["bar"] = bar
        if n_status == "insufficient-n":
            out["clears_bar"] = False
            out["within_noise_of_bar"] = False
            out["bar_status"] = "insufficient-n"
        else:
            assert interval is not None and bound is not None
            # Cleared only when the point estimate is above the bar AND the bar sits
            # strictly below the Wilson lower bound (margin > noise). Inside the interval
            # or within the rough bound of the bar → within-noise, not cleared.
            within_wilson = interval["low"] <= bar <= interval["high"]
            within_rough = abs(acc - bar) <= bound
            within_noise = within_wilson or within_rough
            clears = acc > bar and bar < interval["low"]
            out["clears_bar"] = clears
            out["within_noise_of_bar"] = within_noise and not clears
            if clears:
                out["bar_status"] = "cleared"
            elif within_noise:
                out["bar_status"] = "within-noise"
            else:
                out["bar_status"] = "below-bar"
    return out


def label_for_report(stamp: dict[str, Any]) -> str:
    """One-token label for tables: insufficient-n | within-noise | cleared | below-bar | ok."""
    if stamp.get("n_status") == "insufficient-n":
        return "insufficient-n"
    return str(stamp.get("bar_status") or stamp.get("n_status") or "ok")
