#!/usr/bin/env python3
"""Red-green proofs for Wilson intervals + mechanical n-floor / within-noise labels."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_stats as S  # noqa: E402


def test_wilson_known_values():
    # n=40, hits=32 → p=0.80. Interval must contain 0.80 and be narrower than [0,1].
    interval = S.wilson_interval(32, 40)
    assert interval is not None
    assert interval["method"] == "wilson"
    assert interval["level"] == 0.95
    assert interval["low"] < 0.80 < interval["high"]
    assert 0.0 <= interval["low"] < interval["high"] <= 1.0
    # Rough SE bound 0.5/√40 ≈ 0.079
    assert abs(S.rough_bound(40) - 0.5 / math.sqrt(40)) < 1e-12


def test_insufficient_n_is_mechanical():
    stamp = S.stamp_proportion(32, 40, min_n=50, bar=0.80, estimand="augmentation_holdout")
    assert stamp["n_status"] == "insufficient-n"
    assert stamp["bar_status"] == "insufficient-n"
    assert stamp["clears_bar"] is False
    assert stamp["wilson"] is not None
    assert stamp["estimand"] == "augmentation_holdout"
    assert S.label_for_report(stamp) == "insufficient-n"


def test_within_noise_of_bar_no_hand_arithmetic():
    # 40-item slice at exactly the bar: within-noise / not-cleared (handbook red-green).
    # Use min_n=40 so n-floor does not mask the within-noise label.
    stamp = S.stamp_proportion(32, 40, min_n=40, bar=0.80)
    assert stamp["acc"] == 0.80
    assert stamp["n_status"] == "ok"
    assert stamp["within_noise_of_bar"] is True
    assert stamp["clears_bar"] is False
    assert stamp["bar_status"] == "within-noise"
    assert S.label_for_report(stamp) == "within-noise"


def test_cleared_bar_requires_wilson_lower_above_bar():
    # n=200, hits=180 → p=0.90 against bar 0.80 should clear.
    stamp = S.stamp_proportion(180, 200, min_n=50, bar=0.80)
    assert stamp["n_status"] == "ok"
    assert stamp["bar_status"] == "cleared"
    assert stamp["clears_bar"] is True
    assert stamp["wilson"]["low"] > 0.80


def test_below_bar():
    stamp = S.stamp_proportion(10, 100, min_n=50, bar=0.50)
    assert stamp["bar_status"] == "below-bar"
    assert stamp["clears_bar"] is False


def test_zero_n():
    stamp = S.stamp_proportion(0, 0, min_n=50, bar=0.5)
    assert stamp["wilson"] is None
    assert stamp["n_status"] == "insufficient-n"


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} EVAL-STATS TESTS PASS")
