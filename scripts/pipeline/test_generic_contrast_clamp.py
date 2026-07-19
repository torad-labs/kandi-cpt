#!/usr/bin/env python3
"""Red-green proofs for contrast hard-cap clamp (no provider)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.generic_contrast import clamp_contrast_to_fraction  # noqa: E402


def _recs(n: int) -> list[dict]:
    return [{"text": f"contrast {i}", "i": i} for i in range(n)]


def test_within_cap_no_clamp():
    kept, info = clamp_contrast_to_fraction(_recs(5), base_lines=100, max_fraction=0.10)
    assert len(kept) == 5
    assert info["clamped"] is False
    assert info["reason"] == "within cap"
    assert info["fraction_before"] == 0.05


def test_over_cap_clamps_with_reason():
    # 13% of 100 = 13 records; cap 10% → keep 10
    kept, info = clamp_contrast_to_fraction(_recs(13), base_lines=100, max_fraction=0.10)
    assert len(kept) == 10
    assert info["clamped"] is True
    assert info["emitted_before"] == 13
    assert info["emitted_after"] == 10
    assert info["dropped"] == 3
    assert info["fraction_after"] == 0.10
    assert "exceeded max_fraction" in info["reason"]
    # deterministic: first N in emission order
    assert [r["i"] for r in kept] == list(range(10))


def test_zero_base_left_unclamped():
    kept, info = clamp_contrast_to_fraction(_recs(3), base_lines=0, max_fraction=0.10)
    assert len(kept) == 3
    assert info["clamped"] is False


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} CONTRAST-CLAMP TESTS PASS")
