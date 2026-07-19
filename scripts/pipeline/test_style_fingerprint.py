#!/usr/bin/env python3
"""Red-green proofs for style fingerprint + minimum refusal."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.style_fingerprint as SF  # noqa: E402


def test_mechanical_stats_detect_habits():
    docs = [
        {"text": "hey so basically i just think this is fine lol", "path": "a", "unit": "u"},
        {"text": "anyway honestly not sure tbh. short stuff.", "path": "b", "unit": "u"},
        {"text": "like you know it is what it is ngl", "path": "c", "unit": "u"},
    ]
    stats = SF.mechanical_stats(docs)
    assert stats["n_docs"] == 3
    assert stats["n_tokens_est"] > 10
    assert "discourse_markers" in stats
    assert isinstance(stats["habits"], list)


def test_minimums_refuse_thin_corpus():
    stats = {"n_docs": 5, "n_tokens_est": 400}
    ok, reason = SF.check_minimums(stats, min_tokens=20_000, min_docs=50)
    assert ok is False
    assert "STYLE_MINIMUM_NOT_MET" in reason


def test_minimums_pass_large_corpus():
    stats = {"n_docs": 100, "n_tokens_est": 50_000}
    ok, reason = SF.check_minimums(stats, min_tokens=20_000, min_docs=50)
    assert ok is True


def test_style_card_renders():
    stats = SF.mechanical_stats([
        {"text": "Hello world! This is a longer sentence with punctuation?", "path": "x", "unit": "u"}
    ])
    card = SF.render_style_card("demo", stats)
    assert "Style card" in card
    assert "Checkable habits" in card
    assert "Approval" in card


def test_build_fingerprint_refuse_empty(tmp_path_factory=None):
    # no docs on a fake domain name → refuse
    result = SF.build_fingerprint("__no_such_style_domain__", min_tokens=100, min_docs=2)
    assert result["status"] == "REFUSE"
    assert result["stats"]["n_docs"] == 0


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} STYLE-FINGERPRINT TESTS PASS")
