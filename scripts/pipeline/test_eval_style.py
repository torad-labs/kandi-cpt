#!/usr/bin/env python3
"""Red-green proofs for style fidelity floor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_style as ES  # noqa: E402


def test_habit_no_emoji():
    assert ES.score_habit("no_emoji", "plain text only") is True
    assert ES.score_habit("no_emoji", "hello 😀") is False


def test_habit_exclamation():
    assert ES.score_habit("rarely_uses_exclamation", "calm sentence.") is True
    assert ES.score_habit("rarely_uses_exclamation", "wow!!! amazing!!!") is False


def test_fidelity_report_pass():
    habits = ["no_emoji", "rarely_uses_exclamation"]
    samples = ["just a normal note about the day.", "another quiet line here."]
    rep = ES.fidelity_report(habits, samples)
    assert rep["status"] == "PASS"
    assert rep["n_habits"] == 2


def test_fidelity_skip_without_habits():
    rep = ES.fidelity_report([], ["x"])
    assert rep["status"] == "SKIP"


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} EVAL-STYLE TESTS PASS")
