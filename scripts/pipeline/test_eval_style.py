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


def test_voice_reveal_receipt_is_measured_and_self_consistent():
    receipt = ES.voice_reveal_receipt(
        domain="voice-demo",
        pairs=[
            {"id": "p1", "prompt": "Cancel", "base": "A", "persona": "B", "judge_winner": "persona"},
            {"id": "p2", "prompt": "Celebrate", "base": "C", "persona": "D", "judge_winner": "base"},
            {"id": "p3", "prompt": "Ask", "base": "E", "persona": "F", "judge_winner": "persona"},
        ],
        judge_calibration_stamp="judge:test-v1",
        demo=True,
    )
    assert receipt["schema"] == "kandi.voice_reveal.v1"
    assert receipt["measured"]["persona_wins"] == 2
    assert receipt["measured"]["total"] == 3
    assert receipt["demo"] is True


def test_voice_reveal_refuses_wrong_pair_count_and_missing_stamp():
    pair = {"id": "p1", "prompt": "Cancel", "base": "A", "persona": "B", "judge_winner": "persona"}
    try:
        ES.voice_reveal_receipt(domain="voice", pairs=[pair], judge_calibration_stamp="judge:v1")
        raise AssertionError("expected refusal")
    except ValueError as error:
        assert "exactly three" in str(error)
    try:
        ES.voice_reveal_receipt(domain="voice", pairs=[pair, {**pair, "id": "p2"}, {**pair, "id": "p3"}],
                                judge_calibration_stamp="")
        raise AssertionError("expected refusal")
    except ValueError as error:
        assert "calibration" in str(error)


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} EVAL-STYLE TESTS PASS")
