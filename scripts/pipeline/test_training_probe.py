#!/usr/bin/env python3
"""Stopping must follow measured learning, and must not fire on noise.

CPT loss can fall monotonically while fact acquisition stalls and general ability degrades, so a
loss-plateau stop fires late on a run that stopped learning and never at all on one that is quietly
getting worse at everything except next-token prediction.

Every threshold here is eval_stats.rough_bound (0.5/√n), the bound this repo already uses, so a
"decline" that is inside sampling noise must NOT stop a paid run — a trigger-happy stop is as
expensive as a blind one.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_training_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training_probe as TP
from eval_stats import rough_bound
from eval_probes import summarize_hits


def _row(step: int, loss: float, **families) -> dict:
    scores = {
        name: {"hits": round(acc * n), "n": n, "acc": acc}
        for name, (acc, n) in families.items()
    }
    return TP.record(step=step, loss=loss, scores=scores)


def test_too_little_trajectory_never_stops_a_run() -> None:
    for history in ([], [_row(100, 2.0, install=(0.5, 400))]):
        verdict = TP.decide(history)
        assert verdict["stop"] is False
        assert any("two checkpoints" in r["rule"] for r in verdict["rules"])


def test_healthy_progress_does_not_stop() -> None:
    n = 400
    step = rough_bound(n) * 2
    history = [
        _row(100, 2.0, install=(0.40, n), confusable=(0.60, n), general=(0.70, n)),
        _row(200, 1.8, install=(0.40 + step, n), confusable=(0.60, n), general=(0.70, n)),
        _row(300, 1.6, install=(0.40 + 2 * step, n), confusable=(0.60, n), general=(0.70, n)),
    ]
    verdict = TP.decide(history, general_base_acc=0.70)
    assert verdict["stop"] is False, verdict["reasons"]


def test_install_strength_stalling_stops_the_run() -> None:
    n = 400
    history = [
        _row(100, 2.0, install=(0.60, n)),
        _row(200, 1.8, install=(0.601, n)),      # movement well inside the bound
        _row(300, 1.6, install=(0.602, n)),
    ]
    verdict = TP.decide(history)
    assert verdict["stop"] is True
    assert any("has not improved" in r for r in verdict["reasons"]), verdict["reasons"]


def test_the_divergence_the_research_names() -> None:
    """Loss improving while install strength declines — the case loss-watching cannot see."""
    n = 400
    drop = rough_bound(n) * 2
    history = [
        _row(100, 2.0, install=(0.70, n)),
        _row(200, 1.5, install=(0.70 - drop, n)),
    ]
    verdict = TP.decide(history)
    assert verdict["stop"] is True
    assert any("diverged" in r for r in verdict["reasons"]), verdict["reasons"]


def test_confusable_regression_stops_the_run() -> None:
    n = 400
    drop = rough_bound(n) * 2
    history = [
        _row(100, 2.0, install=(0.50, n), confusable=(0.80, n)),
        _row(200, 1.9, install=(0.90, n), confusable=(0.80 - drop, n)),
    ]
    verdict = TP.decide(history)
    assert verdict["stop"] is True
    assert any("confusable error increased" in r for r in verdict["reasons"]), verdict["reasons"]


def test_general_ability_is_compared_to_BASE_not_to_the_last_checkpoint() -> None:
    """Slow drift never trips a step-to-step check and still ends in a damaged model.

    Each hop here is inside the bound; the cumulative fall from base is not.
    """
    n = 400
    hop = rough_bound(n) * 0.4
    history = [
        _row(100, 2.0, install=(0.50, n), general=(0.90 - hop, n)),
        _row(200, 1.9, install=(0.90, n), general=(0.90 - 6 * hop, n)),
    ]
    assert TP.decide(history)["stop"] is False, "without a base there is nothing to regress against"

    verdict = TP.decide(history, general_base_acc=0.90)
    assert verdict["stop"] is True
    assert any("general ability regressed against base" in r for r in verdict["reasons"])


def test_noise_inside_the_bound_must_not_stop_a_paid_run() -> None:
    n = 400
    wobble = rough_bound(n) * 0.5
    history = [
        _row(100, 2.0, install=(0.50, n), confusable=(0.80, n), general=(0.90, n)),
        _row(200, 1.9, install=(0.50 - wobble, n), confusable=(0.80 - wobble, n),
             general=(0.90 - wobble, n)),
    ]
    verdict = TP.decide(history, general_base_acc=0.90)
    assert verdict["stop"] is False, verdict["reasons"]


def test_a_family_that_was_not_measured_is_not_treated_as_passing() -> None:
    """"We did not measure it" and "it is fine" are different claims."""
    n = 400
    history = [_row(100, 2.0, install=(0.50, n)), _row(200, 1.9, install=(0.90, n))]
    verdict = TP.decide(history)
    fired_rules = {r["rule"] for r in verdict["rules"]}
    assert not any("confusable" in rule for rule in fired_rules), fired_rules
    assert not any("general ability" in rule for rule in fired_rules), fired_rules


def test_previously_installed_facts_must_not_be_traded_away() -> None:
    """A booster that installs new knowledge by destroying old knowledge has not installed
    anything — it has traded. install strength rising while prior_install falls is that trade."""
    n = 400
    drop = rough_bound(n) * 2
    history = [
        _row(100, 2.0, install=(0.50, n), prior_install=(0.90, n)),
        _row(200, 1.9, install=(0.90, n), prior_install=(0.90 - drop, n)),
    ]
    verdict = TP.decide(history)
    assert verdict["stop"] is True
    assert any("previously installed facts regressed" in r for r in verdict["reasons"])

    # Holding steady must not fire, or the rule is noise.
    steady = [
        _row(100, 2.0, install=(0.50, n), prior_install=(0.90, n)),
        _row(200, 1.9, install=(0.90, n), prior_install=(0.90, n)),
    ]
    assert TP.decide(steady)["stop"] is False


def test_families_are_recorded_separately_and_never_merged() -> None:
    row = _row(100, 2.0, install=(0.5, 100), extraction=(0.4, 60), longtail=(0.2, 50),
               confusable=(0.9, 40), general=(0.7, 30), prior_install=(0.8, 45))
    assert set(TP.FAMILIES) == {"install", "extraction", "longtail", "confusable",
                                "general", "prior_install"}
    for family in TP.FAMILIES:
        assert family in row and set(row[family]) == {"hits", "n", "acc"}
    assert "score" not in row and "combined" not in row, (
        "a single merged number hides memorising versus generalising"
    )


def test_checkpoint_probe_summary_uses_independent_fact_counts() -> None:
    probes = [
        {
            "question": f"q{i}",
            "choices": ["a", "b"],
            "answer": "a",
            "predicate": "p",
            "source_triple_id": f"fact-{i // 4}",
        }
        for i in range(60)
    ]
    summary = summarize_hits(probes, [True] * 60, min_n=50, estimand="training_probe:test")
    row = TP.record(step=100, loss=1.0, scores={
        "install": {
            "hits": summary["independent_fact_hits"],
            "n": summary["n_independent_facts"],
            "acc": summary["independent_fact_acc"],
        },
    })
    assert row["install"]["n"] == 15
    assert rough_bound(row["install"]["n"]) > rough_bound(60)


if __name__ == "__main__":
    test_too_little_trajectory_never_stops_a_run()
    test_healthy_progress_does_not_stop()
    test_install_strength_stalling_stops_the_run()
    test_the_divergence_the_research_names()
    test_confusable_regression_stops_the_run()
    test_general_ability_is_compared_to_BASE_not_to_the_last_checkpoint()
    test_noise_inside_the_bound_must_not_stop_a_paid_run()
    test_a_family_that_was_not_measured_is_not_treated_as_passing()
    test_previously_installed_facts_must_not_be_traded_away()
    test_families_are_recorded_separately_and_never_merged()
    test_checkpoint_probe_summary_uses_independent_fact_counts()
    print("OK — stops on measured learning, holds through noise, never merges the families")
