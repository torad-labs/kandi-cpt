#!/usr/bin/env python3
"""The four depths must stay separate, and the deep ones must not be answerable shallowly.

A "reasoning" item whose answer sits beside its subject in one training surface is a RECALL item
wearing a reasoning label. It inflates the reasoning score of a model that learned only the
surface — the same estimand failure as holdout leakage, one level up.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_knowledge_depth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import knowledge_depth as KD


def test_a_single_knowledge_score_is_refused() -> None:
    try:
        KD.overall({"recall": {"hits": 9, "n": 10, "acc": 0.9}})
    except KD.DepthError as error:
        assert "no single knowledge-installed score" in str(error)
    else:
        raise AssertionError("overall() produced a merged score")


def test_report_has_no_aggregate_field_to_quote() -> None:
    report = KD.depth_report({
        "recall": {"hits": 9, "n": 10, "acc": 0.9},
        "reasoning": {"hits": 2, "n": 10, "acc": 0.2},
    })
    assert set(report["depths"]) == set(KD.DEPTHS)
    assert report["depths"]["recall"]["acc"] == 0.9
    assert report["depths"]["reasoning"]["acc"] == 0.2
    # An unmeasured depth is marked, not zeroed and not dropped.
    assert report["depths"]["association"] == {"measured": False}
    for forbidden in ("overall", "combined", "score", "knowledge_installed", "mean"):
        assert forbidden not in report, forbidden
    assert not any(isinstance(v, (int, float)) for v in report.values()), report


def test_unknown_depths_fail_closed() -> None:
    try:
        KD.depth_report({"memorisation": {"hits": 1, "n": 2, "acc": 0.5}})
    except KD.DepthError as error:
        assert "unknown depth" in str(error)
    else:
        raise AssertionError("an invented depth was accepted")


def test_cooccurrence_leak_is_caught_for_deep_items_only() -> None:
    surfaces = [
        "Calem trains in Kanto and his rival is Serena.",     # subject + answer together
        "Brock runs the Pewter City gym.",
    ]
    items = [
        {"probe_surface_id": "leaky", "depth": "reasoning", "subject": "Calem", "answer": "Serena"},
        {"probe_surface_id": "clean", "depth": "reasoning", "subject": "Brock", "answer": "Serena"},
        # A recall item is SUPPOSED to be answerable from one surface — flagging it would be wrong.
        {"probe_surface_id": "recall", "depth": "recall", "subject": "Calem", "answer": "Serena"},
    ]
    leaks = KD.answer_cooccurrence_leaks(items, surfaces)
    assert [leak["probe_surface_id"] for leak in leaks] == ["leaky"], leaks


def test_shortcut_audit_fails_on_a_single_leak() -> None:
    """One leaked item is a reasoning score partly earned by recall."""
    surfaces = ["Calem trains in Kanto and his rival is Serena."]
    clean = [{"probe_surface_id": "a", "depth": "reasoning", "subject": "Brock", "answer": "Misty",
              "choices": ["Misty", "Serena"]}]
    leaky = [{"probe_surface_id": "b", "depth": "reasoning", "subject": "Calem", "answer": "Serena",
              "choices": ["Serena", "Misty"]}]
    assert KD.shortcut_audit(clean, surfaces)["verdict"] == "PASS"
    audit = KD.shortcut_audit(leaky, surfaces)
    assert audit["verdict"] == "FAIL"
    assert audit["checks"][0]["n_leaks"] == 1


def test_frequency_shortcut_is_detected() -> None:
    """If the answer is always the most familiar name, "pick the familiar one" scores for free."""
    surfaces = ["Serena " * 50, "Misty appears once."]
    items = [
        {"probe_surface_id": str(i), "depth": "reasoning", "subject": "Brock",
         "answer": "Serena", "choices": ["Serena", "Misty"]}
        for i in range(4)
    ]
    flagged = KD.distractor_frequency_balance(items, surfaces)
    assert len(flagged) == 4, flagged
    assert flagged[0]["answer_frequency"] > flagged[0]["runner_up_frequency"]
    assert KD.shortcut_audit(items, surfaces)["verdict"] == "FAIL"

    # Balanced frequencies must NOT be flagged, or the check is just noise.
    balanced_surfaces = ["Serena and Misty each appear here.", "Misty and Serena again."]
    assert KD.distractor_frequency_balance(items, balanced_surfaces) == []


def test_derivable_and_non_derivable_are_reported_apart() -> None:
    report = KD.depth_report(
        {"reasoning": {"hits": 5, "n": 10, "acc": 0.5}},
        derivable={"derivable": {"hits": 5, "n": 6, "acc": 0.8333},
                   "non_derivable": {"hits": 0, "n": 4, "acc": 0.0}},
    )
    split = report["derivable_split"]
    assert split["derivable"]["acc"] > split["non_derivable"]["acc"]
    assert "pooled" not in split and "total" not in split


def test_substring_matches_do_not_create_phantom_leaks() -> None:
    """Word-boundary matching: "Ser" inside "Serena" is not the answer "Ser"."""
    surfaces = ["Calem battles Serenade in the hall."]
    items = [{"probe_surface_id": "x", "depth": "reasoning", "subject": "Calem", "answer": "Serena"}]
    assert KD.answer_cooccurrence_leaks(items, surfaces) == []


if __name__ == "__main__":
    test_a_single_knowledge_score_is_refused()
    test_report_has_no_aggregate_field_to_quote()
    test_unknown_depths_fail_closed()
    test_cooccurrence_leak_is_caught_for_deep_items_only()
    test_shortcut_audit_fails_on_a_single_leak()
    test_frequency_shortcut_is_detected()
    test_derivable_and_non_derivable_are_reported_apart()
    test_substring_matches_do_not_create_phantom_leaks()
    print("OK — depths non-mergeable, shortcut leaks caught, balanced sets not flagged")
