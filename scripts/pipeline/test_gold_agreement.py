#!/usr/bin/env python3
"""Red-green proofs for two-model gold agreement."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.generic_probes import two_model_gold_filter  # noqa: E402


def test_intersection_kept_disagreement_queued():
    gold = [
        {"subject_id": "a", "predicate": "p", "object": "1"},
        {"subject_id": "b", "predicate": "p", "object": "2"},
        {"subject_id": "c", "predicate": "p", "object": "3"},
    ]
    second = [
        {"subject_id": "a", "predicate": "p", "object": "1"},
        {"subject_id": "c", "predicate": "p", "object": "3"},
        {"subject_id": "d", "predicate": "p", "object": "4"},  # extra — does not add to gold
    ]
    kept, disagreed = two_model_gold_filter(gold, second)
    assert len(kept) == 2
    assert {t["subject_id"] for t in kept} == {"a", "c"}
    assert len(disagreed) == 1
    assert disagreed[0]["subject_id"] == "b"


def test_empty_second_pass_drops_all():
    gold = [{"subject_id": "a", "predicate": "p", "object": "1"}]
    kept, disagreed = two_model_gold_filter(gold, [])
    assert kept == []
    assert len(disagreed) == 1


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} GOLD-AGREEMENT TESTS PASS")
