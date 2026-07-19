#!/usr/bin/env python3
"""Red-green proofs for diversity metrics (CPU; no embeddings)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.diversity_lib as D  # noqa: E402


def test_distinct_n_detects_collapse():
    collapsed = [f"Giovanni leads Team Rocket ({i})." for i in range(40)]
    diverse = [
        "Giovanni leads Team Rocket.",
        "Who leads Team Rocket? Giovanni.",
        "Team Rocket's leader is Giovanni.",
        "The boss of Team Rocket: Giovanni.",
        "Giovanni is the head of Team Rocket.",
        "Leadership of Team Rocket belongs to Giovanni.",
        "Team Rocket answers to Giovanni.",
        "Giovanni runs the criminal group Team Rocket.",
        "The organization Team Rocket is led by Giovanni.",
        "Giovanni, commander of Team Rocket.",
    ]
    assert D.distinct_n(collapsed, 2) < 0.35
    assert D.distinct_n(diverse, 2) > 0.45


def test_collapsed_corpus_fails_report():
    records = D.collapsed_corpus("Bruno is a member of the Elite Four", n=40)
    report = D.diversity_report(records, min_distinct_2=0.35, max_nn_cosine_p90=0.90)
    assert report["passed"] is False
    assert report["severity"] == "fail"
    assert "collapsed_rel" in report["failed_predicates"]


def test_diverse_corpus_passes_report():
    texts = [
        "Misty is the Gym Leader of Cerulean City.",
        "Who runs the Cerulean Gym? Misty.",
        "Cerulean City's gym is led by Misty.",
        "Misty leads Cerulean's water-type gym.",
        "The Cerulean Gym Leader is Misty.",
        "In Kanto lore, Misty heads the Cerulean Gym.",
        "Misty, of Cerulean City, is its Gym Leader.",
        "Gym authority in Cerulean rests with Misty.",
    ]
    records = [
        {"text": t, "relation_type": "leads", "source_triple_ids": [f"fact:{i%3}"]}
        for i, t in enumerate(texts)
    ]
    report = D.diversity_report(records, min_distinct_2=0.30)
    assert report["passed"] is True
    assert report["total_effective_forms"] >= 3


def test_pick_diverse_respects_floor():
    texts = ["a b c d", "a b c d e", "x y z w", "x y z q"]
    picked = D.pick_diverse_representatives(texts, budget=2)
    assert len(picked) == 2
    # floor: never empty
    assert D.pick_diverse_representatives(texts, budget=1)
    assert D.pick_diverse_representatives(["only"], budget=5) == [0]


def test_effective_forms_clusters_near_dupes():
    texts = [
        "Pikachu evolves into Raichu.",
        "Pikachu evolves into Raichu!",
        "Who does Pikachu become? Raichu.",
    ]
    forms = D.effective_forms(texts, threshold=0.55)
    assert 1 <= forms <= 3


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} DIVERSITY-LIB TESTS PASS")
