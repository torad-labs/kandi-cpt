#!/usr/bin/env python3
"""P0 red-greens: utilization-balanced allocation (SoG secondary sampling / controlled allocation).

Uniform rendering ties a fact's surface count to how many templates happen to fit its predicate,
which is a property of the template file rather than of the domain. Measured on pokemon-lore before
the change: template surfaces 2/fact median vs 21 max (10.5x); llm_synth 10 vs 70 (7.0x).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.generic_expand import balance_by_utilization  # noqa: E402


def _rec(fact: str, i: int, entities: list[str]) -> dict:
    return {
        "text": f"{fact} surface {i}",
        "relation_type": "rel",
        "source_triple_ids": [fact],
        "entity_ids": entities,
    }


def _two_faction_corpus():
    """One popular faction (20 surfaces/fact) and one long tail (2 surfaces/fact)."""
    records = []
    for fact_index in range(5):
        fact = f"popular:{fact_index}"
        for i in range(20):
            records.append(_rec(fact, i, ["Popular Subject", f"obj{fact_index}"]))
    for fact_index in range(5):
        fact = f"tail:{fact_index}"
        for i in range(2):
            records.append(_rec(fact, i, [f"Rare Subject {fact_index}", f"tobj{fact_index}"]))
    return records


def test_before_the_change_the_corpus_is_skewed():
    """Control: the raw corpus really is imbalanced, so a balanced result means something."""
    records = _two_faction_corpus()
    by_fact: dict[str, int] = {}
    for rec in records:
        by_fact[rec["source_triple_ids"][0]] = by_fact.get(rec["source_triple_ids"][0], 0) + 1
    counts = sorted(by_fact.values())
    assert counts[0] == 2 and counts[-1] == 20, counts
    assert counts[-1] / counts[0] == 10.0, "raw spread is 10x"


def test_balancing_bounds_the_per_fact_spread():
    kept, report = balance_by_utilization(_two_faction_corpus())
    by_fact: dict[str, int] = {}
    for rec in kept:
        by_fact[rec["source_triple_ids"][0]] = by_fact.get(rec["source_triple_ids"][0], 0) + 1
    counts = sorted(by_fact.values())
    assert counts[-1] <= report["quota"], f"no fact exceeds quota: {counts[-1]} > {report['quota']}"
    assert counts[-1] / counts[0] < 10.0, f"spread must shrink, got {counts[-1] / counts[0]}x"
    assert report["dropped"] > 0, "the popular faction must actually be trimmed"


def test_long_tail_facts_are_never_culled():
    """A fact at or below quota keeps everything — balancing must not starve the tail further."""
    kept, report = balance_by_utilization(_two_faction_corpus())
    tail = [r for r in kept if r["source_triple_ids"][0].startswith("tail:")]
    by_fact: dict[str, int] = {}
    for rec in tail:
        by_fact[rec["source_triple_ids"][0]] = by_fact.get(rec["source_triple_ids"][0], 0) + 1
    assert set(by_fact.values()) == {2}, by_fact
    assert report["quota"] >= 2


def test_quota_is_derived_not_hardcoded():
    """Doubling every fact's supply doubles the derived quota — it tracks the corpus."""
    small = balance_by_utilization([_rec("f", i, ["e"]) for i in range(4)])[1]["quota"]
    large = balance_by_utilization([_rec("f", i, ["e"]) for i in range(40)])[1]["quota"]
    assert large > small, (small, large)


def test_allocation_is_deterministic():
    first = [r["text"] for r in balance_by_utilization(_two_faction_corpus())[0]]
    second = [r["text"] for r in balance_by_utilization(_two_faction_corpus())[0]]
    assert first == second, "allocation must be reproducible across runs"


def test_entity_utilization_is_levelled():
    """The popular subject must stop soaking up surfaces relative to the rare ones."""
    records = _two_faction_corpus()
    before: dict[str, int] = {}
    for rec in records:
        for entity in rec["entity_ids"]:
            before[entity] = before.get(entity, 0) + 1
    kept, _ = balance_by_utilization(records)
    after: dict[str, int] = {}
    for rec in kept:
        for entity in rec["entity_ids"]:
            after[entity] = after.get(entity, 0) + 1
    ratio_before = before["Popular Subject"] / before["Rare Subject 0"]
    ratio_after = after["Popular Subject"] / after["Rare Subject 0"]
    assert ratio_after < ratio_before, (ratio_before, ratio_after)


def test_empty_input_is_safe():
    kept, report = balance_by_utilization([])
    assert kept == [] and report["facts"] == 0


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} UTILIZATION-BALANCE TESTS PASS")
