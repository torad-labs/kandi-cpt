#!/usr/bin/env python3
"""Red-green proofs for surface dedup floor protection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.dedup_surfaces as DS  # noqa: E402


def _row(text, fact="f1", i=0):
    return {
        "text": text,
        "relation_type": "leads",
        "source_triple_ids": [fact],
        "surface_form": "llm_synth",
        "path": "relations_llm.jsonl",
        "line": i,
        "key": f"relations_llm.jsonl:{i}:{hash(text) & 0xFFFFFFFF:08x}",
        "fact": fact,
    }


def test_floor_never_empties_fact():
    rows = [_row(f"Giovanni leads Team Rocket ({i}).", i=i) for i in range(10)]
    # budget=2 is below floor(10)=5 — floor wins; still culls above floor
    result = DS.dedup_unit(rows, budget=2, cap=60)
    assert result["n_after"] >= 5, result
    assert not result["floor_violations"]
    assert result["n_excluded"] >= 1
    # generous budget still culls near-dupes when forms collapse
    result2 = DS.dedup_unit(rows, budget=40, cap=60)
    assert result2["n_after"] >= 5
    assert not result2["floor_violations"]


def test_rare_fact_keeps_both_similar():
    rows = [
        _row("Obscure fact A is true.", fact="rare", i=0),
        _row("Obscure fact A is true!", fact="rare", i=1),
    ]
    result = DS.dedup_unit(rows, budget=40, cap=60)
    assert result["n_after"] == 2
    assert result["n_excluded"] == 0


def test_exposure_floor_math():
    assert DS.exposure_floor(1, 60) == 1
    assert DS.exposure_floor(2, 60) == 2
    assert DS.exposure_floor(10, 60) == 5   # half of 10
    assert DS.exposure_floor(100, 60) == 50  # half of 100, under cap
    assert DS.exposure_floor(200, 60) == 60  # half would be 100 → capped at 60


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} DEDUP-SURFACES TESTS PASS")
