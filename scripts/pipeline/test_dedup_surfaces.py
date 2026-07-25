#!/usr/bin/env python3
"""Red-green proofs for surface dedup floor protection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def test_surface_key_is_stable_across_processes():
    """The whole exclusion mechanism depends on writer and reader agreeing on this key.

    RED (pre-fix): the key used `hash(text)`, which Python salts per interpreter, so dedup wrote
    keys tokenize_corpus could never reproduce and every "excluded" surface stayed in the shard.
    Measured on edc/dallas/2011: 8063 stored exclusions, 8063 matching path:line prefixes, ZERO
    whole-key matches. An in-process test cannot catch this — the salt is fixed within one run —
    so the key must be recomputed in SEPARATE interpreters with different seeds.
    """
    import subprocess
    import sys as _sys

    snippet = (
        "import sys; sys.path.insert(0, 'scripts');"
        "from pipeline.dedup_surfaces import surface_key;"
        "print(surface_key('relations.jsonl', 7508, 'Kygo plays kineticFIELD at EDC 2026.'))"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [_sys.executable, "-c", snippet],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=60,
        )
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"key differs across PYTHONHASHSEED values: {seen}"
    assert seen.pop().startswith("relations.jsonl:7508:"), "key must keep its path:line prefix"


def test_surface_key_distinguishes_different_text_at_the_same_line():
    a = DS.surface_key("relations.jsonl", 12, "Kygo plays kineticFIELD.")
    b = DS.surface_key("relations.jsonl", 12, "Kygo plays circuitGROUNDS.")
    assert a != b, "same line, different text must not collide"
    assert DS.surface_key("relations.jsonl", 12, "x") == DS.surface_key("relations.jsonl", 12, "x")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} DEDUP-SURFACES TESTS PASS")
