"""Regression test for the deterministic predicate→object contract enforcement (B2 — the safety net under
the prompt rule that fixes plays_at→festival, defect #3 from the corpus audit).

Run:  PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_contract_enforce.py

The enforcement (extract_structured._enforce_contract) guarantees the structured layer obeys the spec
contract regardless of LLM compliance: plays_at→festival/edition is coarsened to in_lineup (a TRUE,
weaker fact — never dropped), other contract violations are rejected, valid/unconstrained triples pass.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_structured as E          # noqa: E402
from pipeline.ontology import Ontology   # noqa: E402

E.ONT = Ontology("edc")
PFX = "edc_las_vegas_2018"


def _run(pred, obj):
    return E._enforce_contract({"predicate": pred, "object": obj}, Counter())


def test_plays_at_festival_remaps_to_in_lineup():
    rec, drop = _run("plays_at", f"{PFX}:festival:{PFX}")
    assert not drop and rec["predicate"] == "in_lineup"   # the safe coarsening — the fact survives, valid


def test_plays_at_edition_remaps_to_in_lineup():
    rec, drop = _run("plays_at", f"{PFX}:edition:2018")
    assert not drop and rec["predicate"] == "in_lineup"


def test_plays_at_stage_unchanged():
    rec, drop = _run("plays_at", f"{PFX}:stage:kinetic_field")
    assert not drop and rec["predicate"] == "plays_at"


def test_plays_at_artist_is_dropped():
    # a hard violation (artist is neither a valid stage object NOR the safe festival coarsening) → reject
    _, drop = _run("plays_at", f"{PFX}:artist:zedd")
    assert drop


def test_literal_predicate_untouched():
    _, drop = _run("serves", "free water stations")
    assert not drop


def test_unconstrained_predicate_untouched():
    _, drop = _run("bounded_by", f"{PFX}:landmark:gate_a")
    assert not drop


def test_in_lineup_to_festival_is_contract_valid():
    # the remap target must itself satisfy the contract (else we'd just move the violation)
    _, drop = _run("in_lineup", f"{PFX}:festival:{PFX}")
    assert not drop


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} CONTRACT-ENFORCEMENT TESTS PASS")
