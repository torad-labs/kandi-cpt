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


# ── b2b artist-name normalization: the hydrator emits b2b_pairing by splitting on " B2B "; a mashed
#    separator from the source/extractor makes the pairing invisible (84 of 92 billings corpus-wide).
def test_b2b_mashed_gets_spaced():
    # real corpus shapes: lowercase-mashed, uppercase-no-space, inside-parens
    assert E._normalize_b2b_name("Yaatb2bIltik") == "Yaat B2B Iltik"
    assert E._normalize_b2b_name("AbanaB2BD. Zeledon") == "Abana B2B D. Zeledon"
    assert E._normalize_b2b_name("Indo Warehouse (Kahanib2bKunal Merchant)") == "Indo Warehouse (Kahani B2B Kunal Merchant)"


def test_b2b_three_way_chain():
    assert E._normalize_b2b_name("Audiofreqb2bCode Blackb2bToneshifterz") == "Audiofreq B2B Code Black B2B Toneshifterz"


def test_b2b_already_clean_is_idempotent():
    for clean in ("GRiZ B2B Wooli", "Audiofreq B2B Code Black B2B Toneshifterz"):
        assert E._normalize_b2b_name(clean) == clean


def test_non_b2b_name_untouched():
    for n in ("Martin Garrix", "Above & Beyond", "deadmau5"):
        assert E._normalize_b2b_name(n) == n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} CONTRACT-ENFORCEMENT TESTS PASS")
