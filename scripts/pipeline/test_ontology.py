"""Parity de-risk for the spec-driven extraction ontology (pipeline/ontology.py).

Run:  PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_ontology.py

The qgre-cpt refactor moved the extraction enums + system prompt out of the hardcoded
`pipeline/structured_prompts.py` constants and into the domain spec (configs/domains/edc.yaml). This test
proves that move is a NO-OP for EDC: `Ontology("edc")` reproduces every hardcoded value byte-for-byte, so
re-extraction behaves identically. (The predicate→object contract fix is a SEPARATE, deliberate change on
top — it does not run through this parity path.)
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ontology import Ontology          # noqa: E402
from pipeline import structured_prompts as SP    # noqa: E402

O = Ontology("edc")


def test_system_prompt_byte_identical():
    assert O.system() == SP.SYSTEM


def test_entity_types_verbatim():
    assert O.entity_types == SP.ENTITY_TYPES


def test_predicate_enum_order_and_count():
    # the prompt's predicate enum must be ALL predicates in schema order (not just the contracted subset).
    # 34 = the 33 schema predicates + in_lineup (plays_at's coarse fallback, B2).
    assert O.predicates == SP.PREDICATES
    assert len(O.predicates) == 34
    assert "in_lineup" in O.predicates


def test_topics_and_stances_verbatim():
    assert O.topics == SP.TOPICS
    assert O.stances == SP.STANCES


def test_confidence_map_verbatim():
    assert O.confidence_map == SP.CONFIDENCE


def test_user_prompt_byte_identical():
    a = O.user_prompt("edc_las_vegas_2026", "Lineup", "Body text here.")
    b = SP.user_prompt("edc_las_vegas_2026", "Lineup", "Body text here.")
    assert a == b


def test_inject_provenance_identical_shape():
    meta = {"domain_class": "official-festival", "url": "https://insomniac.com/x",
            "fetched_at": "2026-01-01", "doc_id": "official/x.md", "doc_sha256": "a" * 64,
            "doc_body": "The lineup names Example Artist."}
    for kind in ("entity", "triple", "discourse"):
        rec = {"id": "x", "anchors": ["e"]}
        if kind == "triple":
            rec["quote"] = "The lineup names Example Artist."
        a = O.inject_provenance(copy.deepcopy(rec), kind, copy.deepcopy(meta), "edc_las_vegas_2026")
        b = SP.inject_provenance(copy.deepcopy(rec), kind, copy.deepcopy(meta), "edc_las_vegas_2026", {})
        # discourse gets a uuid id when missing; both set it deterministically off the SAME input id here
        assert a == b, f"{kind} provenance diverged"


def test_predicate_contract_exposed():
    # the contract (for gate_id_integrity) rides alongside the enum and keeps plays_at constrained
    assert set(O.contract["plays_at"]) == {"stage", "set", "venue"}
    assert O.contract["bounded_by"] == []   # unconstrained → gate skips


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} ONTOLOGY-PARITY TESTS PASS")
