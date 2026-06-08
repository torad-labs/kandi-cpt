"""Domain-general structured-validate: prove edc is byte-identical (the proven pipeline is untouched) and
that a new domain validates against ITS own ontology, sourced from the manifest.

Run: PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_schema_general.py
(plain-assert + __main__ runner, matching test_ontology.py — the venv has no pytest)
"""
from __future__ import annotations

import json

import config as C
from pipeline.schema_validate import _domain, _load, _schema, validate

_KINDS = ("entity", "triple", "discourse", "metadata")
_GOLD = C.ROOT / "data" / "edc" / "las_vegas" / "2026" / "15_structured"


def test_edc_patched_equals_static():
    """edc's manifest reproduces the static schemas byte-for-byte — generalizing changes NOTHING for edc."""
    edc = _domain("edc")
    for name in _KINDS:
        assert _schema(name, edc) == _load(name), f"{name}: edc-patched schema != static schema"


def test_edc_golden_validates_identically():
    """Every record in the proven edc corpus still validates clean under the domain-aware validator."""
    if not (_GOLD / "entities.jsonl").exists():
        print("    (SKIP — edc golden 15_structured corpus not in this checkout)")
        return
    for kind, fn in (("entity", "entities.jsonl"), ("triple", "triples.jsonl"), ("discourse", "discourse.jsonl")):
        path = _GOLD / fn
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            errs = validate(kind, rec, domain="edc")
            assert errs == [], f"edc {kind} {rec.get('id', '?')} now fails: {errs}"


def test_new_domain_uses_its_own_enum_not_festival():
    """A non-festival domain's schema carries ITS entity types/predicates; the static default stays festival."""
    poke_entity = _schema("entity", _domain("pokemon-lore"))
    poke_types = poke_entity["properties"]["entity_type"]["enum"]
    assert "character" in poke_types and "festival" not in poke_types
    assert "character" in poke_entity["properties"]["id"]["pattern"]  # id-pattern regenerated from the types
    poke_pred = _schema("triple", _domain("pokemon-lore"))["properties"]["predicate"]["enum"]
    assert "originates_from" in poke_pred and "plays_at" not in poke_pred
    static_types = _schema("entity", None)["properties"]["entity_type"]["enum"]
    assert "festival" in static_types and "character" not in static_types  # static path unchanged


def test_validate_routes_entity_type_by_domain():
    """The full validate() path: a pokemon 'character' id+type passes its domain, is rejected by the static one."""
    rec = {"id": "kanto:character:ash", "entity_type": "character", "name_canonical": "Ash Ketchum",
           "attributes": {}, "provenance": {}}
    poke_errs = " ".join(validate("entity", rec, domain="pokemon-lore"))
    static_errs = " ".join(validate("entity", rec, domain=None))
    assert "entity_type" in static_errs or "id" in static_errs  # festival default rejects the non-festival type/id
    assert "entity_type" not in poke_errs and "/id:" not in poke_errs  # its own domain accepts them


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} SCHEMA-GENERALITY TESTS PASS")
