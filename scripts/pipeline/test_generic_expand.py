"""F1 gate — the generic multi-relation hydrate generator renders per-predicate templates over ARBITRARY
triples (any predicate, entity-id or literal objects), domain-agnostic and non-festival. Fixture-based with
a deliberately NEUTRAL example (a company/person/place graph — NOT any real domain we ship; the point is
that generic_expand renders whatever triples + templates it is handed). Deterministic. __main__ runner.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_generic_expand.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pipeline.generic_expand import expand_records, load_all_triples


def _build(d: str):
    """A neutral fixture graph — no festival vocabulary, no shipped-domain content; just arbitrary triples."""
    struct = Path(d) / "15_structured"
    struct.mkdir(parents=True)
    (struct / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"id": "x:org:acme_corp", "name_canonical": "Acme Corp"},
        {"id": "x:place:berlin", "name_canonical": "Berlin"},
    ]) + "\n", encoding="utf-8")
    (struct / "triples.jsonl").write_text("\n".join(json.dumps(t) for t in [
        {"subject_id": "x:org:acme_corp", "predicate": "headquartered_in",
         "object": "x:place:berlin", "attrs": {"since": "1998"}},
        {"subject_id": "x:person:rui_santos", "predicate": "works_at",
         "object": "x:org:acme_corp", "attrs": {}},
    ]) + "\n", encoding="utf-8")
    return SimpleNamespace(dir=Path(d), slug="unit-1")


def test_load_all_triples_resolves_names_and_id_tail_fallback():
    with tempfile.TemporaryDirectory() as d:
        tr = load_all_triples(_build(d))
    assert len(tr) == 2
    assert tr[0]["subject"] == "Acme Corp" and tr[0]["object"] == "Berlin"
    assert tr[0]["predicate"] == "headquartered_in" and tr[0]["attrs"] == {"since": "1998"}
    # rui_santos has no entity record → display falls back to the id tail (no crash, no blank)
    assert tr[1]["subject"] == "rui santos" and tr[1]["object"] == "Acme Corp"


def test_expand_records_renders_per_predicate_and_carries_attrs():
    with tempfile.TemporaryDirectory() as d:
        ed = _build(d)
        ptmpls = {
            "headquartered_in": ["{subject} is headquartered in {object}.", "{subject} has been in {object} since {attr_since}."],
            "works_at": ["{subject} works at {object}."],
        }
        recs = expand_records(load_all_triples(ed), ptmpls, "the registry")
    texts = [r["text"] for r in recs]
    assert "Acme Corp is headquartered in Berlin." in texts
    assert "Acme Corp has been in Berlin since 1998." in texts          # {attr_since} threaded from the triple
    assert "rui santos works at Acme Corp." in texts
    r0 = recs[0]
    assert set(r0) >= {"text", "relation_type", "entity_ids", "source_triple_ids", "split"}
    assert r0["relation_type"] == "headquartered_in" and r0["split"] == "cpt"


def test_uncovered_predicate_skipped_not_crashed():
    with tempfile.TemporaryDirectory() as d:
        recs = expand_records(load_all_triples(_build(d)), {"headquartered_in": ["{subject} -> {object}"]}, "u")
    assert [r["relation_type"] for r in recs] == ["headquartered_in"]   # works_at has no template → skipped, no crash


if __name__ == "__main__":
    test_load_all_triples_resolves_names_and_id_tail_fallback()
    test_expand_records_renders_per_predicate_and_carries_attrs()
    test_uncovered_predicate_skipped_not_crashed()
    print("OK — F1: generic_expand renders per-predicate templates over arbitrary triples, domain-agnostic")
