"""G3 gate — generic_probes builds forced-choice items over ARBITRARY triples (question from rel_q, correct
object + same-predicate distractors), domain-agnostic and deterministic. Fixture-based. __main__ runner.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_generic_probes.py
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pipeline.generic_expand import load_all_triples
from pipeline.generic_probes import build_probes, question_surfaces
from pipeline.split import triple_id


def _build(d: str):
    struct = Path(d) / "15_structured"
    struct.mkdir(parents=True)
    ents = [{"id": "x:character:ash", "name_canonical": "Ash"}, {"id": "x:character:gary", "name_canonical": "Gary"},
            {"id": "x:character:may", "name_canonical": "May"}, {"id": "x:region:kanto", "name_canonical": "Kanto"},
            {"id": "x:region:johto", "name_canonical": "Johto"}, {"id": "x:region:hoenn", "name_canonical": "Hoenn"}]
    tris = [{"subject_id": "x:character:ash", "predicate": "originates_from", "object": "x:region:kanto", "attrs": {}},
            {"subject_id": "x:character:gary", "predicate": "originates_from", "object": "x:region:johto", "attrs": {}},
            {"subject_id": "x:character:may", "predicate": "originates_from", "object": "x:region:hoenn", "attrs": {}}]
    (struct / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in ents) + "\n", encoding="utf-8")
    (struct / "triples.jsonl").write_text("\n".join(json.dumps(t) for t in tris) + "\n", encoding="utf-8")
    return SimpleNamespace(dir=Path(d), slug="u")


def test_forced_choice_items_built():
    rel_q = {"originates_from": ["Which region is {s} from?", "{s} originates from {o}."]}
    with tempfile.TemporaryDirectory() as d:
        items = build_probes(load_all_triples(_build(d)), rel_q, "the corpus", n_distractors=3, seed=0)
    assert len(items) == 3                               # all three have a distractor pool
    ash = next(i for i in items if i["subject"] == "Ash")
    assert ash["question"] == "Which region is Ash from?"
    assert ash["answer"] == "Kanto" and ash["answer"] in ash["choices"]      # correct answer among the choices
    assert set(ash["choices"]) <= {"Kanto", "Johto", "Hoenn"}                # distractors are same-type (regions)
    assert ash["predicate"] == "originates_from" and ash["split"] == "held_out"


def test_deterministic_and_skips_no_distractor_or_template():
    rel_q = {"originates_from": ["Which region is {s} from?", "{s} originates from {o}."],
             "leads": ["Which org does {s} lead?", "{s} leads {o}."]}
    with tempfile.TemporaryDirectory() as d:
        tr = load_all_triples(_build(d))
        a = build_probes(tr, rel_q, "u", 3, seed=7)
        b = build_probes(tr, rel_q, "u", 3, seed=7)
    assert a == b                                        # deterministic given the seed
    assert {i["source_triple_id"] for i in a} == {triple_id(t) for t in tr}
    assert all(i["predicate"] == "originates_from" for i in a)   # 'leads' has no triples → no items


def test_duplicate_semantic_triples_count_once():
    rel_q = {"located_in": ["Where is {s}?", "{s} is in {o}."]}
    triples = [
        {"subject_id": "place:a", "subject": "A", "predicate": "located_in", "object": "Region A"},
        {"subject_id": "place:b", "subject": "B", "predicate": "located_in", "object": "Region B"},
        {"subject_id": "place:c", "subject": "C", "predicate": "located_in", "object": "Region C"},
    ]
    items = build_probes(triples + [dict(triples[0])], rel_q, "u", 3, seed=0)
    assert len(items) == 3
    assert len({item["source_triple_id"] for item in items}) == 3


def test_multiple_surfaces_expand_items_without_expanding_gold():
    rel_q = {"located_in": ["Where is {s}?", "{s} is in {o}."]}
    triples = [
        {"subject_id": f"place:{name}", "subject": name, "predicate": "located_in", "object": f"Region {name}"}
        for name in ("A", "B", "C")
    ]
    items = build_probes(triples, rel_q, "u", 2, seed=0, surfaces_per_gold=4)
    assert len(items) == 12
    assert len({item["source_triple_id"] for item in items}) == 3
    assert len({item["probe_surface_id"] for item in items}) == 12
    assert {item["surface_variant"] for item in items} == {0, 1, 2, 3}
    assert question_surfaces("Where is A?", 4)[0] == "Where is A?"


def test_multi_valued_truths_never_become_distractors():
    rel_q = {"member_of": ["Which group includes {s}?", "{s} belongs to {o}."]}
    triples = [
        {"subject_id": "x:person:alex", "subject": "Alex", "predicate": "member_of", "object": "Alpha"},
        {"subject_id": "x:person:alex", "subject": "Alex", "predicate": "member_of", "object": "Beta"},
        {"subject_id": "x:person:blair", "subject": "Blair", "predicate": "member_of", "object": "Gamma"},
    ]
    items = build_probes(triples, rel_q, "u", 3, seed=0)
    alex = [i for i in items if i["subject"] == "Alex"]
    assert len(alex) == 2
    assert all(set(i["choices"]) <= {i["answer"], "Gamma"} for i in alex)
    assert all(not ({"Alpha", "Beta"} - {i["answer"]}) & set(i["choices"]) for i in alex)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exhausted = build_probes(triples[:2], rel_q, "u", 3, seed=0)
    assert exhausted == []
    assert "dropped 2 probe(s) with an empty distractor pool" in output.getvalue()


def test_one_entity_under_two_ids_never_leaks_a_truth_as_distractor():
    """Same rendered name, two subject_ids: a truth recorded under id B is still a TRUE answer to a
    question asked about the NAME, so it must never be offered as a distractor for a probe on id A.
    Measured live in edc/los_angeles/2007, where 4 shipped probes offered 'Exposition Park' and
    'Downtown L.A.' as wrong answers to a question they in fact answer correctly."""
    rel_q = {"located_at": ["Where is {s} located?", "{s} is located at {o}."]}
    triples = [
        {"subject_id": "edc:edition:2007_edition", "subject": "EDC 2007", "predicate": "located_at",
         "object": "Exposition Park & the Coliseum"},
        {"subject_id": "edc:edition:edc_2007", "subject": "EDC 2007", "predicate": "located_at",
         "object": "Exposition Park"},
        {"subject_id": "edc:edition:edc_2007", "subject": "EDC 2007", "predicate": "located_at",
         "object": "Downtown L.A."},
        {"subject_id": "edc:edition:2008", "subject": "EDC 2008", "predicate": "located_at",
         "object": "Las Vegas Motor Speedway"},
    ]
    items = build_probes(triples, rel_q, "u", 3, seed=0)
    truths = {"Exposition Park & the Coliseum", "Exposition Park", "Downtown L.A."}
    for item in (i for i in items if i["subject"] == "EDC 2007"):
        leaked = (set(item["choices"]) & truths) - {item["answer"]}
        assert not leaked, f"true answer(s) {sorted(leaked)} offered as distractors in {item['question']!r}"


def test_resolve_merged_ids_never_leak_a_truth_as_distractor():
    """The other equivalence: resolve merged two ids whose rendered names DIFFER ('Above & Beyond' vs
    'Above and Beyond'). Display-name grouping cannot see that merge — only the resolve id_map can, so
    both equivalences are required. Measured live in edc/orlando/2016, /2019 and edc/phuket/2025."""
    rel_q = {"in_lineup": ["Which festival did {s} play?", "{s} played {o}."]}
    triples = [
        {"subject_id": "x:artist:above_and_beyond", "subject": "Above & Beyond", "predicate": "in_lineup",
         "object": "EDC Orlando"},
        {"subject_id": "x:artist:above_beyond", "subject": "Above and Beyond", "predicate": "in_lineup",
         "object": "EDC Orlando 2019"},
        {"subject_id": "x:artist:tiesto", "subject": "Tiësto", "predicate": "in_lineup",
         "object": "EDC Las Vegas"},
    ]
    id_map = {"x:artist:above_beyond": "x:artist:above_and_beyond"}
    items = build_probes(triples, rel_q, "u", 3, seed=0, id_map=id_map)
    truths = {"EDC Orlando", "EDC Orlando 2019"}
    for item in (i for i in items if i["subject"].startswith("Above")):
        leaked = (set(item["choices"]) & truths) - {item["answer"]}
        assert not leaked, f"merged-id truth {sorted(leaked)} offered as distractor in {item['question']!r}"


def test_distinct_entities_keep_their_distractors():
    """The fix must not over-collapse: unrelated subjects with distinct names and no merge keep serving
    as each other's distractors, or the pool starves and fact axes go untested."""
    rel_q = {"located_in": ["Where is {s}?", "{s} is in {o}."]}
    triples = [
        {"subject_id": "p:a", "subject": "A", "predicate": "located_in", "object": "Region A"},
        {"subject_id": "p:b", "subject": "B", "predicate": "located_in", "object": "Region B"},
        {"subject_id": "p:c", "subject": "C", "predicate": "located_in", "object": "Region C"},
    ]
    items = build_probes(triples, rel_q, "u", 3, seed=0)
    assert len(items) == 3, "distinct entities must still yield probes"
    for item in items:
        assert len(item["choices"]) == 3, "each keeps both other regions as distractors"


if __name__ == "__main__":
    test_forced_choice_items_built()
    test_deterministic_and_skips_no_distractor_or_template()
    test_duplicate_semantic_triples_count_once()
    test_multiple_surfaces_expand_items_without_expanding_gold()
    test_multi_valued_truths_never_become_distractors()
    test_one_entity_under_two_ids_never_leaks_a_truth_as_distractor()
    test_resolve_merged_ids_never_leak_a_truth_as_distractor()
    test_distinct_entities_keep_their_distractors()
    print("OK — G3: generic_probes builds forced-choice items over arbitrary triples, domain-agnostic")
