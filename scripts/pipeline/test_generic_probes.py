"""G3 gate — generic_probes builds forced-choice items over ARBITRARY triples (question from rel_q, correct
object + same-predicate distractors), domain-agnostic and deterministic. Fixture-based. __main__ runner.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_generic_probes.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pipeline.generic_expand import load_all_triples
from pipeline.generic_probes import build_probes


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
    assert all(i["predicate"] == "originates_from" for i in a)   # 'leads' has no triples → no items


if __name__ == "__main__":
    test_forced_choice_items_built()
    test_deterministic_and_skips_no_distractor_or_template()
    print("OK — G3: generic_probes builds forced-choice items over arbitrary triples, domain-agnostic")
