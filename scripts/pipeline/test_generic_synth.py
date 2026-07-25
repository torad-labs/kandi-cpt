"""F5 gate — generic_synth (the LLM-diversity layer) builds correct 25_hydrated records from a triple + an
INJECTED generator (no API call). Domain-agnostic, neutral fixture. The real OpenRouter call is the same
proven pattern as synth_corpus; here we test the skeleton's logic (prompt assembly, record shape, marker
stripping) deterministically. __main__ runner.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_generic_synth.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pipeline.generic_expand import load_all_triples
from pipeline.generic_synth import synth_records


def _build(d: str):
    struct = Path(d) / "15_structured"
    struct.mkdir(parents=True)
    (struct / "entities.jsonl").write_text(
        json.dumps({"id": "x:org:acme", "name_canonical": "Acme Corp"}) + "\n"
        + json.dumps({"id": "x:place:berlin", "name_canonical": "Berlin"}) + "\n", encoding="utf-8")
    (struct / "triples.jsonl").write_text(
        json.dumps({"subject_id": "x:org:acme", "predicate": "headquartered_in",
                    "object": "x:place:berlin", "attrs": {},
                    "provenance": {"doc_id": "registry/acme.md"}}) + "\n", encoding="utf-8")
    return SimpleNamespace(dir=Path(d), slug="u")


def test_synth_records_uses_injected_generator_and_builds_records():
    captured = {}

    def fake_gen(system, user):
        captured["system"] = system
        return ("Acme Corp is based in Berlin.\n"
                "Q: Where is Acme Corp HQ'd? A: Berlin.\n"
                "- Berlin is home to Acme Corp.")   # a list-marker line → stripped

    with tempfile.TemporaryDirectory() as d:
        ed = _build(d)
        ptmpls = {"headquartered_in": ["{subject} is headquartered in {object}."]}
        recs = synth_records(load_all_triples(ed), ptmpls, "PREAMBLE.", "the registry", 3, generate_fn=fake_gen)

    # the canonical fact (predicate_templates[0]) + the domain preamble are in the prompt the model saw
    assert "Acme Corp is headquartered in Berlin." in captured["system"]
    assert captured["system"].startswith("PREAMBLE.")
    texts = [r["text"] for r in recs]
    assert "Acme Corp is based in Berlin." in texts
    assert "Q: Where is Acme Corp HQ'd? A: Berlin." in texts
    assert "Berlin is home to Acme Corp." in texts          # the "- " bullet marker was stripped
    r0 = recs[0]
    assert r0["relation_type"] == "headquartered_in" and r0["surface_form"] == "llm_synth"
    assert r0["entity_ids"] == ["Acme Corp", "Berlin"] and r0["split"] == "cpt"
    assert len(r0["source_triple_ids"]) == 1 and len(r0["source_triple_ids"][0]) == 16
    assert r0["source_doc_ids"] == ["registry/acme.md"]


def test_no_template_predicate_skipped_without_calling_generator():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        ed = _build(d)
        recs = synth_records(load_all_triples(ed), {}, "", "u", 3, generate_fn=lambda s, u: calls.append(1) or "x")
    assert recs == [] and calls == []   # no predicate template → nothing synthesized, no API call attempted




def test_duplicate_triples_cost_one_call_not_many():
    """15_structured can hold the same triple twice; each copy used to buy its own n variations.

    Measured on pokemon-lore/kanto: 869 templated triples, 736 distinct, 68 ids duplicated for 133
    redundant calls — the provider paid twice for one fact AND that fact got 2-7x its neighbours'
    exposure. That is the utilization imbalance in the LLM channel: duplicated INPUT, not policy.
    """
    triple = {"subject": "Giovanni", "predicate": "leads", "object": "Team Rocket", "attrs": {}}
    triples = [dict(triple) for _ in range(5)]          # the same fact five times
    ptmpls = {"leads": ["{subject} leads {object}."]}
    calls = {"n": 0}

    def stub(system, user, **kw):
        calls["n"] += 1
        return "\n".join(f"- variation {i} of that one fact" for i in range(3))

    records = synth_records(triples, ptmpls, "", "unit", 3, generate_fn=stub, domain=None)
    assert calls["n"] == 1, f"one distinct fact must cost ONE call, got {calls['n']}"
    per_fact = len({r["text"] for r in records})
    assert per_fact <= 3, f"exposure must not multiply with duplicates: {per_fact}"


def test_distinct_triples_still_each_get_a_call():
    """The dedupe must key on triple identity, not collapse genuinely different facts."""
    ptmpls = {"leads": ["{subject} leads {object}."]}
    triples = [
        {"subject": "Giovanni", "predicate": "leads", "object": "Team Rocket", "attrs": {}},
        {"subject": "Lance", "predicate": "leads", "object": "Elite Four", "attrs": {}},
    ]
    calls = {"n": 0}

    def stub(system, user, **kw):
        calls["n"] += 1
        return "- a variation of the fact here"

    synth_records(triples, ptmpls, "", "unit", 1, generate_fn=stub, domain=None)
    assert calls["n"] == 2, calls["n"]


if __name__ == "__main__":
    # Auto-discover so an appended test can never again be defined-but-unrun.
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nOK — F5: generic_synth, {len(tests)} tests")
