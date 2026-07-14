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


if __name__ == "__main__":
    test_synth_records_uses_injected_generator_and_builds_records()
    test_no_template_predicate_skipped_without_calling_generator()
    print("OK — F5: generic_synth builds LLM-diversity records via an injected generator, domain-agnostic")
