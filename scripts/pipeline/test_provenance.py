"""CPU-only provenance regression: extraction injection and every generic hydrated writer.

No LLM/API call is made. A fake extraction response is passed through extract_structured's real
post-processing/writer seam, then its triple is loaded and rendered by expand, synth, and contrast.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_provenance.py
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import extract_structured as X
from pipeline.generic_contrast import contrast_records
from pipeline.generic_expand import expand_records, load_all_triples
from pipeline.generic_synth import synth_records
from pipeline.ontology import Ontology
from pipeline.schema_validate import validate


def _write_fake_extraction(root: Path) -> tuple[Path, str, str]:
    clean_root = root / "10_clean"
    md = clean_root / "wiki" / "mini.md"
    md.parent.mkdir(parents=True)
    md.write_text(
        "# Mini source\n\nSynthetic provenance fixture.\n\n---\n"
        "Red opposes Team Rocket. Blue mentors Red.\n",
        encoding="utf-8",
    )
    context = X.document_context(md, clean_root)
    exact_quote = "Red opposes Team Rocket."
    absent_quote = "This sentence is absent from the clean document."

    out_dir = root / "written"
    out_dir.mkdir()
    paths = {"fixture": (
        out_dir / "entities.jsonl",
        out_dir / "triples.jsonl",
        out_dir / "discourse.jsonl",
        out_dir / "rejects.jsonl",
    )}
    item = {
        "slug": "fixture",
        "festival_slug": "pokemon_lore_kanto",
        "meta": {
            "domain_class": "wiki-bulbapedia",
            "url": "https://example.test/mini",
            "final_url": "https://example.test/mini",
            "fetched_at": "2026-07-13T00:00:00Z",
            **context,
        },
    }
    response = {"entities": [], "discourse": [], "triples": [
        {"subject_id": "pokemon_lore_kanto:character:red", "predicate": "opposes",
         "object": "pokemon_lore_kanto:organization:team_rocket", "attrs": {}, "quote": exact_quote},
        {"subject_id": "pokemon_lore_kanto:character:blue", "predicate": "opposes",
         "object": "pokemon_lore_kanto:organization:team_rocket", "attrs": {}, "quote": absent_quote},
    ]}
    X.ONT = Ontology("pokemon-lore")
    X._write_records(response, item, paths, Counter())

    records = [json.loads(line) for line in paths["fixture"][1].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    body = X.body_of(md.read_text(encoding="utf-8"))
    expected_start = body.index(exact_quote)
    exact, absent = records
    for rec in records:
        provenance = rec["provenance"]
        assert provenance["doc_id"] == "wiki/mini.md"
        assert provenance["doc_sha256"] == context["doc_sha256"]
        assert not validate("triple", rec, domain="pokemon-lore")
        assert "quote" not in rec                         # model-only field moved into provenance
    assert exact["provenance"]["quote"] == exact_quote
    assert exact["provenance"]["span"] == {
        "start": expected_start, "end": expected_start + len(exact_quote)
    }
    assert absent["provenance"]["quote"] == absent_quote
    assert absent["provenance"]["span"] is None          # exact-match failure never invents offsets
    return out_dir, context["doc_id"], context["doc_sha256"]


def test_injection_and_generic_propagation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        written, doc_id, doc_sha256 = _write_fake_extraction(root)
        edition = root / "unit"
        struct = edition / "15_structured"
        struct.mkdir(parents=True)
        (struct / "entities.jsonl").write_text(
            json.dumps({"id": "pokemon_lore_kanto:character:red", "name_canonical": "Red"}) + "\n" +
            json.dumps({"id": "pokemon_lore_kanto:organization:team_rocket",
                        "name_canonical": "Team Rocket"}) + "\n",
            encoding="utf-8",
        )
        first_triple = written.joinpath("triples.jsonl").read_text(encoding="utf-8").splitlines()[0]
        (struct / "triples.jsonl").write_text(first_triple + "\n", encoding="utf-8")
        triples = load_all_triples(SimpleNamespace(dir=edition, slug="fixture"))
        assert triples[0]["provenance"]["doc_id"] == doc_id
        assert triples[0]["provenance"]["doc_sha256"] == doc_sha256

        templates = {"opposes": ["{subject} opposes {object}."]}
        expanded = expand_records(triples, templates, "fixture", domain="pokemon-lore")
        synthesized = synth_records(
            triples, templates, "", "fixture", 1,
            generate_fn=lambda _system, _user: "Red remains firmly opposed to Team Rocket.",
            domain="pokemon-lore",
        )
        contrasted, truth_drops, _ = contrast_records(
            triples,
            {"teams": ["Team Rocket", "Team Aqua"]},
            {"opposes": {"group": "teams", "templates": ["{subject} opposes {object}, not {attractor}."]}},
            "fixture",
            "pokemon-lore",
        )
        assert truth_drops == 0 and contrasted
        for records in (expanded, synthesized, contrasted):
            assert records and records[0]["source_doc_ids"] == [doc_id]
            assert not validate("relation", records[0], domain="pokemon-lore")


if __name__ == "__main__":
    test_injection_and_generic_propagation()
    print("OK — exact/null spans and expand/synth/contrast source_doc_ids provenance")
