"""Seam 4 byte-identity gate — build_sft's POSITIVE Q/A templates are config-sourced, edc-identical.

VENUE_TEMPLATES (module global) + REL_Q (was function-local in edition_qas) moved into
configs/templates/edc.yaml `sft_templates`, threaded into edition_qas from main(). The anti-hallucination
NEGATIVE machinery (absent-edition / fake-city / fine-grained) is untouched — proven live recipe.

las_vegas/2026 exercises BOTH lifts (8 venue + all six REL_Q predicates), so the full-output golden is the
gate; Golden 2 is the structural drift guard. Deterministic — no model/GPU. Plain-assert + __main__ runner.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_sft_general.py
"""
from __future__ import annotations

import json
from pathlib import Path

import build_sft as BS
from pipeline.domain import Domain
from pipeline.editions import iter_editions

GOLD = Path(__file__).parent / "_golden"
SAMPLE = "las_vegas/2026"
REL_PREDS = {"located_at", "belongs_to_class", "serves", "requires", "practiced_by", "prohibits"}


def _edc_templates() -> tuple[list, dict]:
    """Load the positive templates exactly as build_sft.main() does."""
    t = (Domain("edc").templates or {}).get("sft_templates", {})
    return [tuple(x) for x in t.get("venue_templates", [])], {k: tuple(v) for k, v in t.get("rel_q", {}).items()}


def _edition(slug: str):
    eds = {e.slug: e for e in iter_editions("edc")}
    assert slug in eds, f"{slug} missing from editions on disk"
    return eds[slug]


def test_sft_qas_byte_identity():
    """edition_qas(las_vegas/2026) from config templates == frozen golden (covers venue + all six REL_Q)."""
    vt, rq = _edc_templates()
    got = BS.edition_qas(_edition(SAMPLE), 6, 8, vt, rq)
    golden = json.loads((GOLD / "sft_qas_las_vegas_2026.json").read_text())
    assert got == golden, "edition_qas diverged from golden — positive-template lift is NOT byte-identical"


def test_sft_templates_match_original():
    """The config sft_templates == the original hardcoded VENUE_TEMPLATES + REL_Q (structural drift guard)."""
    t = (Domain("edc").templates or {}).get("sft_templates", {})
    cfg = {"venue_templates": [list(x) for x in t.get("venue_templates", [])],
           "rel_q": {k: list(v) for k, v in t.get("rel_q", {}).items()}}
    orig = json.loads((GOLD / "sft_templates_original.json").read_text())
    assert cfg == orig, "sft_templates drifted from the original hardcoded shapes"


def test_empty_templates_no_bleed():
    """With no templates (a domain that declares none), edition_qas emits NO venue/relational QA — no bleed."""
    rels = {q["rel"] for q in BS.edition_qas(_edition(SAMPLE), 6, 8, [], {})}
    assert "venue" not in rels, "empty venue_templates still produced venue QA"
    assert not (rels & REL_PREDS), "empty rel_q still produced relational QA (EDC bleed)"


if __name__ == "__main__":
    test_sft_qas_byte_identity()
    test_sft_templates_match_original()
    test_empty_templates_no_bleed()
    print("OK — Seam 4: build_sft positive templates config-sourced, edc byte-identical + no festival bleed")
