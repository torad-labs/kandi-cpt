"""Seam 1 byte-identity gate — gen_festival_meta scaffold is config-sourced, edc-identical.

The canonical stage taxonomy + day calendar moved from a hardcoded read of the LV-2026 gold meta into
configs/templates/edc.yaml `meta_scaffold` (via Domain.templates). This proves the lift is byte-identical
for edc (the hard gate) and that an unconfigured domain gets NO festival bleed. Deterministic — no LLM, no
GPU. Plain-assert + __main__ runner (no pytest in the venv), matching test_ontology.py.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_meta_general.py
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C
import gen_festival_meta as G
from pipeline.editions import iter_editions

GOLDEN = Path(__file__).parent / "_golden" / "meta_london_2013.json"
SAMPLE = "london/2013"   # a non-protected edition with its own structured stages (frozen pre-change)


def _edition(slug: str):
    eds = {e.slug: e for e in iter_editions("edc")}
    assert slug in eds, f"{slug} missing from editions on disk"
    return eds[slug]


def test_meta_byte_identity_edc():
    """build_meta from the config-sourced scaffold == the frozen pre-change golden, byte-for-byte."""
    cs, dd, vbc = G.load_scaffold("edc")
    got = G.build_meta(_edition(SAMPLE), cs, dd, vbc)
    golden = json.loads(GOLDEN.read_text())
    assert got == golden, "build_meta diverged from the golden — the scaffold lift is NOT byte-identical"


def test_config_scaffold_matches_gold_meta_stages():
    """The lifted canonical_stages still equals the LV-2026 gold meta's `stages` (no silent drift)."""
    cs, _, _ = G.load_scaffold("edc")
    gold = json.loads((C.ROOT / "data/edc/las_vegas/2026/festival_meta.json").read_text())
    assert cs == gold["stages"], "canonical_stages drifted from the gold-meta taxonomy"


def test_empty_scaffold_no_festival_bleed():
    """With an empty scaffold (a domain that declares none), build_meta emits ONLY the edition's own
    found stages and NO day calendar — proving canonical stages come solely from config, not the code."""
    ed = _edition(SAMPLE)
    found = set()
    for line in (ed.dir / "15_structured/triples.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            obj = str(json.loads(line).get("object", ""))
            if ":stage:" in obj:
                found.add(obj.split(":stage:")[1])
    meta = G.build_meta(ed, {}, {}, {})
    assert set(meta["stages"]) == found, "empty canonical scaffold leaked stages it didn't find (EDC bleed)"
    assert meta["days"] == {}, "empty default_days still produced a calendar"


if __name__ == "__main__":
    test_meta_byte_identity_edc()
    test_config_scaffold_matches_gold_meta_stages()
    test_empty_scaffold_no_festival_bleed()
    print("OK — Seam 1: gen_festival_meta scaffold lift is edc byte-identical + no festival bleed")
