"""Seam 2+3 byte-identity gate — the hydrate template SHAPES are config-sourced (configs/templates/edc.yaml
`hydrate_templates`) and render against a per-unit `Ctx`, edc byte-identical. Per-generator sha256 vs the
frozen golden manifest + the disjoint co-mention audit still EMPTY + the domain-general context_from_meta
path renders. Deterministic — no LLM/GPU. Plain-assert + __main__ runner.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_hydrate_general.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.editions import iter_editions
from pipeline import fact_expand as FE
from pipeline import contrasts as CO
from pipeline import disjoint_context as DJ

GOLD = json.loads((Path(__file__).parent / "_golden" / "hydrate_gold_hashes.json").read_text())


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _gold_ed():
    return {e.slug: e for e in iter_editions("edc")}["las_vegas/2026"]


def test_hydrate_byte_identity():
    """Every deterministic hydrate generator's output for the gold == the frozen pre-change hash."""
    ed = _gold_ed()
    triples = FE.load_triples(ed)
    stages_present = sorted({s for _, s, _ in triples})
    days_present = sorted({d for _, _, d in triples if d})
    got = {
        "fact_templates": h({f"{a}|{s}|{d}": FE.fact_templates(a, s, d) for a, s, d in triples}),
        "calendar_doc": h(FE.calendar_doc()),
        "festival_overview_doc": h(FE.festival_overview_doc()),
        "stage_roster": h({s: FE.stage_roster(s, triples) for s in stages_present}),
        "day_roster": h({d: FE.day_roster(d, triples) for d in days_present}),
        "stage_attribute_lines": h({s: FE.stage_attribute_lines(s) for s in sorted(FE.STAGE_META)}),
        "b2b_lines": h(FE.b2b_lines(triples)),
        "multistage_lines": h(FE.multistage_lines(triples)),
        "contrasts": h({f"{a}|{s}|{d}": CO.generate_for_fact(a, s, d) for a, s, d in triples}),
        "disjoint": h({grp["key"]: DJ.generate_for_group(grp) for grp in DJ.GROUPS}),
    }
    for k, hv in got.items():
        assert hv == GOLD[k], f"{k} diverged from golden — hydrate redesign is NOT byte-identical for edc"


def test_disjoint_audit_still_empty():
    """The v3 disjoint-context guardrail must not regress: the gold's own disjoint lines never co-mention."""
    all_dj = [r for grp in DJ.GROUPS for r in DJ.generate_for_group(grp)]
    assert DJ.audit_cooccurrence(all_dj) == [], "disjoint co-mention audit regressed (the v3 guardrail)"


def test_context_from_meta_renders():
    """The domain-general path: context_from_meta(meta) builds a ctx and the generators render with it.
    The meta-built venue is the unit's bare venue (no article) — the gold's 'the ' is default-ctx only."""
    ed = _gold_ed()
    meta = json.loads((ed.dir / "festival_meta.json").read_text())
    ctx = FE.context_from_meta(meta)
    a, s, d = FE.load_triples(ed)[0]
    out = FE.fact_templates(a, s, d, ctx)
    assert out and all(isinstance(x, str) and x for x in out), "context_from_meta render produced no strings"
    assert ctx.venue == meta["venue"] and not ctx.venue.startswith("the "), "venue should be meta.venue verbatim"


if __name__ == "__main__":
    test_hydrate_byte_identity()
    test_disjoint_audit_still_empty()
    test_context_from_meta_renders()
    print("OK — Seams 2+3: hydrate template shapes config-sourced + ctx-rendered, edc byte-identical; "
          "disjoint audit empty; context_from_meta path renders")
