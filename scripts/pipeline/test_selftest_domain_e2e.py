#!/usr/bin/env python3
"""End-to-end gate for the NON-EDC path — the executable counter-instance the byte-identity goldens can't be.

The per-function sha256 goldens (test_*_general.py) only ever run EDC; test_new_domain_swap asserts isolated
render functions with hand-fed inputs. NEITHER executes a full non-festival run through the REAL seams
(iter_editions → load_all_triples → the generator main() → the write path → the gates). So a festival
assumption that leaks onto the "generic" path is invisible to them. This runs the generic generators over a
committed, deterministic, non-festival fixture domain (`_selftest`: instruments/materials, zero festival
tokens) and asserts:
  (a) generic_expand + generic_probes RUN end-to-end and produce a non-empty corpus;
  (b) that corpus contains ZERO festival/EDC bleed (the de-festivalization invariant, mechanically);
  (c) generic_synth's record-builder produces clean records with a stubbed generator (no API);
  (d) run_pipeline --check is GREEN on the filled fixture and RED on a placeholdered copy;
  (e) the gate/audit modules import without requiring EDC data present (the R1-B1 regression).

A festival default that leaks onto the generic path turns this RED — mechanically, not by audit. Plain-assert
+ __main__ runner (no pytest in the venv):  PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_selftest_domain_e2e.py
"""
from __future__ import annotations

import json
import os
import subprocess

import config as C
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples, predicate_templates
from pipeline.generic_synth import synth_records

PY = ".venv/bin/python"
ENV = {**os.environ, "PYTHONPATH": "scripts"}
ED_DIR = C.ROOT / "data/_selftest/instruments"      # single-level unit (no city/year) — the generalized shape

# Tokens that must NEVER appear in a non-festival domain's generated corpus — if any shows up, a festival
# constant / the gold meta / an iter_editions("edc") default leaked onto the generic path.
FESTIVAL_BLEED = [
    "edc", "las vegas", "las_vegas", "motor speedway", "kinetic", "cosmic", "neon garden",
    "circuit grounds", "quantum valley", "bass pod", "stereo bloom", "armin", "insomniac",
    "friday", "saturday", "sunday", "festival", "lineup", "headliner",
]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *args], cwd=str(C.ROOT), env=ENV, capture_output=True, text=True)


def _assert_no_bleed(text: str, where: str) -> None:
    low = text.lower()
    hits = [tok for tok in FESTIVAL_BLEED if tok in low]
    assert not hits, f"FESTIVAL BLEED in {where}: {hits} leaked into the non-festival corpus"


def test_fixture_loads_and_is_discovered():
    """The seam the façade test skips: iter_editions finds the SINGLE-LEVEL unit (parts=('instruments',),
    no city/year), load_all_triples reads its triples — proving the generalized editions layer."""
    eds = list(iter_editions("_selftest"))
    assert len(eds) == 1 and eds[0].slug == "instruments", f"fixture not discovered: {[e.slug for e in eds]}"
    assert eds[0].parts == ("instruments",), f"expected a single-level unit, got parts={eds[0].parts}"
    triples = load_all_triples(eds[0])
    assert len(triples) == 6, f"expected 6 fixture triples, got {len(triples)}"
    assert predicate_templates("_selftest"), "no predicate_templates loaded for _selftest"
    print(f"  discovery: 1 single-level unit ({eds[0].slug!r}), {len(triples)} triples, templates loaded OK")


def test_generic_hydrate_runs_clean():
    r = _run(["scripts/pipeline/generic_expand.py", "--domain", "_selftest"])
    assert r.returncode == 0, f"generic_expand failed (exit {r.returncode}): {r.stderr}"
    out = ED_DIR / "25_hydrated/relations.jsonl"
    assert out.exists(), "no relations.jsonl written"
    text = out.read_text(encoding="utf-8")
    recs = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert len(recs) >= 6, f"expected >=6 hydrate lines, got {len(recs)}"
    assert {r["relation_type"] for r in recs} == {"made_of", "measures"}, "wrong relation coverage"
    _assert_no_bleed(text, "generic_expand hydrate corpus")
    print(f"  hydrate: {len(recs)} clean non-festival lines OK")


def test_generic_probes_runs_clean():
    r = _run(["scripts/pipeline/generic_probes.py", "--domain", "_selftest"])
    assert r.returncode == 0, f"generic_probes failed (exit {r.returncode}): {r.stderr}"
    out = ED_DIR / "probes.jsonl"
    assert out.exists(), "no probes.jsonl written"
    text = out.read_text(encoding="utf-8")
    items = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert items, "empty probe set"
    for it in items:                                  # the answer must be among the choices (a valid forced choice)
        assert it["answer"] in it["choices"], f"probe answer not in choices: {it}"
    _assert_no_bleed(text, "generic_probes eval items")
    print(f"  probes: {len(items)} valid forced-choice items OK")


def test_generic_synth_records_clean_with_stub():
    """synth_records with a no-API stub generator → clean records, no festival bleed (the diversity path)."""
    eds = list(iter_editions("_selftest"))
    triples = load_all_triples(eds[0])
    ptmpls = predicate_templates("_selftest")

    def stub(system: str, _user: str) -> str:          # deterministic, no network — echoes the seeded fact twice
        fact = system.splitlines()[-1] if system else "a fact"
        return f"{fact}\nRestated: {fact}"

    recs = synth_records(triples, ptmpls, preamble="", unit_name="instruments 2026", n=2, generate_fn=stub)
    assert recs, "synth produced no records"
    assert all(r["surface_form"] == "llm_synth" for r in recs)
    _assert_no_bleed("\n".join(r["text"] for r in recs), "generic_synth diversity records")
    print(f"  synth(stub): {len(recs)} clean records OK")


def test_check_green_on_filled_red_on_placeholder():
    r = _run(["scripts/run_pipeline.py", "--domain", "_selftest", "--check"])
    rep = json.loads(r.stdout)
    assert rep["ok"], f"--check should be GREEN on the filled fixture: {rep['summary']} :: " \
                      f"{[c for c in rep['checks'] if not c['ok']]}"
    # A placeholdered copy must be caught by --check (the onboarding footgun gate).
    ph = C.ROOT / "configs/domains/_selftest_ph.yaml"
    src = (C.ROOT / "configs/domains/_selftest.yaml").read_text()
    ph.write_text(src.replace("[instrument, material]", "[instrument, <material_type>]"))
    try:
        r2 = _run(["scripts/run_pipeline.py", "--domain", "_selftest_ph", "--check"])
        rep2 = json.loads(r2.stdout)
        assert not rep2["ok"], "--check must be RED on an unfilled <placeholder> manifest"
        assert any("placeholder" in c.get("check", "") and not c["ok"] for c in rep2["checks"]), \
            "the no_placeholders check did not fire"
    finally:
        ph.unlink(missing_ok=True)
    print("  --check: green on filled, red on placeholder OK")


def test_audit_gate_imports_without_edc_data():
    """R1-B1 regression: importing the audit/gate path must not crash on the gold festival_meta read."""
    r = _run(["-c", "from pipeline import disjoint_context; from qgre import gates; "
                    "import data_audit; print('imports-ok')"])
    assert r.returncode == 0 and "imports-ok" in r.stdout, f"audit/gate import failed: {r.stderr}"
    print("  gate/audit imports (no EDC-data dependency at import) OK")


if __name__ == "__main__":
    for fn in [test_fixture_loads_and_is_discovered, test_generic_hydrate_runs_clean,
               test_generic_probes_runs_clean, test_generic_synth_records_clean_with_stub,
               test_check_green_on_filled_red_on_placeholder, test_audit_gate_imports_without_edc_data]:
        fn()
    print("OK — the non-EDC path runs end-to-end with zero festival bleed")
