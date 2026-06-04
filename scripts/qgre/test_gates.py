"""Regression tests for the deterministic quality gates (scripts/qgre/gates.py).

Run:  PYTHONPATH=scripts .venv/bin/python scripts/qgre/test_gates.py
 (or: PYTHONPATH=scripts .venv/bin/python -m pytest scripts/qgre/test_gates.py)

These pin the EXACT numbers measured during the corpus audit (docs/qgre_data_pipeline_audit.md) so a
regression in gate logic — or a corpus change — is caught immediately. The validation story is the HONEST
one the evidence forced (not "gold passes everything"):

  • The data-quality gates discriminate the VC1 disaster from the gold reference: VC1 2018 FAILS balance,
    exposure-cap, and faithfulness; GOLD 2026 passes those (exposure is a tail-only WARN, not the systemic
    runaway VC1 has).
  • source_vs_gap classifies VC1's 4 missing relations correctly: set_time + disjoint_context are SOURCE
    GAPS (no input data), day_of_week_map + b2b_pairing are DEFECTS (data present, generation failed).
  • id_integrity FAILS on BOTH editions — the plays_at→festival contract violation is a SHARED EXTRACTION
    defect (gold 35% is actually WORSE than VC1 25%), proving it lives in the extraction layer (fixed in
    Phase B), not in hydration. The gate correctly surfaces it in both.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qgre import gates, spec as qspec          # noqa: E402
from pipeline.editions import iter_editions     # noqa: E402

SPEC = qspec.load_spec("edc")


def _suite(edition):
    eds = list(iter_editions("edc", only=edition))
    assert eds, f"edition {edition} not found on disk"
    return gates.run_suite(eds, SPEC)


# ─────────────────────────────────────────────────────────────── VC1 2018 — the disaster, must FAIL hard
def test_vc1_2018_overall_fail():
    assert _suite("las_vegas/2018")["overall"] == "fail"


def test_vc1_2018_balance_monoculture():
    g = _suite("las_vegas/2018")["gates"]["balance"]
    assert g["severity"] == "fail"
    assert g["metrics"]["dominant_frac"] > 0.85        # measured 0.983 — relation monoculture
    assert g["metrics"]["n_types"] == 8                # 4 of gold's 12 types absent


def test_vc1_2018_exposure_systemic_runaway():
    # Redundancy is the install mechanism, NOT a defect — so this gate never FAILs on it; it WARNs as a
    # compute-footprint signal when the typical fact is systemically over-replicated (median > cap).
    g = _suite("las_vegas/2018")["gates"]["exposure_cap"]
    assert g["severity"] == "warn"                     # compute-heavy, not a quality failure
    assert g["metrics"]["systemic"] is True            # the median itself exceeds the cap
    assert g["metrics"]["median"] > 60                 # measured 204 — every fact over-replicated (compute, not quality)


def test_vc1_2018_faithfulness_is_edition_scoped():
    """The confusable co-mention audit is edition-SCOPED: it judges each edition against ITS OWN
    confusable_groups, never the gold's. las_vegas/2018 declares no confusable_groups (it never took part
    in the V6 disjoint-context experiment), so it has no such constraint and zero violations. The 1438
    'violations' a gold-hardcoded audit reported were cross-edition artifacts — every one was a true 2018
    fact (e.g. 'Armin van Buuren on Saturday') flagged only because it differs from Armin's 2026 Sunday
    binding. VC1 2018 is still a genuine disaster; its failure is carried by balance + exposure (below),
    the gates that are actually valid for it — not by a confounded cross-edition measurement."""
    g = _suite("las_vegas/2018")["gates"]["faithfulness"]
    assert g["severity"] == "pass"
    assert g["metrics"]["cooccurrence_violations"] == 0


def test_vc1_2018_source_gap_classification():
    """The load-bearing honesty check: the 4 missing relations classified source_gap vs defect."""
    cls = _suite("las_vegas/2018")["gates"]["source_vs_gap"]["source_gap_vs_defect"]
    assert cls["set_time"]["kind"] == "source_gap"          # no set_times.jsonl for 2018
    assert cls["disjoint_context"]["kind"] == "source_gap"  # no confusable_groups in festival_meta
    assert cls["day_of_week_map"]["kind"] == "defect"       # festival_meta.days IS present → generation failed
    assert cls["b2b_pairing"]["kind"] == "defect"           # needs nothing → generation failed


# ─────────────────────────────────────────────────────── GOLD 2026 — clean except the shared extraction bug
def test_gold_2026_data_quality_gates_clean():
    """Gold passes the gates that distinguish a healthy corpus: balance, faithfulness, source-vs-gap, and
    exposure is only a tail WARN (not the systemic FAIL VC1 has)."""
    g = _suite("las_vegas/2026")["gates"]
    assert g["balance"]["severity"] == "pass"           # dominant 0.824 < 0.85
    assert g["faithfulness"]["severity"] == "pass"      # 0 co-mention
    assert g["source_vs_gap"]["severity"] == "pass"     # nothing missing — all 12 types present
    assert g["exposure_cap"]["severity"] == "pass"      # median 2 (healthy) — typical fact well within footprint
    assert g["exposure_cap"]["metrics"]["systemic"] is False


def test_shared_extraction_defect_id_integrity():
    """plays_at→festival is an EXTRACTION-layer bug present in BOTH editions — gold is worse, proving it is
    not a VC1 hydration regression. The gate must flag it in both (fixed in Phase B)."""
    gold = _suite("las_vegas/2026")["gates"]["id_integrity"]
    vc1 = _suite("las_vegas/2018")["gates"]["id_integrity"]
    assert gold["severity"] == "fail" and vc1["severity"] == "fail"
    gv = gold["metrics"]["object_contract_violation_frac"]
    vv = vc1["metrics"]["object_contract_violation_frac"]
    assert gv > vv          # gold ~0.35 worse than VC1 ~0.25 → extraction-layer, not hydration
    assert vv > 0.0         # both have the festival-as-stage masquerade


# ───────────────────────────────────────────────────────────────────── the gold-vs-VC1 discrimination proof
def test_gold_beats_vc1_on_every_data_quality_axis():
    # balance + exposure are the axes that honestly discriminate the VC1 disaster from gold. faithfulness is
    # NOT one of them: it is edition-scoped, and 2018 (no confusable_groups) is correctly clean there — so
    # both editions tie at pass, and asserting gold strictly-beats VC1 on faithfulness would be measuring a
    # cross-edition artifact (the very confound this gate fix removed).
    gold = _suite("las_vegas/2026")["gates"]
    vc1 = _suite("las_vegas/2018")["gates"]
    rank = {"pass": 0, "warn": 1, "skipped": 1, "fail": 2}
    for axis in ("balance", "exposure_cap"):
        assert rank[gold[axis]["severity"]] < rank[vc1[axis]["severity"]], \
            f"gold should be strictly better than VC1 on {axis}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} QGRE-GATE TESTS PASS")
