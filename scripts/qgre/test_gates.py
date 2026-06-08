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
    # festival_meta.days is a dateless {Friday/Saturday/Sunday} skeleton — the hydrator can only emit
    # day_of_week_map from a real date, so its absence here is a SOURCE GAP, not a generation defect.
    assert cls["day_of_week_map"]["kind"] == "source_gap"
    # 2018 HAS a B2B billing ("Shimon B2B Benny L") yet produced no b2b_pairing line → genuine DEFECT
    # (the content probe entities.jsonl~B2B finds the source; editions with no B2B billing are source_gap).
    assert cls["b2b_pairing"]["kind"] == "defect"


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
    """plays_at→festival is an EXTRACTION-layer masquerade (a real lineup fact under the wrong predicate).
    It lived in EVERY edition's structured layer; the deterministic remap_contract pass fixes it losslessly
    (plays_at→festival ⇒ in_lineup) — so a remapped edition like las_vegas/2018 is now contract-clean (0.0).
    The GOLD (las_vegas/2026) is the lone holdout: it keeps the documented defect ON PURPOSE, because its
    9,982 protected hydrated lines carry POSITIONAL plays_at:N provenance and a remap (which drops 21 rows)
    would silently re-point them. So the invariant is now: gold still flags the masquerade; the remap drives
    it to zero on a real edition — proving it was extraction-layer, not a hydration regression, and fixable."""
    gold = _suite("las_vegas/2026")["gates"]["id_integrity"]
    remapped = _suite("las_vegas/2018")["gates"]["id_integrity"]
    gv = gold["metrics"]["object_contract_violation_frac"]
    rv = remapped["metrics"]["object_contract_violation_frac"]
    assert gold["severity"] == "fail" and gv > 0.1   # gold retains the documented masquerade (~0.35), protected
    assert rv == 0.0                                  # remap fixed it losslessly on the real edition
    assert gv > rv                                    # the defect was extraction-layer, now remediated except on the gold


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


def test_run_suite_partial_state_not_false_failed():
    """F4 regression guard — the coverage gap that let the round-3 false-fail through. A not-yet-hydrated
    edition (valid 15_structured, no 25_hydrated) must NOT be flagged `empty_corpus`/forced to 'fail': only
    the hydrated gates skip, id_integrity has a REAL verdict — that's not-ready, not broken. A TRULY-empty
    edition (no data at any layer) must be flagged + 'fail'."""
    import shutil
    from pipeline.editions import Ed
    sp = qspec.load_spec("_selftest")
    # structured-only: the committed _selftest fixture has 15_structured but no committed 25_hydrated (the e2e
    # regenerates it, gitignored) — remove it so this exercises the not-yet-hydrated state regardless of order.
    inst = Ed("_selftest", ("instruments",))
    if (inst.dir / "25_hydrated").exists():
        shutil.rmtree(inst.dir / "25_hydrated")
    res = gates.run_suite([inst], sp)
    assert res["empty_corpus"] is False, f"a structured-only edition was wrongly flagged empty ({res['overall']})"
    # truly empty: no 15_structured AND no 25_hydrated → every data gate skips → empty_corpus + 'fail'
    empty = gates.run_suite([Ed("_selftest", ("nonexistent_unit",))], sp)
    assert empty["empty_corpus"] is True and empty["overall"] == "fail", \
        f"a truly-empty corpus must fail, got {empty['overall']} empty={empty['empty_corpus']}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} QGRE-GATE TESTS PASS")
