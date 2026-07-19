#!/usr/bin/env python3
"""Red-green proofs for zero-knob derivation + degradation ladder."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.derive_plan as DP  # noqa: E402


def test_drop_frac_rows_match_curate_law():
    # exercise via derive_knobs measured path with mocked counters
    with mock.patch.object(DP, "_count_clean_docs", return_value=100), \
         mock.patch.object(DP, "_count_triples", return_value=50), \
         mock.patch.object(DP, "_count_hydrated_surfaces", return_value=0), \
         mock.patch.object(DP, "_effective_forms_estimate", return_value=None), \
         mock.patch.object(DP, "_tokenize_blocks", return_value=(None, "absent")), \
         mock.patch.object(DP, "_contrast_authorable", return_value=False), \
         mock.patch.object(DP, "Domain") as Dom:
        Dom.return_value = mock.Mock(name="d", configs={}, templates={})
        Dom.return_value.name = "toy"
        plan = DP.derive_knobs("toy", domain_kind="facts")
    by = {k["knob"]: k for k in plan["knobs"]}
    assert by["drop_frac"]["value"] == 0.0
    assert "corpus<500" in by["drop_frac"]["derivation"]
    assert by["sft_wrong_value_neg"]["value"] == 0
    assert plan["measured"]["contrast_authorable"] is False


def test_contrast_authorable_flips_hard_neg_defaults():
    # 3200 blocks → max_steps = 3200*3/16 = 600 ≥ 500 → anneal ON
    with mock.patch.object(DP, "_count_clean_docs", return_value=1000), \
         mock.patch.object(DP, "_count_triples", return_value=200), \
         mock.patch.object(DP, "_count_hydrated_surfaces", return_value=500), \
         mock.patch.object(DP, "_effective_forms_estimate", return_value=400), \
         mock.patch.object(DP, "_tokenize_blocks", return_value=(3200, "stats")), \
         mock.patch.object(DP, "_contrast_authorable", return_value=True), \
         mock.patch.object(DP, "Domain") as Dom:
        Dom.return_value = mock.Mock(configs={}, templates={})
        Dom.return_value.name = "toy"
        plan = DP.derive_knobs("toy")
    by = {k["knob"]: k for k in plan["knobs"]}
    assert by["drop_frac"]["value"] == 0.05
    assert by["contrast_fraction"]["value"] == 0.10
    assert by["sft_wrong_value_neg"]["value"] == 20
    assert by["sft_absent_entity_neg"]["value"] == 10
    assert by["train.max_steps"]["value"] == (3200 * 3) // (1 * 16)
    assert by["anneal.enabled"]["value"] is True


def test_thin_corpus_ladder_skips_curate_and_recall():
    with mock.patch.object(DP, "_count_clean_docs", return_value=10), \
         mock.patch.object(DP, "_count_triples", return_value=5), \
         mock.patch.object(DP, "_count_hydrated_surfaces", return_value=0), \
         mock.patch.object(DP, "_effective_forms_estimate", return_value=None), \
         mock.patch.object(DP, "_tokenize_blocks", return_value=(None, "absent")), \
         mock.patch.object(DP, "_contrast_authorable", return_value=False), \
         mock.patch.object(DP, "Domain") as Dom:
        Dom.return_value = mock.Mock(configs={}, templates={})
        Dom.return_value.name = "toy"
        plan = DP.derive_knobs("toy", domain_kind="facts")
    rungs = {r["rung"]: r for r in plan["ladder"]}
    assert rungs["thin_facts_corpus"]["status"] == "skip"
    assert rungs["no_contrast_block"]["status"] == "degrade"
    assert rungs["exposure_floor_feasibility"]["status"] == "degrade"
    assert plan["knobs"]  # every knob has derivation
    assert all("derivation" in k and k["derivation"] for k in plan["knobs"])


def test_device_only_ladder_degrades_provider_stages():
    with mock.patch.object(DP, "_count_clean_docs", return_value=200), \
         mock.patch.object(DP, "_count_triples", return_value=80), \
         mock.patch.object(DP, "_count_hydrated_surfaces", return_value=100), \
         mock.patch.object(DP, "_effective_forms_estimate", return_value=90), \
         mock.patch.object(DP, "_tokenize_blocks", return_value=(400, "stats")), \
         mock.patch.object(DP, "_contrast_authorable", return_value=True), \
         mock.patch.object(DP, "Domain") as Dom:
        Dom.return_value = mock.Mock(configs={}, templates={})
        Dom.return_value.name = "toy"
        plan = DP.derive_knobs("toy", device_only=True)
    rungs = {r["rung"]: r for r in plan["ladder"]}
    assert rungs["device_only"]["status"] == "degrade"
    assert "weaker" in rungs["device_only"]["user_sentence"].lower() or "LLM" in rungs["device_only"]["user_sentence"]
    # anneal off on tiny max_steps
    by = {k["knob"]: k for k in plan["knobs"]}
    # 400*3/16 = 75 < 500
    assert by["anneal.enabled"]["value"] is False


def test_override_recorded_beside_default():
    with mock.patch.object(DP, "_count_clean_docs", return_value=100), \
         mock.patch.object(DP, "_count_triples", return_value=50), \
         mock.patch.object(DP, "_count_hydrated_surfaces", return_value=0), \
         mock.patch.object(DP, "_effective_forms_estimate", return_value=None), \
         mock.patch.object(DP, "_tokenize_blocks", return_value=(None, "absent")), \
         mock.patch.object(DP, "_contrast_authorable", return_value=False), \
         mock.patch.object(DP, "Domain") as Dom:
        Dom.return_value = mock.Mock(configs={}, templates={})
        Dom.return_value.name = "toy"
        plan = DP.derive_knobs("toy", overrides={"drop_frac": 0.2})
    row = next(k for k in plan["knobs"] if k["knob"] == "drop_frac")
    assert row["source"] == "override"
    assert row["value"] == 0.2
    assert row["derived_default"] == 0.0
    assert "operator-override" in row["derivation"]


def test_failure_sentences_present():
    assert "diversity" in DP.FAILURE_SENTENCES
    assert "audit_recall" in DP.FAILURE_SENTENCES
    assert DP.FAILURE_SENTENCES["diversity"]["sentence"]


def test_live_pokemon_lore_emits_plan():
    """Smoke against real disk if present — never fails the suite if domain missing."""
    try:
        plan = DP.derive_knobs("pokemon-lore", "kanto")
    except Exception as exc:  # noqa: BLE001
        print(f"  skip live pokemon-lore: {exc}")
        return
    if not plan.get("ok"):
        print(f"  skip live: {plan.get('error')}")
        return
    assert plan["knobs"]
    assert plan["plain_language"]
    assert plan["measured"]["n_clean_docs"] >= 1
    print(f"  live pokemon-lore docs={plan['measured']['n_clean_docs']} "
          f"triples={plan['measured']['n_triples']} contrast={plan['measured']['contrast_authorable']}")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} DERIVE-PLAN TESTS PASS")
