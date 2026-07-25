#!/usr/bin/env python3
"""h7c red-greens: disjoint_static must judge each edition against ITS OWN confusable pins.

logs/phase_h7_edc_gate_diagnosis.json: data_audit called DC.audit_cooccurrence(lines) with no
index, so every one of the 97 EDC editions was judged against las_vegas/2026's pins. The 938
flagged surfaces were las_vegas/2016 lines naming circuitGROUNDS — factually correct THERE, and
only "wrong" because 2026 pins Armin van Buuren to kineticFIELD.

The RED cases come first and stay first: narrowing a check's scope without proving its teeth
still bite is how a law gets disarmed while looking fixed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import disjoint_context as DC  # noqa: E402

# las_vegas/2026 — the gold edition, the only one that declares confusable_groups.
GOLD_META = {
    "confusable_groups": [
        {"key": "armin_sunday_finale", "members": ["Armin van Buuren"],
         "stage": "kinetic_field", "day": "Sunday"},
        {"key": "aboveandbeyond_saturday_heart", "members": ["Above & Beyond"],
         "stage": "kinetic_field", "day": "Saturday"},
        {"key": "underworld_friday_opening", "members": ["Underworld"],
         "stage": "cosmic_meadow", "day": "Friday"},
    ],
    "days": {"Friday": {"date": "2026-05-15"}, "Saturday": {"date": "2026-05-16"},
             "Sunday": {"date": "2026-05-17"}},
    "stages": {"kinetic_field": {"camel": "kineticFIELD"},
               "circuit_grounds": {"camel": "circuitGROUNDS"},
               "cosmic_meadow": {"camel": "cosmicMEADOW"}},
}

# las_vegas/2016 — a real non-gold edition: stages and days of its own, NO confusable_groups.
NON_GOLD_META = {
    "confusable_groups": [],
    "days": {"Friday": {"date": "2016-06-17"}, "Saturday": {"date": "2016-06-18"},
             "Sunday": {"date": "2016-06-19"}},
    "stages": {"kinetic_field": {"camel": "kineticFIELD"},
               "circuit_grounds": {"camel": "circuitGROUNDS"}},
}


def _violates(text: str, meta: dict) -> str | None:
    return DC.violates(text, *DC.index_for_meta(meta))


# ── RED: the disjoint-context law must keep its teeth ────────────────────────

def test_red_1_gold_edition_wrong_stage_still_fails():
    """A 2026 surface pinning Armin to a non-kineticFIELD stage is real contamination."""
    why = _violates("Armin van Buuren plays the circuitGROUNDS stage at EDC Las Vegas 2026.", GOLD_META)
    assert why is not None, "the 2026 wrong-stage co-mention MUST still be caught"
    assert "wrong stage" in why, why


def test_red_1b_gold_edition_wrong_day_still_fails():
    why = _violates("Armin van Buuren closes out Saturday at EDC Las Vegas 2026.", GOLD_META)
    assert why is not None, "the 2026 wrong-day co-mention MUST still be caught"
    assert "wrong day" in why, why


def test_red_2_two_confusable_comention_still_fails():
    """Two members of DIFFERENT groups in one window — the co-mention the law exists to stop."""
    why = _violates(
        "Armin van Buuren and Above & Beyond both headline kineticFIELD at EDC Las Vegas 2026.",
        GOLD_META,
    )
    assert why is not None, "a two-confusable co-mention MUST still be caught"
    assert "different confusable groups" in why, why


# ── GREEN: another edition's true facts are not contamination ────────────────

def test_green_1_non_gold_edition_circuitgrounds_passes():
    """The exact shape of the 938 false violations: 2016 lines judged by 2016's own (empty) pins."""
    assert _violates(
        "Armin van Buuren plays the circuitGROUNDS stage at EDC Las Vegas 2016.", NON_GOLD_META,
    ) is None


def test_green_2_non_gold_edition_other_phrasings_pass():
    for text in (
        "Armin van Buuren performs at circuitGROUNDS at EDC Las Vegas 2016.",
        "Armin van Buuren is on the lineup for circuitGROUNDS at EDC Las Vegas 2016.",
        "Armin van Buuren will play at the circuitGROUNDS stage during EDC Las Vegas 2016.",
    ):
        assert _violates(text, NON_GOLD_META) is None, text


def test_the_bug_itself_gold_pins_applied_to_a_2016_line():
    """Pin the regression: judging a 2016 line by the GOLD index is what produced the 938."""
    assert _violates(
        "Armin van Buuren plays the circuitGROUNDS stage at EDC Las Vegas 2016.", GOLD_META,
    ) is not None, "under the gold index this line is flagged — the scope mismatch"


def test_edition_without_pins_has_no_constraint():
    """An edition declaring no confusable_groups is unconstrained, not silently gold-bound."""
    confusable, group_of, _days, _stages = DC.index_for_meta(NON_GOLD_META)
    assert confusable == {} and group_of == {}


# ── The gate level: where the scope mismatch actually lived ──────────────────


def _edition(tmp: Path, slug: str, meta: dict, texts: list[str]):
    """A minimal on-disk edition: festival_meta.json + 25_hydrated/relations.jsonl."""
    from types import SimpleNamespace
    unit = tmp / slug.replace("/", "__")
    (unit / "25_hydrated").mkdir(parents=True, exist_ok=True)
    (unit / "festival_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (unit / "25_hydrated" / "relations.jsonl").write_text(
        "\n".join(json.dumps({"text": t, "relation_type": "artist_to_stage"}) for t in texts) + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(slug=slug, dir=unit)


def test_gate_scopes_per_edition_and_reports_unchecked():
    """RED pre-fix: the 2016 lines were flagged because the gate used the gold index for all.

    GREEN post-fix: 2016 is judged by its own (empty) pins, so it is UNCHECKED with zero
    violations, while the gold edition's genuine wrong-stage line is still caught.
    """
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import data_audit as DA

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        eds = [
            _edition(tmp, "las_vegas/2016", NON_GOLD_META, [
                "Armin van Buuren plays the circuitGROUNDS stage at EDC Las Vegas 2016.",
                "Armin van Buuren performs at circuitGROUNDS at EDC Las Vegas 2016.",
            ]),
            _edition(tmp, "las_vegas/2026", GOLD_META, [
                "Armin van Buuren plays the circuitGROUNDS stage at EDC Las Vegas 2026.",
            ]),
        ]
        violations, per_edition, unchecked = DA._cooccurrence_by_edition(eds)

    assert unchecked == ["las_vegas/2016"], unchecked
    assert "las_vegas/2016" not in per_edition, "an unpinned edition must not be scored at all"
    assert per_edition.get("las_vegas/2026") == 1, per_edition
    assert len(violations) == 1, f"only the gold edition's real violation survives: {violations}"


def test_no_pinned_edition_in_scope_is_skipped_not_passed():
    """Correct scoping must not launder a false FAIL into a vacuous PASS.

    On the real EDC run all 97 non-protected editions declare no confusable_groups and the one
    edition that does (las_vegas/2026) is excluded as protected — so zero editions get audited.
    Reporting that as PASS would claim the disjoint-context law was satisfied when it was never
    enforced.
    """
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import data_audit as DA

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        eds = [_edition(tmp, "las_vegas/2016", NON_GOLD_META, [
            "Armin van Buuren plays the circuitGROUNDS stage at EDC Las Vegas 2016.",
        ])]
        out = DA.dim_disjoint_static(eds)

    assert out["verdict"] == "skipped", out["verdict"]
    assert out["n_editions_checked"] == 0
    assert "UNENFORCED" in out["note"], out["note"]


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} DISJOINT-EDITION-SCOPE TESTS PASS")
