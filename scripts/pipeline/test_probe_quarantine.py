#!/usr/bin/env python3
"""Red-green + calibration proofs for the floor-stated-holdout-pair detector.

A ZERO FROM AN UNCALIBRATED DETECTOR IS WORTHLESS, and this detector reports a zero on 97 of the
repo's 99 split-bearing units. Four controls below each feed it a case it MUST NOT fire on, so its
clean reads are measurements rather than vacuous truths; the positive cases are the two real kanto
shapes (reverse direction, and a different predicate on the same pair).

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_probe_quarantine.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.probe_quarantine as Q  # noqa: E402
from pipeline.generic_reasoning import entity_classes  # noqa: E402
from pipeline.split import triple_id  # noqa: E402


def _t(subject_id: str, subject: str, predicate: str, object_id: str, obj: str) -> dict:
    """A 15_structured triple as `generic_expand.load_all_triples` hands it downstream."""
    return {
        "subject_id": subject_id,
        "subject": subject,
        "predicate": predicate,
        "object_id": object_id,
        "object": obj,
        "attrs": {},
    }


def _index(triples: list[dict], holdout: list[dict], id_map: dict | None = None):
    """(uniq, train_ids, holdout_ids, classes) for a triple list, keyed exactly as the scan does."""
    uniq: dict[str, dict] = {}
    for t in triples:
        uniq.setdefault(triple_id(t), t)
    holdout_ids = {triple_id(t) for t in holdout}
    train_ids = set(uniq) - holdout_ids
    classes = entity_classes(list(uniq.values()), id_map or {})
    return uniq, train_ids, holdout_ids, classes


# The real kanto pair 1, reduced: TRAIN states the reverse of the HOLDOUT triple.
_RED = _t("d:c:red", "Red", "opposes", "d:c:lt_surge", "Lt. Surge")
_SURGE = _t("d:c:lt_surge", "Lt. Surge", "opposes", "d:c:red", "Red")


def test_detector_flags_a_holdout_pair_a_floor_renderable_train_triple_states():
    uniq, train_ids, holdout_ids, classes = _index([_SURGE, _RED], [_RED])
    result = Q.floor_stated_holdout_pairs(
        uniq, train_ids, holdout_ids, classes, {"opposes"})
    assert result["instrument_blind"] is False
    assert result["floor_channel"] == Q.FLOOR_CHANNEL_GENERIC
    assert result["counts"]["holdout_class_pairs"] == 1
    assert result["counts"]["pairs_with_any_train_witness"] == 1
    assert result["counts"]["pairs_floor_stated"] == 1
    assert result["counts"]["quarantined_sources"] == 1
    row = result["quarantined"][0]
    assert row["source_triple_id"] == triple_id(_RED)
    assert row["reason"] == Q.REASON_FLOOR_PAIR
    assert row["entity_class_pair"] == sorted(["d:c:red", "d:c:lt_surge"])
    assert row["train_witness_triple_ids"] == [triple_id(_SURGE)]
    assert row["holdout_triple"] == {"subject": "Red", "predicate": "opposes", "object": "Lt. Surge"}


def test_detector_flags_a_pair_stated_under_a_different_predicate_with_every_witness():
    """kanto pair 2: HOLDOUT (Team Rocket|allied_with|Giovanni) against three train witnesses
    across TWO predicates. Every witness is recorded — a single one would understate the leak."""
    holdout = _t("d:o:tr", "Team Rocket", "allied_with", "d:c:gio", "Giovanni")
    leads = _t("d:c:gio", "Giovanni", "leads", "d:o:tr", "Team Rocket")
    member = _t("d:c:gio", "Giovanni", "member_of", "d:o:tr", "Team Rocket")
    uniq, train_ids, holdout_ids, classes = _index([holdout, leads, member], [holdout])
    result = Q.floor_stated_holdout_pairs(
        uniq, train_ids, holdout_ids, classes, {"leads", "member_of", "allied_with"})
    assert result["counts"]["quarantined_sources"] == 1
    assert result["quarantined"][0]["train_witness_triple_ids"] == sorted(
        [triple_id(leads), triple_id(member)])
    assert {row["predicate"] for row in result["quarantined"][0]["train_witness_triples"]} == {
        "leads", "member_of"}


def test_detector_ignores_a_train_witness_whose_predicate_has_no_floor_template():
    """CONTROL (a) — the load-bearing qualifier. The same graph, the same class pair, but the
    train witness's predicate has no hydrate template, so the floor never states it. Measured on
    the real corpus this qualifier takes the pool from 10 pairs to 2."""
    uniq, train_ids, holdout_ids, classes = _index([_SURGE, _RED], [_RED])
    result = Q.floor_stated_holdout_pairs(
        uniq, train_ids, holdout_ids, classes, {"located_in"})
    assert result["instrument_blind"] is False          # the map is non-empty: a real measurement
    assert result["counts"]["pairs_with_any_train_witness"] == 1     # the wider pool still counts it
    assert result["counts"]["pairs_floor_stated"] == 0
    assert result["counts"]["pairs_train_witness_not_floor_stated"] == 1
    assert result["quarantined"] == []
    assert result["pairs_with_any_train_witness"][0]["floor_stated"] is False


def test_detector_ignores_a_holdout_pair_with_no_train_witness():
    """CONTROL (b) — a clean holdout pair. No train triple joins its two endpoint classes."""
    holdout = _t("d:c:agatha", "Agatha", "opposes", "d:c:red", "Red")
    elsewhere = _t("d:c:blue", "Blue", "opposes", "d:c:lance", "Lance")
    uniq, train_ids, holdout_ids, classes = _index([holdout, elsewhere], [holdout])
    result = Q.floor_stated_holdout_pairs(
        uniq, train_ids, holdout_ids, classes, {"opposes"})
    assert result["counts"]["holdout_class_pairs"] == 1
    assert result["counts"]["pairs_with_any_train_witness"] == 0
    assert result["counts"]["pairs_floor_stated"] == 0
    assert result["quarantined"] == []


def test_detector_ignores_a_self_loop_pair():
    """CONTROL (c) — a triple whose two endpoints are ONE entity class is not a pair at all, and
    a self-loop must never manufacture a leak against every other self-loop."""
    holdout = _t("d:c:red", "Red", "rival_of", "d:c:red_alias", "Red")
    train = _t("d:c:red_alias", "Red", "mentors", "d:c:red", "Red")
    uniq, train_ids, holdout_ids, classes = _index([holdout, train], [holdout])
    assert classes["d:c:red"] == classes["d:c:red_alias"]        # same display name → one class
    result = Q.floor_stated_holdout_pairs(
        uniq, train_ids, holdout_ids, classes, {"rival_of", "mentors"})
    assert result["counts"]["holdout_class_pairs"] == 0
    assert result["quarantined"] == []


def test_an_empty_floor_template_map_is_instrument_blind_never_zero_leaks():
    """CONTROL (d) — the 97-unit case. `predicate_templates("edc")` is {} because EDC's floor is
    the Rust factory, so the floor-renderable test is vacuously false. Publishing that as
    `leaks: 0` would be an uncalibrated zero; it must come back as blind, with no count at all."""
    uniq, train_ids, holdout_ids, classes = _index([_SURGE, _RED], [_RED])
    result = Q.floor_stated_holdout_pairs(uniq, train_ids, holdout_ids, classes, set())
    assert result["instrument_blind"] is True
    assert result["floor_channel"] == Q.FLOOR_CHANNEL_UNREADABLE
    assert result["counts"]["pairs_floor_stated"] is None
    assert result["counts"]["pairs_train_witness_not_floor_stated"] is None
    assert result["quarantined"] == []
    # The wider pool needs no template map and is still measured — that is what keeps the
    # non-floor overlaps visible on a blind unit instead of vanishing.
    assert result["counts"]["pairs_with_any_train_witness"] == 1


def test_two_ids_for_one_entity_are_paired_as_one_class_before_the_leak_test():
    """Endpoint identity is RESOLVED, not raw: a train witness recorded under an alias id still
    states the holdout pair, and keying on raw ids would miss it."""
    holdout = _t("d:o:tr", "Team Rocket", "allied_with", "d:c:gio", "Giovanni")
    aliased = _t("d:c:gio_boss", "Giovanni Boss", "leads", "d:o:tr", "Team Rocket")
    id_map = {"d:c:gio_boss": "d:c:gio"}
    uniq, train_ids, holdout_ids, classes = _index([holdout, aliased], [holdout], id_map)
    result = Q.floor_stated_holdout_pairs(
        uniq, train_ids, holdout_ids, classes, {"leads", "allied_with"})
    assert result["counts"]["quarantined_sources"] == 1
    # Same graph WITHOUT the resolve merge: two distinct classes, so no pair and no leak.
    uniq2, train2, holdout2, classes2 = _index([holdout, aliased], [holdout])
    plain = Q.floor_stated_holdout_pairs(
        uniq2, train2, holdout2, classes2, {"leads", "allied_with"})
    assert plain["counts"]["quarantined_sources"] == 0


def test_scan_unit_writes_a_record_the_loader_reads_back_with_counted_reasons():
    original_root = Q.C.ROOT
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        Q.C.ROOT = root
        try:
            unit_dir = root / "data" / "fixture-domain" / "kanto"
            (unit_dir / "15_structured").mkdir(parents=True)
            (unit_dir / "splits").mkdir(parents=True)
            (unit_dir / "15_structured" / "triples.jsonl").write_text(
                "\n".join(json.dumps({"subject_id": t["subject_id"], "predicate": t["predicate"],
                                      "object": t["object_id"]}) for t in (_SURGE, _RED)) + "\n",
                encoding="utf-8")
            (unit_dir / "15_structured" / "entities.jsonl").write_text(
                "\n".join(json.dumps({"id": i, "name_canonical": n}) for i, n in (
                    ("d:c:red", "Red"), ("d:c:lt_surge", "Lt. Surge"))) + "\n", encoding="utf-8")
            (unit_dir / "splits" / "split_v1.json").write_text(json.dumps({
                "version": "v1",
                "train_ids": [triple_id(_SURGE)],
                "holdout_ids": [triple_id(_RED)],
            }), encoding="utf-8")
            ed = SimpleNamespace(slug="kanto", dir=unit_dir)
            record = Q.scan_unit("fixture-domain", ed, {"opposes"})
            assert record is not None and record["counts"]["quarantined_sources"] == 1
            path = Q.quarantine_path("fixture-domain", "kanto")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

            loaded = Q.load_quarantine("fixture-domain", "kanto")
            assert loaded is not None and loaded["split_version"] == "v1"
            assert Q.quarantined_reasons(loaded) == {triple_id(_RED): Q.REASON_FLOOR_PAIR}
        finally:
            Q.C.ROOT = original_root


def test_load_quarantine_returns_none_when_absent_or_schema_mismatch():
    original_root = Q.C.ROOT
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        Q.C.ROOT = root
        try:
            assert Q.load_quarantine("fixture-domain", "kanto") is None
            assert Q.quarantined_reasons(None) == {}
            path = Q.quarantine_path("fixture-domain", "kanto")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema": "something/else", "quarantined": [
                {"source_triple_id": "x", "reason": "made up"}]}), encoding="utf-8")
            assert Q.load_quarantine("fixture-domain", "kanto") is None
        finally:
            Q.C.ROOT = original_root


def test_unit_slug_with_a_path_separator_flattens_to_one_file():
    """EDC units are `city/year`; a raw join would write into a nested directory named after the
    city and collide across years."""
    assert Q.quarantine_path("edc", "las_vegas/2025").name == "las_vegas_2025.json"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} PROBE-QUARANTINE TESTS PASS")
