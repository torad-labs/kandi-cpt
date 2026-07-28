#!/usr/bin/env python3
"""A stage that RAN is not a stage that PASSED — the run-gate half.

run_stage's generic checks are exit-zero + declared outputs exist + a nonzero output delta +
recipe identity held. None of that can see semantic failure: audit_recall writes
report_<domain>.json with status FAIL, exits 0, and the file it was supposed to produce is
right there and changed, so the stage records PASS. The readiness ladder was fixed to read
those verdicts; this is the same hole one layer down, where the stage record is minted.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_typed_postconditions.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stage
from execution_contract import build_contract


def _kinds(stage_id: str, domain: str = "pokemon-lore", editions: str = "kanto") -> list[str]:
    stage = next(s for s in build_contract(domain, editions)["stages"] if s["id"] == stage_id)
    return [str(p.get("kind")) for p in stage["postconditions"]]


def test_contract_emits_semantic_postconditions_for_verdict_stages() -> None:
    assert "report_verdict" in _kinds("audit_recall"), _kinds("audit_recall")
    assert "report_verdict" in _kinds("pii_scan"), _kinds("pii_scan")
    assert "jsonl_no_fail_verdict" in _kinds("audit"), _kinds("audit")
    assert "json_field_positive" in _kinds("tokenize"), _kinds("tokenize")

    # Contrast: a stage with no recorded verdict must NOT acquire one, or the emission rule is
    # just decorating everything and the assertions above prove nothing.
    assert _kinds("extract") == ["output_exists"] * len(_kinds("extract")), _kinds("extract")


def test_report_verdict_rejects_a_failing_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = run_stage.C.ROOT
        run_stage.C.ROOT = root
        try:
            report = root / "report.json"
            spec = {"check": "verdict", "kind": "report_verdict", "path": "report.json",
                    "field": "status", "expect": ["PASS"]}

            report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is True

            # The whole point: the file EXISTS and CHANGED, which is all the generic checks see.
            report.write_text(json.dumps({"status": "FAIL", "overall": {"ratio": 0.35}}), encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False

            # Fail closed on every degenerate shape rather than treating "unreadable" as "fine".
            report.write_text(json.dumps({"overall": {}}), encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "missing field must not pass"
            report.write_text("{not json", encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "unparseable must not pass"
            report.unlink()
            assert run_stage._postcondition(spec)["pass"] is False, "absent must not pass"
        finally:
            run_stage.C.ROOT = original


def test_jsonl_no_fail_verdict_and_json_field_positive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = run_stage.C.ROOT
        run_stage.C.ROOT = root
        try:
            rows = root / "rows.jsonl"
            spec = {"check": "audit", "kind": "jsonl_no_fail_verdict", "path": "rows.jsonl",
                    "field": "verdicts"}
            rows.write_text(
                json.dumps({"editions": "a", "verdicts": {"balance": "pass", "phrasing": "warn"}}) + "\n",
                encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is True, "warn is not fail"
            rows.write_text(
                json.dumps({"editions": "a", "verdicts": {"balance": "pass"}}) + "\n"
                + json.dumps({"editions": "b", "verdicts": {"coverage": "fail"}}) + "\n",
                encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "a fail in ANY row must fail"
            rows.write_text("", encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "no rows is not a pass"

            # The audit file is shared across domains and keyed only by edition slug, so the check
            # must be scoped to the editions the stage ran. Measured, not hypothetical: unscoped,
            # millbrook/main failed because a booster_1 row in the same file had coverage=fail.
            rows.write_text(
                json.dumps({"editions": "main", "verdicts": {"balance": "pass"}}) + "\n"
                + json.dumps({"editions": "booster_1", "verdicts": {"coverage": "fail"}}) + "\n",
                encoding="utf-8")
            assert run_stage._postcondition({**spec, "editions": ["main"]})["pass"] is True
            assert run_stage._postcondition({**spec, "editions": ["booster_1"]})["pass"] is False
            assert run_stage._postcondition({**spec, "editions": ["absent"]})["pass"] is False, (
                "no row for the scoped editions means no verdict, which is not a pass"
            )

            stats = root / "stats.json"
            positive = {"check": "blocks", "kind": "json_field_positive", "path": "stats.json",
                        "field": "num_chunks"}
            stats.write_text(json.dumps({"num_chunks": 1480}), encoding="utf-8")
            assert run_stage._postcondition(positive)["pass"] is True
            stats.write_text(json.dumps({"num_chunks": 0}), encoding="utf-8")
            assert run_stage._postcondition(positive)["pass"] is False, "zero blocks is a failed tokenize"
        finally:
            run_stage.C.ROOT = original


if __name__ == "__main__":
    test_contract_emits_semantic_postconditions_for_verdict_stages()
    test_report_verdict_rejects_a_failing_report()
    test_jsonl_no_fail_verdict_and_json_field_positive()
    print("OK — a FAIL verdict now fails its stage, and degenerate reports fail closed")
