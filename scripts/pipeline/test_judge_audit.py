#!/usr/bin/env python3
"""A heuristic verdict must not be able to impersonate a calibrated judge.

judge_one never raises on unparseable judge output — it runs a gold-containment heuristic and
returns a label from the same vocabulary as an LLM verdict. That keeps a run alive through a
transient blip, and it also means a provider outage produces a COMPLETE label set and a report that
reads exactly like a working calibrated judge. Nothing counted the substitutions.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_judge_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_generative import JudgeAudit


def _audit(llm: int = 0, heuristic: int = 0, empty: int = 0) -> JudgeAudit:
    audit = JudgeAudit()
    for source, count in (("llm", llm), ("heuristic", heuristic), ("empty", empty)):
        for _ in range(count):
            audit.record(source=source)
    return audit


def test_a_clean_run_is_not_penalised() -> None:
    audit = _audit(llm=100)
    assert audit.mode == "llm" and audit.fallback_ratio == 0.0 and audit.excessive is False


def test_a_provider_outage_is_not_reported_as_a_working_judge() -> None:
    audit = _audit(heuristic=50)
    assert audit.mode == "heuristic"
    assert audit.fallback_ratio == 1.0
    assert audit.excessive is True, "a fully heuristic run must not pass as judged"


def test_the_threshold_bites_between_tolerable_and_not() -> None:
    assert _audit(llm=95, heuristic=5).excessive is False       # 5% — a transient blip
    assert _audit(llm=80, heuristic=20).excessive is True       # 20% — not a judged run
    assert _audit(llm=90, heuristic=10).excessive is False, "exactly at the bound is allowed"
    assert _audit(llm=89, heuristic=11).excessive is True


def test_empty_answers_never_count_against_the_judge() -> None:
    """An empty model answer is labelled `deflected` WITHOUT calling the judge.

    Counting those as judge failures would make a heavily-deflecting model look like a broken
    judge, and would mask a real outage behind a plausible explanation.
    """
    audit = _audit(llm=10, empty=90)
    assert audit.judged == 10
    assert audit.fallback_ratio == 0.0 and audit.excessive is False
    assert audit.report()["empty_answer_verdicts"] == 90
    assert audit.report()["judge_verdicts"] == 100


def test_report_carries_every_field_the_review_names() -> None:
    audit = _audit(llm=90, heuristic=10, empty=5)
    report = audit.report(calibration_stamp="kappa-v1:abc123")
    for field in ("judge_mode", "judge_parse_failures", "heuristic_fallback_count",
                  "fallback_ratio", "judge_calibration_stamp"):
        assert field in report, field
    assert report["judge_mode"] == "mixed"
    assert report["judge_parse_failures"] == 10
    assert report["judge_calibration_stamp"] == "kappa-v1:abc123"
    # parse failures and heuristic substitutions are the same event, reported under both names the
    # review asks for; they must never drift apart.
    assert report["judge_parse_failures"] == report["heuristic_fallback_count"]


def test_no_verdicts_at_all_is_not_a_failure_signal() -> None:
    audit = _audit()
    assert audit.mode == "llm" and audit.excessive is False and audit.fallback_ratio == 0.0


if __name__ == "__main__":
    test_a_clean_run_is_not_penalised()
    test_a_provider_outage_is_not_reported_as_a_working_judge()
    test_the_threshold_bites_between_tolerable_and_not()
    test_empty_answers_never_count_against_the_judge()
    test_report_carries_every_field_the_review_names()
    test_no_verdicts_at_all_is_not_a_failure_signal()
    print("OK — heuristic fallback is counted, reported, and gated instead of impersonating a judge")
