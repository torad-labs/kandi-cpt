#!/usr/bin/env python3
"""Lever execution receipts: an ENABLED lever that never runs must fail the run.

The case this exists for is the one an inert-lever inventory cannot see — a lever that is fully
implemented and correctly enabled, whose effect branch is simply never reached because the corpus
lacks the column it needs. train_cpt.py logs that condition for lever4 and continues.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lever_receipts as LR


def test_enabled_lever_that_never_fires_fails_the_run() -> None:
    LR.reset()
    levers = {"lever1_entity_loss": {"enabled": True}, "lever4_exposure_tracking": {"enabled": True}}
    LR.fire("lever1_entity_loss")                      # lever1 ran; lever4's column was missing

    # audit() must NOT raise — the record has to reach the run contract before the run dies.
    record = LR.audit(levers, optimizer_steps=500)
    assert record["status"] == "FAIL" and record["silent"] == ["lever4_exposure_tracking"], record
    assert record["fired"] == {"lever1_entity_loss": 1, "lever4_exposure_tracking": 0}

    try:
        LR.enforce(record)
    except LR.LeverReceiptError as error:
        assert "lever4_exposure_tracking" in str(error)
        assert "lever1_entity_loss" not in str(error), "a lever that DID fire must not be blamed"
    else:
        raise AssertionError("enabled-but-silent lever4 did not fail enforcement")


def test_all_enabled_levers_firing_passes() -> None:
    LR.reset()
    levers = {"lever1_entity_loss": {"enabled": True}, "lever4_exposure_tracking": {"enabled": True}}
    for _ in range(3):
        LR.fire("lever1_entity_loss")
        LR.fire("lever4_exposure_tracking")
    record = LR.audit(levers, optimizer_steps=500)
    assert record["status"] == "PASS", record
    assert record["fired"] == {"lever1_entity_loss": 3, "lever4_exposure_tracking": 3}
    LR.enforce(record)                                 # must not raise


def test_disabled_levers_are_not_required_to_fire() -> None:
    LR.reset()
    record = LR.audit({"lever2_qa_mix": {"enabled": False}}, optimizer_steps=500)
    assert record["status"] == "PASS" and record["enabled"] == [], record


def test_zero_step_run_is_not_failed() -> None:
    """A run that took no optimizer steps has no receipts for an honest reason."""
    LR.reset()
    record = LR.audit({"lever1_entity_loss": {"enabled": True}}, optimizer_steps=0)
    assert record["status"] == "SKIPPED_NO_STEPS", record


def test_receipt_counts_are_execution_not_configuration() -> None:
    """Firing is per-execution, so the count distinguishes 'ran once' from 'ran all run'.

    A lever whose branch is reached for only the first few batches (metadata present in one shard,
    absent in the rest) passes the boolean check but shows a count far below the step count.
    """
    LR.reset()
    LR.fire("lever4_exposure_tracking")
    assert LR.receipts()["lever4_exposure_tracking"] == 1
    LR.fire("lever4_exposure_tracking")
    assert LR.receipts()["lever4_exposure_tracking"] == 2
    LR.reset()
    assert LR.receipts() == {}


if __name__ == "__main__":
    test_enabled_lever_that_never_fires_fails_the_run()
    test_all_enabled_levers_firing_passes()
    test_disabled_levers_are_not_required_to_fire()
    test_zero_step_run_is_not_failed()
    test_receipt_counts_are_execution_not_configuration()
    print("OK — enabled-but-silent levers fail, fired levers pass, zero-step runs exempt")
