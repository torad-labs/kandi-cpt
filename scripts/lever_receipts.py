#!/usr/bin/env python3
"""Execution receipts for CPT levers — prove what RAN instead of remembering what is dead.

`LeversConfig._INERT_LEVERS` (cpt_config.py) is a HAND-MAINTAINED inventory of levers known not
to be wired. It fails fast, costs nothing, and stays. But it is the same shape as every stale-list
defect this pipeline has produced — the dedup exclusion list that sat inert for months,
`split.load_id_map` returning {} for 51 units — correct only while a human remembers to update it,
and structurally blind to the failure it most needs to catch: a lever that IS wired, IS enabled,
and still never executes.

That case is live, not hypothetical. train_cpt.py's loader LOGS
"[lever4] exposure_tracking enabled but corpus has no fact_ids column" and then CONTINUES. The step
loop's `if lever4_on and "fact_ids" in batch` never fires, `exposure` stays empty, and the run
record reports lever4 as an active intervention for the entire run. lever1 carries the same shape
through `if lever1_on and emask is not None`. A declared-but-unenforced check is this pipeline's
most repeated defect class; an inventory of dead levers cannot see either of these, because in both
the lever is genuinely implemented — it just never runs.

A receipt inverts the burden of proof. The lever's own implementation calls `fire()` at the point
its effect actually lands — INSIDE the effect branch, never at config parse. At the end of a run,
every ENABLED lever must hold a receipt or the run fails and does not get to claim the intervention.

This cannot go stale: delete a lever's code and its `fire()` call goes with it, so the next run
turns red on its own. Nothing has to be remembered.
"""
from __future__ import annotations

_FIRED: dict[str, int] = {}


class LeverReceiptError(RuntimeError):
    """An enabled lever produced no execution receipt."""


def fire(lever: str) -> None:
    """Record that `lever`'s effect actually executed.

    Call from inside the branch that performs the effect, so the receipt is evidence of execution
    rather than of configuration. Hot path: one dict get + store per microbatch.
    """
    _FIRED[lever] = _FIRED.get(lever, 0) + 1


def receipts() -> dict[str, int]:
    """Fire counts so far, by lever."""
    return dict(_FIRED)


def reset() -> None:
    """Clear the registry. For tests and for a process that runs more than one training job."""
    _FIRED.clear()


def audit(levers: dict | None, *, optimizer_steps: int) -> dict:
    """Reconcile enabled levers against fired receipts and return the record. Never raises.

    Computing and enforcing are separate so the caller can embed this record in the run contract
    BEFORE failing on it. A run whose lever went silent should leave a checkpoint that says so;
    raising first would leave the artifact on disk with no explanation of why the run died.

    `optimizer_steps` guards the degenerate case: a run that took no steps legitimately has no
    receipts, and failing it would be noise rather than signal.
    """
    enabled = sorted(
        name for name, spec in (levers or {}).items()
        if isinstance(spec, dict) and spec.get("enabled")
    )
    fired = receipts()
    record: dict = {
        "enabled": enabled,
        "fired": {name: fired.get(name, 0) for name in enabled},
        "optimizer_steps": optimizer_steps,
    }
    if optimizer_steps <= 0:
        record["status"] = "SKIPPED_NO_STEPS"
        return record

    silent = [name for name in enabled if not fired.get(name)]
    record["silent"] = silent
    record["status"] = "PASS" if not silent else "FAIL"
    return record


def enforce(record: dict) -> None:
    """Fail the run if `audit`'s record shows an enabled lever that never executed."""
    silent = record.get("silent") or []
    if record.get("status") != "FAIL" or not silent:
        return
    raise LeverReceiptError(
        f"levers enabled but never executed: {', '.join(silent)}. The run completed "
        f"{record.get('optimizer_steps')} optimizer steps and these produced no receipt, so the "
        "run record would report an intervention that did not happen. Usually the corpus lacks the "
        "column the lever needs (re-tokenize) or the effect branch is unreachable. Fix the run — "
        "disabling the receipt only restores the silence it was built to break."
    )
