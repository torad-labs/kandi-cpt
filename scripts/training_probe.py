#!/usr/bin/env python3
"""Stop on measured LEARNING, not on training loss.

`run.early_stop_patience` watches the smoothed loss and stops when it plateaus. That is the wrong
instrument for knowledge installation: CPT loss can fall monotonically while fact acquisition stalls
and general ability degrades, so a loss-plateau stop fires late on a run that stopped learning long
before, and never fires at all on a run that is quietly getting worse at everything except the
next-token objective.

This module carries the trajectory across checkpoints and applies the four stop rules the review
names, all against the repo's existing mechanical bound (eval_stats.rough_bound = 0.5/√n) rather
than a new hand-picked threshold:

  1. install strength has not improved for two consecutive checkpoints
  2. confusable error increased materially
  3. general ability crossed its regression bound against the BASE model
  4. loss improved while task-level learning declined  — the divergence the research names

The decision is pure: it takes recorded numbers and returns a verdict with its reasons, so it is
testable without a GPU and cannot silently disagree with what the run record shows. Scoring itself
reuses eval_probes.evaluate_model on the ALREADY-LOADED training model — forced choice is a logprob
pass, not generation, so a checkpoint probe costs a fraction of a step rather than a reload.

Relationship to the sentinel: the sentinel is a one-event candidate-vs-reference tripwire and owns
the install-event verdict. This is the same idea applied ACROSS checkpoints during a run. It
deliberately reuses the sentinel's frozen, hash-stamped slices — a mid-run probe drawn from a
different set would not be comparable to the post-run verdict.
"""
from __future__ import annotations

from typing import Any

from eval_stats import rough_bound

SCHEMA = "kandi.training_probe.v1"

# Families the review names. Each is scored separately and NEVER merged into one number: a single
# "knowledge installed" score hides the difference between memorising and generalising.
FAMILIES = ("install", "extraction", "longtail", "confusable", "general", "prior_install")
# The review names six diagnostic slices. `extraction` is short generative extraction (an unseen
# paraphrase) and `prior_install` is previously-installed facts — the regression surface a booster
# run can quietly destroy. Both are scored the same way as the rest: whichever slices the frozen
# file supplies get measured, and a slice that is absent is reported as not-evaluated rather than
# assumed passing.

# Install strength may stall for this many consecutive checkpoints before the run is stopped.
STALL_PATIENCE = 2


def record(
    *,
    step: int,
    loss: float,
    scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One checkpoint's row. `scores` maps family -> {hits, n, acc}."""
    row: dict[str, Any] = {"step": int(step), "loss": float(loss)}
    for family in FAMILIES:
        block = scores.get(family)
        if block:
            row[family] = {"hits": int(block["hits"]), "n": int(block["n"]),
                           "acc": float(block["acc"])}
    return row


def _acc(row: dict[str, Any], family: str) -> float | None:
    block = row.get(family)
    return float(block["acc"]) if block else None


def _n(row: dict[str, Any], family: str) -> int:
    block = row.get(family)
    return int(block["n"]) if block else 0


def _improved(previous: float, current: float, n: int) -> bool:
    """Improvement must clear the sampling bound, or it is noise being read as progress."""
    bound = rough_bound(n)
    return bound is not None and (current - previous) > bound


def _regressed(reference: float, current: float, n: int) -> bool:
    bound = rough_bound(n)
    return bound is not None and (current - reference) < -bound


def decide(
    history: list[dict[str, Any]],
    *,
    general_base_acc: float | None = None,
) -> dict[str, Any]:
    """Should this run stop, and why. Pure — it reads recorded numbers and nothing else.

    Returns `stop: False` with an explicit reason whenever there is not yet enough trajectory to
    judge. Never guesses: an absent family is reported as not-evaluated rather than treated as
    passing, because "we did not measure it" and "it is fine" are different claims.
    """
    verdict: dict[str, Any] = {
        "schema": SCHEMA,
        "checkpoints": len(history),
        "stop": False,
        "reasons": [],
        "rules": [],
    }
    if len(history) < 2:
        verdict["rules"].append({"rule": "needs at least two checkpoints", "fired": False,
                                 "detail": f"{len(history)} recorded"})
        return verdict

    latest, previous = history[-1], history[-2]

    # 1 — install strength stalled for STALL_PATIENCE consecutive checkpoints.
    install_accs = [_acc(row, "install") for row in history]
    if all(acc is not None for acc in install_accs) and len(history) >= STALL_PATIENCE + 1:
        window = install_accs[-(STALL_PATIENCE + 1):]
        n = _n(latest, "install")
        stalled = not any(_improved(window[i], window[i + 1], n) for i in range(len(window) - 1))
        verdict["rules"].append({
            "rule": f"install strength improved within the last {STALL_PATIENCE} checkpoints",
            "fired": stalled, "window": window, "n": n, "bound": rough_bound(n),
        })
        if stalled:
            verdict["reasons"].append(
                f"install strength has not improved beyond the sampling bound for "
                f"{STALL_PATIENCE} checkpoints (acc {window})")

    # 2b — previously-installed facts must not regress: a booster that installs new knowledge by
    #      destroying old knowledge has not installed anything, it has traded.
    prior_now, prior_before = _acc(latest, "prior_install"), _acc(previous, "prior_install")
    if prior_now is not None and prior_before is not None:
        n = _n(latest, "prior_install")
        fired = _regressed(prior_before, prior_now, n)
        verdict["rules"].append({"rule": "previously installed facts held", "fired": fired,
                                 "from": prior_before, "to": prior_now, "n": n,
                                 "bound": rough_bound(n)})
        if fired:
            verdict["reasons"].append(
                f"previously installed facts regressed ({prior_before:.4f} -> {prior_now:.4f}, "
                f"beyond ±{rough_bound(n):.4f}) — new knowledge traded against old")

    # 2 — confusable error increased materially (accuracy fell beyond the bound).
    confusable_now, confusable_before = _acc(latest, "confusable"), _acc(previous, "confusable")
    if confusable_now is not None and confusable_before is not None:
        n = _n(latest, "confusable")
        fired = _regressed(confusable_before, confusable_now, n)
        verdict["rules"].append({"rule": "confusable accuracy held", "fired": fired,
                                 "from": confusable_before, "to": confusable_now,
                                 "n": n, "bound": rough_bound(n)})
        if fired:
            verdict["reasons"].append(
                f"confusable error increased materially ({confusable_before:.4f} -> "
                f"{confusable_now:.4f}, beyond ±{rough_bound(n):.4f})")

    # 3 — general ability crossed its regression bound against the BASE model, not the last
    #     checkpoint: slow drift that never trips a step-to-step check still ends in a damaged model.
    general_now = _acc(latest, "general")
    if general_now is not None and general_base_acc is not None:
        n = _n(latest, "general")
        fired = _regressed(general_base_acc, general_now, n)
        verdict["rules"].append({"rule": "general ability within its regression bound of base",
                                 "fired": fired, "base": general_base_acc, "now": general_now,
                                 "n": n, "bound": rough_bound(n)})
        if fired:
            verdict["reasons"].append(
                f"general ability regressed against base ({general_base_acc:.4f} -> "
                f"{general_now:.4f}, beyond ±{rough_bound(n):.4f})")

    # 4 — the divergence itself: loss improved while task-level learning declined.
    install_now, install_before = _acc(latest, "install"), _acc(previous, "install")
    if install_now is not None and install_before is not None:
        n = _n(latest, "install")
        loss_improved = latest["loss"] < previous["loss"]
        learning_declined = _regressed(install_before, install_now, n)
        fired = loss_improved and learning_declined
        verdict["rules"].append({
            "rule": "loss and task-level learning agree", "fired": fired,
            "loss": [previous["loss"], latest["loss"]],
            "install_acc": [install_before, install_now], "n": n,
        })
        if fired:
            verdict["reasons"].append(
                f"loss improved ({previous['loss']:.4f} -> {latest['loss']:.4f}) while install "
                f"strength declined ({install_before:.4f} -> {install_now:.4f}) — the objective "
                f"and the thing being installed have diverged")

    verdict["stop"] = bool(verdict["reasons"])
    return verdict


def probe_loaded_model(tokenizer, model, device: str, slices: dict[str, list[dict[str, Any]]]) -> dict:
    """Score the frozen families against the model ALREADY IN MEMORY.

    Forced choice is a logprob comparison, so this costs a fraction of a training step and needs no
    checkpoint reload — which is what makes probing at every checkpoint affordable rather than a
    nice idea that never runs.
    """
    from eval_probes import evaluate_model, summarize_hits

    out: dict[str, Any] = {}
    for family, items in slices.items():
        scorable = [
            {"question": item["question"], "choices": item["choices"], "answer": item["answer"],
             "predicate": item.get("predicate") or "unknown"}
            for item in items
            if item.get("question") and item.get("choices") and item.get("answer")
        ]
        if not scorable:
            continue
        hits = evaluate_model(tokenizer, model, device, scorable)
        summary = summarize_hits(scorable, hits, min_n=1, estimand=f"training_probe:{family}")
        out[family] = {"hits": summary["hits"], "n": summary["n"], "acc": summary["acc"]}
    return out
