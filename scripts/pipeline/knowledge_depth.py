#!/usr/bin/env python3
"""Four depths of knowledge installation, kept separate — and shortcut controls for the deep ones.

A single "knowledge installed" number hides the difference between memorising a surface and
combining two facts. DeepKnowledge's four levels are reported as four metrics that CANNOT be
merged:

  recall       reproduce the installed fact
  extraction   answer an unseen paraphrase of it
  reasoning    combine two installed facts
  association  combine installed knowledge with pre-existing knowledge

The depths only mean something if the deep ones cannot be answered shallowly. A "reasoning" item
whose answer is stated verbatim beside its subject in one training surface is a RECALL item wearing
a reasoning label, and it will inflate the reasoning score of a model that learned nothing but the
surface. That is the same estimand failure as holdout leakage, one level up, so it is audited the
same way: mechanically, against the actual surfaces, before the number is believed.

Implemented controls (computable from the corpus):
  * answer co-occurrence — the answer must not appear together with the subject in any single
    training surface, which is the shortcut that makes a two-hop question one-hop
  * distractor frequency balance — the correct answer must not stand out by corpus frequency, or
    "pick the most familiar name" scores without any knowledge
  * derivable / non-derivable separation — reported apart, never pooled

NOT implemented here, and named rather than implied: entity-swapped controls, counterfactual
controls, and direct-versus-indirect question pairs are generation-side — they require authoring
paired items, which belongs in the probe generator, not in an audit over an existing set.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

SCHEMA = "kandi.knowledge_depth.v1"

DEPTHS = ("recall", "extraction", "reasoning", "association")
# Depths whose whole claim is that a shallow route does not reach the answer.
DEEP_DEPTHS = ("reasoning", "association")


class DepthError(ValueError):
    """A depth report was asked for something it must not provide."""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _contains_phrase(haystack_words: list[str], phrase: str) -> bool:
    needle = _words(phrase)
    if not needle:
        return False
    return any(haystack_words[i:i + len(needle)] == needle
               for i in range(len(haystack_words) - len(needle) + 1))


def answer_cooccurrence_leaks(
    items: Iterable[dict[str, Any]],
    surfaces: Iterable[str],
) -> list[dict[str, Any]]:
    """Deep items whose subject and answer appear TOGETHER in one training surface.

    That co-occurrence is the shortcut: the model can answer by recalling a single sentence rather
    than combining two facts, so the item measures recall while being counted as reasoning.
    """
    surface_words = [_words(text) for text in surfaces]
    leaks = []
    for item in items:
        if item.get("depth") not in DEEP_DEPTHS:
            continue
        subject, answer = item.get("subject") or "", item.get("answer") or ""
        if not subject or not answer:
            continue
        for index, words in enumerate(surface_words):
            if _contains_phrase(words, subject) and _contains_phrase(words, answer):
                leaks.append({
                    "probe_surface_id": item.get("probe_surface_id"),
                    "depth": item["depth"],
                    "subject": subject,
                    "answer": answer,
                    "surface_index": index,
                })
                break
    return leaks


def distractor_frequency_balance(
    items: Iterable[dict[str, Any]],
    surfaces: Iterable[str],
) -> list[dict[str, Any]]:
    """Items where the correct answer is the most corpus-frequent choice too often.

    If the answer is reliably the most familiar name, "pick the most familiar" scores well with no
    knowledge at all. Reported per item; the caller decides what share is tolerable.
    """
    counts: Counter[str] = Counter()
    for text in surfaces:
        counts.update(_words(text))

    def frequency(choice: str | None) -> int:
        tokens = _words(choice)
        return min((counts[t] for t in tokens), default=0) if tokens else 0

    flagged = []
    for item in items:
        choices = list(item.get("choices") or [])
        answer = item.get("answer")
        if len(choices) < 2 or answer not in choices:
            continue
        ranked = sorted(choices, key=frequency, reverse=True)
        # STRICTLY more frequent, not merely first past a stable sort: a tie carries no shortcut,
        # and flagging ties would make a balanced set look rigged.
        if ranked[0] == answer and frequency(answer) > frequency(ranked[1]):
            flagged.append({
                "probe_surface_id": item.get("probe_surface_id"),
                "depth": item.get("depth"),
                "answer": answer,
                "answer_frequency": frequency(answer),
                "runner_up_frequency": frequency(ranked[1]),
            })
    return flagged


def shortcut_audit(
    items: list[dict[str, Any]],
    surfaces: list[str],
    *,
    max_frequency_advantage_share: float = 0.5,
) -> dict[str, Any]:
    """Mechanical verdict on whether the deep items can be answered shallowly.

    Fails on ANY answer co-occurrence: one leaked item is a reasoning score partly earned by
    recall, and the whole point of separating the depths is that they mean different things.
    """
    deep = [i for i in items if i.get("depth") in DEEP_DEPTHS]
    leaks = answer_cooccurrence_leaks(items, surfaces)
    frequency_flags = distractor_frequency_balance(items, surfaces)
    share = (len(frequency_flags) / len(items)) if items else 0.0
    checks = [
        {"check": "no deep item's subject and answer co-occur in a training surface",
         "pass": not leaks, "n_leaks": len(leaks), "n_deep_items": len(deep)},
        {"check": f"answer is the most frequent choice in <{max_frequency_advantage_share:.0%} of items",
         "pass": share < max_frequency_advantage_share,
         "share": round(share, 4), "n_flagged": len(frequency_flags)},
    ]
    return {
        "schema": SCHEMA,
        "checks": checks,
        "verdict": "PASS" if all(c["pass"] for c in checks) else "FAIL",
        "leaks": leaks[:20],
        "frequency_flags": frequency_flags[:20],
    }


def depth_report(scores: dict[str, dict[str, Any]], *, derivable: dict[str, dict[str, Any]] | None = None) -> dict:
    """Per-depth blocks, structurally non-mergeable.

    There is deliberately no overall/combined field: an aggregate is exactly the number the review
    says must not exist, and an absent key is the only reliable way to stop one being quoted.
    Derivable and non-derivable items are reported apart for the same reason.
    """
    unknown = sorted(set(scores) - set(DEPTHS))
    if unknown:
        raise DepthError(f"unknown depth(s) {unknown}; expected {list(DEPTHS)}")
    report: dict[str, Any] = {"schema": SCHEMA, "depths": {}}
    for depth in DEPTHS:
        block = scores.get(depth)
        # An unmeasured depth is recorded as such — never as zero and never omitted, because both
        # read as an answer to a question nobody asked.
        report["depths"][depth] = (
            {"hits": int(block["hits"]), "n": int(block["n"]), "acc": float(block["acc"])}
            if block else {"measured": False}
        )
    if derivable:
        report["derivable_split"] = {
            key: {"hits": int(value["hits"]), "n": int(value["n"]), "acc": float(value["acc"])}
            for key, value in derivable.items()
        }
    return report


def overall(*_args, **_kwargs):  # noqa: D401 - intentionally unavailable
    """Refuse to produce a single knowledge score.

    Present as a function so the refusal is discoverable at the call site rather than a missing
    attribute someone works around by summing the blocks themselves.
    """
    raise DepthError(
        "there is no single knowledge-installed score: recall, extraction, reasoning and "
        "association measure different things, and pooling them hides memorising behind reasoning. "
        "Quote the depth you mean."
    )
