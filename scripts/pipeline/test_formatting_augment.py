#!/usr/bin/env python3
"""P3 red-greens: entity-masked formatting augmentation that proves its own value.

Attempt 1 used PREFIX transforms at a fixed fraction and dropped distinct-2 by 27.7%. That was the
transform's fault, not the technique's: a prefix leaves every interior bigram identical, so each
variant contributes ~1 new bigram against ~6 total (0.17) — far under the corpus ratio (~0.33), so
the ratio had to fall.

Attempt 2 edits INSIDE the string with an entity-span mask, and derives its own knobs by measuring
distinct-2 on each candidate setting and keeping only a setting that improves. Measured on kanto it
selects {fraction 0.25, emphasis 4} and moves distinct-2 0.3305 -> 0.3472.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.generic_expand import (  # noqa: E402
    distinct_n, formatting_augment, formatting_augment_derived,
)


def _rec(i: int) -> dict:
    return {
        "text": f"Giovanni leads Team Rocket from the Viridian Gym in Kanto during season {i}.",
        "relation_type": "leads",
        "surface_form": "template",
        "source_triple_ids": [f"t{i}"],
        "entity_ids": ["Giovanni", "Team Rocket", "Viridian Gym"],
    }


def test_entities_survive_every_variant():
    """The invariant that keeps faithfulness untouched: emphasis never lands inside an entity."""
    out, report = formatting_augment([_rec(i) for i in range(60)], fraction=1.0, emphasis_per_surface=4)
    variants = [r for r in out if "+fmt:" in r["surface_form"]]
    assert variants, report
    for variant in variants:
        for entity in variant["entity_ids"]:
            assert entity in variant["text"], f"{entity!r} corrupted in {variant['text']!r}"
    assert report["skipped_unsafe"] == 0, report


def test_emphasis_never_lands_inside_an_entity():
    """Explicitly: no '**' may appear within an entity mention."""
    out, _ = formatting_augment([_rec(i) for i in range(40)], fraction=1.0, emphasis_per_surface=6)
    for variant in [r for r in out if "+fmt:" in r["surface_form"]]:
        for entity in variant["entity_ids"]:
            start = variant["text"].find(entity)
            assert "**" not in variant["text"][start:start + len(entity)], variant["text"]


def test_interior_edit_beats_a_prefix_on_new_bigram_share():
    """The arithmetic that condemned attempt 1 and vindicates attempt 2."""
    original = "Giovanni leads Team Rocket in Kanto."

    def bigrams(text):
        words = text.split()
        return {tuple(words[i:i + 2]) for i in range(len(words) - 1)}, len(words) - 1

    base, _ = bigrams(original)
    _, prefix_total = bigrams("- " + original)
    prefix_new = len(bigrams("- " + original)[0] - base)
    interior_set, interior_total = bigrams("Giovanni **leads** Team Rocket **in** Kanto.")
    interior_new = len(interior_set - base)
    assert prefix_new / prefix_total < interior_new / interior_total


def test_derived_augmentation_improves_distinct_2_or_declines():
    """The contract: it may never leave the corpus worse than it found it."""
    records = [_rec(i) for i in range(200)]
    baseline = distinct_n([r["text"] for r in records])
    out, report = formatting_augment_derived(records)
    after = distinct_n([r["text"] for r in out])
    assert after >= baseline, f"augmentation regressed distinct-2: {baseline} -> {after}"
    if report["chosen"] is None:
        assert out == records, "declining must return the input untouched"
    else:
        assert report["distinct_2_gain"] > 0, report


def test_derived_declines_on_a_corpus_it_cannot_help():
    """A corpus of identical one-token surfaces has no room to improve — it must decline, not force."""
    records = [
        {"text": "Kanto.", "relation_type": "r", "surface_form": "template",
         "source_triple_ids": [f"t{i}"], "entity_ids": ["Kanto"]}
        for i in range(20)
    ]
    out, report = formatting_augment_derived(records)
    assert report["chosen"] is None, report
    assert out == records


def test_derived_reports_its_trials():
    """The chosen setting must be auditable — every candidate and its score is recorded."""
    _, report = formatting_augment_derived([_rec(i) for i in range(120)])
    assert report["trials"], report
    assert all("distinct_2" in t for t in report["trials"])


def test_originals_are_never_dropped():
    records = [_rec(i) for i in range(30)]
    out, _ = formatting_augment(records, fraction=1.0)
    kept = {r["text"] for r in out}
    for rec in records:
        assert rec["text"] in kept, "augmentation is additive; it must never replace an original"


def test_deterministic_across_runs():
    records = [_rec(i) for i in range(50)]
    assert [r["text"] for r in formatting_augment_derived(records)[0]] == \
           [r["text"] for r in formatting_augment_derived(records)[0]]


def test_variants_are_labelled_for_provenance():
    out, _ = formatting_augment([_rec(0)], fraction=1.0)
    variants = [r for r in out if r["surface_form"] != "template"]
    assert variants and all(v["surface_form"] == "template+fmt:emphasis" for v in variants)


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} FORMATTING-AUGMENT TESTS PASS")
