#!/usr/bin/env python3
"""Red-green: a contrast attractor equal to the SUBJECT denies its own sentence.

"Quillspire Keep is located in Millbrook Hollow, not Quillspire Keep." — the object-equality
guard does not catch this, because the attractor equals the subject. Three such records were on
disk when this was found (millbrook/main 2/38, kanto 1/168): strictly false text in the shard.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.generic_contrast import contrast_records  # noqa: E402

_TRIPLES = [{"subject_id": "d:place:keep", "subject": "Quillspire Keep",
             "predicate": "located_in", "object": "Millbrook Hollow", "attrs": {}}]
_GROUPS = {"places": ["Millbrook Hollow", "Quillspire Keep", "Amberfen Crossing"]}
_PREDS = {"located_in": {"group": "places",
                         "templates": ["{subject} is located in {object}, not {attractor}."]}}


def test_attractor_equal_to_the_subject_is_dropped():
    records, _truth, _cap = contrast_records(
        _TRIPLES, _GROUPS, _PREDS, "unit", None, truth_triples=_TRIPLES,
    )
    for rec in records:
        assert "not Quillspire Keep" not in rec["text"], f"self-denying surface emitted: {rec['text']}"


def test_a_genuine_attractor_still_emits():
    """The guard must suppress ONLY the self-denying attractor, never a real contrast."""
    records, _truth, _cap = contrast_records(
        _TRIPLES, _GROUPS, _PREDS, "unit", None, truth_triples=_TRIPLES,
    )
    assert records, "a valid attractor must still produce a surface"
    assert any("Amberfen Crossing" in r["text"] for r in records), [r["text"] for r in records]


def test_object_equality_guard_still_fires():
    """Do not regress the pre-existing guard while adding the new one."""
    records, _truth, _cap = contrast_records(
        _TRIPLES, _GROUPS, _PREDS, "unit", None, truth_triples=_TRIPLES,
    )
    for rec in records:
        assert "not Millbrook Hollow" not in rec["text"], rec["text"]


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} CONTRAST-SELF-DENIAL TESTS PASS")
