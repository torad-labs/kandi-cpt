#!/usr/bin/env python3
"""Red-green proofs for extraction recall audit."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.audit_recall as A  # noqa: E402


def test_derive_sample_n():
    assert A.derive_sample_n(0) == 0
    assert A.derive_sample_n(100) == 20  # max(20, 2)
    assert A.derive_sample_n(5000) == 100  # capped
    assert A.derive_sample_n(2000) == 40


def test_coverage_ratio_red_green():
    # 10 assertable, 3 in graph → 0.3 fails 0.6 gate
    assertable = [
        {"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}
        for i in range(10)
    ]
    graph_keys = {A.fact_key(f"s{i}", "rel", f"o{i}") for i in range(3)}
    cov = A.coverage_for_doc(assertable, graph_keys)
    assert cov["n"] == 10
    assert cov["hits"] == 3
    assert abs(cov["ratio"] - 0.3) < 1e-9


def test_run_audit_fails_below_min():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        unit = root / "data" / "dom" / "u"
        clean = unit / "10_clean"
        clean.mkdir(parents=True)
        (clean / "doc.md").write_text("---\n---\nSome lore text here.\n", encoding="utf-8")
        structured = unit / "15_structured"
        structured.mkdir(parents=True)
        # Graph only has 3 of 10 facts
        triples = [
            {"subject_id": f"s{i}", "predicate": "rel", "object": f"o{i}", "attrs": {}, "provenance": {}}
            for i in range(3)
        ]
        (structured / "triples.jsonl").write_text(
            "\n".join(json.dumps(t) for t in triples) + "\n", encoding="utf-8",
        )
        facts = {
            "doc.md": [
                {"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}
                for i in range(10)
            ]
        }
        fix = root / "facts.json"
        fix.write_text(json.dumps(facts), encoding="utf-8")
        original = A.iter_editions
        A.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit)]
        try:
            report = A.run_audit("dom", None, min_coverage=0.6, sample_n=1, facts_json=fix)
        finally:
            A.iter_editions = original
        assert report["status"] == "FAIL"
        assert abs(report["overall"]["ratio"] - 0.3) < 1e-9
        assert report["RECALL_AUDIT_UNAVAILABLE"] is False


def test_run_audit_pass_when_covered():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        unit = root / "data" / "dom" / "u"
        clean = unit / "10_clean"
        clean.mkdir(parents=True)
        (clean / "doc.md").write_text("---\n---\nLore.\n", encoding="utf-8")
        structured = unit / "15_structured"
        structured.mkdir(parents=True)
        triples = [
            {"subject_id": f"s{i}", "predicate": "rel", "object": f"o{i}", "attrs": {}, "provenance": {}}
            for i in range(10)
        ]
        (structured / "triples.jsonl").write_text(
            "\n".join(json.dumps(t) for t in triples) + "\n", encoding="utf-8",
        )
        facts = {
            "doc.md": [
                {"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}
                for i in range(10)
            ]
        }
        fix = root / "facts.json"
        fix.write_text(json.dumps(facts), encoding="utf-8")
        original = A.iter_editions
        A.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit)]
        try:
            report = A.run_audit("dom", None, min_coverage=0.6, sample_n=1, facts_json=fix)
        finally:
            A.iter_editions = original
        assert report["status"] == "PASS"
        assert report["overall"]["ratio"] == 1.0


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} AUDIT-RECALL TESTS PASS")
