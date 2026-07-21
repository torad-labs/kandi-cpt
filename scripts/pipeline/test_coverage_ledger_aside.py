#!/usr/bin/env python3
"""Red-green proof for reversible document-scoped extraction retries."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from pipeline import coverage_ledger_aside as L  # noqa: E402


def test_scopes_triples_and_ledger_by_pipeline_owned_document_identity():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        old = C.ROOT, C.DATA, C.BUILD
        C.ROOT, C.DATA, C.BUILD = root, root / "data", root / "data" / "_build"
        try:
            clean = C.DATA / "domain" / "unit" / "10_clean" / "wiki"
            structured = C.DATA / "domain" / "unit" / "15_structured"
            report_dir = C.BUILD / "recall" / "domain"
            clean.mkdir(parents=True)
            structured.mkdir(parents=True)
            report_dir.mkdir(parents=True)
            (root / "logs").mkdir()
            doc = clean / "weak.md"
            doc.write_text("weak passage", encoding="utf-8")
            sha = __import__("hashlib").sha256(doc.read_bytes()).hexdigest()
            rows = [
                {"predicate": "weak", "provenance": {"doc_id": "wiki/weak.md"}},
                {"predicate": "healthy", "provenance": {"doc_id": "wiki/weak.md"}},
                {"predicate": "healthy", "provenance": {"doc_id": "wiki/keep.md"}},
            ]
            (structured / "triples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            (structured / ".extracted.json").write_text(json.dumps([sha, "keep-sha"]), encoding="utf-8")
            (report_dir / "unit.json").write_text(json.dumps({
                "docs": [{"path": "weak.md", "misses": [{"predicate": "weak"}]}],
            }), encoding="utf-8")
            receipt = L.prepare("domain", "unit", "weak", "iteration")
            assert receipt["triples_removed"] == 1
            assert receipt["ledger_removed"] == 1
            assert len(L.load_jsonl(structured / "triples.jsonl")) == 2
            assert json.loads((structured / ".extracted.json").read_text()) == ["keep-sha"]
            assert (C.BUILD / "recall" / "domain" / "iteration_backup" / "triples.jsonl").is_file()
            restored = L.restore("domain", "unit", "iteration")
            assert restored["restored"] == [
                "data/domain/unit/15_structured/triples.jsonl",
                "data/domain/unit/15_structured/.extracted.json",
                "data/_build/recall/domain/unit.json",
            ]
            assert L.load_jsonl(structured / "triples.jsonl") == rows
            assert json.loads((structured / ".extracted.json").read_text()) == [sha, "keep-sha"]
        finally:
            C.ROOT, C.DATA, C.BUILD = old


def test_multi_predicate_scope_is_a_union_and_preserves_other_predicates():
    report = {
        "docs": [
            {"path": "a.md", "misses": [{"predicate": "one"}]},
            {"path": "b.md", "misses": [{"predicate": "two"}]},
            {"path": "c.md", "misses": [{"predicate": "healthy"}]},
        ]
    }
    assert L.parse_predicates("two, one,two") == ["one", "two"]
    assert L.selected_docs(report, ["one", "two"]) == ["a.md", "b.md"]


def test_scope_repair_parks_only_older_pending_documents():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        old = C.ROOT, C.DATA, C.BUILD
        C.ROOT, C.DATA, C.BUILD = root, root / "data", root / "data" / "_build"
        try:
            clean = C.DATA / "domain" / "unit" / "10_clean"
            structured = C.DATA / "domain" / "unit" / "15_structured"
            clean.mkdir(parents=True)
            structured.mkdir(parents=True)
            (root / "logs").mkdir()
            selected = clean / "selected.md"
            older = clean / "older.md"
            selected.write_text("selected", encoding="utf-8")
            older.write_text("older", encoding="utf-8")
            selected_sha = __import__("hashlib").sha256(selected.read_bytes()).hexdigest()
            older_sha = __import__("hashlib").sha256(older.read_bytes()).hexdigest()
            (structured / ".extracted.json").write_text("[]", encoding="utf-8")
            scope = root / "logs" / "scope_ledger_aside.json"
            scope.write_text(json.dumps({"document_shas": [selected_sha]}), encoding="utf-8")
            receipt = L.park_unselected_pending("domain", "unit", scope)
            assert receipt["parked_n"] == 1
            assert receipt["parked_preexisting_pending"][0]["sha256"] == older_sha
            assert json.loads((structured / ".extracted.json").read_text()) == [older_sha]
        finally:
            C.ROOT, C.DATA, C.BUILD = old


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} COVERAGE-LEDGER-ASIDE TESTS PASS")
