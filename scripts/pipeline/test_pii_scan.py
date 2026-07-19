#!/usr/bin/env python3
"""Red-green proofs for pii_scan (no provider, no NER required)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.pii_scan as P  # noqa: E402


def _write_corpus(root: Path, docs: dict[str, str]) -> Path:
    unit = root / "data" / "dom" / "unit" / "10_clean"
    unit.mkdir(parents=True)
    for name, body in docs.items():
        (unit / f"{name}.md").write_text(f"---\ntitle: {name}\n---\n{body}\n", encoding="utf-8")
    return root / "data" / "dom" / "unit"


def test_find_email_and_handle():
    text = "reach me at alice@example.com or @alice_w on twitter"
    hits = P.find_pii(text)
    types = {h["type"] for h in hits}
    assert "email" in types
    assert "handle" in types


def test_mask_text_replaces_spans():
    text = "mail alice@example.com please"
    hits = P.find_pii(text)
    masked = P.mask_text(text, hits)
    assert "alice@example.com" not in masked
    assert "[EMAIL]" in masked


def test_quarantine_dense_chat_export():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        C.BUILD.mkdir(parents=True)
        # Dense: email + phone + handle + ssn → quarantine
        dense = (
            "hey it's bob bob@corp.com +1-415-555-0100 @bobbie ssn 123-45-6789 "
            "also card 4111 1111 1111 1111"
        )
        sparse = "wiki footer contact: editor@wiki.org"
        clean = "Giovanni leads Team Rocket in Kanto."
        unit_dir = _write_corpus(root, {"chat": dense, "wiki": sparse, "lore": clean})
        original = P.iter_editions
        P.iter_editions = lambda *a, **k: [SimpleNamespace(slug="unit", dir=unit_dir)]
        try:
            out = P.run_scan("dom", None, dense_threshold=3)
            assert out["report"]["PII_SCAN_UNAVAILABLE"] is False
            assert out["report"]["quarantined"] >= 1
            assert out["report"]["masked"] >= 1
            assert out["report"]["clean"] >= 1
            q = P.load_quarantine("dom")
            assert q is not None
            # dense chat path is quarantined
            assert any("chat.md" in p for p in q)
            # sparse got a masked copy on disk
            masked_files = list((C.BUILD / "pii" / "masked").rglob("*.md"))
            assert masked_files, "expected at least one masked copy"
            # original 10_clean untouched
            original_chat = (unit_dir / "10_clean" / "chat.md").read_text()
            assert "bob@corp.com" in original_chat
        finally:
            P.iter_editions = original


def test_mandatory_empty_refuses():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        C.BUILD.mkdir(parents=True)
        original = P.iter_editions
        P.iter_editions = lambda *a, **k: []
        try:
            try:
                P.run_scan("dom", None, mandatory=True)
            except ValueError as error:
                assert "mandatory" in str(error).lower() or "refuse" in str(error).lower()
            else:
                raise AssertionError("mandatory empty scan should refuse")
        finally:
            P.iter_editions = original


def test_load_quarantine_missing():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        assert P.load_quarantine("nope") is None


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} PII-SCAN TESTS PASS")
