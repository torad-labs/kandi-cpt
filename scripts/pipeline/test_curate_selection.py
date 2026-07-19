#!/usr/bin/env python3
"""Red-green proofs for curate drop product, rarity guard, PII quarantine (no GPU)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import curate_corpus as CC  # noqa: E402
import pipeline.curate_lib as L  # noqa: E402


def test_derive_drop_frac_table():
    assert CC.derive_drop_frac(100, None) == (0.0, "corpus<500→0")
    assert CC.derive_drop_frac(1000, None)[0] == 0.05
    assert CC.derive_drop_frac(9000, None)[0] == 0.10
    frac, src = CC.derive_drop_frac(100, 0.2)
    assert frac == 0.2 and "override" in src


def test_replay_cross_dedup_marks_near_dups():
    domain = [
        "The quick brown fox jumps over the lazy dog near the river bank every morning at dawn.",
        "Completely unique domain lore about Giovanni and Team Rocket in Kanto only.",
    ]
    replay = [
        "The quick brown fox jumps over the lazy dog near the river bank every morning at dawn!!",
        "Unrelated general text about cooking pasta with olive oil and garlic.",
    ]
    hit = L.indices_near_dup_of_reference(domain, replay)
    assert 0 in hit
    assert 1 not in hit


def test_replay_cross_dedup_empty_reference():
    assert L.indices_near_dup_of_reference(["anything here"], []) == set()
    assert L.indices_near_dup_of_reference([], ["x"]) == set()


def test_rarity_guard_protects_bottom_decile_carrier():
    # One doc carries the only mentions of a rare token; others are common filler.
    rare_doc = "zyxzyx unique creature appears here once " + ("common " * 20)
    common_docs = ["common filler text about ordinary things " * 10 for _ in range(20)]
    texts = [rare_doc] + common_docs
    counts = CC.mention_counts(texts)
    rare = CC.rare_entity_set(counts)
    assert "zyxzyx" in rare
    assert CC.doc_carries_rare(rare_doc, rare) is True
    assert CC.doc_carries_rare(common_docs[0], rare) is False


def test_curate_uses_educational_not_perplexity_and_protects_rare(tmp_path_factory=None):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        C.DATA = root / "data"
        C.BUILD.mkdir(parents=True)
        unit = root / "data" / "dom" / "u" / "10_clean"
        unit.mkdir(parents=True)

        # Doc 0: rare-entity carrier, low "fluency" (short) — must be KEPT by rarity guard.
        rare_body = "zyxzyx the obscure entity " + ("word " * 30)
        (unit / "rare.md").write_text(f"---\n---\n{rare_body}\n", encoding="utf-8")
        (unit / "rare.clean.json").write_text(json.dumps({"domain_class": "user-contributed"}), encoding="utf-8")

        # Docs 1..10: common, long, high educational scores — some will be droppable.
        for i in range(10):
            body = ("common ordinary filler content about nothing special " * 15)
            (unit / f"c{i}.md").write_text(f"---\n---\n{body}\n", encoding="utf-8")
            (unit / f"c{i}.clean.json").write_text(json.dumps({"domain_class": "user-contributed"}), encoding="utf-8")

        # Educational scores: rare doc gets BOTTOM score; commons get high — old perplexity-style
        # product would drop the rare doc; new path protects it.
        scores = {}
        for md in unit.glob("*.md"):
            rel = str(md.relative_to(root))
            scores[rel] = 1.0 if md.name == "rare.md" else 5.0
        ed_dir = C.BUILD / "educational"
        ed_dir.mkdir(parents=True)
        (ed_dir / "scores_dom.json").write_text(json.dumps(scores), encoding="utf-8")

        # No PII quarantine file → PII_SCAN_NOT_RUN (ok for this test)

        original_iter = CC.iter_editions
        CC.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit.parent)]

        # Stub GPU bits: embeddings = identity-ish zeros, no real model.
        import torch
        def fake_embed(texts, **kwargs):
            return torch.zeros(len(texts), 8)
        def fake_near_dup(texts, **kwargs):
            return []
        def fake_cluster(emb, k=20):
            return [0] * len(emb)

        try:
            with mock.patch.object(L, "embed_docs", fake_embed), \
                 mock.patch.object(L, "near_dup_groups", fake_near_dup), \
                 mock.patch.object(L, "cluster", fake_cluster):
                # Force a non-zero drop frac so something would be dropped without the guard.
                sys.argv = ["curate_corpus.py", "--domain", "dom", "--drop-frac", "0.5", "--no-difficulty", "--min-words", "5"]
                CC.main()
        finally:
            CC.iter_editions = original_iter

        keep = json.loads((C.BUILD / "curation" / "keep_list_dom.json").read_text())
        meta = json.loads((C.BUILD / "curation" / "curation_meta_dom.json").read_text())
        assert meta["educational_status"] == "EDUCATIONAL_SCORE_OK"
        assert meta["difficulty_role"] == "diagnostic"
        assert meta["rarity_protected_count"] >= 1
        # Rare doc path must be in the keep list despite bottom educational score.
        assert any("rare.md" in p for p in keep), f"rare carrier was dropped: {keep}"


def test_curate_honors_pii_quarantine():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        C.DATA = root / "data"
        C.BUILD.mkdir(parents=True)
        unit = root / "data" / "dom" / "u" / "10_clean"
        unit.mkdir(parents=True)
        for name in ("ok", "secret"):
            body = "Giovanni leads Team Rocket in the Kanto region with many facts " * 10
            (unit / f"{name}.md").write_text(f"---\n---\n{body}\n", encoding="utf-8")
            (unit / f"{name}.clean.json").write_text("{}", encoding="utf-8")

        secret_rel = str((unit / "secret.md").relative_to(root))
        pii_dir = C.BUILD / "pii"
        pii_dir.mkdir(parents=True)
        (pii_dir / "quarantine_dom.json").write_text(json.dumps([secret_rel]), encoding="utf-8")

        original_iter = CC.iter_editions
        CC.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit.parent)]
        import torch
        try:
            with mock.patch.object(L, "embed_docs", lambda texts, **k: torch.zeros(len(texts), 4)), \
                 mock.patch.object(L, "near_dup_groups", lambda texts, **k: []), \
                 mock.patch.object(L, "cluster", lambda emb, k=20: [0] * len(emb)):
                sys.argv = [
                    "curate_corpus.py", "--domain", "dom", "--drop-frac", "0",
                    "--no-difficulty", "--min-words", "5",
                ]
                CC.main()
        finally:
            CC.iter_editions = original_iter

        keep = json.loads((C.BUILD / "curation" / "keep_list_dom.json").read_text())
        meta = json.loads((C.BUILD / "curation" / "curation_meta_dom.json").read_text())
        assert meta["pii_status"] == "PII_SCAN_OK"
        assert meta["quarantined"] == 1
        assert not any("secret.md" in p for p in keep)
        assert any("ok.md" in p for p in keep)


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} CURATE-SELECTION TESTS PASS")
