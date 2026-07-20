#!/usr/bin/env python3
"""Red-green proofs for three-tier routing (trust-loop Phase 4)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.tier_routing as T  # noqa: E402


def test_undeclared_defaults_hot():
    assert T.tier_for("nope", {}) == "hot"


def test_filter_drops_hot():
    tiers = {"leads": "stable", "current_season": "hot"}
    triples = [
        {"predicate": "leads", "subject_id": "a", "object": "b"},
        {"predicate": "current_season", "subject_id": "s", "object": "e"},
        {"predicate": "undeclared_vol", "subject_id": "x", "object": "y"},  # default hot
    ]
    kept, drops = T.filter_triples_by_tier(triples, tiers)
    assert len(kept) == 1 and kept[0]["predicate"] == "leads"
    assert drops["hot"] == 2


def test_pokemon_lore_current_season_is_hot():
    # Live manifest — restore real project paths (other tests rebind C.*)
    root = Path(__file__).resolve().parents[2]
    C.ROOT = root
    C.CONFIGS = root / "configs"
    C.BUILD = root / "data" / "_build"
    tiers = T.load_predicate_tiers("pokemon-lore")
    assert tiers.get("current_season") == "hot"
    assert tiers.get("leads") == "stable"
    rows = T.derivation_rows("pokemon-lore")
    by = {r["predicate"]: r for r in rows}
    assert by["current_season"]["tier"] == "hot"
    assert "hot" in by["current_season"]["reason"].lower() or by["current_season"]["declared"]


def test_style_domain_rejects_hot_predicate():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.CONFIGS = root / "configs"
        C.BUILD = root / "data" / "_build"
        cfg = root / "configs" / "domains"
        cfg.mkdir(parents=True)
        (cfg / "voicedemo.yaml").write_text(
            "name: voicedemo\nkind: style\n"
            "entity_types: [speaker]\n"
            "predicates:\n  speaks_like: {object: [speaker], tier: hot}\n",
            encoding="utf-8",
        )
        try:
            T.assert_style_all_stable("voicedemo")
            raise AssertionError("expected TierError")
        except T.TierError as error:
            assert "uniform-stable" in str(error)


def test_hot_index_and_retrieve():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.CONFIGS = root / "configs"
        C.BUILD = root / "data" / "_build"
        unit = root / "data" / "dom" / "u" / "10_clean" / "wiki"
        unit.mkdir(parents=True)
        (unit / "Alpha.md").write_text("---\n---\nAlpha dragon lives in the mountain.\n", encoding="utf-8")
        (unit / "Beta.md").write_text("---\n---\nBeta river flows to the sea.\n", encoding="utf-8")
        # minimal domain yaml so Domain() works if needed — build_hot_index uses editions
        # stub iter_editions via writing editions registry? simpler: call build with monkeypatch
        import pipeline.tier_routing as TR
        from types import SimpleNamespace
        original = TR.iter_editions
        TR.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=root / "data" / "dom" / "u")]
        try:
            idx = T.build_hot_index("dom")
            assert idx["n_docs"] == 2
            hits = T.retrieve_hot("dom", "dragon mountain")
            assert hits and hits[0]["lane"] == "hot"
            assert hits[0]["provenance"]["sha256"]
        finally:
            TR.iter_editions = original


def test_adaptive_probe_budget_widest_first():
    alloc = T.adaptive_probe_budget(
        {"a": 10, "b": 10, "c": 10},
        total_budget=9,
        wilson_widths={"a": 0.4, "b": 0.1, "c": 0.2},
        booster_touched={"c"},
    )
    assert sum(alloc.values()) == 9
    assert alloc["a"] >= 1 and alloc["b"] >= 1 and alloc["c"] >= 1
    # a has widest width → should get a large share
    assert alloc["a"] >= alloc["b"]


def test_audit_shard_flags_hot_in_hydrate():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.CONFIGS = root / "configs"
        C.BUILD = root / "data" / "_build"
        cfg = root / "configs" / "domains"
        cfg.mkdir(parents=True)
        (cfg / "dom.yaml").write_text(
            "name: dom\nentity_types: [t]\n"
            "predicates:\n  leads: {object: [t], tier: stable}\n"
            "  current_season: {object: [t], tier: hot}\n",
            encoding="utf-8",
        )
        hyd = root / "data" / "dom" / "u" / "25_hydrated"
        hyd.mkdir(parents=True)
        (hyd / "relations.jsonl").write_text(
            json.dumps({"predicate": "current_season", "subject": "s", "object": "o"}) + "\n"
            + json.dumps({"predicate": "leads", "subject": "a", "object": "b"}) + "\n",
            encoding="utf-8",
        )
        import pipeline.tier_routing as TR
        from types import SimpleNamespace
        original = TR.iter_editions
        TR.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=root / "data" / "dom" / "u")]
        try:
            rep = T.audit_shard_has_no_hot("dom")
            assert rep["pass"] is False
            assert rep["n_violations"] == 1
        finally:
            TR.iter_editions = original


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} TIER-ROUTING TESTS PASS")
