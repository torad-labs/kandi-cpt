#!/usr/bin/env python3
"""Red-green proofs for entity resolution (CPU; no embeddings)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.resolve_entities as R  # noqa: E402


def _ent(eid, name, et="character", continuity=None):
    attrs = {}
    if continuity:
        attrs["continuity"] = continuity
    return {"id": eid, "entity_type": et, "name_canonical": name, "attributes": attrs, "provenance": {}}


def _tri(s, p, o):
    return {"subject_id": s, "predicate": p, "object": o, "attrs": {}, "provenance": {}}


def test_continuity_of_markers():
    assert R.continuity_of(_ent("x", "Giovanni (anime)")) == "anime"
    assert R.continuity_of(_ent("x", "Giovanni (game)")) == "game"
    assert R.continuity_of(_ent("x", "Giovanni", continuity="manga")) == "manga"
    assert R.continuity_of(_ent("x", "Giovanni")) == "unspecified"


def test_exact_name_merges_same_continuity():
    ents = [
        _ent("dom:character:giovanni", "Giovanni"),
        _ent("dom:character:giovanni_boss", "Giovanni"),  # same normalized name
        _ent("dom:character:proton", "Proton"),
    ]
    trips = [
        _tri("dom:character:giovanni", "leads", "dom:org:team_rocket"),
        _tri("dom:character:giovanni_boss", "member_of", "dom:org:team_rocket"),
    ]
    result = R.resolve_unit(
        ents, trips,
        auto_band=0.92, review_band=0.85, review_cap=50,
        device="cpu", use_embeddings=False,
    )
    assert result["cross_merge_violations"] == []
    assert result["n_entities_out"] == 2  # giovanni merged, proton kept
    # One of the two giovanni ids is canonical; the other is in id_map
    gio_ids = {"dom:character:giovanni", "dom:character:giovanni_boss"}
    mapped = gio_ids & set(result["id_map"].keys())
    assert len(mapped) == 1, result["id_map"]
    canon = result["id_map"][next(iter(mapped))]
    assert canon in gio_ids
    # triples rewritten onto the canonical id
    subjects = {t["subject_id"] for t in result["resolved_triples"]}
    assert subjects & gio_ids == {canon}


def test_cross_continuity_never_merges():
    ents = [
        _ent("dom:character:giovanni_game", "Giovanni (game)", continuity="game"),
        _ent("dom:character:giovanni_anime", "Giovanni (anime)", continuity="anime"),
    ]
    trips = [
        _tri("dom:character:giovanni_game", "leads", "dom:org:tr"),
        _tri("dom:character:giovanni_anime", "leads", "dom:org:tr"),
    ]
    result = R.resolve_unit(
        ents, trips,
        auto_band=0.92, review_band=0.85, review_cap=50,
        device="cpu", use_embeddings=False,
    )
    assert result["id_map"] == {}  # no merges
    assert result["n_entities_out"] == 2
    assert len(result["cross_continuity_links"]) >= 1
    assert all(link["merge"] is False for link in result["cross_continuity_links"])
    assert result["cross_merge_violations"] == []


def test_normalize_name_strips_parens():
    assert R._normalize_name("Giovanni (anime)") == "giovanni"
    assert R._normalize_name("Team Rocket's Boss") == "team rocket s boss"


def test_cli_writes_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        unit = root / "data" / "dom" / "u" / "15_structured"
        unit.mkdir(parents=True)
        ents = [
            _ent("dom_u:character:ash", "Ash Ketchum"),
            _ent("dom_u:character:ash_ketchum", "Ash Ketchum"),
            _ent("dom_u:character:misty", "Misty", continuity="anime"),
            _ent("dom_u:character:misty_game", "Misty", continuity="game"),
        ]
        trips = [
            _tri("dom_u:character:ash", "lives_in", "dom_u:location:pallet"),
            _tri("dom_u:character:ash_ketchum", "owns", "dom_u:species:pikachu"),
        ]
        (unit / "entities.jsonl").write_text(
            "\n".join(json.dumps(e) for e in ents) + "\n", encoding="utf-8",
        )
        (unit / "triples.jsonl").write_text(
            "\n".join(json.dumps(t) for t in trips) + "\n", encoding="utf-8",
        )
        original = R.iter_editions
        R.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit.parent)]
        try:
            sys.argv = [
                "resolve_entities.py", "--domain", "dom", "--no-embeddings",
            ]
            R.main()
        finally:
            R.iter_editions = original
        out = C.BUILD / "resolve" / "dom" / "u"
        assert (out / "id_map.json").is_file()
        assert (out / "summary.json").is_file()
        id_map = json.loads((out / "id_map.json").read_text())
        assert id_map  # ash variants merged
        assert (unit / "entities.resolved.jsonl").is_file()
        assert (unit / "triples.resolved.jsonl").is_file()
        summary = json.loads((out / "summary.json").read_text())
        assert summary["cross_merge_violations"] == 0
        # misty game/anime not merged
        misty_ids = {"dom_u:character:misty", "dom_u:character:misty_game"}
        assert not (misty_ids & set(id_map.keys()) == misty_ids and len({id_map.get(i, i) for i in misty_ids}) == 1)




def test_pick_canonical_prefers_type_consistent_id():
    """Longer mismatched id must lose to shorter type-consistent id (kanto location→region)."""
    members = [
        _ent("pokemon_lore_kanto:location:kanto", "Kanto", et="region"),
        _ent("pokemon_lore_kanto:region:kanto", "Kanto", et="region"),
        _ent("kanto:region:kanto", "Kanto", et="region"),
    ]
    # longest is location:… — old tiebreak would pick it; new prefers type match then length
    canon = R.pick_canonical(members)
    assert canon["id"] == "pokemon_lore_kanto:region:kanto", canon["id"]
    assert R.id_type_consistent(canon)


def test_pick_canonical_still_uses_length_among_consistent():
    members = [
        _ent("dom:character:gio", "Giovanni"),
        _ent("dom:character:giovanni", "Giovanni"),
    ]
    canon = R.pick_canonical(members)
    assert canon["id"] == "dom:character:giovanni"


def test_warn_id_type_mismatches_detects_location_region():
    ents = [
        _ent("pokemon_lore_kanto:location:kanto", "Kanto", et="region"),
        _ent("pokemon_lore_kanto:region:kanto", "Kanto", et="region"),
        _ent("dom:character:ash", "Ash"),
    ]
    bad = R.warn_id_type_mismatches(ents, unit="kanto")
    assert len(bad) == 1
    assert bad[0]["id"] == "pokemon_lore_kanto:location:kanto"
    assert bad[0]["id_type_segment"] == "location"
    assert bad[0]["entity_type"] == "region"


def test_resolve_unit_prefers_type_consistent_canonical_and_rewrites_triples():
    ents = [
        _ent("pokemon_lore_kanto:location:kanto", "Kanto", et="region"),
        _ent("pokemon_lore_kanto:region:kanto", "Kanto", et="region"),
        _ent("dom:character:ash", "Ash"),
    ]
    trips = [
        _tri("dom:character:ash", "originates_from", "pokemon_lore_kanto:location:kanto"),
        _tri("dom:character:ash", "lives_in", "pokemon_lore_kanto:region:kanto"),
    ]
    result = R.resolve_unit(
        ents, trips,
        auto_band=0.92, review_band=0.85, review_cap=50,
        device="cpu", use_embeddings=False,
    )
    assert len(result["id_type_mismatches"]) == 1
    # location id should map onto region id
    assert result["id_map"].get("pokemon_lore_kanto:location:kanto") == "pokemon_lore_kanto:region:kanto"
    canon_ids = {e["id"] for e in result["canonical_entities"]}
    assert "pokemon_lore_kanto:region:kanto" in canon_ids
    assert "pokemon_lore_kanto:location:kanto" not in canon_ids
    objs = {t["object"] for t in result["resolved_triples"]}
    assert "pokemon_lore_kanto:location:kanto" not in objs
    assert "pokemon_lore_kanto:region:kanto" in objs




def test_auto_cluster_too_large_threshold():
    assert R.auto_cluster_too_large(100, 100) is True   # whole bucket
    assert R.auto_cluster_too_large(30, 100) is True     # >25%
    assert R.auto_cluster_too_large(25, 100) is False    # ==25% allowed (strict >)
    assert R.auto_cluster_too_large(10, 100) is False
    assert R.auto_cluster_too_large(1, 100) is False


def test_embedding_pass_too_aggressive_threshold():
    assert R.embedding_pass_too_aggressive(400, 1000) is True   # 40% > 30%
    assert R.embedding_pass_too_aggressive(300, 1000) is False  # ==30% allowed
    assert R.embedding_pass_too_aggressive(50, 1000) is False


def test_embedding_mega_cluster_refused_to_review():
    """Synthetic embeddings that collapse a bucket must NOT auto-merge — demote to review.

    Mirrors the kanto 2026-07-19 failure mode (auto_band 0.92 bare-name roberta → one
    mega-cluster per type). We stub pairwise_clusters to return one group of all live ids.
    """
    ents = [
        _ent(f"dom:character:c{i}", f"Char {i}") for i in range(20)
    ]
    trips = [_tri(f"dom:character:c{i}", "member_of", "dom:org:x") for i in range(20)]

    # Force embedding path: no exact-name merges (all names distinct), stub clusterer.
    real_pw = R.pairwise_clusters
    real_embed = R.embed_names

    def _collapse(emb, auto_band, review_band):
        n = emb.shape[0] if hasattr(emb, "shape") else len(ents)
        # one mega auto group, no review pairs
        return [list(range(n))], []

    def _fake_embed(names, device="cpu"):
        import torch
        # orthonormal-ish dummy; content unused because pairwise_clusters is stubbed
        return torch.eye(len(names), 8)

    R.pairwise_clusters = _collapse
    R.embed_names = _fake_embed
    try:
        result = R.resolve_unit(
            ents, trips,
            auto_band=0.92, review_band=0.85, review_cap=500,
            device="cpu", use_embeddings=True,
        )
    finally:
        R.pairwise_clusters = real_pw
        R.embed_names = real_embed

    auto_merges = [m for m in result["merges"] if m.get("band") == "auto"]
    assert auto_merges == [], f"mega-cluster must not auto-merge: {auto_merges}"
    # All 20 entities survive as distinct canonicals
    assert result["n_entities_out"] == 20, result["n_entities_out"]
    # Review queue carries the demoted pairs (guard reason)
    assert result["review"], "refused cluster must land on review queue"
    assert any(r.get("reason") == "embedding-collapse-guard" for r in result["review"])
    assert result["id_map"] == {}


def test_small_auto_cluster_still_merges():
    """A tight 2-entity auto group inside a large bucket is below the 25% guard — still merges."""
    ents = [_ent(f"dom:character:c{i}", f"Char {i}") for i in range(20)]
    # two near-dupes that the stub will pair; rest singleton
    trips = []

    real_pw = R.pairwise_clusters
    real_embed = R.embed_names

    def _pair_first_two(emb, auto_band, review_band):
        return [[0, 1]], []  # only c0+c1 auto; well under 25% of 20

    def _fake_embed(names, device="cpu"):
        import torch
        return torch.eye(len(names), 8)

    R.pairwise_clusters = _pair_first_two
    R.embed_names = _fake_embed
    try:
        result = R.resolve_unit(
            ents, trips,
            auto_band=0.92, review_band=0.85, review_cap=50,
            device="cpu", use_embeddings=True,
        )
    finally:
        R.pairwise_clusters = real_pw
        R.embed_names = real_embed

    auto_merges = [m for m in result["merges"] if m.get("band") == "auto"]
    assert len(auto_merges) == 1, auto_merges
    assert len(auto_merges[0]["aliases"]) == 1
    assert result["n_entities_out"] == 19


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} RESOLVE-ENTITIES TESTS PASS")
