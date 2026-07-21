#!/usr/bin/env python3
"""Red-green proofs for booster digest (trust-loop Phase 3)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.booster_digest as B  # noqa: E402


def _graph_domain(root: Path) -> str:
    C.ROOT = root
    C.BUILD = root / "data" / "_build"
    domain = "dom"
    unit = root / "data" / domain / "u" / "15_structured"
    unit.mkdir(parents=True)
    ents = [
        {"id": "dom:character:giovanni", "entity_type": "character",
         "name_canonical": "Giovanni", "aliases": []},
        {"id": "dom:organization:team_rocket", "entity_type": "organization",
         "name_canonical": "Team Rocket", "aliases": []},
        {"id": "dom:species_lore:pikachu", "entity_type": "species_lore",
         "name_canonical": "Pikachu", "aliases": []},
    ]
    triples = [
        {"subject_id": "dom:character:giovanni", "predicate": "leads",
         "object": "dom:organization:team_rocket", "attrs": {}, "provenance": {}},
    ]
    (unit / "entities.resolved.jsonl").write_text(
        "\n".join(json.dumps(e) for e in ents) + "\n", encoding="utf-8",
    )
    (unit / "triples.resolved.jsonl").write_text(
        "\n".join(json.dumps(t) for t in triples) + "\n", encoding="utf-8",
    )
    import pipeline.booster_digest as BD
    from types import SimpleNamespace
    BD.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=root / "data" / domain / "u")]
    return domain


def test_trolling_correction_quarantined():
    """Synthetic trolling ('Pikachu is a Digimon') truth-rejected — never admitted."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain = _graph_domain(root)
        cand = B.harvest_candidate(
            domain,
            signal="correction",
            text="no, Pikachu is a Digimon",
            claimed_subject="Pikachu",
            claimed_predicate="member_of",
            claimed_object="Digimon",  # not in graph
        )
        assert cand["status"] == "quarantined"
        assert "truth" in (cand.get("quarantine_reason") or "")
        # Pool contains it as quarantined, not admitted
        pending = B.load_candidates(domain, pending_only=True)
        assert any(c["id"] == cand["id"] and c["status"] == "quarantined" for c in pending)


def test_supported_fact_not_reinstalled():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain = _graph_domain(root)
        cand = B.harvest_candidate(
            domain,
            signal="correction",
            text="yes Giovanni leads Team Rocket",
            claimed_subject="Giovanni",
            claimed_predicate="leads",
            claimed_object="Team Rocket",
        )
        assert cand["status"] == "quarantined"
        assert "already in graph" in (cand.get("quarantine_reason") or "")


def test_novel_admitted_and_digest_approve_decline():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain = _graph_domain(root)
        # Novel: Giovanni opposes Pikachu (silly but entities exist, fact absent)
        for i in range(3):
            B.harvest_candidate(
                domain,
                signal="correction",
                text=f"correction {i}",
                claimed_subject="Giovanni",
                claimed_predicate="opposes",
                claimed_object="Pikachu",
            )
        dig = B.build_digest(domain, booster_n=1, unit="u")
        assert dig is not None
        assert dig["status"] == "pending_approval"
        assert dig["counts"]["admitted"] >= 1
        # Decline persists candidates
        dec = B.decline_digest(domain, dig["digest_id"])
        assert dec["status"] == "declined"
        assert dec["n_persisted"] >= 1
        pending = B.load_candidates(domain, pending_only=True)
        assert any(c.get("digest_id") in (None, "") for c in pending)

        # Rebuild + approve
        dig2 = B.build_digest(domain, booster_n=2, unit="u")
        assert dig2 is not None
        ap = B.approve_digest(domain, dig2["digest_id"])
        assert ap["status"] == "approved"
        assert ap["n_install_payload"] >= 1


def test_dispute_when_graph_disagrees():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain = _graph_domain(root)
        # Graph: Giovanni leads Team Rocket. Claim: Giovanni leads Pikachu (wrong org type but entity exists)
        cand = B.harvest_candidate(
            domain,
            signal="correction",
            text="no he leads Pikachu",
            claimed_subject="Giovanni",
            claimed_predicate="leads",
            claimed_object="Pikachu",
        )
        assert cand["status"] == "dispute"


def test_empty_mass_sends_no_digest():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain = _graph_domain(root)
        assert B.build_digest(domain) is None


def test_delta_harvest_uses_sha_sidecars_and_skips_unchanged():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain = _graph_domain(root)
        C.DATA = root / "data"
        unit = root / "data" / domain / "u"
        raw = unit / "00_raw"
        clean = unit / "10_clean" / "wiki"
        raw.mkdir(parents=True)
        clean.mkdir(parents=True)

        unchanged = raw / "unchanged.txt"
        unchanged.write_text("steady fact", encoding="utf-8")
        import hashlib
        (clean / "unchanged.clean.json").write_text(json.dumps({
            "raw_sha256": hashlib.sha256(unchanged.read_bytes()).hexdigest(),
        }), encoding="utf-8")

        changed = raw / "changed.txt"
        changed.write_text("new value", encoding="utf-8")
        (clean / "changed.clean.json").write_text(json.dumps({"raw_sha256": "0" * 64}), encoding="utf-8")
        new = raw / "new.txt"
        new.write_text("brand new", encoding="utf-8")

        result = B.harvest_delta_corpus(domain, editions="u")
        assert result["counts"] == {"new": 1, "changed": 1, "unchanged": 1, "harvested": 2}
        candidates = B.load_candidates(domain)
        assert {Path(c["source_path"]).name for c in candidates} == {"changed.txt", "new.txt"}
        assert {c["meta"]["delta_status"] for c in candidates} == {"new", "changed"}
        assert all(len(c["meta"]["raw_sha256"]) == 64 for c in candidates)
        assert all(c["gates"]["admission_basis"] == "own-data" for c in candidates)

        plain = B.render_digest_plain({
            "domain": domain,
            "booster_n": 1,
            "candidates": candidates,
        })
        assert "1 new files, 1 changed files" in plain
        assert "facts from practice" in plain

        retried = B.harvest_delta_corpus(domain, editions="u")
        assert retried["candidate_ids"] == result["candidate_ids"]
        assert len(B.load_candidates(domain)) == 2


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} BOOSTER-DIGEST TESTS PASS")
