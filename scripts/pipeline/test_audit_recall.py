#!/usr/bin/env python3
"""Red-green proofs for extraction recall audit (Phase F2 class split)."""
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
    # 10 assertable, 3 in graph → 0.3 fails 0.6 gate (no contract/index → RAW==CLEAN)
    assertable = [
        {"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}
        for i in range(10)
    ]
    graph_keys = {A.fact_key(f"s{i}", "rel", f"o{i}") for i in range(3)}
    cov = A.coverage_for_doc(assertable, graph_keys)
    assert cov["n"] == 10
    assert cov["hits"] == 3
    assert abs(cov["ratio"] - 0.3) < 1e-9
    assert cov["raw"]["n"] == 10
    assert cov["raw"]["hits"] == 3


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


# ── Phase F2 class-split proofs ────────────────────────────────────────────────


def _pokemon_contract() -> dict[str, list[str]]:
    """Minimal manifest-shaped contract (mirrors pokemon-lore object types; not hardcoded in prod)."""
    return {
        "occurs_at": ["location"],
        "involves": ["character", "organization", "species_lore"],
        "originates_from": ["region"],
        "located_in": ["region"],
        "member_of": ["organization"],
        "leads": ["organization"],
    }


def test_judge_rejected_type_invalid_wednesday():
    """occurs_at → Wednesday is type-invalid (object not a location) → judge_rejected, never a miss."""
    contract = _pokemon_contract()
    entities = [
        {
            "id": "dom:character:giovanni",
            "entity_type": "character",
            "name_canonical": "Giovanni",
            "aliases": [],
        },
        {
            "id": "dom:location:viridian_gym",
            "entity_type": "location",
            "name_canonical": "Viridian Gym",
            "aliases": [],
        },
    ]
    index = A.build_entity_index(entities, {})
    assertable = [
        {"subject": "Giovanni", "predicate": "occurs_at", "object": "Wednesday"},
        # one real in-scope fact so the clean denominator is non-empty
        {"subject": "Giovanni", "predicate": "occurs_at", "object": "Viridian Gym"},
    ]
    graph_keys = {A.fact_key("Giovanni", "occurs_at", "Viridian Gym")}
    cov = A.coverage_for_doc(
        assertable,
        graph_keys,
        predicates=list(contract),
        contract=contract,
        entity_index=index,
    )
    assert len(cov["judge_rejected"]) == 1
    rej = cov["judge_rejected"][0]
    assert rej["object"] == "Wednesday"
    assert "object_type" in (rej.get("reject_reason") or "")
    assert cov["misses"] == []
    assert not any(m.get("object") == "Wednesday" for m in cov["misses"])
    assert cov["hits"] == 1
    assert cov["n"] == 1  # Wednesday never entered the clean denominator
    # RAW still counted it
    assert cov["raw"]["n"] == 2


def test_out_of_unit_hoenn_assertion_on_kanto_doc():
    """A Hoenn-scoped assertion on a kanto doc lands in out_of_unit, not in misses."""
    contract = _pokemon_contract()
    # Unit entity set is Kanto-only — no Hoenn / Wallace.
    entities = [
        {
            "id": "pokemon_lore_kanto:character:giovanni",
            "entity_type": "character",
            "name_canonical": "Giovanni",
            "aliases": [],
        },
        {
            "id": "pokemon_lore_kanto:region:kanto",
            "entity_type": "region",
            "name_canonical": "Kanto",
            "aliases": [],
        },
        {
            "id": "pokemon_lore_kanto:organization:team_rocket",
            "entity_type": "organization",
            "name_canonical": "Team Rocket",
            "aliases": [],
        },
    ]
    index = A.build_entity_index(entities, {})
    assertable = [
        {"subject": "Wallace", "predicate": "originates_from", "object": "Hoenn"},
        {"subject": "Giovanni", "predicate": "member_of", "object": "Team Rocket"},
    ]
    graph_keys = {A.fact_key("Giovanni", "member_of", "Team Rocket")}
    cov = A.coverage_for_doc(
        assertable,
        graph_keys,
        predicates=list(contract),
        contract=contract,
        entity_index=index,
    )
    assert len(cov["out_of_unit"]) == 1
    assert cov["out_of_unit"][0]["subject"] == "Wallace"
    assert not any(m.get("subject") == "Wallace" for m in cov["misses"])
    assert cov["hits"] == 1
    assert cov["n"] == 1  # Wallace excluded from clean denominator


def test_canonical_match_via_id_map():
    """Graph holds the fact under a canonical alias; id_map match records match_mode=canonical."""
    contract = _pokemon_contract()
    entities = [
        {
            "id": "dom:character:giovanni",
            "entity_type": "character",
            "name_canonical": "Giovanni",
            "aliases": ["The Boss"],
        },
        {
            "id": "dom:organization:team_rocket",
            "entity_type": "organization",
            "name_canonical": "Team Rocket",
            "aliases": [],
        },
        # Alias entity merged away by resolve — only present via id_map
        {
            "id": "dom:character:gio_boss",
            "entity_type": "character",
            "name_canonical": "Gio Boss",
            "aliases": [],
        },
    ]
    id_map = {"dom:character:gio_boss": "dom:character:giovanni"}
    index = A.build_entity_index(entities, id_map)
    # Graph keyed under the canonical display name only.
    graph_keys = {A.fact_key("Giovanni", "member_of", "Team Rocket")}
    assertable = [
        {"subject": "Gio Boss", "predicate": "member_of", "object": "Team Rocket"},
    ]
    cov = A.coverage_for_doc(
        assertable,
        graph_keys,
        predicates=list(contract),
        contract=contract,
        entity_index=index,
    )
    assert cov["hits"] == 1
    assert cov["n"] == 1
    assert cov["misses"] == []
    assert cov["match_mode_counts"].get("canonical", 0) == 1
    assert cov["hits_detail"][0]["match_mode"] == "canonical"
    # RAW (no id_map subject match) still misses — proves the matcher class was the gap.
    assert cov["raw"]["hits"] == 0
    assert cov["raw"]["n"] == 1


def test_genuine_in_scope_absence_still_miss():
    """A genuinely absent in-scope fact still counts as a miss under the clean estimand."""
    contract = _pokemon_contract()
    entities = [
        {
            "id": "dom:character:giovanni",
            "entity_type": "character",
            "name_canonical": "Giovanni",
            "aliases": [],
        },
        {
            "id": "dom:character:agatha",
            "entity_type": "character",
            "name_canonical": "Agatha",
            "aliases": [],
        },
        {
            "id": "dom:organization:team_rocket",
            "entity_type": "organization",
            "name_canonical": "Team Rocket",
            "aliases": [],
        },
        {
            "id": "dom:organization:elite_four",
            "entity_type": "organization",
            "name_canonical": "Elite Four",
            "aliases": [],
        },
    ]
    index = A.build_entity_index(entities, {})
    # Graph has Giovanni→Team Rocket but NOT Agatha→Elite Four.
    graph_keys = {A.fact_key("Giovanni", "member_of", "Team Rocket")}
    assertable = [
        {"subject": "Giovanni", "predicate": "member_of", "object": "Team Rocket"},
        {"subject": "Agatha", "predicate": "member_of", "object": "Elite Four"},
    ]
    cov = A.coverage_for_doc(
        assertable,
        graph_keys,
        predicates=list(contract),
        contract=contract,
        entity_index=index,
    )
    assert cov["hits"] == 1
    assert cov["n"] == 2
    assert abs(cov["ratio"] - 0.5) < 1e-9
    assert len(cov["misses"]) == 1
    assert cov["misses"][0]["subject"] == "Agatha"
    assert cov["judge_rejected"] == []
    assert cov["out_of_unit"] == []


def test_validate_judge_fact_manifest_predicate():
    """Unknown predicate is rejected; valid location object for occurs_at passes."""
    contract = _pokemon_contract()
    entities = [
        {
            "id": "dom:location:indigo_plateau",
            "entity_type": "location",
            "name_canonical": "Indigo Plateau",
            "aliases": [],
        },
    ]
    index = A.build_entity_index(entities, {})
    ok, reason = A.validate_judge_fact(
        {"subject": "X", "predicate": "not_a_real_pred", "object": "Y"},
        predicates=list(contract),
        contract=contract,
        entity_index=index,
    )
    assert ok is False and reason and "predicate_not_in_manifest" in reason
    ok, reason = A.validate_judge_fact(
        {"subject": "X", "predicate": "occurs_at", "object": "Indigo Plateau"},
        predicates=list(contract),
        contract=contract,
        entity_index=index,
    )
    assert ok is True and reason is None


# ── Judge-availability proofs (H1: judge failure is not zero coverage) ────────


def _live_audit(root: Path, n_docs: int, judge_for, *, min_coverage: float = 0.6) -> dict:
    """Build an n_docs corpus whose graph holds every fact, then audit with a stubbed judge.

    ``judge_for(index)`` returns the judge result for doc i: a fact list, or None for
    "the judge never answered". Every scored doc therefore has coverage 1.0, so any drop in
    the reported ratio comes from unavailable docs leaking into the estimand.
    """
    C.ROOT = root
    C.BUILD = root / "data" / "_build"
    unit = root / "data" / "dom" / "u"
    clean = unit / "10_clean"
    clean.mkdir(parents=True)
    for i in range(n_docs):
        (clean / f"doc{i}.md").write_text(f"---\n---\nBody {i}.\n", encoding="utf-8")
    structured = unit / "15_structured"
    structured.mkdir(parents=True)
    triples = [
        {"subject_id": f"s{i}", "predicate": "rel", "object": f"o{i}", "attrs": {}, "provenance": {}}
        for i in range(n_docs)
    ]
    (structured / "triples.jsonl").write_text(
        "\n".join(json.dumps(t) for t in triples) + "\n", encoding="utf-8",
    )

    def _stub(text, _entity_types, _predicates=None):
        index = int(str(text).strip().rstrip(".").split()[-1])
        return judge_for(index)

    original_editions, original_judge = A.iter_editions, A._judge_facts
    A.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit)]
    A._judge_facts = _stub
    try:
        return A.run_audit("dom", None, min_coverage=min_coverage, sample_n=n_docs)
    finally:
        A.iter_editions, A._judge_facts = original_editions, original_judge


def test_unavailable_judge_doc_is_excluded_not_scored_as_a_miss():
    """Red: an unanswered doc scored 0 coverage, dragging the ratio down (the H1 defect)."""
    with tempfile.TemporaryDirectory() as temporary:
        report = _live_audit(
            Path(temporary), 20,
            lambda i: None if i == 0 else [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}],
        )
        assert report["judge_unavailable"]["n_docs"] == 1, report["judge_unavailable"]
        assert report["overall"]["ratio"] == 1.0, (
            f"unavailable doc leaked into the estimand: ratio={report['overall']['ratio']}"
        )
        assert report["overall"]["n"] == 19, "only answered docs enter the denominator"
        assert report["status"] == "PASS", report.get("reason")


def test_partial_judge_availability_fails_closed():
    """Dropping the hard documents must never read as an extraction improvement."""
    with tempfile.TemporaryDirectory() as temporary:
        report = _live_audit(
            Path(temporary), 10,
            lambda i: None if i < 3 else [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}],
        )
        assert report["status"] == "FAIL", report
        assert "judge unavailable" in report.get("reason", ""), report.get("reason")
        assert report["judge_unavailable"]["ratio"] == 0.3
        # The surviving docs are perfectly covered — the FAIL is about trustworthiness, not coverage.
        assert report["overall"]["ratio"] == 1.0


def test_no_judge_output_at_all_is_skip_not_fail():
    """An offline / no-key run legitimately cannot score this estimand."""
    with tempfile.TemporaryDirectory() as temporary:
        report = _live_audit(Path(temporary), 5, lambda _i: None)
        assert report["status"] == "SKIP", report
        assert report["RECALL_AUDIT_UNAVAILABLE"] is True


def test_genuinely_empty_judge_answer_is_scored_not_excluded():
    """[] means "this document asserts nothing" — an answer, not a failure."""
    with tempfile.TemporaryDirectory() as temporary:
        report = _live_audit(
            Path(temporary), 10,
            lambda i: [] if i == 0 else [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}],
        )
        assert report["judge_unavailable"]["n_docs"] == 0, "an empty answer is not unavailable"
        assert report["units"][0]["n_docs_scored"] == 10
        assert report["status"] == "PASS", report.get("reason")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} AUDIT-RECALL TESTS PASS")
