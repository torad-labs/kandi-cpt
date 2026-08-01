#!/usr/bin/env python3
"""Red-green proofs for extraction recall audit (Phase F2 class split)."""
from __future__ import annotations

import json
import os
import subprocess
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


# ── Miss attribution (H1 round 2: is the gap matcher-side or extractor-side?) ──


def _attr_fixture():
    entities = [
        {"id": "dom:character:blaine", "entity_type": "character", "name_canonical": "Blaine",
         "aliases": ["The Hot-Headed Quiz Master"]},
        {"id": "dom:character:misty", "entity_type": "character", "name_canonical": "Misty", "aliases": []},
        {"id": "dom:character:erika", "entity_type": "character", "name_canonical": "Erika", "aliases": []},
    ]
    triples = [
        {"subject_id": "dom:character:blaine", "predicate": "allied_with", "object": "dom:character:misty"},
    ]
    return triples, A.loose_surface_index(entities, {})


def test_miss_present_under_an_alias_is_matcher_recoverable():
    """The fact IS in the graph, just keyed under a surface the matcher did not try."""
    triples, surfaces = _attr_fixture()
    out = A.classify_misses(
        [{"subject": "The Hot-Headed Quiz Master", "predicate": "allied_with", "object": "Misty"}],
        triples, surfaces,
    )
    assert out["matcher_recoverable"] == 1, out
    assert out["absent"] == 0 and out["different_object"] == 0


def test_same_subject_predicate_other_object_is_its_own_class():
    """The extractor saw the relation and disagreed about the target — not a keying failure."""
    triples, surfaces = _attr_fixture()
    out = A.classify_misses(
        [{"subject": "Blaine", "predicate": "allied_with", "object": "Erika"}], triples, surfaces,
    )
    assert out["different_object"] == 1, out
    assert out["matcher_recoverable"] == 0


def test_nothing_for_the_subject_predicate_is_absent():
    """Only re-extraction recovers this; no matcher change can."""
    triples, surfaces = _attr_fixture()
    out = A.classify_misses(
        [{"subject": "Clefairy", "predicate": "evolves_into_lore", "object": "Clefable"}], triples, surfaces,
    )
    assert out["absent"] == 1, out
    assert out["matcher_recoverable"] == 0


def test_attribution_control_graph_triples_are_all_recoverable():
    """Control: replaying the graph's OWN triples must be 100% matcher_recoverable.

    If this ever drops below 1.0 the loose index is broken and every attribution built on it —
    including 'the gap is extraction-side' — is void.
    """
    triples, surfaces = _attr_fixture()
    own = [{"subject": "Blaine", "predicate": "allied_with", "object": "Misty"}]
    out = A.classify_misses(own, triples, surfaces)
    assert out["matcher_recoverable"] == len(own), out
    assert out["matcher_recoverable_ratio"] == 1.0




# ── r3: the coverage ruler is pinned, and the pin is falsifiable ───────────────
#
# The assertion fixture froze 376 assertions and NOTHING else, so the CLEAN denominator was
# still a function of the entity table (subject_in_unit) and the domain manifest (object-type
# contract), and the sampled document list was a function of the .md tree. Measured on kanto
# against fixture 0b50daf3… with a BYTE-IDENTICAL triples file: +55 entity rows drove clean
# 0.1923 → 1.0000 and turned the FAIL into a PASS; a manifest-only edit did the same; one .md
# inserted at sorted position 0 made the seed-0 sample overlap the fixture 3/20 while the report
# still printed n_docs_sampled 20 / judge_unavailable 0.
#
# Each test below asserts BOTH directions from one fixture: the mutation genuinely moves the
# number when the pin is not enforced (so a green result cannot come from an inert mutation —
# the uncalibrated-detector failure), and the pin turns that same mutation into a fail-closed
# run that quotes nothing.


def _ruler_root(temporary: str) -> Path:
    root = Path(temporary)
    C.ROOT = root
    C.BUILD = root / "data" / "_build"
    return root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _ruler_unit(
    root: Path,
    *,
    n_facts: int = 10,
    n_triples: int = 8,
    contract_object: str = "thing",
    extra_entities: list[dict] | None = None,
    extra_docs: int = 0,
) -> dict:
    """A whole scorable unit: docs, entities, triples, id_map, domain manifest, fixture.

    Every fact is (sN, rel, oN) with sN/oN in the entity table, so subject_in_unit passes and the
    object-type contract is satisfiable — which makes an entity or manifest edit the ONLY way to
    move the denominator, which is exactly what these tests need to demonstrate.
    """
    unit = root / "data" / "dom" / "u"
    clean = unit / "10_clean"
    clean.mkdir(parents=True, exist_ok=True)
    (clean / "doc.md").write_text("---\n---\nSome lore text here.\n", encoding="utf-8")
    for i in range(extra_docs):
        # sorts BEFORE doc.md, so it displaces doc.md from a size-1 sample
        (clean / f"000{i}_inserted.md").write_text("---\n---\nInserted body.\n", encoding="utf-8")

    structured = unit / "15_structured"
    structured.mkdir(parents=True, exist_ok=True)
    _write_jsonl(structured / "triples.jsonl", [
        {"subject_id": f"s{i}", "predicate": "rel", "object": f"o{i}", "attrs": {}, "provenance": {}}
        for i in range(n_triples)
    ])
    entities = [
        {"id": f"dom_u:{contract_object}:{name}", "name_canonical": name,
         "entity_type": contract_object, "aliases": []}
        for name in [f"s{i}" for i in range(n_facts)] + [f"o{i}" for i in range(n_facts)]
    ]
    entities.extend(extra_entities or [])
    _write_jsonl(structured / "entities.jsonl", entities)

    id_map_dir = C.BUILD / "resolve" / "dom" / "u"
    id_map_dir.mkdir(parents=True, exist_ok=True)
    (id_map_dir / "id_map.json").write_text(json.dumps({"old:1": "new:1"}), encoding="utf-8")

    manifest = root / "configs" / "domains" / "dom.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "name: dom\n"
        "ontology:\n  entity_types: [%s]\n" % contract_object
        + "predicates:\n  rel: { object: [%s] }\n" % contract_object,
        encoding="utf-8",
    )

    # The unit REGISTRY — the sixth ruler input. It decides how many units exist, so it moves the
    # aggregate denominator without touching any per-unit file.
    registry = root / "configs" / "editions_dom.json"
    registry.write_text(json.dumps([{"unit": "u", "status": "active"}]), encoding="utf-8")

    fixture_path = root / "facts.json"
    fixture_path.write_text(json.dumps({
        "doc.md": [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"} for i in range(n_facts)],
    }), encoding="utf-8")
    return {"unit": unit, "fixture": fixture_path, "manifest": manifest, "registry": registry,
            "entities": structured / "entities.jsonl", "triples": structured / "triples.jsonl",
            "id_map": id_map_dir / "id_map.json"}


def _score(unit: Path, fixture: Path, sample_n: int = 1, **kwargs) -> dict:
    original = A.iter_editions
    A.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit)]
    try:
        return A.run_audit(
            "dom", None, min_coverage=0.6, sample_n=sample_n, facts_json=fixture, **kwargs)
    finally:
        A.iter_editions = original


def _shadow_rows(n: int) -> list[dict]:
    """Pure-addition entity rows that retype existing objects — 0 new triples, 0 new facts."""
    return [
        {"id": f"dom_u:shadow_type:shadow{i}", "name_canonical": f"o{i}",
         "entity_type": "shadow_type", "aliases": []}
        for i in range(n)
    ]


def test_ruler_fingerprint_records_every_scoring_input():
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        report = _score(made["unit"], made["fixture"])
        ruler = report["ruler"]["units"][0]
        for field in ("fixture", "entities", "id_map", "domain_manifest", "triples"):
            assert ruler[field]["sha256"], f"{field} has no digest: {ruler[field]}"
        assert ruler["entities"]["rows"] == 20, ruler["entities"]
        assert ruler["triples"]["rows"] == 8, ruler["triples"]
        assert ruler["id_map"]["entries"] == 1, ruler["id_map"]
        assert ruler["fixture"]["n_assertions"] == 10, ruler["fixture"]
        # The graph is the thing under measurement — recorded, never pinned.
        assert ruler["triples"]["pinned"] is False
        assert "triples" not in A.PINNED_RULER_FIELDS


def test_ruler_pin_blocks_entity_only_edit():
    """(a) shadow entities + 0 triples must not move the REPORTED ratio."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        assert pinned["ruler"]["verified"] is True
        baseline = pinned["overall"]["ratio"]
        assert baseline == 0.8 and pinned["overall"]["n"] == 10, pinned["overall"]

        rows = [json.loads(l) for l in made["entities"].read_text().splitlines() if l.strip()]
        _write_jsonl(made["entities"], rows + _shadow_rows(6))

        # RED calibration: the mutation is REAL — unpinned it moves the number a long way.
        # A pin that "held" against an inert edit would prove nothing.
        unpinned = _score(made["unit"], made["fixture"], use_ruler_pin=False)
        assert unpinned["overall"]["ratio"] != baseline, unpinned["overall"]
        assert unpinned["overall"]["n"] < 10, unpinned["overall"]

        after = _score(made["unit"], made["fixture"])
        assert after["status"] == "FAIL", after["status"]
        assert after["quotable"] is False
        assert after["ruler"]["verified"] is False
        assert any("entities" in p for p in after["ruler"]["problems"]), after["ruler"]["problems"]
        assert "ruler pin mismatch" in after["reason"]


def test_ruler_pin_blocks_contract_only_edit():
    """(b) a contract widening with the entity table untouched must not move it."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        baseline = pinned["overall"]["ratio"]
        entities_sha_before = pinned["ruler"]["units"][0]["entities"]["sha256"]

        made["manifest"].write_text(
            made["manifest"].read_text().replace("{ object: [thing] }", "{ object: [nonexistent] }"),
            encoding="utf-8",
        )

        unpinned = _score(made["unit"], made["fixture"], use_ruler_pin=False)
        assert unpinned["overall"]["n"] != 10, unpinned["overall"]   # denominator really moved
        assert unpinned["overall"]["ratio"] != baseline, unpinned["overall"]

        after = _score(made["unit"], made["fixture"])
        assert after["status"] == "FAIL"
        assert after["quotable"] is False
        assert any("domain_manifest" in p for p in after["ruler"]["problems"]), after["ruler"]
        # the entity table was NOT touched, and the pin says so — the two halves are independent
        assert after["ruler"]["units"][0]["entities"]["sha256"] == entities_sha_before
        assert not any("entities" in p for p in after["ruler"]["problems"])


def test_ruler_pin_blocks_id_map_edit():
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        _score(made["unit"], made["fixture"], pin_ruler=True)
        made["id_map"].write_text(json.dumps({"old:1": "new:1", "old:2": "new:2"}), encoding="utf-8")
        after = _score(made["unit"], made["fixture"])
        assert after["status"] == "FAIL"
        assert any("id_map" in p for p in after["ruler"]["problems"]), after["ruler"]["problems"]


def test_ruler_pin_allows_graph_only_change_in_both_directions():
    """(c) a genuine added triple MUST still move it — and a deleted one must move it back.

    This is the equal-and-opposite failure the item names: a denominator pinned so hard that real
    extraction changes stop registering is as broken as one an entity edit can drive to 1.0.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        baseline = pinned["overall"]["ratio"]
        assert baseline == 0.8

        triples = [json.loads(l) for l in made["triples"].read_text().splitlines() if l.strip()]
        _write_jsonl(made["triples"], triples + [
            {"subject_id": "s8", "predicate": "rel", "object": "o8", "attrs": {}, "provenance": {}},
        ])
        gained = _score(made["unit"], made["fixture"])
        assert gained["ruler"]["verified"] is True, gained["ruler"]["problems"]
        assert gained["quotable"] is True
        assert gained["overall"]["ratio"] == 0.9, gained["overall"]
        assert gained["overall"]["n"] == 10, "denominator must not dilute on a graph-only gain"

        _write_jsonl(made["triples"], triples[:4])
        lost = _score(made["unit"], made["fixture"])
        assert lost["ruler"]["verified"] is True
        assert lost["quotable"] is True
        assert lost["overall"]["ratio"] == 0.4, lost["overall"]


def test_fixture_sample_mismatch_fails_closed():
    """(d) sample/fixture misalignment must FAIL, not score unmatched docs as empty."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        before = _score(made["unit"], made["fixture"])
        assert before["fixture_sample"]["n_docs_matched"] == 1
        assert before["fixture_sample"]["n_docs_missing"] == 0

        # A new .md enters the corpus. The fixture, pinned earlier, does not cover it. Both docs
        # are sampled so the misalignment is deterministic rather than seed-dependent.
        (made["unit"] / "10_clean" / "0000_inserted.md").write_text(
            "---\n---\nInserted body.\n", encoding="utf-8")
        after = _score(made["unit"], made["fixture"], sample_n=2)
        assert after["status"] == "FAIL", after["status"]
        assert after["quotable"] is False
        assert after["fixture_sample"]["n_docs_missing"] == 1, after["fixture_sample"]
        assert after["fixture_sample"]["n_docs_matched"] == 1
        assert "fixture covers 1/2 sampled documents" in after["reason"], after["reason"]
        assert "0000_inserted.md" in after["reason"], after["reason"]
        # the silent-collapse signature: sampled count and judge availability BOTH still read clean
        assert after["judge_unavailable"]["n_docs"] == 0
        assert after["judge_unavailable"]["n_docs_sampled"] == 2
        # RED: without the check the unmatched doc scores as `assertable = []` and simply
        # vanishes — the ratio is IDENTICAL to the one-document run it silently became.
        assert after["overall"]["n"] == before["overall"]["n"], "the 2nd doc contributed 0/0"


def test_empty_clean_denominator_is_fail_not_pass():
    """A gate cleared by a denominator of 0 is the zero-from-an-uncalibrated-detector shape."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        made["manifest"].write_text(
            made["manifest"].read_text().replace("{ object: [thing] }", "{ object: [nonexistent] }"),
            encoding="utf-8",
        )
        report = _score(made["unit"], made["fixture"])
        assert report["overall"]["n"] == 0, report["overall"]
        assert report["raw"]["n"] == 10, report["raw"]
        assert report["overall"]["ratio"] == 1.0, "the vacuous ratio is still computed…"
        assert report["status"] == "FAIL", "…but it must never clear the gate"
        assert report["quotable"] is False
        assert "empty denominator" in report["reason"], report["reason"]


def test_document_corpus_is_pinned_so_a_corpus_only_edit_fails_closed():
    """(round-3 objection 2) The DOCUMENT CORPUS is a denominator input and was unpinned.

    `derive_sample_n` and the seed-0 sample are functions of what `collect_clean_docs` returns, so
    changing the corpus changes which documents are scored while every other pinned digest stays
    byte-identical. Measured on real kanto by a round-2 reviewer: parking 104 of 105 .md files
    reproduced the full ratio exploit with all four then-pinned digests unchanged.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        assert pinned["ruler"]["verified"] is True, pinned["ruler"]
        before = A.documents_digest(made["unit"])
        assert before["sha256"] and before["n_docs"] == 1, before

        # Corpus-only edit: add a document. No entity, triple, id_map or manifest byte changes.
        (made["unit"] / "10_clean" / "0000_inserted.md").write_text(
            "---\n---\nInserted body.\n", encoding="utf-8")
        after = A.documents_digest(made["unit"])
        assert after["sha256"] != before["sha256"], "corpus edit must move the documents digest"
        assert after["n_docs"] == 2, after

        rescored = _score(made["unit"], made["fixture"])
        problems = rescored["ruler"]["units"][0]["pin"]["problems"]
        assert any(p.startswith("documents:") for p in problems), problems
        assert rescored["ruler"]["verified"] is False, rescored["ruler"]


def test_documents_pin_still_lets_a_genuine_graph_change_score():
    """The pin must not be so tight that real extraction gains stop registering.

    The graph is the thing under measurement: adding a triple changes no ruler input, so the
    documents pin stays verified and the ratio is free to move.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        assert pinned["ruler"]["verified"] is True
        before_digest = A.documents_digest(made["unit"])["sha256"]

        rows = [json.loads(l) for l in made["triples"].read_text().splitlines() if l.strip()]
        rows.append({"subject_id": "s9", "predicate": "rel", "object": "o9",
                     "attrs": {}, "provenance": {}})
        _write_jsonl(made["triples"], rows)

        rescored = _score(made["unit"], made["fixture"])
        assert A.documents_digest(made["unit"])["sha256"] == before_digest, \
            "a graph change must NOT move the documents digest"
        assert rescored["ruler"]["verified"] is True, rescored["ruler"]["units"][0]["pin"]
        assert rescored["overall"]["ratio"] > pinned["overall"]["ratio"], (
            f"the genuine new triple must still move the ratio: "
            f"{pinned['overall']['ratio']} -> {rescored['overall']['ratio']}")


def test_fixture_digest_binds_only_when_a_fixture_was_supplied():
    """(round-3 objection 1) The fixture is a REPLAY input, not a property of the corpus.

    The production invocation scores against the live judge and passes no --facts-json. Enforcing
    the fixture digest unconditionally failed every real run closed with
    "fixture: file absent on disk" and quoted no ratio at all — a pin that bricks the path it
    exists to protect. The other four digests still bind when the fixture is absent.
    """
    fingerprint = {
        "fixture": {"sha256": None},
        "entities": {"sha256": "e" * 64},
        "id_map": {"sha256": "i" * 64},
        "domain_manifest": {"sha256": "m" * 64},
        "documents": {"sha256": "d" * 64},
        "editions_registry": {"sha256": "r" * 64},
        "sample": {"sha256": "s" * 64},
    }
    pin = {"pinned": {"fixture": "f" * 64, "entities": "e" * 64, "id_map": "i" * 64,
                      "domain_manifest": "m" * 64, "documents": "d" * 64,
                      "editions_registry": "r" * 64, "sample": "s" * 64}}

    supplied = A.compare_ruler_pin(pin, fingerprint, fixture_supplied=True)
    assert any(p.startswith("fixture:") for p in supplied), supplied

    replayless = A.compare_ruler_pin(pin, fingerprint, fixture_supplied=False)
    assert replayless == [], replayless

    # …and skipping the fixture must NOT smuggle the other four past the check.
    fingerprint["entities"]["sha256"] = "x" * 64
    still_bound = A.compare_ruler_pin(pin, fingerprint, fixture_supplied=False)
    assert any(p.startswith("entities:") for p in still_bound), still_bound


def test_unit_registry_is_pinned_so_a_registry_only_edit_fails_closed():
    """(round-3 objection) The SIXTH input: the registry decides how many units exist.

    Five pins covered the inputs of ONE unit. overall_ratio re-aggregates over whatever
    iter_editions yields, and iter_editions yields whatever configs/editions_<domain>.json lists —
    so the registry moves the aggregate denominator with every per-unit digest byte-identical.
    All three round-3 reviewers found this independently.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        assert pinned["ruler"]["verified"] is True, pinned["ruler"]

        before = {json.loads(l)["id"] for l in made["entities"].read_text().splitlines() if l.strip()}
        made["registry"].write_text(
            json.dumps([{"unit": "u", "status": "active"}, {"unit": "u2", "status": "active"}]),
            encoding="utf-8")
        after = {json.loads(l)["id"] for l in made["entities"].read_text().splitlines() if l.strip()}
        assert before == after, "the registry edit must touch no per-unit file"

        rescored = _score(made["unit"], made["fixture"])
        problems = rescored["ruler"]["units"][0]["pin"]["problems"]
        assert any(p.startswith("editions_registry:") for p in problems), problems
        assert rescored["ruler"]["verified"] is False
        assert rescored["quotable"] is False, rescored["quotable_note"]


def test_a_unit_with_no_pin_is_not_quotable():
    """Absence of a pin was verification by default — the root cause under the registry finding.

    `unit_problems` is only computed when a pin exists, so an unpinned unit produced no problems
    and sailed through as quotable. Since the unit SLUG decides the pin filename, renaming or
    adding a unit silently unpinned it instead of failing closed.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        unpinned = _score(made["unit"], made["fixture"])
        assert unpinned["ruler"]["units"][0]["pin"]["present"] is False
        assert unpinned["ruler"]["units"][0]["pin"]["problems"] == []   # nothing to disagree with
        assert unpinned["quotable"] is False, (
            "an unpinned unit was never checked against anything and must not be quotable: "
            + str(unpinned.get("quotable_note")))

        # …and pinning it makes it quotable again, so this is fail-closed, not blanket refusal.
        pinned = _score(made["unit"], made["fixture"], pin_ruler=True)
        assert pinned["ruler"]["verified"] is True and pinned["quotable"] is True, pinned["ruler"]


def test_no_ruler_pin_flag_is_not_a_bypass():
    """Skipping the check is itself reportable, or the flag is a bypass with a polite name."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        _score(made["unit"], made["fixture"], pin_ruler=True)
        rows = [json.loads(l) for l in made["entities"].read_text().splitlines() if l.strip()]
        _write_jsonl(made["entities"], rows + _shadow_rows(6))

        ignored = _score(made["unit"], made["fixture"], use_ruler_pin=False)
        assert ignored["ruler"]["problems"] == [], "the pin was not enforced, so there is no mismatch"
        assert ignored["ruler"]["pins_ignored"] == ["u"], ignored["ruler"]["pins_ignored"]
        assert ignored["quotable"] is False, "an ignored pin must not yield a quotable number"
        assert "DO NOT QUOTE" in ignored["quotable_note"]
        assert ignored["ruler"]["units"][0]["pin"]["ignored"] is True


def test_repin_records_what_it_superseded():
    """Re-pinning is the escape hatch and it is not sealed — but it cannot be silent."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        first = _score(made["unit"], made["fixture"], pin_ruler=True)
        old_entities_sha = first["ruler"]["units"][0]["entities"]["sha256"]

        rows = [json.loads(l) for l in made["entities"].read_text().splitlines() if l.strip()]
        _write_jsonl(made["entities"], rows + _shadow_rows(6))
        _score(made["unit"], made["fixture"], pin_ruler=True)

        pin = json.loads(A.ruler_pin_path("dom", "u").read_text())
        assert pin["supersedes"]["pinned"]["entities"] == old_entities_sha, pin["supersedes"]
        assert pin["supersedes"]["recorded"]["entity_rows"] == 20
        assert pin["recorded"]["entity_rows"] == 26, pin["recorded"]
        # the graph is recorded at pin time but is NOT one of the enforced digests
        assert "triples" not in pin["pinned"]
        assert pin["recorded"]["triples_rows_at_pin_time"] == 8

        after = _score(made["unit"], made["fixture"])
        assert after["ruler"]["verified"] is True, "a re-pinned ruler verifies against itself"


def test_incomplete_pin_is_a_mismatch_not_a_pass():
    """A pin that does not cover a field must not silently verify it."""
    fingerprint = {f: {"sha256": "beef"} for f in A.PINNED_RULER_FIELDS}
    complete = {"pinned": {f: "beef" for f in A.PINNED_RULER_FIELDS}}
    assert A.compare_ruler_pin(complete, fingerprint) == []
    partial = {"pinned": {"fixture": "beef"}}
    problems = A.compare_ruler_pin(partial, fingerprint)
    assert len(problems) == len(A.PINNED_RULER_FIELDS) - 1, problems
    assert all("pin carries no digest" in p for p in problems), problems
    missing_file = {f: {"sha256": None} for f in A.PINNED_RULER_FIELDS}
    assert all("file absent" in p for p in A.compare_ruler_pin(complete, missing_file))


def test_clean_unfiltered_arm_reports_the_contract_reject_mass():
    """The estimand choice must be an argued number, not a silent default."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        made["manifest"].write_text(
            made["manifest"].read_text().replace("{ object: [thing] }", "{ object: [other] }"),
            encoding="utf-8",
        )
        report = _score(made["unit"], made["fixture"])
        # CLEAN drops every fact on the object-type contract; the unfiltered arm keeps them.
        assert report["overall"]["n"] == 0, report["overall"]
        assert report["clean_unfiltered"]["n"] == 10, report["clean_unfiltered"]
        assert report["clean_unfiltered"]["hits"] == 8, report["clean_unfiltered"]
        assert report["clean_unfiltered"]["ratio"] == 0.8
        assert report["reject_reasons"].get("object_type_mismatch") == 10, report["reject_reasons"]
        assert report["raw"]["n"] == 10


# ── the replay-determinism lock ──────────────────────────────────────────────────────────────
# Replay reproducibility is the WHOLE point of the fixture: --facts-json exists so a later
# iteration re-runs the identical assertion set and any movement is attributable to the graph
# alone. That property holds on HEAD but nothing gates it, so a later change that serializes a
# SET, a sampler draw, a uuid or a TIMESTAMP into the report would destroy it silently and every
# number after it would be uncomparable in a way no existing test can see.
#
# THREE SEPARATE PROCESSES at DIFFERENT PYTHONHASHSEED, comparing BYTES ON DISK — not calls in
# one interpreter. Set iteration order over strings is a function of the per-process hash seed,
# so a set leaking into the report is IDENTICAL on two in-process calls and only diverges across
# processes. Not hypothetical in this tree: test_dedup_surfaces.py:58 guards the same class after
# a live `hash(text)` leak wrote dedup keys tokenize_corpus could never reproduce (8063 stored
# exclusions, ZERO whole-key matches). This mirrors that test's shape, including its three seeds.
#
# The corpus is deliberately WIDE (6 documents, 3 sampled, 38 fixture entries) — but wide for
# the FIXTURE-KEY collections only (assertions, per-doc keys, coverage rows). The happy path
# leaves missing_sample / judge_rejected_sample / out_of_unit_sample / reject_reasons all EMPTY,
# and a set of size <= 1 iterates identically under every hash seed, so this arm alone cannot
# catch a leak in the FAILURE-path collections. That is what the wide-MISS arm below exists for:
# a 30-doc unit whose fixture covers one doc drives n_docs_missing >= 20, making the miss-path
# sets wide enough that seed-dependent iteration would show. Neither arm covers everything;
# together they cover both families.
_REPLAY_WORKER = '''
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# SHIFTED CLOCK, a DIFFERENT epoch per worker, installed BEFORE audit_recall is imported so a
# `from datetime import date` inside it binds the fake too. Without it the lock is blind to
# COARSE timestamps: the replays run milliseconds apart, so a date-, hour- or year-granularity
# stamp is byte-identical today and destroys the property the moment the next iteration runs on
# another day — the multi-week cadence the coverage campaign runs on. The clock is SHIFTED, not
# frozen: it still ticks, so an elapsed-duration leak (`time.time() - t0`) stays nondeterministic
# and stays caught, which a hard-frozen clock would silently collapse to a constant 0.0.
import datetime as _dt
import time as _time
_real_time = _time.time
_OFFSET = float(sys.argv[5]) - _real_time()
_time.time = lambda: _real_time() + _OFFSET
_real_localtime, _real_gmtime = _time.localtime, _time.gmtime
_real_strftime, _real_ctime, _real_asctime = _time.strftime, _time.ctime, _time.asctime
_time.localtime = lambda t=None: _real_localtime(_time.time() if t is None else t)
_time.gmtime = lambda t=None: _real_gmtime(_time.time() if t is None else t)
# time.strftime(fmt) resolves its default tuple in C, not via the module attribute, so each
# entry point is wrapped individually — patching localtime alone leaves strftime on the real clock.
_time.strftime = lambda f, t=None: _real_strftime(f, _real_localtime(_time.time()) if t is None else t)
_time.ctime = lambda t=None: _real_ctime(_time.time() if t is None else t)
_time.asctime = lambda t=None: _real_asctime(_real_localtime(_time.time()) if t is None else t)
_time.time_ns = lambda: int(_time.time() * 1_000_000_000)


class _ShiftedDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(_time.time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.fromtimestamp(_time.time(), _dt.timezone.utc).replace(tzinfo=None)

    @classmethod
    def today(cls):
        return cls.fromtimestamp(_time.time())


class _ShiftedDate(_dt.date):
    @classmethod
    def today(cls):
        return cls.fromtimestamp(_time.time())


_dt.datetime = _ShiftedDateTime
_dt.date = _ShiftedDate

import config as C
import pipeline.audit_recall as A

root = Path(sys.argv[1])
C.ROOT = root
C.BUILD = root / "data" / "_build"
A.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=root / "data" / "dom" / "u")]


def _forbidden(*a, **k):
    raise AssertionError("replay consulted the judge")


def _no_network(*a, **k):
    raise AssertionError("replay opened a network connection")


# The ONLY egress point in run_audit is _judge_facts (audit_recall.py:1597, the sole caller of
# try_resolve_provider). Do NOT swap this for scrubbing $OPENROUTER_API_KEY: llm_provider.py also
# resolves the key from `secret-tool lookup service=openrouter`, so scrubbing looks like
# protection and is not. This class has already fired for real in this repo — a red-proof that
# deleted the replay guard made the suite issue live HTTP requests from inside the gate sweep.
assert hasattr(A, "_judge_facts"), "the judge entry point moved - this tripwire is stale"
A._judge_facts = _forbidden
# Belt and braces, because one arm below deliberately drives the LIVE path to prove the tripwire
# above is reachable rather than decorative. A future rename of the judge entry point would turn
# that arm into real provider egress from inside run_tests.sh, so the TRANSPORT is stopped too:
# urllib.request.urlopen is the only one audit_recall uses (:1646) and socket.socket is under it.
import socket as _socket
import urllib.request as _urlreq
_urlreq.urlopen = _no_network
_socket.socket = _no_network

# A worker that does not write its own artifacts would silently compare a PREVIOUS worker's file
# with itself, so a total non-write would read as perfectly reproducible. Delete first: a
# non-write is then a loud FileNotFoundError.
report_path = C.BUILD / "recall" / "report_dom.json"
unit_path = C.BUILD / "recall" / "dom" / "u.json"
report_path.unlink(missing_ok=True)
unit_path.unlink(missing_ok=True)

facts_json = None if sys.argv[6] == "-" else root / sys.argv[6]
A.run_audit("dom", None, min_coverage=0.6, sample_n=int(sys.argv[7]), facts_json=facts_json)
if len(sys.argv) > 8 and sys.argv[8] == "inject-stamps":
    # The two calendar forms the shim widening + non-day-aligned stride exist to catch: a date
    # stamp (moves on the 400-day component) and a seconds-of-day stamp (moves on the 3661s
    # component). Injected AFTER scoring so the probe perturbs only the artifact under comparison.
    _probe = json.loads(report_path.read_text())
    _probe["_probe_calendar_stamp"] = _time.strftime("%Y-%m-%d")
    _probe["_probe_seconds_of_day"] = int(_time.time()) % 86400
    report_path.write_text(json.dumps(_probe, indent=1, sort_keys=True))
Path(sys.argv[2]).write_bytes(report_path.read_bytes())
Path(sys.argv[3]).write_bytes(unit_path.read_bytes())
# Canary: this process's set-iteration order. Workers whose canaries MATCH were not actually
# running under different hash seeds, which would make byte-identity vacuous.
Path(sys.argv[4]).write_text(json.dumps(list({"canary-%02d" % i for i in range(32)})))
'''


def _replay(worker: Path, root: Path, out: Path, tag: str, hashseed: str,
            epoch: float = 1.6e9, fixture: str = "facts.json", expect_rc: int = 0,
            sample_n: int = 3, inject_stamps: bool = False):
    """One replay in its own interpreter. Returns (report, unit, canary, stdout, stderr).

    `expect_rc` is the calibration lever, not a convenience: expect_rc=1 asserts the worker DIED,
    which is how the no-fixture arm below proves `_judge_facts = _forbidden` is a live tripwire
    rather than three decorative lines that never execute.
    """
    # PYTHONPATH comes from the module UNDER TEST, not from this file's location: the worker must
    # import the same audit_recall this file imported.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(A.__file__).resolve().parents[1])
    env["PYTHONHASHSEED"] = hashseed
    report, unit, canary = out / f"report.{tag}", out / f"unit.{tag}", out / f"canary.{tag}"
    done = subprocess.run(
        [sys.executable, str(worker), str(root), str(report), str(unit), str(canary),
         str(epoch), fixture, str(sample_n)] + (["inject-stamps"] if inject_stamps else []),
        env=env, capture_output=True, text=True, timeout=300,
    )
    if done.returncode != expect_rc and (
            "replay consulted the judge" in done.stderr
            or "replay opened a network connection" in done.stderr):
        raise AssertionError(
            f"replay consulted the judge ({tag}) - the no-judge guard at audit_recall.py:1144 is "
            f"gone and the gate sweep is one keyring lookup from live provider egress: "
            f"{done.stderr[-600:]}")
    assert done.returncode == expect_rc, \
        f"replay {tag} rc={done.returncode} (expected {expect_rc}): {done.stderr[-2000:]}"
    if expect_rc != 0:
        return None, None, None, done.stdout, done.stderr
    # stdout too: the printed receipt at audit_recall.py:1566 is "the surface receipts copy from"
    # and is the third of three replay-reachable serialization sites (:1332 <unit>.json,
    # :1563 report_<domain>.json, :1566 the print). The two file comparisons are blind to it.
    return report.read_bytes(), unit.read_bytes(), canary.read_text(), done.stdout, done.stderr


def _replay_corpus(root: Path) -> dict:
    """A MULTI-DOC, OVER-COVERED fixture. Both properties are load-bearing.

    * 6 docs sampled 3 puts the SAMPLE SELECTOR inside the lock. With one document
      rng.sample(docs, 1) returns that document under every seed, so sampler nondeterminism
      (audit_recall.py:274, the exact drift select_sample's docstring exists to prevent) is
      invisible no matter how the report is compared.
    * 32 phantom entries make the fixture-derived collections large enough to REORDER. A set of
      size <= 1 iterates identically under every hash seed, so against a 1-entry fixture only a
      leak whose size is independent of the data could ever fail — the lock would lock nothing.

    Every document on disk is covered, so the extra documents are not fixture_missing and the
    calibration below fails for the reason it names rather than for a corpus mistake.
    """
    made = _ruler_unit(root, extra_docs=5)
    facts = [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"} for i in range(10)]
    fixture_doc = {p.name: facts for p in sorted((made["unit"] / "10_clean").glob("*.md"))}
    for i in range(32):
        fixture_doc[f"phantom-{i:02d}.md"] = []
    made["fixture"].write_text(json.dumps(fixture_doc), encoding="utf-8")
    return made


def test_replay_is_byte_identical_across_processes():
    """Consecutive replays over the same pinned ruler must produce byte-identical output."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _replay_corpus(root)
        pinned = _score(made["unit"], made["fixture"], sample_n=3, pin_ruler=True)
        assert pinned["ruler"]["verified"] is True, pinned["ruler"]["problems"]

        worker = root / "replay_worker.py"
        worker.write_text(_REPLAY_WORKER, encoding="utf-8")
        out = root / "replays"
        out.mkdir()
        # THREE seeds, matching test_dedup_surfaces.py:58. Measured over 60 string prefixes, the
        # fraction of leaks that iterate identically (and so slip through): a 3-element leak
        # 7/60 under (0, 12345) vs 4/60 under (0, 1, 12345); 4-element 2/60 vs 0/60; 5 or more
        # 0/60 either way. Residual, stated plainly: a 2-element leak is still 16/60 blind even
        # with three seeds — inherent to short sets, not worth chasing. The epochs are 400 days
        # PLUS 3661 seconds apart: the day component moves date- and year-granularity stamps, and
        # the 3661s (1h 1m 1s) moves hour/minute/second stamps — a whole-day stride left every
        # %H:%M and seconds-of-day stamp IDENTICAL across arms, so only the SHIM covered the time
        # module and nothing proved it. The stride diversifies the stamps; the shim shifts the
        # clock; the arms below need both.
        seeds = ("0", "1", "12345")
        runs = [_replay(worker, root, out, tag, seed, 1.6e9 + i * (400 * 86400 + 3661))
                for i, (tag, seed) in enumerate(zip("abc", seeds))]

        # RED calibration 1 — the canary. Workers that shared a hash seed would agree for free.
        canaries = [run[2] for run in runs]
        assert len(set(canaries)) == len(canaries), \
            "replays shared a hash seed - byte-identity is vacuous"
        # RED calibration 2 — two identical FAILURES are also byte-identical. Every replay must be
        # a real, quotable, ruler-verified, fully-covered score of the pinned ruler, over a corpus
        # wide enough for the comparison to mean something.
        scored = json.loads(runs[0][0])
        assert scored["quotable"] is True, scored["quotable_note"]
        assert scored["ruler"]["verified"] is True, scored["ruler"]["problems"]
        assert scored["overall"]["ratio"] == 0.8 and scored["overall"]["n"] == 30, scored["overall"]
        assert scored["fixture_sample"]["n_docs_sampled"] == 3, scored["fixture_sample"]
        assert scored["fixture_sample"]["n_docs_missing"] == 0, scored["fixture_sample"]
        assert scored["fixture_sample"]["n_fixture_entries_unused"] >= 8, scored["fixture_sample"]
        assert scored["judge_unavailable"]["n_docs"] == 0, scored["judge_unavailable"]

        for tag, run in zip("abc", runs):
            assert run[0] == runs[0][0], f"report_dom.json is not reproducible across replays ({tag})"
            assert run[1] == runs[0][1], f"u.json is not reproducible across replays ({tag})"
            assert run[3] == runs[0][3], f"the printed receipt is not reproducible ({tag})"


def test_replay_never_reaches_the_judge_when_a_sampled_doc_is_uncovered():
    """`and not fixture` (audit_recall.py:1144) is the only thing between replay mode and a live
    provider fan-out; delete it and run_tests.sh calls OpenRouter from inside the gate sweep.

    Three arms, because the property is worthless without its own calibration:
      (1) NO fixture at all — the judge MUST be reached. If this arm passes quietly, the
          `_forbidden` stub in the worker is decoration and arms (2) and (3) prove nothing.
      (2) a NON-EMPTY fixture covering nothing on disk — every sampled document is uncovered
          whatever the sampler picks, so this is the guard's exact state with no dependence on
          random.sample internals. The judge must NOT be reached and the run must fail CLOSED.
      (3) a GENUINELY EMPTY assertion list for a document that IS on disk — the shape the real
          kanto fixture holds three of. `is not None` (audit_recall.py:1191), not truthiness, is
          what keeps this an answer instead of an absence; read as absence it would fall through
          to the judge.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        made = _ruler_unit(root)
        worker = root / "replay_worker.py"
        worker.write_text(_REPLAY_WORKER, encoding="utf-8")
        out = root / "replays"
        out.mkdir()

        # (1) the tripwire is REACHABLE — this is the red calibration for the other two arms
        _r, _u, _c, _o, stderr = _replay(worker, root, out, "live", "0", fixture="-", expect_rc=1)
        assert "replay consulted the judge" in stderr, stderr[-2000:]

        # (2) uncovered sampled document — a non-zero rc here IS the judge having been consulted
        made["fixture"].write_text(json.dumps({
            "phantom-not-on-disk.md": [{"subject": "s0", "predicate": "rel", "object": "o0"}],
        }), encoding="utf-8")
        report = _replay(worker, root, out, "uncovered", "0")[0]
        scored = json.loads(report)
        assert scored["quotable"] is False, scored["quotable_note"]
        assert scored["fixture_sample"]["n_docs_missing"] == 1, scored["fixture_sample"]

        # (3) an empty assertion list is an ANSWER, not an absence: still no judge, still not
        # missing, and an empty denominator that fails closed rather than scoring a vacuous 1.0
        made["fixture"].write_text(json.dumps({"doc.md": []}), encoding="utf-8")
        report = _replay(worker, root, out, "empty", "0")[0]
        scored = json.loads(report)
        assert scored["fixture_sample"]["n_docs_missing"] == 0, scored["fixture_sample"]
        assert scored["fixture_sample"]["n_fixture_entries_unused"] == 0, scored["fixture_sample"]
        assert scored["judge_unavailable"]["n_docs"] == 0, scored["judge_unavailable"]
        assert scored["overall"]["n"] == 0 and scored["quotable"] is False, scored["overall"]


def test_replay_is_byte_identical_when_the_fixture_MISSES_sampled_documents():
    """The lock's own law — a collection of size <= 1 iterates identically under every hash
    seed — must hold for the FAILURE-path collections too, not only the fixture keys.
    In the happy-path corpus missing_sample/judge_rejected_sample/out_of_unit_sample/
    reject_reasons are all empty, so a set leaking into any of them is invisible there."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        _ruler_unit(root, extra_docs=29)          # 30 docs; the fixture covers only doc.md
        worker = root / "replay_worker.py"
        worker.write_text(_REPLAY_WORKER, encoding="utf-8")
        out = root / "replays"
        out.mkdir()
        runs = [_replay(worker, root, out, tag, seed, 1.6e9 + i * (400 * 86400 + 3661),
                        sample_n=25)
                for i, (tag, seed) in enumerate(zip("abc", ("0", "1", "12345")))]
        assert len({r[2] for r in runs}) == 3, "replays shared a hash seed - byte-identity is vacuous"
        scored = json.loads(runs[0][0])
        # calibration: this arm is worthless unless the miss-path collection is WIDE
        assert scored["fixture_sample"]["n_docs_missing"] >= 20, scored["fixture_sample"]
        assert len(scored["fixture_sample"]["missing_sample"]) >= 20, scored["fixture_sample"]
        assert scored["quotable"] is False, scored.get("quotable_note")   # a miss is FAIL by design
        for tag, run in zip("abc", runs):
            assert run[0] == runs[0][0], f"report_dom.json is not reproducible across replays ({tag})"
            assert run[1] == runs[0][1], f"u.json is not reproducible across replays ({tag})"
            assert run[3] == runs[0][3], f"the printed receipt is not reproducible ({tag})"



def test_the_clock_shim_covers_calendar_stamps_and_the_stride_dealiases_them():
    """The lock can only catch a timestamp leak it can SEE diverge. Two forms were measured
    invisible before this probe existed: time.strftime("%Y-%m-%d") (strftime resolves its default
    tuple in C, so patching time.time alone leaves it on the real clock) and int(time.time())%86400
    (identical across arms when the epoch stride is a whole number of days). With the widened shim
    and the 400d+3661s stride, BOTH must differ pairwise across the three arms — this is the
    cannot-fail probe for the byte-identity lock, made permanent so neither regresses."""
    with tempfile.TemporaryDirectory() as temporary:
        root = _ruler_root(temporary)
        _ruler_unit(root)
        worker = root / "replay_worker.py"
        worker.write_text(_REPLAY_WORKER, encoding="utf-8")
        out = root / "replays"
        out.mkdir()
        runs = [_replay(worker, root, out, tag, seed, 1.6e9 + i * (400 * 86400 + 3661),
                        inject_stamps=True)
                for i, (tag, seed) in enumerate(zip("abc", ("0", "1", "12345")))]
        stamps = [json.loads(run[0])["_probe_calendar_stamp"] for run in runs]
        seconds = [json.loads(run[0])["_probe_seconds_of_day"] for run in runs]
        assert len(set(stamps)) == 3, (
            f"calendar stamps agree across shifted arms ({stamps}) - the strftime shim is gone "
            "and a %Y-%m-%d leak in the report would pass the byte-identity lock silently")
        assert len(set(seconds)) == 3, (
            f"seconds-of-day agree across shifted arms ({seconds}) - the epoch stride is a whole "
            "number of days again and any seconds-of-day stamp is invisible to the lock")



if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} AUDIT-RECALL TESTS PASS")
