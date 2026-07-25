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

    fixture_path = root / "facts.json"
    fixture_path.write_text(json.dumps({
        "doc.md": [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"} for i in range(n_facts)],
    }), encoding="utf-8")
    return {"unit": unit, "fixture": fixture_path, "manifest": manifest,
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
    }
    pin = {"pinned": {"fixture": "f" * 64, "entities": "e" * 64, "id_map": "i" * 64,
                      "domain_manifest": "m" * 64, "documents": "d" * 64}}

    supplied = A.compare_ruler_pin(pin, fingerprint, fixture_supplied=True)
    assert any(p.startswith("fixture:") for p in supplied), supplied

    replayless = A.compare_ruler_pin(pin, fingerprint, fixture_supplied=False)
    assert replayless == [], replayless

    # …and skipping the fixture must NOT smuggle the other four past the check.
    fingerprint["entities"]["sha256"] = "x" * 64
    still_bound = A.compare_ruler_pin(pin, fingerprint, fixture_supplied=False)
    assert any(p.startswith("entities:") for p in still_bound), still_bound


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


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} AUDIT-RECALL TESTS PASS")
