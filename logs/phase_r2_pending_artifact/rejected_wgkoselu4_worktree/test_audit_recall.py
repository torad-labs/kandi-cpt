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


def test_object_type_invalid_wednesday_is_flagged_but_stays_in_the_denominator():
    """r3 supersedes the Phase-F2 rule that dropped `occurs_at → Wednesday` from the denominator.

    RED (Phase F2): judge_rejected == 1, misses == [], n == 1 — the object-type contract chose the
    denominator, so a manifest edit moved the gated ratio with a byte-identical graph.
    GREEN (r3): the fact is ANNOTATED (object_type_flagged) and scored as the miss it is.
    `clean_legacy` still reports the old 1/1 so the h5 trail stays comparable.
    """
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
    assert cov["judge_rejected"] == [], "object-type is no longer a rejection class"
    assert len(cov["object_type_flagged"]) == 1
    flagged = cov["object_type_flagged"][0]
    assert flagged["object"] == "Wednesday"
    assert "object_type" in (flagged.get("type_flag") or "")
    assert [m["object"] for m in cov["misses"]] == ["Wednesday"], (
        "an object the entity table cannot type is an extraction gap, not judge noise"
    )
    assert cov["hits"] == 1
    assert cov["n"] == 2, "the object-type contract must not choose the denominator"
    # RAW unchanged, and the superseded Phase-F2 number is still reported.
    assert cov["raw"]["n"] == 2
    assert cov["clean_legacy"] == {"hits": 1, "n": 1, "ratio": 1.0}


# ── r3: the ruler is the fixture + the manifest, and nothing else ─────────────
#
# Every case below holds the "fixture" (the assertable list) and the graph fixed and edits only
# the OTHER inputs the ratio used to depend on. Baseline: clean 1/4 = 0.25, legacy 1/2 = 0.50.


def _denominator_fixture() -> tuple[list[dict], dict[str, list[str]], list[dict], set[str]]:
    """(entities, contract, assertable, graph_keys) mirroring the kanto shape in miniature.

    Assertions 2 and 3 have objects absent from the entity table (`object_type_unresolved` — the
    154/376 class on the pinned kanto fixture); assertion 4 is resolvable, type-valid and simply
    not in the graph. Only assertion 1 is a hit.
    """
    entities = [
        {"id": "dom:character:giovanni", "entity_type": "character",
         "name_canonical": "Giovanni", "aliases": []},
        {"id": "dom:character:blaine", "entity_type": "character",
         "name_canonical": "Blaine", "aliases": []},
        {"id": "dom:organization:team_rocket", "entity_type": "organization",
         "name_canonical": "Team Rocket", "aliases": []},
        {"id": "dom:location:viridian_gym", "entity_type": "location",
         "name_canonical": "Viridian Gym", "aliases": []},
    ]
    contract = {
        "member_of": ["organization"],
        "occurs_at": ["location"],
        "owns_pokemon": ["species_lore"],
        "opposes": ["character", "organization"],
    }
    assertable = [
        {"subject": "Giovanni", "predicate": "member_of", "object": "Team Rocket"},
        {"subject": "Blaine", "predicate": "owns_pokemon", "object": "Aerodactyl"},
        {"subject": "Blaine", "predicate": "opposes", "object": "Team Rocket Grunts"},
        {"subject": "Blaine", "predicate": "occurs_at", "object": "Viridian Gym"},
    ]
    graph_keys = {A.fact_key("Giovanni", "member_of", "Team Rocket")}
    return entities, contract, assertable, graph_keys


def _ratio(entities, contract, assertable, graph_keys) -> dict:
    return A.coverage_for_doc(
        assertable,
        graph_keys,
        predicates=list(contract),
        contract=contract,
        entity_index=A.build_entity_index(entities, {}),
    )


def test_r3_baseline_denominator_is_the_fixture_not_the_entity_table():
    entities, contract, assertable, graph_keys = _denominator_fixture()
    cov = _ratio(entities, contract, assertable, graph_keys)
    assert (cov["hits"], cov["n"]) == (1, 4), cov
    assert abs(cov["ratio"] - 0.25) < 1e-9
    assert len(cov["object_type_flagged"]) == 2
    assert cov["clean_legacy"] == {"hits": 1, "n": 2, "ratio": 0.5}


def test_r3_entity_growth_with_zero_new_triples_cannot_move_the_gated_ratio():
    """Red-green (a). Appending ONE shadow row retypes an existing surface (a later row wins in
    build_entity_index), which under Phase F2 pushed that assertion out of the denominator:
    clean 1/2 = 0.5000 → 1/1 = 1.0000, +0.5000, with the graph byte-identical. On the real
    kanto fixture the same construction at 51 rows scored +0.5577.
    """
    entities, contract, assertable, graph_keys = _denominator_fixture()
    before = _ratio(entities, contract, assertable, graph_keys)
    shadow = [{"id": "shadow:not_a_contract_type:viridian_gym",
               "entity_type": "not_a_contract_type",
               "name_canonical": "Viridian Gym", "aliases": []}]
    after = _ratio(entities + shadow, contract, assertable, graph_keys)
    assert after["ratio"] == before["ratio"], (
        f"entity-table growth moved the GATED ratio {before['ratio']:.4f} → {after['ratio']:.4f} "
        f"with zero new triples (n {before['n']} → {after['n']})"
    )
    assert after["n"] == before["n"] == 4
    assert after["hits"] == before["hits"] == 1
    # …and the superseded estimand is exactly the one that moves — that is why it is not gated.
    assert after["clean_legacy"]["ratio"] == 1.0 and before["clean_legacy"]["ratio"] == 0.5


def test_r3_many_added_entities_including_unrelated_junk_are_still_inert():
    """Variant of (a): bulk addition, alias flood, and rows unrelated to any assertion."""
    entities, contract, assertable, graph_keys = _denominator_fixture()
    before = _ratio(entities, contract, assertable, graph_keys)
    bulk = [{"id": f"shadow:not_a_contract_type:s{i}", "entity_type": "not_a_contract_type",
             "name_canonical": name, "aliases": ["Team Rocket Grunts", "Aerodactyl"]}
            for i, name in enumerate(
                ["Viridian Gym", "Team Rocket", "Aerodactyl", "Team Rocket Grunts"])]
    junk = [{"id": f"dom:character:junk{i}", "entity_type": "character",
             "name_canonical": f"Junk Entity {i}", "aliases": []} for i in range(50)]
    after = _ratio(entities + bulk + junk, contract, assertable, graph_keys)
    assert after["n"] == before["n"] == 4, "the denominator must not be a function of table size"
    assert after["ratio"] == before["ratio"], (
        f"bulk entity growth moved the gated ratio {before['ratio']:.4f} → {after['ratio']:.4f}"
    )


def test_r3_domain_manifest_contract_edits_cannot_move_the_gated_ratio():
    """Red-green (b). The contract is the OTHER half of the old ruler and is edited in a config
    file, not the graph. Phase F2: poisoning it emptied the denominator (1/2 = 0.5000 → 0/0 =
    1.0000, +0.5000 — the '+0.8077 → 1.0000' case on the real kanto fixture); widening it to
    ['literal'] moved the same number the other way.
    """
    entities, contract, assertable, graph_keys = _denominator_fixture()
    before = _ratio(entities, contract, assertable, graph_keys)
    for name, edited in (
        ("poisoned", {k: ["nonexistent_type"] for k in contract}),
        ("widened_to_literal", {k: ["literal"] for k in contract}),
        ("emptied", {k: [] for k in contract}),
    ):
        after = _ratio(entities, edited, assertable, graph_keys)
        assert (after["n"], after["hits"]) == (before["n"], before["hits"]), (
            f"contract edit {name!r} moved the gated estimand "
            f"{before['hits']}/{before['n']} → {after['hits']}/{after['n']}"
        )


def test_r3_a_genuine_added_triple_still_moves_the_gated_ratio():
    """Red-green (c) — the equally-broken direction. A denominator pinned so hard that real
    extraction stops registering is the failure mode a frozen entity+contract snapshot has.
    """
    entities, contract, assertable, graph_keys = _denominator_fixture()
    before = _ratio(entities, contract, assertable, graph_keys)
    after = _ratio(entities, contract, assertable,
                   graph_keys | {A.fact_key("Blaine", "occurs_at", "Viridian Gym")})
    assert after["ratio"] > before["ratio"], "a genuinely extracted fact must be visible"
    assert (after["hits"], after["n"]) == (2, 4)


def test_r3_added_entity_plus_triple_moves_the_ratio_without_diluting_the_denominator():
    """Red-green (d) — the shape a real re-extraction takes: the object was unresolvable, so the
    fix adds BOTH the entity and the triple. Under Phase F2 the rescued assertion also entered
    the denominator, diluting it (kanto: 78 → 79, so a real gain scored only +0.0102).
    A frozen snapshot scores this +0.0000 — the 154/376 object_type_unresolved blind spot.
    """
    entities, contract, assertable, graph_keys = _denominator_fixture()
    before = _ratio(entities, contract, assertable, graph_keys)
    rescued = [{"id": "dom:species_lore:aerodactyl", "entity_type": "species_lore",
                "name_canonical": "Aerodactyl", "aliases": []}]
    after = _ratio(entities + rescued, contract, assertable,
                   graph_keys | {A.fact_key("Blaine", "owns_pokemon", "Aerodactyl")})
    assert after["n"] == before["n"] == 4, "the denominator must not dilute when an entity lands"
    assert after["hits"] == 2 and after["ratio"] > before["ratio"], (
        f"re-extraction gain invisible: {before['hits']}/{before['n']} → "
        f"{after['hits']}/{after['n']}"
    )
    assert len(after["object_type_flagged"]) == 1, "one of the two unresolved objects was rescued"


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


# ── r3: fixture-sample integrity + ruler provenance on the report ────────────


def _fixture_audit(root: Path, n_docs: int, fixture: dict, *, sample_n: int) -> dict:
    """A 3-file unit (docs + entities + triples) replayed against `fixture`, provider-free."""
    old = (C.ROOT, C.BUILD)
    C.ROOT = root
    C.BUILD = root / "data" / "_build"
    unit = root / "data" / "dom" / "u"
    clean = unit / "10_clean"
    clean.mkdir(parents=True)
    for i in range(n_docs):
        (clean / f"doc{i}.md").write_text(f"---\n---\nBody {i}.\n", encoding="utf-8")
    structured = unit / "15_structured"
    structured.mkdir(parents=True)
    (structured / "triples.jsonl").write_text(
        "".join(json.dumps(
            {"subject_id": f"s{i}", "predicate": "rel", "object": f"o{i}"}) + "\n"
            for i in range(n_docs)),
        encoding="utf-8",
    )
    (structured / "entities.jsonl").write_text(
        "".join(json.dumps(
            {"id": f"dom:thing:s{i}", "entity_type": "thing", "name_canonical": f"s{i}",
             "aliases": []}) + "\n" for i in range(n_docs)),
        encoding="utf-8",
    )
    fix = root / "facts.json"
    fix.write_text(json.dumps(fixture), encoding="utf-8")
    original = A.iter_editions
    A.iter_editions = lambda *a, **k: [SimpleNamespace(slug="u", dir=unit)]
    try:
        return A.run_audit("dom", None, min_coverage=0.6, sample_n=sample_n, facts_json=fix)
    finally:
        A.iter_editions = original
        C.ROOT, C.BUILD = old


def _covering_fixture(n_docs: int) -> dict:
    return {f"doc{i}.md": [{"subject": f"s{i}", "predicate": "rel", "object": f"o{i}"}]
            for i in range(n_docs)}


def test_r3_fixture_that_misses_a_sampled_document_fails_closed():
    """Red-green (e), the silent-collapse guard — larger than the entity-table lever and quieter.

    RED: the uncovered documents scored as `assertable = []`, contributed 0/0, and vanished. The
    report still printed n_docs_sampled 3 and judge_unavailable 0 while the ratio was computed
    over ONE document — a clean bill of health from a ruler that examined a third of its sample.
    """
    with tempfile.TemporaryDirectory() as temporary:
        report = _fixture_audit(Path(temporary), 3, _covering_fixture(1), sample_n=3)
    assert report["status"] == "FAIL", report.get("reason")
    assert "fixture covers 1/3 sampled documents" in report.get("reason", ""), report.get("reason")
    assert report["fixture_sample"]["n_docs_matched"] == 1
    assert report["fixture_sample"]["n_docs_missing"] == 2
    assert sorted(r["path"] for r in report["fixture_sample"]["missing_sample"]) == [
        "doc1.md", "doc2.md",
    ]
    assert report["units"][0]["n_docs_fixture_matched"] == 1
    # The old tell-tales still look healthy — which is exactly why they could not catch this.
    assert report["units"][0]["n_docs_sampled"] == 3
    assert report["judge_unavailable"]["n_docs"] == 0


def test_r3_fixture_covering_every_sampled_document_still_passes():
    """The paired green: the guard must not fail a fully-covered replay (over-suppression check)."""
    with tempfile.TemporaryDirectory() as temporary:
        report = _fixture_audit(Path(temporary), 3, _covering_fixture(3), sample_n=3)
    assert report["status"] == "PASS", report.get("reason")
    assert report["fixture_sample"]["n_docs_missing"] == 0
    assert report["fixture_sample"]["n_fixture_entries_unused"] == 0
    assert report["overall"]["ratio"] == 1.0


def test_r3_a_narrower_sample_n_is_legitimate_not_a_fixture_miss():
    """Unused fixture entries are reported, never fatal — the guard is about UNCOVERED samples."""
    with tempfile.TemporaryDirectory() as temporary:
        report = _fixture_audit(Path(temporary), 3, _covering_fixture(3), sample_n=1)
    assert report["status"] == "PASS", report.get("reason")
    assert report["fixture_sample"]["n_docs_missing"] == 0
    assert report["fixture_sample"]["n_fixture_entries_unused"] == 2


def test_r3_an_empty_fixture_entry_is_a_covered_document_not_a_missing_one():
    """`[]` means "this document asserts nothing" — an answer. The `x or y or z` lookup chain
    turned it back into None, so a legitimately empty entry read as an uncovered document.
    """
    fixture = _covering_fixture(3)
    fixture["doc1.md"] = []
    with tempfile.TemporaryDirectory() as temporary:
        report = _fixture_audit(Path(temporary), 3, fixture, sample_n=3)
    assert report["fixture_sample"]["n_docs_missing"] == 0, report["fixture_sample"]
    assert report["status"] == "PASS", report.get("reason")


def test_r3_report_carries_ruler_provenance_for_every_input():
    """The ruler's other halves are recorded, and the digests are the files' real ones.

    Calibration (never trust an uncalibrated field): each sha256 is recomputed here from the
    file bytes, and a mutated triples file must produce a different digest.
    """
    import hashlib
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = _fixture_audit(root, 3, _covering_fixture(3), sample_n=3)
        ruler = report["ruler"]
        assert ruler["fixture"]["sha256"] == hashlib.sha256(
            (root / "facts.json").read_bytes()).hexdigest()
        assert ruler["fixture"]["n_docs"] == 3 and ruler["fixture"]["n_assertions"] == 3
        unit_ruler = ruler["units"][0]
        triples_file = root / "data" / "dom" / "u" / "15_structured" / "triples.jsonl"
        entities_file = root / "data" / "dom" / "u" / "15_structured" / "entities.jsonl"
        assert unit_ruler["triples"]["sha256"] == hashlib.sha256(
            triples_file.read_bytes()).hexdigest()
        assert unit_ruler["entities"]["sha256"] == hashlib.sha256(
            entities_file.read_bytes()).hexdigest()
        assert unit_ruler["triples"]["rows"] == 3 and unit_ruler["entities"]["rows"] == 3
        assert unit_ruler["id_map"]["entries"] == 0
        assert ruler["domain_manifest"]["n_predicates"] == 0  # no manifest in the tmp root
        # ≥5 provenance fields, and the report echoes the same block per unit.
        assert report["units"][0]["ruler"] == unit_ruler
        recorded = unit_ruler["triples"]["sha256"]

        # Synthetic violation: the detector must FIRE, not report a comforting zero.
        triples_file.write_text(triples_file.read_text() + json.dumps(
            {"subject_id": "sX", "predicate": "rel", "object": "oX"}) + "\n", encoding="utf-8")
        assert A.sha256_of(triples_file) != recorded, "provenance digest never moved"
        assert A.sha256_of(root / "does_not_exist.json") is None


def test_r3_clean_legacy_reproduces_the_superseded_phase_f2_number():
    """The trail stays comparable: clean_legacy is the contract-filtered denominator, still
    reported, explicitly not gated.
    """
    entities, contract, assertable, graph_keys = _denominator_fixture()
    cov = _ratio(entities, contract, assertable, graph_keys)
    assert cov["clean_legacy"]["n"] == 2 < cov["n"] == 4
    assert cov["clean_legacy"]["ratio"] > cov["ratio"], (
        "the superseded estimand reads HIGHER because it drops extraction gaps from its denominator"
    )


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} AUDIT-RECALL TESTS PASS")
