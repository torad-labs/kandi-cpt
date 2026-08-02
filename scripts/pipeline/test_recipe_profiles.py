#!/usr/bin/env python3
"""Validate the profile a domain DECLARES, not one capability model forced on all of them.

EDC generates its surfaces with a bespoke script (rehydrate_vc1.sh) and never reads the generic
hydrate templates, yet RECIPE_COMPLETE demanded them — two permanent gaps that could not be closed
because they described work the domain does not do. A permanently-failing check is a check people
learn to skip.

The declaration must not become an escape hatch either: `supports.X = false` beside a stage that
actually runs the generic tool X is a manifest lying about itself, and would silence the very
check that catches a missing template.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_recipe_profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.domain import RECIPE_PROFILES, Domain
from plan_install import build_plan


def _recipe(domain: str, editions: str) -> dict:
    return next(s for s in build_plan(domain, editions)["ladder"] if s["state"] == "RECIPE_COMPLETE")


def _profile_evidence(state: dict) -> dict:
    return next(e for e in state["evidence"] if isinstance(e, dict) and "recipe_profile" in e)


def _strip_tiers(domain: Domain) -> Domain:
    """The same domain with every tier declaration removed — IN MEMORY, never on disk.

    Strips BOTH legal routes (the per-predicate `tier:` key and the domain-level `tiers:` block),
    because stripping only one leaves the other declaring and the negative case never fires.
    Domain re-reads its YAML on every construction (domain.py:42), so this mutates one instance.
    The fixture is synthetic on purpose: pinning the 0-of-N branch to whichever manifest happens
    to be undeclared today makes the guard evaporate the moment that manifest is reconciled, which
    is exactly what happened when EDC was the fixture.
    """
    predicates = domain.spec.get("predicates") or {}
    domain.spec["predicates"] = {
        name: {key: value for key, value in (meta or {}).items() if key != "tier"}
        for name, meta in predicates.items()
    }
    domain.spec.pop("tiers", None)
    return domain


def test_unset_profile_keeps_the_generic_meaning() -> None:
    """Every existing manifest must keep its current verdict — this is not a silent loosening."""
    for domain, editions in (("pokemon-lore", "kanto"), ("millbrook-hollow", "main")):
        state = _recipe(domain, editions)
        assert _profile_evidence(state)["recipe_profile"] == "generic_facts", domain
        assert state["value"] == "PASS", (domain, state["gaps"])


def test_bespoke_domain_is_not_asked_for_templates_it_never_reads() -> None:
    state = _recipe("edc", "las_vegas/2011")
    assert _profile_evidence(state)["recipe_profile"] == "bespoke_facts"
    assert not [g for g in state["gaps"] if "hydrate_templates" in g], (
        "the bespoke hydrate gap should be gone — EDC runs rehydrate_vc1.sh, not the expander"
    )
    # The load-bearing half: a REAL gap must survive. EDC's contrast stage does run
    # generic_contrast.py, it declares contrast: true, and it authors no contrast block.
    assert state["value"] == "FAIL"
    assert any("contrast" in g and "template" in g for g in state["gaps"]), state["gaps"]


def test_declaring_a_capability_false_cannot_silence_a_stage_that_uses_it() -> None:
    """The anti-escape-hatch guard, measured on the real manifest.

    EDC's contrast stage runs generic_contrast.py. Flipping its declaration to false must be
    REFUSED rather than quietly dropping the template check.
    """
    domain = Domain("edc")
    assert domain.supports.get("contrast") is True, domain.supports
    original = dict(domain.supports)
    try:
        domain.supports["contrast"] = False
        from plan_install import _recipe_state
        state = _recipe_state(domain, "las_vegas/2011")
        assert any("contradicts what the stage actually does" in g for g in state["gaps"]), state["gaps"]
    finally:
        domain.supports = original


def test_profile_defaults_and_unknown_keys_fail_closed() -> None:
    assert set(RECIPE_PROFILES) == {"generic_facts", "bespoke_facts", "style"}
    assert RECIPE_PROFILES["generic_facts"]["generic_hydrate"] is True
    assert RECIPE_PROFILES["bespoke_facts"]["generic_hydrate"] is False
    assert RECIPE_PROFILES["style"]["sft"] is False

    domain = Domain("pokemon-lore")
    original = dict(domain.supports)
    try:
        domain.supports["contrats"] = True          # a typo must not pass silently
        from plan_install import _recipe_state
        state = _recipe_state(domain, "kanto")
        assert any("unknown capabilities" in g for g in state["gaps"]), state["gaps"]

        domain.supports = original
        domain.recipe_profile = "not_a_profile"
        state = _recipe_state(domain, "kanto")
        assert any("is not one of" in g for g in state["gaps"]), state["gaps"]
    finally:
        domain.supports = original
        domain.recipe_profile = "generic_facts"


def test_a_domain_that_declares_no_predicate_tiers_installs_nothing() -> None:
    """Undeclared defaults to `hot`, and hot NEVER reaches the train shard.

    The default is right — misroute-to-hot is cheap, misroute-to-weights is a governance incident.
    But a facts domain declaring a CPT route and NO tiers routes every predicate to retrieval, so
    its install route silently carries nothing.

    Both halves run against the SAME manifest, stripped and unstripped, for every facts domain that
    declares a tokenize stage. That is deliberate: the previous version used EDC's then-undeclared
    manifest as the negative case, so reconciling EDC would have left the 0-of-N branch untested
    while the suite stayed green. A fixture that only holds while some manifest stays broken is not
    a guard. The strip is in memory; the last assertion proves it never reached disk.
    """
    from pipeline.tier_routing import DEFAULT_TIER, load_predicate_tiers
    from plan_install import _recipe_state

    def tier_evidence(state):
        return next(e for e in state["evidence"]
                    if isinstance(e, dict) and "predicate_tiers_declared" in e)

    for name, editions, n_predicates in (("pokemon-lore", "kanto", 19), ("edc", "las_vegas/2011", 34)):
        # A domain that DOES declare its tiers must not be flagged, or the check is noise. Pinned at
        # FULL coverage, not merely non-zero: plan_install flags only the all-undeclared case, so
        # `declared == total` is what stops a manifest drifting to one token declaration.
        state = _recipe_state(Domain(name), editions)
        evidence = tier_evidence(state)
        assert evidence["predicates_total"] == n_predicates, (name, evidence)
        assert evidence["predicate_tiers_declared"] == n_predicates, (name, evidence)
        assert not [g for g in state["gaps"] if "predicate tiers" in g], (name, state["gaps"])
        tiers = load_predicate_tiers(name)
        assert set(tiers) == set(Domain(name).spec["predicates"]), name

        # The load-bearing half: strip the declarations and the gap must come back.
        state = _recipe_state(_strip_tiers(Domain(name)), editions)
        evidence = tier_evidence(state)
        assert evidence["predicate_tiers_declared"] == 0, (name, evidence)
        assert evidence["predicates_total"] == n_predicates, (name, evidence)
        assert evidence["undeclared_default_tier"] == DEFAULT_TIER == "hot"
        assert any("predicate tiers" in g for g in state["gaps"]), (name, state["gaps"])

    # Declared-and-legal is not the same as declared-and-useful: 34 of 34 hot would satisfy every
    # assertion above and still install nothing, and plan_install only flags the ALL-undeclared case.
    # So pin the mass. Measured over all 98 editions' 15_structured/triples.jsonl: 19,229 triple rows,
    # of which in_lineup 11,461 + plays_at 6,909 = 95.5%. Routing either to hot empties EDC's install
    # route while leaving the readiness verdict green. Re-judge with a fresh count, not by taste.
    edc_tiers = load_predicate_tiers("edc")
    for predicate in ("in_lineup", "plays_at"):
        assert edc_tiers.get(predicate) not in (None, "hot"), (
            f"{predicate} carries the bulk of EDC's 19,229 triples (re-measure before re-judging); "
            f"absent or hot means the install route is empty"
        )

    # The strip is in-memory only: a fresh load still declares every tier off disk, by EITHER
    # legal route (the per-predicate `tier:` key or the domain-level `tiers:` block —
    # tier_routing.py:54-61, unioned by plan_install.py:200). Reading raw YAML shape here
    # instead would red a manifest the gate itself certifies.
    assert set(load_predicate_tiers("edc")) == set(Domain("edc").spec["predicates"])


def test_an_illegal_routing_tier_fails_closed_at_load() -> None:
    """tier_routing refuses a tier outside hot|stable|proven, by BOTH declaration routes.

    The two raises (per-predicate at tier_routing.py:52, tiers block at tier_routing.py:60) had NO
    test owner anywhere in this tree — `grep -rn TierError scripts/ --include=test_*.py` hits only
    test_tier_routing.py, whose one TierError block owns the assert_style_all_stable raise. This
    fixture is that owner. A values-subset assertion over whatever load_predicate_tiers RETURNS
    could never have been that owner: it normalizes with str().strip().lower() and raises TierError
    before returning, so its values are a subset of TIERS by construction. That is an argument about
    a shape, not a claim about this repository's history — `git log --all -S` finds no such
    assertion ever committed to this tree, so there was never one here to replace.
    Synthetic manifests only — no real manifest is read or mutated.

    Every expectation below is derived (sorted(T.TIERS)) or literal-and-checked; there is no
    per-case escape clause, so each assertion is live for the case that reaches it.
    """
    import tempfile

    import config as C
    import pipeline.tier_routing as T

    legal = str(sorted(T.TIERS))                    # exactly what the per-predicate raise renders
    saved = (C.ROOT, C.CONFIGS, C.BUILD)

    def _install(root: Path, body: str) -> None:
        C.ROOT, C.CONFIGS, C.BUILD = root, root / "configs", root / "data" / "_build"
        domains = C.CONFIGS / "domains"
        domains.mkdir(parents=True)
        (domains / "vocabdemo.yaml").write_text(
            "name: vocabdemo\nentity_types: [t]\n" + body, encoding="utf-8")

    try:
        for label, body, wanted in (
            ("per-predicate (tier_routing.py:52)",
             "predicates:\n  p: {object: [t], tier: warm}\n",
             ("predicate 'p'", "'warm'", legal)),
            ("tiers block (tier_routing.py:60)",
             "predicates:\n  p: {object: [t]}\ntiers:\n  p: warm\n",
             ("tiers block:", "'p'", "'warm'")),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                _install(Path(temporary), body)
                try:
                    T.load_predicate_tiers("vocabdemo")
                except T.TierError as error:
                    for want in wanted:
                        assert want in str(error), (label, want, str(error))
                else:
                    raise AssertionError(f"{label}: expected TierError, got a silent accept")

        # Non-string tiers must fail closed the same way — YAML types a bare 123 as int and a bare
        # `true` as bool, and str().strip().lower() would otherwise launder them into a name.
        for scalar in ("123", "true"):
            with tempfile.TemporaryDirectory() as temporary:
                _install(Path(temporary), f"predicates:\n  p: {{object: [t], tier: {scalar}}}\n")
                try:
                    T.load_predicate_tiers("vocabdemo")
                except T.TierError as error:
                    assert legal in str(error), (scalar, str(error))
                else:
                    raise AssertionError(f"non-string tier {scalar}: expected TierError, got a silent accept")

        # Case-folding is the one legal-looking input a values-subset check would have waved
        # through: it must be ACCEPTED and normalized, not refused — by either route.
        for label, body in (
            ("per-predicate", "predicates:\n  p: {object: [t], tier: STABLE}\n"),
            ("tiers block", "predicates:\n  p: {object: [t]}\ntiers:\n  p: STABLE\n"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                _install(Path(temporary), body)
                assert T.load_predicate_tiers("vocabdemo") == {"p": "stable"}, label
    finally:
        C.ROOT, C.CONFIGS, C.BUILD = saved


def test_the_manifest_is_build_identity_input() -> None:
    """The blast-radius claim in edc.yaml, made mechanical (D-obj-2's required case).

    Editing the manifest — comments included — moves the build id, because recipe_read_set
    content-addresses the manifest file. This fixture pins the ROLE so a future reader cannot
    restate the blast radius without this check moving. Red-proof (run, not asserted): dropping
    the domain_manifest entry from the read set makes this fail on the roles assertion.
    """
    from run_stage import recipe_read_set

    roles = {entry["role"] for entry in recipe_read_set(Domain("edc"))}
    assert "domain_manifest" in roles, roles


if __name__ == "__main__":
    test_unset_profile_keeps_the_generic_meaning()
    test_bespoke_domain_is_not_asked_for_templates_it_never_reads()
    test_declaring_a_capability_false_cannot_silence_a_stage_that_uses_it()
    test_profile_defaults_and_unknown_keys_fail_closed()
    test_a_domain_that_declares_no_predicate_tiers_installs_nothing()
    test_an_illegal_routing_tier_fails_closed_at_load()
    test_the_manifest_is_build_identity_input()
    print("OK — declared profiles validated, false gaps gone, real gaps and lies both refused")
