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


if __name__ == "__main__":
    test_unset_profile_keeps_the_generic_meaning()
    test_bespoke_domain_is_not_asked_for_templates_it_never_reads()
    test_declaring_a_capability_false_cannot_silence_a_stage_that_uses_it()
    test_profile_defaults_and_unknown_keys_fail_closed()
    print("OK — declared profiles validated, false gaps gone, real gaps and lies both refused")
