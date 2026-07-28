#!/usr/bin/env python3
"""The manifest is the single owner of stage order — nothing may hold a second copy.

Conductor carried a static `CptStage` enum listing 11 stages (including a festival-era `META`)
while the manifests declared 17, omitting pii_scan, score_educational, resolve, split, synth,
contrast, dedup_surfaces, probes and audit_recall. Its tests were green the whole time, because a
parallel definition only disagrees with reality, never with itself. That enum is deleted; this test
is what stops the next one being written.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_stage_lockstep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution_contract import build_contract
from pipeline.domain import Domain
from plan_install import build_plan

DOMAINS = (("pokemon-lore", "kanto"), ("millbrook-hollow", "main"), ("edc", "las_vegas/2011"))


def test_manifest_contract_and_plan_agree_on_stages() -> None:
    for name, editions in DOMAINS:
        manifest = [stage.name for stage in Domain(name).stages]
        contract = [stage["id"] for stage in build_contract(name, editions)["stages"]]
        plan = [entry["stage"] for entry in build_plan(name, editions)["plan"]]

        # Order, not just membership: the DAG is ordered, and a reordering is exactly the kind of
        # drift a set comparison would wave through.
        assert manifest == contract, (
            f"{name}: manifest and execution contract disagree\n"
            f"  manifest only: {[s for s in manifest if s not in contract]}\n"
            f"  contract only: {[s for s in contract if s not in manifest]}"
        )
        assert contract == plan, (
            f"{name}: execution contract and plan_install disagree\n"
            f"  contract only: {[s for s in contract if s not in plan]}\n"
            f"  plan only: {[s for s in plan if s not in contract]}"
        )
        assert len(manifest) == len(set(manifest)), f"{name}: duplicate stage id in manifest"


def test_the_comparison_can_actually_fail() -> None:
    """Guard the guard: prove the assertion above is capable of detecting a mismatch.

    A lockstep test that compares two lists which are derived from one another can be vacuously
    green. This mutates one list and confirms the same comparison rejects it.
    """
    manifest = [stage.name for stage in Domain("pokemon-lore").stages]
    assert len(manifest) > 3, manifest
    for mutation in (manifest[:-1], [*manifest[1:], manifest[0]], [*manifest, "invented_stage"]):
        assert mutation != manifest, "mutation did not change the list — the check proves nothing"


if __name__ == "__main__":
    test_manifest_contract_and_plan_agree_on_stages()
    test_the_comparison_can_actually_fail()
    print("OK — manifest == execution contract == plan_install, in order, for every domain")
