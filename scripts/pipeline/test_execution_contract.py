"""CPU-only unit coverage for the read-only ExecutionContract emitter.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_execution_contract.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import config as C
from execution_contract import _effects, build_contract
from pipeline.domain import Domain


def test_execution_contract_matches_manifest() -> None:
    result = subprocess.run(
        [sys.executable, str(C.ROOT / "scripts/execution_contract.py"), "--domain", "edc"],
        cwd=C.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["contract_version"] == "1"
    assert contract["engine"]["commit"]
    contracts = {
        "edc": contract,
        "pokemon-lore": build_contract("pokemon-lore"),
        "millbrook-hollow": build_contract("millbrook-hollow"),
    }
    for domain_name, emitted_contract in contracts.items():
        manifest = Domain(domain_name)
        assert [stage["id"] for stage in emitted_contract["stages"]] == manifest.stage_names()
        emitted_by_id = {stage["id"]: stage for stage in emitted_contract["stages"]}
        for stage in manifest.stages:
            emitted = emitted_by_id[stage.name]
            assert emitted["gpu"] is stage.gpu
            assert ("gpu" in emitted["effects"]) is stage.gpu
            assert emitted["outputs"], f"{domain_name}/{stage.name} has no declared outputs"
            assert emitted["cmd"]
            assert all("{" not in argument and "}" not in argument for argument in emitted["cmd"])
            digest_payload = {key: value for key, value in emitted.items() if key != "stage_digest"}
            digest = json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            assert emitted["stage_digest"] == hashlib.sha256(digest).hexdigest()
            assert emitted["postconditions"]
            assert all(check["kind"] == "output_exists" for check in emitted["postconditions"])

    egress_stages = [
        stage
        for emitted_contract in contracts.values()
        for stage in emitted_contract["stages"]
        if "egress:llm" in stage["effects"]
    ]
    assert {(domain, stage["id"]) for domain, contract in contracts.items() for stage in contract["stages"]
            if "egress:llm" in stage["effects"]} == {
        ("edc", "extract"),
        ("edc", "score_educational"),
        ("edc", "structured"),
        ("edc", "hydrate"),
        # Added 2026-07-25 by the EDC manifest migration to the overhauled stage spine. This set is
        # an EGRESS INVENTORY, so a new provider-calling stage must be declared here deliberately —
        # the assertion failing was the control working, not noise to silence.
        ("edc", "synth"),
        ("edc", "audit_recall"),
        ("pokemon-lore", "score_educational"),
        ("pokemon-lore", "structured"),
        ("pokemon-lore", "synth"),
        ("pokemon-lore", "audit_recall"),
        ("millbrook-hollow", "synth"),
    }
    # Phase B/C stages present on pokemon-lore
    poke_ids = {s["id"] for s in contracts["pokemon-lore"]["stages"]}
    assert "resolve" in poke_ids
    assert "audit_recall" in poke_ids
    assert "dedup_surfaces" in poke_ids
    assert any(
        "--version" in " ".join(s["cmd"]) and "v2" in " ".join(s["cmd"])
        for s in contracts["pokemon-lore"]["stages"] if s["id"] == "split"
    )
    for domain_name in ("pokemon-lore", "millbrook-hollow"):
        split = next(stage for stage in contracts[domain_name]["stages"] if stage["id"] == "split")
        assert split["outputs"]
        assert all(output.endswith("/splits/split_v2.json") for output in split["outputs"])
    assert egress_stages
    assert all(stage["provider"] == "openrouter" for stage in egress_stages)

    scoped = build_contract("edc", "las_vegas/2026")
    widened = build_contract("edc", "las_vegas/2026,orlando/2026")
    assert scoped["editions"] == ["las_vegas/2026"]
    scoped_extract = next(stage for stage in scoped["stages"] if stage["id"] == "extract")
    widened_extract = next(stage for stage in widened["stages"] if stage["id"] == "extract")
    assert "las_vegas/2026" in scoped_extract["cmd"]
    assert scoped_extract["outputs"] == ["data/edc/las_vegas/2026/10_clean/**"]
    assert scoped_extract["cmd"] != widened_extract["cmd"]
    assert scoped_extract["stage_digest"] != widened_extract["stage_digest"]

    try:
        build_contract("pokemon-lore", "unknown-region")
    except ValueError as error:
        assert "unknown unit" in str(error)
    else:
        raise AssertionError("unknown unit scope reached an executable contract")

    try:
        _effects("unknown-stage", [], False)
    except ValueError as error:
        assert "declare its effects" in str(error)
    else:
        raise AssertionError("unknown stages defaulted to write instead of failing closed")


if __name__ == "__main__":
    test_execution_contract_matches_manifest()
    print("OK — execution contract matches manifest GPU flags and egress providers")
