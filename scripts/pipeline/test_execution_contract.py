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
            # Every declared output still needs its existence check; stages that RECORD a verdict
            # additionally carry a semantic one, because exit-zero plus a present file cannot see a
            # stage that ran correctly and reported failure.
            kinds = {check["kind"] for check in emitted["postconditions"]}
            assert kinds <= {"output_exists", "report_verdict", "jsonl_no_fail_verdict",
                             "json_field_positive", "tokenize_stats_sound", "split_integrity",
                             "checkpoint_integrity", "probe_quarantine_sound",
                             "fact_concentration_sound"}, kinds
            existence = {check["path"] for check in emitted["postconditions"]
                         if check["kind"] == "output_exists"}
            assert existence == set(emitted["outputs"])

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
        # Added 2026-07-25 with the reasoning-walk surface class: it renders through the
        # LLM channel (generic_synth's injectable generate_fn seam), so it SPENDS and must
        # appear in this inventory deliberately. It is approval-gated by egress:llm.
        ("pokemon-lore", "reasoning"),
        ("pokemon-lore", "audit_recall"),
        ("millbrook-hollow", "synth"),
    }
    # Phase B/C stages present on pokemon-lore
    poke_ids = {s["id"] for s in contracts["pokemon-lore"]["stages"]}
    assert "resolve" in poke_ids
    assert "audit_recall" in poke_ids
    assert "dedup_surfaces" in poke_ids
    assert "probe_quarantine" in poke_ids
    assert any(
        "--version" in " ".join(s["cmd"]) and "v2" in " ".join(s["cmd"])
        for s in contracts["pokemon-lore"]["stages"] if s["id"] == "split"
    )
    for domain_name in ("pokemon-lore", "millbrook-hollow"):
        quarantine = next(stage for stage in contracts[domain_name]["stages"]
                          if stage["id"] == "probe_quarantine")
        assert quarantine["effects"] == ["write"]
        assert quarantine["outputs"] == [
            f"data/_build/probe_quarantine/{domain_name}/**.json",
            f"data/_build/probe_quarantine/report_{domain_name}.json",
        ]
        assert any(check["kind"] == "probe_quarantine_sound"
                   for check in quarantine["postconditions"])
    for domain_name in ("pokemon-lore", "millbrook-hollow"):
        split = next(stage for stage in contracts[domain_name]["stages"] if stage["id"] == "split")
        assert split["outputs"]
        assert all(output.endswith("/splits/split_v2.json") for output in split["outputs"])
    assert egress_stages
    # Google AI Studio is plan A; OpenRouter is plan B (runtime fallback only).
    assert all(stage["provider"] == "googleai" for stage in egress_stages)
    assert all(stage.get("provider_fallback") == "openrouter" for stage in egress_stages)

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


def test_git_identity_falls_back_to_the_baked_engine_stamp() -> None:
    """The cloud image ships no .git; the build stamps <engine parent>/ENGINE_COMMIT. Without the
    fallback every in-container stage record carries commit=null — a rented-box training run would
    arrive home with no engine provenance (observed: this module red in conductor-trainer)."""
    import tempfile
    from pathlib import Path

    from execution_contract import _git_identity

    sha = "9017fde5d3b73c56fd9a246599964832bed9df21"
    with tempfile.TemporaryDirectory() as tmp:
        engine = Path(tmp) / "kandi-cpt"
        engine.mkdir()
        stamp = Path(tmp) / "ENGINE_COMMIT"

        # no .git above system tmp and no stamp file -> identity is honestly unknown
        assert _git_identity(root=engine) == (None, None)

        stamp.write_text(f"commit={sha}\ndirty=false\n")
        assert _git_identity(root=engine) == (sha, False)

        stamp.write_text(f"commit={sha}\ndirty=true\n")
        assert _git_identity(root=engine) == (sha, True)

        # the Dockerfile's ARG default must never masquerade as provenance
        stamp.write_text("commit=unknown\ndirty=unknown\n")
        assert _git_identity(root=engine) == (None, None)


if __name__ == "__main__":
    test_execution_contract_matches_manifest()
    test_git_identity_falls_back_to_the_baked_engine_stamp()
    print("OK — execution contract matches manifest GPU flags and egress providers")
