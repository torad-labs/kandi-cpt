"""CPU-only fixture coverage for run_stage build identity, device pinning, records, and inspect.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_run_stage.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

import config as C
import run_stage
from pipeline.domain import Domain


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fixture(root: Path) -> None:
    (root / "configs/domains").mkdir(parents=True)
    (root / "configs/templates").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "data").mkdir()
    (root / "configs/templates/fixture.yaml").write_text("fixture: true\n", encoding="utf-8")
    manifest = {
        "name": "fixture",
        "hierarchy": ["unit"],
        "registry": "configs/editions_fixture.json",
        "templates": "configs/templates/fixture.yaml",
        "stages": [],
        "protected_editions": ["gold"],
        "entity_types": ["thing"],
        "predicates": {"related_to": {"object": ["thing"]}},
    }
    (root / "configs/domains/fixture.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / "configs/editions_fixture.json").write_text('{"editions": []}\n', encoding="utf-8")
    (root / "scripts/stub.py").write_text(
        """import os
import pathlib
import sys

mode, output = sys.argv[1:3]
path = pathlib.Path(output)
if mode == "nonzero":
    raise SystemExit(7)
if mode == "recipe-drift":
    pathlib.Path("configs/templates/fixture.yaml").write_text("changed during stage\\n", encoding="utf-8")
if mode != "missing":
    path.parent.mkdir(parents=True, exist_ok=True)
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>") if mode == "env" else mode
    path.write_text(value, encoding="utf-8")
""",
        encoding="utf-8",
    )


def _contract(root: Path, stage: str, mode: str, effects: list[str] | None = None,
              output: str = "data/fixture/result.txt", postconditions: list[dict] | None = None,
              editions: list[str] | None = None) -> dict:
    command = [sys.executable, str(root / "scripts/stub.py"), mode, output]
    stage_entry = {
        "id": stage,
        "editions": editions if editions is not None else ["unit-a"],
        "cmd_template": command,
        "cmd": command,
        "effects": effects or ["write"],
        "outputs": [output],
        "postconditions": postconditions or [{
            "check": f"output exists: {output}",
            "kind": "output_exists",
            "path": output,
        }],
    }
    payload = json.dumps(stage_entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    stage_entry["stage_digest"] = hashlib.sha256(payload).hexdigest()
    return {
        "contract_version": "1",
        "engine": {"name": "fixture", "commit": "abc123", "dirty": True},
        "stages": [stage_entry],
    }


def _execute(root: Path, stage: str, mode: str, devices: str = "", effects: list[str] | None = None,
             output: str = "data/fixture/result.txt", postconditions: list[dict] | None = None,
             build_id: str | None = None,
             host_gpu_preflight: dict | None = None) -> tuple[dict, int, str]:
    contract = _contract(root, stage, mode, effects, output, postconditions)
    return run_stage.execute_stage(
        "fixture",
        stage,
        devices,
        requested_build_id=build_id,
        expected_stage_digest=contract["stages"][0]["stage_digest"],
        contract_builder=lambda _domain, _editions: contract,
        host_gpu_preflight=host_gpu_preflight,
    )


def _gpu_preflight(*selectors: str) -> dict:
    return {
        "allowed": True,
        "sample_interval_ms": 1000,
        "sampling_duration_ms": 2077,
        "devices": [{
            "selector": selector,
            "index": int(selector),
            "uuid": f"GPU-{selector}",
            "total_memory_mib": 16303,
            "required_free_memory_mib": 13312,
            "minimum_free_memory_mib": 13981,
            "free_memory_samples_mib": [13981, 13980, 13981],
            "sm_utilization_watermark_pct": 18,
            "sm_utilization_samples_pct": [9, 8, 9],
            "sample_count": 3,
            "sustained_sm_above_watermark": False,
            "derivation": ["required_free_mib = 13 * 1024 = 13312 (admin-gpu.md; cpt/07-train.md)"],
        } for selector in selectors],
    }


def test_run_stage() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "suite must be launched CPU-only"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fixture(root)
        original = C.ROOT, C.CONFIGS, C.DATA
        C.ROOT = root
        C.CONFIGS = root / "configs"
        C.DATA = root / "data"
        try:
            domain = Domain("fixture")
            build_a, hashes_a = run_stage.build_identity(domain, "1")
            build_b, hashes_b = run_stage.build_identity(domain, "1")
            assert build_a == build_b and hashes_a == hashes_b
            assert len(build_a) == 16 and set(hashes_a) == {
                "recipe_sha256", "contract_version_sha256", "read_set",
            }
            # The read set is the content-addressed recipe manifest; every entry carries its role,
            # its repo-relative path, and the digest the build id was computed over.
            assert all({"role", "path", "sha256"} == set(entry) for entry in hashes_a["read_set"])

            env_record, env_exit, build_id = _execute(root, "env-pin", "env", devices="3,4")
            assert env_exit == 0 and env_record["status"] == "PASS"
            assert (root / "data/fixture/result.txt").read_text(encoding="utf-8") == "3,4"
            build = json.loads((root / f"data/fixture/builds/{build_id}/build.json").read_text(encoding="utf-8"))
            assert build["build_id"] == build_id and build["contract_version"] == "1"
            assert build["inputs"] == hashes_a and build["born_at"].endswith("Z")

            old_build = build_id
            templates = root / "configs/templates/fixture.yaml"
            templates.write_text("fixture: changed\n", encoding="utf-8")
            drift_record, drift_exit, drift_build = _execute(
                root, "drift", "pass", build_id=old_build, output="data/fixture/drift.txt",
            )
            assert drift_build == old_build and drift_exit != 0
            assert drift_record["status"] == "FAIL" and drift_record["error"] == "BUILD_DRIFT"
            assert drift_record["editions"] == ["unit-a"] and drift_record["stage_digest"]
            assert any("BUILD_DRIFT" in check["check"] for check in drift_record["postconditions"])
            assert not (root / "data/fixture/drift.txt").exists(), "BUILD_DRIFT spawned the child"
            assert run_stage.inspect_stage("fixture", old_build, "env-pin") == env_record
            assert run_stage.inspect_stage("fixture", old_build, "drift") == drift_record
            removed_stage_record, removed_stage_exit, _ = run_stage.execute_stage(
                "fixture",
                "removed-stage",
                "",
                requested_build_id=old_build,
                expected_stage_digest="0" * 64,
                contract_builder=lambda _domain, _editions: {"contract_version": "1", "engine": {}, "stages": []},
            )
            assert removed_stage_exit != 0 and removed_stage_record["error"] == "BUILD_DRIFT"

            stdout, stderr = io.StringIO(), io.StringIO()
            argv, contract_builder = sys.argv, run_stage.build_contract
            drift_cli_contract = _contract(root, "drift-cli", "pass", output="data/fixture/drift-cli.txt")
            run_stage.build_contract = lambda _domain, _editions: drift_cli_contract
            sys.argv = [
                "run_stage.py", "--domain", "fixture", "--stage", "drift-cli", "--devices", "",
                "--approved-stage-digest", drift_cli_contract["stages"][0]["stage_digest"],
                "--build-id", old_build,
            ]
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    assert run_stage.main() != 0
            finally:
                sys.argv, run_stage.build_contract = argv, contract_builder
            assert json.loads(stderr.getvalue())["error"] == "BUILD_DRIFT"
            assert json.loads(stdout.getvalue())["error"] == "BUILD_DRIFT"
            assert not (root / "data/fixture/drift-cli.txt").exists(), "CLI BUILD_DRIFT spawned the child"

            approved_contract = _contract(
                root, "contract-drift", "pass", output="data/fixture/contract-drift.txt",
            )
            current_contract = _contract(
                root,
                "contract-drift",
                "pass",
                effects=["egress:upload", "write"],
                output="data/fixture/contract-drift.txt",
            )
            contract_record, contract_exit, _ = run_stage.execute_stage(
                "fixture",
                "contract-drift",
                "",
                expected_stage_digest=approved_contract["stages"][0]["stage_digest"],
                contract_builder=lambda _domain, _editions: current_contract,
            )
            assert contract_exit != 0 and contract_record["error"] == "CONTRACT_DRIFT"
            assert not (root / "data/fixture/contract-drift.txt").exists(), "contract drift spawned the child"

            templates.write_text("fixture: empty-scope\n", encoding="utf-8")
            current_build, _ = run_stage.build_identity(Domain("fixture"), "1")
            empty_scope_contract = _contract(
                root,
                "empty-scope",
                "pass",
                output="data/fixture/empty-scope.txt",
                editions=[],
            )
            empty_scope_record, empty_scope_exit, _ = run_stage.execute_stage(
                "fixture",
                "empty-scope",
                "",
                expected_stage_digest=empty_scope_contract["stages"][0]["stage_digest"],
                contract_builder=lambda _domain, _editions: empty_scope_contract,
            )
            assert empty_scope_exit != 0 and empty_scope_record["error"] == "NO_UNITS"
            assert not (root / "data/fixture/empty-scope.txt").exists(), "empty unit scope spawned the child"
            assert not (root / f"data/fixture/builds/{current_build}/build.json").exists(), \
                "a pre-spawn refusal claimed a recipe build was born"

            protected_contract = _contract(
                root,
                "protected",
                "pass",
                output="data/fixture/gold/protected.txt",
                editions=["gold"],
            )
            protected_record, protected_exit, _ = run_stage.execute_stage(
                "fixture",
                "protected",
                "",
                expected_stage_digest=protected_contract["stages"][0]["stage_digest"],
                contract_builder=lambda _domain, _editions: protected_contract,
            )
            assert protected_exit != 0 and protected_record["error"] == "PROTECTED_UNITS"
            assert not (root / "data/fixture/gold/protected.txt").exists(), "protected unit spawned the child"
            assert not (root / f"data/fixture/builds/{current_build}/build.json").exists(), \
                "a protected-unit refusal claimed a recipe build was born"

            templates.write_text("fixture: true\n", encoding="utf-8")
            changed_env_contract = _contract(
                root,
                "env-pin",
                "env",
                effects=["egress:upload", "write"],
            )
            refusal, refusal_exit, _ = run_stage.execute_stage(
                "fixture",
                "env-pin",
                "3,4",
                expected_stage_digest="0" * 64,
                contract_builder=lambda _domain, _editions: changed_env_contract,
            )
            assert refusal_exit != 0 and refusal["error"] == "CONTRACT_DRIFT"
            assert run_stage.inspect_stage("fixture", build_id, "env-pin") == env_record
            attempts = root / f"data/fixture/builds/{build_id}/stages/attempts/env-pin"
            assert any(json.loads(path.read_text(encoding="utf-8"))["error"] == "CONTRACT_DRIFT"
                       for path in attempts.glob("*.json"))

            marker = "data/fixture/gpu-spawned.txt"
            gpu_record, gpu_exit, _ = _execute(root, "gpu", "pass", effects=["gpu", "write"], output=marker)
            assert gpu_exit != 0 and gpu_record["error"] == "GPU_DEVICES_REQUIRED"
            assert not (root / marker).exists(), "empty-device GPU refusal spawned the child"

            missing_receipt, missing_receipt_exit, _ = _execute(
                root,
                "gpu-no-receipt",
                "pass",
                devices="3",
                effects=["gpu", "write"],
                output="data/fixture/gpu-no-receipt.txt",
            )
            assert missing_receipt_exit != 0 and missing_receipt["error"] == "HOST_GPU_PREFLIGHT_REQUIRED"
            assert not (root / "data/fixture/gpu-no-receipt.txt").exists()

            gpu_allowance = _gpu_preflight("3", "4")
            measured_gpu_record, measured_gpu_exit, _ = _execute(
                root,
                "gpu-measured",
                "pass",
                devices="3,4",
                effects=["gpu", "write"],
                output="data/fixture/gpu-measured.txt",
                host_gpu_preflight=gpu_allowance,
            )
            assert measured_gpu_exit == 0 and measured_gpu_record["status"] == "PASS"
            assert measured_gpu_record["host_gpu_preflight"] == gpu_allowance
            assert measured_gpu_record["host_gpu_preflight"]["sampling_duration_ms"] == 2077

            nonzero_record, nonzero_exit, _ = _execute(
                root, "nonzero", "nonzero", output="data/fixture/nonzero.txt",
            )
            assert nonzero_exit != 0 and nonzero_record["status"] == "FAIL"
            assert nonzero_record["exit_code"] == 7

            missing_record, missing_exit, _ = _execute(
                root, "missing", "missing", output="data/fixture/missing.txt",
            )
            assert missing_exit != 0 and missing_record["status"] == "FAIL"
            assert missing_record["exit_code"] == 0 and missing_record["outputs"] == []
            assert any(not check["pass"] for check in missing_record["postconditions"])

            stale_path = root / "data/fixture/stale.txt"
            stale_path.write_text("OLD OUTPUT", encoding="utf-8")
            stale_record, stale_exit, _ = _execute(
                root, "stale", "missing", output="data/fixture/stale.txt",
            )
            assert stale_exit != 0 and stale_record["status"] == "FAIL"
            assert stale_record["output_delta"]["unchanged"] == 1
            assert stale_record["output_delta"]["created"] == 0
            assert any(check == {"check": "declared output delta is nonzero", "pass": False}
                       for check in stale_record["postconditions"])

            recipe_drift_record, recipe_drift_exit, _ = _execute(
                root,
                "recipe-drift",
                "recipe-drift",
                output="data/fixture/recipe-drift.txt",
            )
            assert recipe_drift_exit != 0 and recipe_drift_record["status"] == "FAIL"
            assert {"check": "recipe identity stayed unchanged during execution", "pass": False} \
                in recipe_drift_record["postconditions"]
            templates.write_text("fixture: changed\n", encoding="utf-8")

            failed_check = [{
                "check": "output contains expected marker",
                "kind": "file_contains",
                "path": "data/fixture/postcondition.txt",
                "contains": "expected",
            }]
            post_record, post_exit, _ = _execute(
                root,
                "postcondition",
                "actual",
                output="data/fixture/postcondition.txt",
                postconditions=failed_check,
            )
            assert post_exit != 0 and post_record["status"] == "FAIL"
            assert post_record["exit_code"] == 0 and post_record["outputs"]
            assert post_record["postconditions"] == [
                {"check": "output contains expected marker", "pass": False},
                {"check": "declared output delta is nonzero", "pass": True},
                {"check": "recipe identity stayed unchanged during execution", "pass": True},
                {"check": "approved stage stayed unchanged during execution", "pass": True},
            ]

            expected_keys = {
                "cmd", "devices", "editions", "stage_digest", "exit_code", "duration_s", "outputs",
                "output_delta", "postconditions", "engine", "status",
            }
            assert set(env_record) == expected_keys
            assert env_record["engine"] == {"commit": "abc123", "dirty": True}
            assert set(env_record["outputs"][0]) == {"path", "sha256", "bytes"}
            assert env_record["output_delta"]["created"] == 1
            assert set(env_record["postconditions"][0]) == {"check", "pass"}

            large = root / "data/fixture/large.bin"
            large.write_bytes(b"0123456789abcdef" * (256 * 1024))
            artifact = run_stage._artifact(large)
            assert artifact["bytes"] == large.stat().st_size
            assert artifact["sha256"] == hashlib.sha256(large.read_bytes()).hexdigest()

            try:
                run_stage.inspect_stage("..", build_id, "env-pin")
            except run_stage.StageError as error:
                assert error.code == "INVALID_IDENTIFIER"
            else:
                raise AssertionError("dot-segment domain escaped the data root")

            inspected = run_stage.inspect_stage("fixture", build_id, "env-pin")
            assert inspected == env_record
            stdout = io.StringIO()
            argv = sys.argv
            sys.argv = [
                "run_stage.py", "--inspect", "--domain", "fixture", "--build-id", build_id, "--stage", "env-pin",
            ]
            before = _tree_hash(root)
            try:
                with contextlib.redirect_stdout(stdout):
                    assert run_stage.main() == 0
            finally:
                sys.argv = argv
            after = _tree_hash(root)
            assert json.loads(stdout.getvalue()) == env_record
            assert before == after, "inspect wrote to the fixture tree"

            stderr = io.StringIO()
            argv = sys.argv
            sys.argv = [
                "run_stage.py", "--inspect", "--domain", "fixture", "--build-id", build_id, "--stage", "absent",
            ]
            before = _tree_hash(root)
            try:
                with contextlib.redirect_stderr(stderr):
                    assert run_stage.main() != 0
            finally:
                sys.argv = argv
            assert json.loads(stderr.getvalue())["error"] == "STAGE_RECORD_NOT_FOUND"
            assert before == _tree_hash(root), "missing-record inspect wrote to the fixture tree"
        finally:
            C.ROOT, C.CONFIGS, C.DATA = original


def test_build_identity_covers_the_whole_recipe_read_set() -> None:
    """Approving a recipe must bind everything whose content can change what actually runs.

    build_identity hashed only manifest + templates + contract version. A stage digest hashes the
    RESOLVED COMMAND, and editing the contents of a YAML the command merely references does not
    change that command — so an operator could approve one recipe and then run materially different
    parameters under the same build identity. Proven as a class, not an instance: four different
    roles in the read set, each mutated and restored independently.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fixture(root)
        (root / "configs/train_fixture.yaml").write_text("train:\n  lr: 1.0e-5\n", encoding="utf-8")
        (root / "schema").mkdir()
        (root / "schema/entity.schema.json").write_text('{"a": 1}\n', encoding="utf-8")
        manifest = yaml.safe_load((root / "configs/domains/fixture.yaml").read_text(encoding="utf-8"))
        manifest["configs"] = {"cpt": "configs/train_fixture.yaml"}
        manifest["ontology"] = {"schema_dir": "schema"}
        manifest["stages"] = [
            {"name": "stub", "cmd": ["{py}", "scripts/stub.py", "pass", "data/fixture/x.txt"]},
        ]
        (root / "configs/domains/fixture.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

        original = C.ROOT, C.CONFIGS, C.DATA
        C.ROOT, C.CONFIGS, C.DATA = root, root / "configs", root / "data"
        try:
            base, inputs = run_stage.build_identity(Domain("fixture"), "1")
            roles = {entry["role"] for entry in inputs["read_set"]}
            assert {"domain_manifest", "templates", "unit_registry", "ontology_schema",
                    "train_config:cpt"} <= roles, roles
            assert any(role.startswith("stage_code:") for role in roles), roles

            moved = {}
            for label, path, replacement in (
                ("train_config", root / "configs/train_fixture.yaml", "train:\n  lr: 9.0e-9\n"),
                ("unit_registry", root / "configs/editions_fixture.json", '{"editions": ["gold"]}\n'),
                ("ontology_schema", root / "schema/entity.schema.json", '{"a": 2}\n'),
                ("stage_code", root / "scripts/stub.py", "raise SystemExit(0)\n"),
            ):
                previous = path.read_text(encoding="utf-8")
                path.write_text(replacement, encoding="utf-8")
                moved[label] = run_stage.build_identity(Domain("fixture"), "1")[0] != base
                path.write_text(previous, encoding="utf-8")
                assert run_stage.build_identity(Domain("fixture"), "1")[0] == base, (
                    f"{label}: build id did not return to base after restore — hash is not content-addressed"
                )
            assert all(moved.values()), f"recipe content changed without moving the build id: {moved}"
        finally:
            C.ROOT, C.CONFIGS, C.DATA = original


if __name__ == "__main__":
    test_run_stage()
    test_build_identity_covers_the_whole_recipe_read_set()
    print("OK — recipe-wide build id, drift/device refusals, fail-closed records, and zero-write inspect")
