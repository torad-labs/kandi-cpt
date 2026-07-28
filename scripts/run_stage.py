#!/usr/bin/env python3
"""Execute one contract-declared stage or inspect its machine-stamped result."""
from __future__ import annotations

import sys

# Inspect mode is strictly read-only, including import-cache behavior.
sys.dont_write_bytecode = True

import argparse
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config as C
from execution_contract import build_contract
from pipeline.domain import Domain


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_BUILD_ID = re.compile(r"^[0-9a-f]{16}$")
_STAGE_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_HOST_GPU_PREFLIGHT_ENV = "CONDUCTOR_HOST_GPU_PREFLIGHT"


class StageError(Exception):
    """A clean machine-readable run_stage refusal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _rooted(ref: str) -> Path:
    path = Path(str(ref))
    return path if path.is_absolute() else C.ROOT / path


def _templates_path(domain: Domain) -> Path:
    return _rooted(domain.spec.get("templates") or f"configs/templates/{domain.name}.yaml")


MISSING_DIGEST = "<missing>"
# Tokens in a stage command that name an implementation file. "{py}" and "{editions}" carry braces
# and cannot match, so the interpreter and the placeholders are excluded by construction.
_CODE_TOKEN = re.compile(r"^[\w./-]+\.(?:py|sh)$")


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    """Content digest of a file, or of a directory tree keyed by sorted relative path."""
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_stream(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    return _sha256_stream(path)


def _stage_code_paths(domain: Domain) -> list[Path]:
    """Implementation files named by the manifest's own stage commands.

    Read from the manifest rather than the execution contract so the read set stays
    edition-independent: the same recipe must hash identically no matter which units are selected.
    """
    seen: set[str] = set()
    for stage in domain.stages:
        for token in stage.cmd:
            if isinstance(token, str) and _CODE_TOKEN.match(token):
                seen.add(str(_rooted(token)))
    return sorted(Path(value) for value in seen)


def recipe_read_set(domain: Domain) -> list[dict[str, str]]:
    """Every AUTHORED file whose content can change what the pipeline does, content-addressed.

    Authored, not derived. The tokenized shard and the split report are PRODUCED by the pipeline,
    so folding them in here would change the build id the moment `tokenize` runs and orphan every
    stage record written before it — readiness would then never find a record belonging to the
    current build. Their fingerprints belong to the readiness ladder's "shard, split, model and
    configuration fingerprints agree" clause, which is checked against a build, not baked into it.

    Admission has no file: it is an operator attestation (INPUTS_ADMITTED), so there is nothing to
    hash and no path is invented for it.
    """
    entries: list[tuple[str, Path]] = [
        ("domain_manifest", domain.path),
        ("templates", _templates_path(domain)),
    ]
    if domain.registry:
        entries.append(("unit_registry", _rooted(domain.registry)))
    schema_dir = (domain.ontology or {}).get("schema_dir")
    if schema_dir:
        entries.append(("ontology_schema", _rooted(str(schema_dir))))
    for key in sorted(domain.configs):
        entries.append((f"train_config:{key}", _rooted(str(domain.configs[key]))))
    for path in _stage_code_paths(domain):
        entries.append((f"stage_code:{path.name}", path))

    return [
        {
            "role": role,
            "path": _display_path(path),
            # A missing file is recorded as missing rather than skipped: deleting a referenced
            # config must move the build id, not silently shrink the read set back to a match.
            "sha256": _sha256_path(path) if path.exists() else MISSING_DIGEST,
        }
        for role, path in entries
    ]


def build_identity(domain: Domain, contract_version: str) -> tuple[str, dict[str, Any]]:
    """Build id over the COMPLETE recipe read set, not just the manifest and templates.

    A stage digest hashes the resolved command; changing the contents of a YAML that command
    merely references leaves the command — and previously the build id — identical, so an operator
    could approve one recipe and run materially different parameters under the same identity.
    """
    read_set = recipe_read_set(domain)
    payload = json.dumps(
        {"read_set": read_set, "contract_version": contract_version},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16], {
        "recipe_sha256": _sha256_bytes(payload),
        "contract_version_sha256": _sha256_bytes(contract_version.encode("utf-8")),
        "read_set": read_set,
    }


def _validate_identifier(value: str, label: str, pattern: re.Pattern[str] = _IDENTIFIER) -> None:
    if value in {".", ".."} or not pattern.fullmatch(value):
        raise StageError("INVALID_IDENTIFIER", f"{label} has an invalid value: {value!r}")


def _record_path(domain: str, build_id: str, stage_id: str) -> Path:
    _validate_identifier(domain, "domain")
    _validate_identifier(build_id, "build_id", _BUILD_ID)
    _validate_identifier(stage_id, "stage")
    data_root = C.DATA.resolve()
    path = (data_root / domain / "builds" / build_id / "stages" / f"{stage_id}.json").resolve()
    if not path.is_relative_to(data_root):
        raise StageError("INVALID_RECORD_PATH", "stage record path escapes the data root")
    return path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _attempt_path(path: Path) -> Path:
    attempts = path.parent / "attempts" / path.stem
    attempts.mkdir(parents=True, exist_ok=True)
    return attempts / f"{time.time_ns()}.json"


def _write_stage_record(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replacement = path.with_suffix(path.suffix + ".next")
    replacement.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        shutil.copy2(path, _attempt_path(path))
    os.replace(replacement, path)


def _write_refusal_record(path: Path, value: dict[str, Any]) -> None:
    _write_json(_attempt_path(path), value) if path.exists() else _write_json(path, value)


def _born_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_host_gpu_preflight(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise StageError("HOST_GPU_PREFLIGHT_INVALID", f"cannot decode host GPU preflight receipt: {error}") from error
    if not isinstance(value, dict) or value.get("allowed") is not True:
        raise StageError("HOST_GPU_PREFLIGHT_INVALID", "host GPU preflight receipt must be an allowed object")
    devices = value.get("devices")
    if not isinstance(devices, list) or not devices or not all(isinstance(device, dict) for device in devices):
        raise StageError("HOST_GPU_PREFLIGHT_INVALID", "host GPU preflight receipt has no measured devices")
    return value


def _preflight_selectors(receipt: dict[str, Any]) -> list[str]:
    return [str(device.get("selector") or "") for device in receipt["devices"]]


def _ensure_build(domain: Domain, build_id: str, inputs: dict[str, str], contract_version: str) -> None:
    path = C.DATA / domain.name / "builds" / build_id / "build.json"
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StageError("BUILD_METADATA_INVALID", f"cannot read {path}: {error}") from error
        if current.get("build_id") != build_id or current.get("inputs") != inputs:
            raise StageError("BUILD_METADATA_CONFLICT", f"existing build metadata does not match recipe {build_id}")
        return
    _write_json(path, {
        "build_id": build_id,
        "inputs": inputs,
        "born_at": _born_at(),
        "contract_version": contract_version,
    })


def _stage_entry(contract: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in contract.get("stages", []):
        if stage.get("id") == stage_id:
            return stage
    raise StageError("UNKNOWN_STAGE", f"stage {stage_id!r} is absent from the execution contract")


def _resolved_command(stage: dict[str, Any]) -> list[str]:
    command = stage.get("cmd")
    if not isinstance(command, list) or not command or not all(isinstance(argument, str) for argument in command):
        raise StageError("CONTRACT_STAGE_INVALID", f"stage {stage.get('id')!r} has no resolved command")
    return command


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(C.ROOT))
    except ValueError:
        return str(path)


def _matched_files(pattern: str) -> list[Path]:
    path = Path(pattern)
    rooted = path if path.is_absolute() else C.ROOT / path
    matches = [Path(match) for match in glob.glob(str(rooted), recursive=True)]
    files: list[Path] = []
    for match in matches:
        if match.is_file():
            files.append(match)
        elif match.is_dir():
            files.extend(candidate for candidate in match.rglob("*") if candidate.is_file())
    return sorted(set(files))


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {"path": _display_path(path), "sha256": digest.hexdigest(), "bytes": byte_count}


def _collect_outputs(patterns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matched = _matched_files(pattern)
        if not matched:
            missing.append(pattern)
        for path in matched:
            if path not in seen:
                seen.add(path)
                artifacts.append(_artifact(path))
    return artifacts, missing


def _output_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, int]:
    before_by_path = {artifact["path"]: artifact for artifact in before}
    after_by_path = {artifact["path"]: artifact for artifact in after}
    shared = before_by_path.keys() & after_by_path.keys()
    changed = sum(
        before_by_path[path]["sha256"] != after_by_path[path]["sha256"]
        for path in shared
    )
    return {
        "created": len(after_by_path.keys() - before_by_path.keys()),
        "changed": changed,
        "unchanged": len(shared) - changed,
        "removed": len(before_by_path.keys() - after_by_path.keys()),
        "bytes_before": sum(artifact["bytes"] for artifact in before),
        "bytes_after": sum(artifact["bytes"] for artifact in after),
    }


def _postcondition(spec: dict[str, Any]) -> dict[str, Any]:
    check = str(spec.get("check") or spec.get("kind") or "unnamed postcondition")
    kind = spec.get("kind")
    path = str(spec.get("path") or "")
    matched = _matched_files(path) if path else []
    if kind == "output_exists":
        passed = bool(matched)
    elif kind == "file_nonempty":
        passed = bool(matched) and all(candidate.stat().st_size > 0 for candidate in matched)
    elif kind == "file_contains":
        needle = str(spec.get("contains") or "")
        passed = bool(matched) and bool(needle) and all(
            needle in candidate.read_text(encoding="utf-8", errors="replace") for candidate in matched
        )
    elif kind == "report_verdict":
        passed = bool(matched) and all(_report_verdict_ok(candidate, spec) for candidate in matched)
    elif kind == "jsonl_no_fail_verdict":
        passed = bool(matched) and all(_jsonl_verdicts_ok(candidate, spec) for candidate in matched)
    elif kind == "tokenize_stats_sound":
        passed = bool(matched) and all(_tokenize_stats_ok(candidate, spec) for candidate in matched)
    elif kind == "json_field_positive":
        passed = bool(matched) and all(_json_field_positive(candidate, spec) for candidate in matched)
    elif kind == "split_integrity":
        passed = bool(matched) and all(_split_integrity_ok(candidate) for candidate in matched)
    elif kind == "checkpoint_integrity":
        target = _rooted(path)
        passed = target.is_dir() and _checkpoint_integrity_ok(target) and _checkpoint_recipe_ok(target, spec)
    else:
        passed = False
    return {"check": check, "pass": passed}


# --- semantic postconditions -------------------------------------------------------------------
# The generic checks are exit-zero, declared outputs exist, output delta nonzero, recipe identity
# held. None of them can see a stage that did its job and reported failure: audit_recall writes
# report_<domain>.json with status FAIL, exits 0, and the file is present and changed — so the
# stage record said PASS. These read the verdict the stage itself recorded. Every one fails CLOSED:
# absent, unparseable, or missing-field is a failure, never a pass, because "I could not tell"
# must never be recorded as "it passed".

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _report_verdict_ok(path: Path, spec: dict[str, Any]) -> bool:
    """A report whose recorded verdict is not in `expect` fails its stage."""
    report = _load_json(path)
    if not isinstance(report, dict):
        return False
    field = str(spec.get("field") or "status")
    if field not in report:
        return False
    expected = {str(value).upper() for value in (spec.get("expect") or ["PASS"])}
    return str(report[field]).upper() in expected


def _jsonl_verdicts_ok(path: Path, spec: dict[str, Any]) -> bool:
    """No row for THIS stage's editions may carry a failing dimension.

    The audit results file is shared across domains and keyed only by edition slug, so the rows are
    filtered by the editions the contract scoped this stage to. Checking every row instead would
    fail millbrook/main because a booster_1 row failed — measured, not hypothetical. `warn` is not
    `fail`; no matching row at all is not a pass.
    """
    field = str(spec.get("field") or "verdicts")
    key = str(spec.get("row_key") or "editions")
    wanted = {str(value) for value in (spec.get("editions") or [])}
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return False
    matched = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return False
        if wanted and str(row.get(key)) not in wanted:
            continue
        matched += 1
        verdicts = row.get(field)
        if not isinstance(verdicts, dict):
            return False
        if any(str(value).lower() == "fail" for value in verdicts.values()):
            return False
    return matched > 0


def _json_field_positive(path: Path, spec: dict[str, Any]) -> bool:
    """A numeric field that must be > 0 — a tokenize that packed zero blocks did not succeed."""
    report = _load_json(path)
    field = str(spec.get("field") or "")
    if not isinstance(report, dict) or field not in report:
        return False
    value = report[field]
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _checkpoint_recipe_ok(directory: Path, spec: dict[str, Any]) -> bool:
    """The run ENDED where it was planned to, and at the precision that was approved.

    A checkpoint that stopped early looks identical on disk to one that ran to completion — same
    files, same manifest, same tensor set — so "the artifact is valid" cannot answer "is this the
    run we approved". run_contract.json is the trainer's own record of what it actually did; the
    expected values come from the contract, which read them from the training config.

    An early stop is legitimate, but only when RECORDED: scored_checkpoint names it. Silence plus a
    short run is the case this refuses.
    """
    contract = _load_json(directory / "run_contract.json")
    if not isinstance(contract, dict):
        return False

    final_step, planned = contract.get("final_step"), contract.get("max_steps")
    if final_step is None or planned is None:
        return False                       # a run that cannot say where it stopped is not verifiable
    if int(final_step) != int(planned):
        # Short is allowed only with a recorded reason.
        if not str(contract.get("scored_checkpoint") or "").strip() or int(final_step) <= 0:
            return False

    for field in ("precision", "int8_master"):
        expected = spec.get(f"expect_{field}")
        if expected is not None and str(contract.get(field)) != str(expected):
            return False
    return True


def _tokenize_stats_ok(path: Path, spec: dict[str, Any]) -> bool:
    """Nonzero blocks, the replay proportion the shard actually realized, and lever metadata.

    "replay_frac was requested" is not "replay_frac was delivered": the packer truncates its replay
    pool to a token target, so a shard can silently under-deliver. tokenize_corpus stamps BOTH the
    requested and the realized fraction; a shard that carries neither cannot prove its own
    composition, and a postcondition runs on a FRESHLY written shard, so absence means the writer
    did not stamp it rather than that the shard is merely old.
    """
    stats = _load_json(path)
    if not isinstance(stats, dict):
        return False
    if not isinstance(stats.get("num_chunks"), int) or stats["num_chunks"] <= 0:
        return False
    if spec.get("require_lever_metadata") and not stats.get("lever_metadata"):
        return False

    requested = stats.get("requested_final_replay_fraction")
    realized = stats.get("realized_final_replay_fraction")
    if requested is None and realized is None:
        # No replay was requested for this corpus — nothing to reconcile.
        return not stats.get("replay_tokens")
    if requested is None or realized is None:
        return False
    tolerance = float(spec.get("replay_tolerance", 0.02))
    return abs(float(realized) - float(requested)) <= tolerance


def _split_integrity_ok(path: Path) -> bool:
    """A split whose holdout leaks into train measures nothing; it just reports a number.

    Checked against what the manifest itself declares, which is what a postcondition can do
    cheaply and honestly: train and holdout disjoint, holdout non-empty, zero alias leaks on the
    alias-closed versions, and a realized holdout fraction near the declared one.

    NOT checked here: coverage against the expected fact population. train_ids/holdout_ids hold
    unique triple IDs while n_triples counts triples, so 553+77 against n_triples 721 is 91
    DUPLICATE ids, not 91 dropped facts — asserting equality would be a false alarm. Establishing
    real coverage needs the triples reloaded, which belongs in the split stage, not a postcondition.
    """
    manifest = _load_json(path)
    if not isinstance(manifest, dict):
        return False
    train, holdout = manifest.get("train_ids"), manifest.get("holdout_ids")
    if not isinstance(train, list) or not isinstance(holdout, list) or not holdout:
        return False
    if set(train) & set(holdout):
        return False
    if str(manifest.get("version")) in ("v2", "v3"):
        decontamination = manifest.get("decontamination")
        if not isinstance(decontamination, dict):
            return False
        if decontamination.get("n_alias_leaks") != 0 or decontamination.get("zero_alias_leaks") is not True:
            return False
    # Population coverage, against the denominator the split now RECORDS. Previously unmeasurable:
    # train_ids/holdout_ids hold unique triple IDs while n_triples counts triples, so comparing
    # them read as dropped facts when nothing was dropped. With n_unique_triple_ids recorded the
    # real question is answerable — did every fact get classified as train or holdout.
    unique_ids = manifest.get("n_unique_triple_ids")
    if unique_ids is not None:
        assigned = len(set(train) | set(holdout))
        if int(unique_ids) <= 0 or assigned < int(unique_ids):
            return False

    declared = manifest.get("holdout_frac")
    if isinstance(declared, (int, float)) and not isinstance(declared, bool) and declared > 0:
        realized = len(holdout) / max(1, len(train) + len(holdout))
        # Group-based splitting cannot hit the fraction exactly; a wide band still catches a split
        # that produced a token holdout or handed over most of the corpus.
        if not (declared * 0.5) <= realized <= (declared * 2.0):
            return False
    return True


def _safetensors_tensor_names(path: Path) -> set[str] | None:
    """Tensor names from the safetensors header alone — 8-byte LE length then JSON, no tensor data.

    Parsed directly rather than through torch so the check stays a header read of a few hundred
    kilobytes instead of loading a multi-gigabyte checkpoint into a validator.
    """
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None
            length = int.from_bytes(raw_length, "little")
            if not 0 < length <= 200 * 1024 * 1024:
                return None
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return {name for name in header if name != "__metadata__"}


_INCOMPLETE_SUFFIXES = (".partial", ".tmp", ".next", ".incomplete")


def _checkpoint_integrity_ok(directory: Path) -> bool:
    """The checkpoint opens, its quant manifest matches the tensors, and nothing half-written remains.

    "The output directory changed" is what the generic delta check sees, and a changed log file
    satisfies it without a loadable checkpoint ever being written. This opens the real artifact.
    """
    weights = directory / "model.safetensors"
    quant_keys = directory / "quant_keys.json"
    if not weights.is_file() or not quant_keys.is_file():
        return False
    for required in ("config.json", "tokenizer.json"):
        if not (directory / required).is_file():
            return False
    if any(child.name.endswith(_INCOMPLETE_SUFFIXES) for child in directory.iterdir()):
        return False

    names = _safetensors_tensor_names(weights)
    if names is None:
        return False
    manifest = _load_json(quant_keys)
    if not isinstance(manifest, dict):
        return False
    quantized = [*(manifest.get("keys") or []), *(manifest.get("embedding_keys") or [])]
    if not quantized:
        return False
    for key in quantized:
        # Every declared quantized tensor must be present WITH its scale. A manifest naming a
        # tensor the file does not hold is precisely the "shipped a format nothing had loaded"
        # failure: bytes moved, the artifact was unusable.
        if key not in names or f"{key}.weight_scale" not in names:
            return False
    return True


def _base_record(
    command: list[str],
    devices: str,
    exit_code: int | None,
    duration_s: float,
    outputs: list[dict[str, Any]],
    output_delta: dict[str, int],
    postconditions: list[dict[str, Any]],
    engine: dict[str, Any],
    status: str,
    host_gpu_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "cmd": command,
        "devices": devices,
        "exit_code": exit_code,
        "duration_s": round(duration_s, 6),
        "outputs": outputs,
        "output_delta": output_delta,
        "postconditions": postconditions,
        "engine": {"commit": engine.get("commit"), "dirty": engine.get("dirty")},
        "status": status,
    }
    if host_gpu_preflight is not None:
        record["host_gpu_preflight"] = host_gpu_preflight
    return record


def _post_run_contract_checks(
    domain_name: str,
    editions: str | None,
    build_id: str,
    stage_id: str,
    expected_stage_digest: str,
    contract_builder: Callable[[str, str | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        contract = contract_builder(domain_name, editions)
        contract_version = str(contract.get("contract_version") or "")
        current_build_id, _ = build_identity(Domain(domain_name), contract_version)
        current_digest = _stage_entry(contract, stage_id).get("stage_digest")
    except Exception as error:  # fail closed: a stage cannot PASS when its identity became unreadable mid-run
        return [{"check": f"recipe and approved stage stayed readable: {error}", "pass": False}]
    return [
        {"check": "recipe identity stayed unchanged during execution", "pass": current_build_id == build_id},
        {"check": "approved stage stayed unchanged during execution", "pass": current_digest == expected_stage_digest},
    ]


def _scope_record(record: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    record["editions"] = stage.get("editions", [])
    record["stage_digest"] = stage.get("stage_digest")
    return record


def _refusal_record(
    command: list[str],
    devices: str,
    engine: dict[str, Any],
    code: str,
    message: str,
    host_gpu_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = _base_record(
        command=command,
        devices=devices,
        exit_code=None,
        duration_s=0.0,
        outputs=[],
        output_delta={"created": 0, "changed": 0, "unchanged": 0, "removed": 0, "bytes_before": 0, "bytes_after": 0},
        postconditions=[{"check": message, "pass": False}],
        engine=engine,
        status="FAIL",
        host_gpu_preflight=host_gpu_preflight,
    )
    record["error"] = code
    return record


def execute_stage(
    domain_name: str,
    stage_id: str,
    devices: str,
    editions: str | None = None,
    requested_build_id: str | None = None,
    expected_stage_digest: str | None = None,
    contract_builder: Callable[[str, str | None], dict[str, Any]] | None = None,
    host_gpu_preflight: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, str]:
    _validate_identifier(domain_name, "domain")
    lock_path = C.DATA / domain_name / "builds" / ".run-stage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _execute_stage_locked(
            domain_name,
            stage_id,
            devices,
            editions,
            requested_build_id,
            expected_stage_digest,
            contract_builder,
            host_gpu_preflight,
        )


def _execute_stage_locked(
    domain_name: str,
    stage_id: str,
    devices: str,
    editions: str | None,
    requested_build_id: str | None,
    expected_stage_digest: str | None,
    contract_builder: Callable[[str, str | None], dict[str, Any]] | None,
    host_gpu_preflight: dict[str, Any] | None,
) -> tuple[dict[str, Any], int, str]:
    _validate_identifier(domain_name, "domain")
    _validate_identifier(stage_id, "stage")
    if requested_build_id is not None:
        _validate_identifier(requested_build_id, "build_id", _BUILD_ID)

    domain = Domain(domain_name)
    active_contract_builder = contract_builder or build_contract
    contract = active_contract_builder(domain_name, editions)
    contract_version = str(contract.get("contract_version") or "")
    if not contract_version:
        raise StageError("CONTRACT_INVALID", "execution contract has no contract_version")
    computed_build_id, inputs = build_identity(domain, contract_version)
    build_id = requested_build_id or computed_build_id
    raw_engine = contract.get("engine")
    engine: dict[str, Any] = raw_engine if isinstance(raw_engine, dict) else {}

    if requested_build_id is not None and requested_build_id != computed_build_id:
        message = f"BUILD_DRIFT: requested {requested_build_id}, current recipe is {computed_build_id}"
        try:
            drift_stage = _stage_entry(contract, stage_id)
            command = _resolved_command(drift_stage)
        except StageError:
            drift_stage = {
                "editions": contract.get("editions", []),
                "stage_digest": expected_stage_digest,
            }
            command = []
        record = _scope_record(
            _refusal_record(command, devices, engine, "BUILD_DRIFT", message, host_gpu_preflight),
            drift_stage,
        )
        _write_refusal_record(_record_path(domain_name, build_id, stage_id), record)
        return record, 2, build_id

    stage = _stage_entry(contract, stage_id)
    command = _resolved_command(stage)

    def refusal(code: str, message: str) -> dict[str, Any]:
        return _scope_record(
            _refusal_record(command, devices, engine, code, message, host_gpu_preflight),
            stage,
        )

    record_path = _record_path(domain_name, build_id, stage_id)
    scoped_editions = stage.get("editions")
    if stage_id != "scrape" and (not isinstance(scoped_editions, list) or not scoped_editions):
        message = "NO_UNITS: a non-scrape stage cannot run with an empty unit scope"
        record = refusal("NO_UNITS", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    stage_digest = stage.get("stage_digest")
    if expected_stage_digest is None:
        message = "APPROVED_STAGE_REQUIRED: execute mode requires the host-approved stage digest"
        record = refusal("APPROVED_STAGE_REQUIRED", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    _validate_identifier(expected_stage_digest, "approved_stage_digest", _STAGE_DIGEST)
    if stage_digest != expected_stage_digest:
        message = f"CONTRACT_DRIFT: approved {expected_stage_digest}, current stage is {stage_digest!r}"
        record = refusal("CONTRACT_DRIFT", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id

    raw_effects = stage.get("effects")
    effects: list[Any] = raw_effects if isinstance(raw_effects, list) else []
    if "gpu" in effects and not devices:
        message = "GPU_DEVICES_REQUIRED: a gpu-effect stage cannot run with empty --devices"
        record = refusal("GPU_DEVICES_REQUIRED", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    if "gpu" in effects and host_gpu_preflight is None:
        message = "HOST_GPU_PREFLIGHT_REQUIRED: a gpu-effect stage requires the host's measured allowance receipt"
        record = refusal("HOST_GPU_PREFLIGHT_REQUIRED", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    if "gpu" in effects and _preflight_selectors(host_gpu_preflight or {}) != devices.split(","):
        message = "HOST_GPU_PREFLIGHT_MISMATCH: measured selectors do not match --devices"
        record = refusal("HOST_GPU_PREFLIGHT_MISMATCH", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    if "gpu" not in effects and host_gpu_preflight is not None:
        message = "HOST_GPU_PREFLIGHT_UNEXPECTED: a non-gpu stage cannot carry a GPU allowance receipt"
        record = refusal("HOST_GPU_PREFLIGHT_UNEXPECTED", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id

    raw_output_patterns = stage.get("outputs")
    output_patterns: list[Any] = raw_output_patterns if isinstance(raw_output_patterns, list) else []
    raw_specs = stage.get("postconditions")
    specs: list[Any] = raw_specs if isinstance(raw_specs, list) else []
    if not output_patterns or not specs:
        message = "CONTRACT_STAGE_INVALID: execution requires declared outputs and postconditions"
        record = refusal("CONTRACT_STAGE_INVALID", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    protected = set(domain.spec.get("protected_editions") or [])
    targeted_protected = protected.intersection(scoped_editions or [])
    protected_outputs = [
        output
        for output in output_patterns
        if any(str(output).startswith(f"data/{domain.name}/{unit}/") for unit in targeted_protected)
    ]
    if protected_outputs:
        units = ", ".join(sorted(targeted_protected))
        message = f"PROTECTED_UNITS: stage outputs would modify gate-only unit(s): {units}"
        record = refusal("PROTECTED_UNITS", message)
        _write_refusal_record(record_path, record)
        return record, 2, build_id
    _ensure_build(domain, build_id, inputs, contract_version)
    try:
        before_outputs, _ = _collect_outputs([str(pattern) for pattern in output_patterns])
    except OSError as error:
        record = refusal("ARTIFACT_READ_FAILED", f"ARTIFACT_READ_FAILED: {error}")
        _write_stage_record(record_path, record)
        return record, 2, build_id
    child_env = {**os.environ, "PYTHONPATH": str(C.ROOT / "scripts"), "CUDA_VISIBLE_DEVICES": devices}
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=C.ROOT, env=child_env, check=False)
        exit_code = completed.returncode
    except OSError as error:
        duration_s = time.monotonic() - started
        record = refusal("SPAWN_FAILED", f"SPAWN_FAILED: {error}")
        record["duration_s"] = round(duration_s, 6)
        _write_stage_record(record_path, record)
        return record, 2, build_id
    duration_s = time.monotonic() - started

    try:
        outputs, missing = _collect_outputs([str(pattern) for pattern in output_patterns])
    except OSError as error:
        record = refusal("ARTIFACT_READ_FAILED", f"ARTIFACT_READ_FAILED: {error}")
        record["exit_code"] = exit_code
        record["duration_s"] = round(duration_s, 6)
        _write_stage_record(record_path, record)
        return record, 1, build_id
    output_delta = _output_delta(before_outputs, outputs)
    postconditions = [
        _postcondition(spec) if isinstance(spec, dict) else {"check": str(spec), "pass": False}
        for spec in specs
    ]
    if not output_patterns:
        postconditions.append({"check": "contract declares at least one output", "pass": False})
    postconditions.extend({"check": f"declared output exists: {path}", "pass": False} for path in missing)
    delta_passed = output_delta["created"] + output_delta["changed"] > 0
    postconditions.append({"check": "declared output delta is nonzero", "pass": delta_passed})
    postconditions.extend(
        _post_run_contract_checks(
            domain_name,
            editions,
            build_id,
            stage_id,
            expected_stage_digest,
            active_contract_builder,
        ),
    )
    passed = exit_code == 0 and not missing and bool(output_patterns) and all(
        condition["pass"] for condition in postconditions
    )
    record = _scope_record(
        _base_record(
            command=command,
            devices=devices,
            exit_code=exit_code,
            duration_s=duration_s,
            outputs=outputs,
            output_delta=output_delta,
            postconditions=postconditions,
            engine=engine,
            status="PASS" if passed else "FAIL",
            host_gpu_preflight=host_gpu_preflight,
        ),
        stage,
    )
    _write_stage_record(record_path, record)
    return record, 0 if passed else 1, build_id


def inspect_stage(domain_name: str, build_id: str, stage_id: str) -> dict[str, Any]:
    path = _record_path(domain_name, build_id, stage_id)
    if not path.is_file():
        raise StageError("STAGE_RECORD_NOT_FOUND", f"stage record does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError("STAGE_RECORD_INVALID", f"cannot read stage record {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageError("STAGE_RECORD_INVALID", f"stage record is not a JSON object: {path}")
    return value


def _error(error: StageError) -> int:
    print(json.dumps({"error": error.code, "message": error.message}), file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="print an existing stage record; never write")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--editions", help="comma-separated unit scope approved for this stage")
    parser.add_argument("--devices", help="exact CUDA_VISIBLE_DEVICES value for the child")
    parser.add_argument("--approved-stage-digest", help="exact stage contract approved by the host")
    parser.add_argument("--build-id")
    args = parser.parse_args()

    try:
        if args.inspect:
            if not args.build_id:
                raise StageError("BUILD_ID_REQUIRED", "--inspect requires --build-id")
            json.dump(inspect_stage(args.domain, args.build_id, args.stage), sys.stdout, separators=(",", ":"))
            sys.stdout.write("\n")
            return 0
        if args.devices is None:
            raise StageError("DEVICES_REQUIRED", "execute mode requires --devices (use an empty string for CPU stages)")
        record, exit_code, build_id = execute_stage(
            args.domain,
            args.stage,
            args.devices,
            editions=args.editions,
            requested_build_id=args.build_id,
            expected_stage_digest=args.approved_stage_digest,
            host_gpu_preflight=_parse_host_gpu_preflight(os.environ.get(_HOST_GPU_PREFLIGHT_ENV)),
        )
        summary = {
            "build_id": build_id,
            "stage": args.stage,
            "status": record["status"],
            "error": record.get("error"),
        }
        print(json.dumps(summary, separators=(",", ":")))
        if record.get("error"):
            print(json.dumps({"error": record["error"], "build_id": build_id, "stage": args.stage}), file=sys.stderr)
        return exit_code
    except StageError as error:
        return _error(error)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return _error(StageError("RUN_STAGE_ERROR", str(error)))


if __name__ == "__main__":
    raise SystemExit(main())
