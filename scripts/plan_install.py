#!/usr/bin/env python3
"""Compute a read-only install-readiness plan from a domain manifest and current disk state."""
from __future__ import annotations

import sys

# A read-only planner must not even leave import-cache files behind.
sys.dont_write_bytecode = True

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import yaml

import config as C
from execution_contract import build_contract
from pipeline.domain import Domain
from pipeline.editions import iter_editions
from run_pipeline import check_domain


# The rungs are deliberately separate because they fail for unrelated reasons and an operator
# needs to read WHICH. A single TRAINING_READY that meant "the files a stage would read exist"
# reported PASS for Kanto while its own extraction audit said FAIL: mechanically launchable and
# actually ready are different claims, and collapsing them is what makes an unattended GPU launch
# look authorised.
LADDER_STATES = [
    "MANIFEST_VALID",
    "INPUTS_ADMITTED",
    "RECIPE_COMPLETE",
    "MECHANICALLY_LAUNCHABLE",
    "DATA_GATES_PASS",
    "TRAIN_CONFIG_FROZEN",
    "TRAINING_READY",
]

# This is the planner's read-only stage-input vocabulary.  It describes paths a
# stage must be able to read now; it does not imply that the stage has run.
STAGE_INPUTS = {
    "scrape": (),
    "extract": ("unit:00_raw",),
    # Phase A data-selection (after extract / before or with curate).
    "pii_scan": ("unit:10_clean",),
    "score_educational": ("unit:10_clean",),
    "style_fingerprint": (),  # reads 10_clean and/or data/<domain>/style_inbox/
    "curate": ("unit:10_clean",),
    "structured": ("unit:10_clean",),
    # Phase B graph correctness (after structured, before/around split).
    "resolve": ("unit:15_structured",),
    "split": ("unit:15_structured",),
    "hydrate": ("unit:15_structured",),
    "contrast": ("unit:15_structured",),
    "reasoning": ("unit:15_structured", "unit:25_hydrated"),
    "synth": ("unit:15_structured",),
    # Phase C synthetic layer (after hydrate/synth surfaces exist).
    "dedup_surfaces": ("unit:25_hydrated",),
    "probes": ("unit:15_structured",),
    "audit": ("unit:15_structured", "unit:25_hydrated"),
    "audit_recall": ("unit:10_clean", "unit:15_structured"),
    "tokenize": ("unit:25_hydrated",),
    "cpt": ("config:cpt:data.tokenized",),
    "build_sft": ("unit:15_structured", "command:--score-model"),
    "sft": ("config:sft:data.tokenized", "config:sft:data.init_from"),
}

TRAIN_STAGES = {"cpt", "build_sft", "sft"}
SCRIPT_SUFFIXES = (".py", ".sh", ".rs")


def _rooted(ref: str) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else C.ROOT / path


def _gap_from_check(check: dict[str, Any]) -> str:
    detail = check.get("detail") or check.get("path") or "required check failed"
    return f"{check.get('check', 'unknown check')}: {detail}"


def _manifest_state(domain_name: str, editions: str) -> tuple[dict[str, Any], bool]:
    report, ok = check_domain(domain_name, editions)
    gaps = [
        _gap_from_check(check)
        for check in report.get("checks", [])
        if check.get("severity") == "REQUIRED" and not check.get("ok")
    ]
    return {
        "state": "MANIFEST_VALID",
        "value": "PASS" if ok else "FAIL",
        "evidence": [report],
        "gaps": gaps,
    }, ok


def _script_paths(cmd: list[str]) -> list[Path]:
    paths: list[Path] = []
    if "-m" in cmd:
        index = cmd.index("-m")
        if index + 1 < len(cmd):
            module = Path("scripts") / Path(cmd[index + 1].replace(".", "/"))
            package_entry = C.ROOT / module / "__main__.py"
            paths.append(package_entry if package_entry.exists() else C.ROOT / module.with_suffix(".py"))
    paths.extend(
        _rooted(arg)
        for arg in cmd
        if "/" in arg and arg.endswith(SCRIPT_SUFFIXES)
    )
    return list(dict.fromkeys(paths))


def _has_authored_predicate_templates(templates: dict[str, Any]) -> bool:
    predicates = templates.get("hydrate_templates", {}).get("predicate_templates", {})
    return isinstance(predicates, dict) and any(bool(surfaces) for surfaces in predicates.values())


def _recipe_state(domain: Domain, editions: str) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    gaps: list[str] = []
    for stage in domain.stages:
        try:
            cmd = domain.resolve(stage.cmd, domain=domain.name, editions=editions)
            scripts = _script_paths(cmd)
            missing = [str(path) for path in scripts if not path.exists()]
            if not scripts:
                missing.append("no resolvable in-tree script")
            evidence.append({"stage": stage.name, "scripts": [str(path) for path in scripts], "missing": missing})
            gaps.extend(f"stage {stage.name}: missing script {path}" for path in missing)
        except KeyError as error:
            evidence.append({"stage": stage.name, "scripts": [], "missing": [str(error)]})
            gaps.append(f"stage {stage.name}: command does not resolve: {error}")

    stage_names = {stage.name for stage in domain.stages}
    template_checks = []
    if "hydrate" in stage_names:
        ok = _has_authored_predicate_templates(domain.templates)
        template_checks.append({"template": "hydrate_templates.predicate_templates", "ok": ok})
        if not ok:
            gaps.append("hydrate: templates must author hydrate_templates.predicate_templates for at least one predicate")
    if "contrast" in stage_names:
        ok = bool(domain.templates.get("contrast"))
        template_checks.append({"template": "contrast", "ok": ok})
        if not ok:
            gaps.append("contrast: templates must contain a non-empty contrast block")
    if "sft" in stage_names:
        ok = bool(domain.templates.get("sft_templates", {}).get("rel_q"))
        template_checks.append({"template": "sft_templates.rel_q", "ok": ok})
        if not ok:
            gaps.append("sft: templates must contain a non-empty sft_templates.rel_q block")
    if "reasoning" in stage_names:
        ok = bool(domain.templates.get("sft_templates", {}).get("rel_q"))
        template_checks.append({"template": "reasoning:sft_templates.rel_q", "ok": ok})
        if not ok:
            gaps.append("reasoning: templates must contain a non-empty sft_templates.rel_q block "
                        "(the declaratives the walk states to the generator)")
    evidence.extend(template_checks)
    return {
        "state": "RECIPE_COMPLETE",
        "value": "PASS" if not gaps else "FAIL",
        "evidence": evidence,
        "gaps": gaps,
    }


def _load_config(domain: Domain, key: str) -> tuple[dict[str, Any] | None, list[str]]:
    ref = domain.configs.get(key)
    if not ref:
        return None, [f"manifest configs.{key} is unset"]
    path = _rooted(str(ref))
    if not path.is_file():
        return None, [f"manifest configs.{key} does not exist: {path}"]
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        return None, [f"cannot read configs.{key} at {path}: {error}"]
    if not isinstance(loaded, dict):
        return None, [f"configs.{key} must contain a YAML mapping: {path}"]
    return loaded, []


def _config_input_missing(domain: Domain, config_key: str, dotted_key: str) -> list[str]:
    config, gaps = _load_config(domain, config_key)
    if config is None:
        return gaps
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return [f"configs.{config_key} is missing {dotted_key}"]
        value = value[part]
    if not isinstance(value, str) or not value:
        return [f"configs.{config_key} {dotted_key} is unset"]
    path = _rooted(value)
    return [] if path.exists() else [f"configs.{config_key} {dotted_key} does not exist: {path}"]


def _command_input_missing(domain: Domain, stage_name: str, flag: str, editions: str) -> list[str]:
    stage = domain.stage(stage_name)
    if stage is None:
        return []
    try:
        cmd = domain.resolve(stage.cmd, domain=domain.name, editions=editions)
    except KeyError as error:
        return [f"stage {stage_name} command does not resolve: {error}"]
    if flag not in cmd:
        return []
    index = cmd.index(flag)
    if index + 1 >= len(cmd):
        return [f"stage {stage_name} {flag} has no value"]
    path = _rooted(cmd[index + 1])
    return [] if path.exists() else [f"stage {stage_name} {flag} input does not exist: {path}"]


def _stage_readiness(domain: Domain, units: list[Any], editions: str, unit_error: str | None) -> list[dict[str, Any]]:
    readiness = []
    for stage in domain.stages:
        inputs = STAGE_INPUTS.get(stage.name)
        if inputs is None:
            missing = [f"inputs unknown for stage '{stage.name}'"]
        else:
            missing: list[str] = []
            for input_spec in inputs:
                kind, value = input_spec.split(":", 1)
                if kind == "unit":
                    if unit_error:
                        missing.append(unit_error)
                    elif not units:
                        missing.append("no registry units matched the requested --editions filter")
                    else:
                        for unit in units:
                            path = unit.dir / value
                            if not path.exists():
                                missing.append(f"{unit.slug}: {path}")
                elif kind == "config":
                    config_key, dotted_key = value.split(":", 1)
                    missing.extend(_config_input_missing(domain, config_key, dotted_key))
                elif kind == "command":
                    missing.extend(_command_input_missing(domain, stage.name, value, editions))
            missing = list(dict.fromkeys(missing))
        readiness.append({"stage": stage.name, "ready": not missing, "missing": missing})
    return readiness


def _split_status(domain: Domain, units: list[Any]) -> list[dict[str, Any]]:
    tokenize = domain.stage("tokenize")
    excludes_clean = bool(tokenize and "--no-clean" in tokenize.cmd)
    split_stage = domain.stage("split")
    split_version = "v1"
    if split_stage and "--version" in split_stage.cmd:
        version_index = split_stage.cmd.index("--version") + 1
        if version_index < len(split_stage.cmd):
            split_version = str(split_stage.cmd[version_index])
    status = []
    for unit in units:
        split_path = unit.dir / "splits" / f"split_{split_version}.json"
        if not split_path.exists():
            measurement = "recall-only"
            note = "no split manifest; probes can measure trained-fact recall only"
        elif excludes_clean:
            measurement = "fact-holdout available"
            note = "split manifest exists and tokenize is configured with --no-clean"
        else:
            measurement = "augmentation-holdout only"
            note = "split exists, but clean prose is not excluded; fact-holdout is not established"
        status.append({
            "unit": unit.slug,
            "split_manifest": str(split_path),
            "exists": split_path.exists(),
            "measurement": measurement,
            "note": note,
        })
    return status


# Stages whose EXIT CODE IS NOT THEIR VERDICT: each writes a report carrying the judgement in
# "status" and exits 0 either way. Readiness therefore has to read the verdict; statting the
# declared output only proves the stage ran. Kanto shipped as TRAINING_READY=PASS for a day with
# audit_recall status=FAIL (coverage 0.3529 against its own 0.6 gate) sitting in the report.
_QUALITY_STAGES = ("pii_scan", "audit_recall")


def _quality_verdicts(contract_stages: list[dict[str, Any]], units: list[Any]) -> list[dict[str, Any]]:
    """Recover each quality stage's recorded PASS/FAIL from the report it declares as an output.

    The path comes from the execution contract, not a literal here, so a stage that moves its
    report cannot silently detach the gate from the artifact it is supposed to be reading.
    """
    by_id = {stage["id"]: stage for stage in contract_stages}
    verdicts: list[dict[str, Any]] = []
    for name in _QUALITY_STAGES:
        stage = by_id.get(name)
        if stage is None:
            # Recorded, not gapped. A domain that never declares audit_recall has no extraction-
            # quality evidence at all, which is weaker than one that ran it and failed — but
            # millbrook-hollow is a deliberately minimal selftest domain, so making an undeclared
            # quality stage BLOCK readiness is an operator policy call, not this gate's to make.
            # Surfaced here so "no gate" is legible instead of indistinguishable from "gate passed".
            verdicts.append({"stage": name, "report": None, "status": "NOT_DECLARED",
                             "detail": "domain does not declare this quality stage"})
            continue
        reports = [out for out in stage["outputs"] if Path(out).name.startswith("report_")]
        if not reports:
            verdicts.append({"stage": name, "report": None, "status": "NO_REPORT_DECLARED",
                             "detail": "contract declares no report_*.json output for this stage"})
            continue
        path = _rooted(reports[0])
        entry: dict[str, Any] = {"stage": name, "report": str(path)}
        if not path.is_file():
            entry.update(status="NOT_RUN", detail="stage has recorded no verdict")
        else:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                entry.update(status="UNREADABLE", detail=str(error))
            else:
                entry["status"] = str(report.get("status", "NO_STATUS_FIELD")).upper()
                if isinstance(report.get("overall"), dict):
                    entry["overall"] = report["overall"]
        verdicts.append(entry)
    verdicts.extend(_audit_verdicts(by_id, units))
    return verdicts


def _audit_verdicts(by_id: dict[str, dict[str, Any]], units: list[Any]) -> list[dict[str, Any]]:
    """`audit` records a different shape: JSONL, one row per edition, dimension -> pass/fail/warn.

    Read separately rather than forced into the report_*.json reader, because pretending the shapes
    match is how a gate ends up reading a field that is never present and passing everything.
    """
    stage = by_id.get("audit")
    if stage is None:
        return [{"stage": "audit", "report": None, "status": "NOT_DECLARED",
                 "detail": "domain does not declare this quality stage"}]
    rows_path = next((_rooted(out) for out in stage["outputs"] if out.endswith(".jsonl")), None)
    entry: dict[str, Any] = {"stage": "audit", "report": str(rows_path) if rows_path else None}
    if rows_path is None or not rows_path.is_file():
        entry.update(status="NOT_RUN", detail="stage has recorded no verdict")
        return [entry]
    wanted = {unit.slug for unit in units}
    failed: dict[str, list[str]] = {}
    matched = 0
    try:
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # The file is shared across domains and keyed only by edition slug, so a row is ours
            # only when its editions match the units actually selected for this plan.
            if str(row.get("editions")) not in wanted:
                continue
            matched += 1
            bad = sorted(k for k, v in (row.get("verdicts") or {}).items() if str(v).lower() == "fail")
            if bad:
                failed[str(row["editions"])] = bad
    except (OSError, ValueError) as error:
        entry.update(status="UNREADABLE", detail=str(error))
        return [entry]
    if not matched:
        entry.update(status="NOT_RUN", detail=f"no audit row for editions {sorted(wanted)}")
    elif failed:
        entry.update(status="FAIL", detail=f"failing dimensions by edition: {failed}")
    else:
        entry.update(status="PASS", detail=f"{matched} edition row(s), no failing dimension")
    return [entry]


def _config_frozen_checks(domain: Domain, derivations: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconcile every DERIVED knob that names a training-config path against what is configured.

    A knob name containing a dot IS the config path (`train.max_steps`), so the mapping is derived
    rather than hand-maintained — a hand-written table would go stale the way every other list in
    this pipeline has. Kanto is the live case: the planner derives train.max_steps=277 while
    configs/train_pokemon_lore_kanto_v1.yaml sets 450, and nothing reconciled the two.
    """
    ref = domain.configs.get("cpt")
    path = _rooted(str(ref)) if ref else None
    if path is None or not path.is_file():
        return [{"knob": "*", "status": "NO_CONFIG",
                 "detail": f"cpt training config unresolved: {path or 'manifest value unset'}"}]
    try:
        configured = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        return [{"knob": "*", "status": "UNREADABLE", "detail": str(error)}]

    def lookup(dotted: str) -> tuple[bool, Any]:
        # Try from the config root, then under `train.` — AnnealConfig and friends are nested there
        # while the knob names them bare.
        for prefix in ((), ("train",)):
            node: Any = configured
            for part in (*prefix, *dotted.split(".")):
                if not isinstance(node, dict) or part not in node:
                    node = _UNSET
                    break
                node = node[part]
            if node is not _UNSET:
                return True, node
        return False, None

    checks = []
    for knob in derivations.get("knobs") or []:
        name = str(knob.get("knob", ""))
        if "." not in name:                       # not a config path (corpus/eval sizing knobs)
            continue
        found, value = lookup(name)
        check = {"knob": name, "derived": knob.get("value"), "configured": value,
                 "derivation": knob.get("derivation")}
        if not found:
            check["status"] = "NOT_IN_CONFIG"     # config relies on the code default — not a drift
        elif value == knob.get("value"):
            check["status"] = "AGREES"
        else:
            check["status"] = "DRIFTED"
        checks.append(check)
    return checks


_UNSET = object()


def _mechanically_launchable(domain: Domain, readiness: list[dict[str, Any]]) -> dict[str, Any]:
    """Could the commands run at all — do the inputs and configs a stage would read exist.

    This is the ONLY question the old TRAINING_READY actually answered. It is a real and useful
    rung; it is just not readiness, and naming it honestly is the point of the split.
    """
    by_stage = {entry["stage"]: entry for entry in readiness}
    stage_names = [stage.name for stage in domain.stages]
    gaps: list[str] = []
    if "tokenize" not in stage_names:
        data_stages: list[str] = []
        gaps.append("tokenize stage is not declared")
    else:
        data_stages = stage_names[:stage_names.index("tokenize") + 1]
        for name in data_stages:
            entry = by_stage[name]
            if not entry["ready"]:
                gaps.append(f"data stage {name} is not ready: {', '.join(entry['missing'])}")

    declared_train_stages = [name for name in stage_names if name in TRAIN_STAGES]
    if not declared_train_stages:
        gaps.append("no training stage is declared")
    for name in declared_train_stages:
        entry = by_stage[name]
        if not entry["ready"]:
            gaps.append(f"training stage {name} is not ready: {', '.join(entry['missing'])}")

    config_evidence = []
    for config_key, stage_name in (("cpt", "cpt"), ("sft", "sft")):
        if stage_name not in stage_names:
            continue
        ref = domain.configs.get(config_key)
        path = _rooted(str(ref)) if ref else None
        exists = bool(path and path.is_file())
        config_evidence.append({"config": config_key, "path": str(path) if path else None, "exists": exists})
        if not exists:
            gaps.append(f"training config {config_key} is missing: {path or 'manifest value unset'}")

    return {
        "state": "MECHANICALLY_LAUNCHABLE",
        "value": "PASS" if not gaps else "FAIL",
        "evidence": [{
            "data_stages_through_tokenize": data_stages,
            "training_stages": declared_train_stages,
            "train_configs": config_evidence,
            "stage_readiness": readiness,
        }],
        "gaps": list(dict.fromkeys(gaps)),
    }


def _data_gates_state(contract_stages: list[dict[str, Any]], units: list[Any]) -> dict[str, Any]:
    """Did the quality stages PASS — read from the verdict each one recorded, not from its exit code.

    pii_scan, audit_recall and audit all exit 0 whether they pass or fail; the judgement lives in
    the report. A rung that stats the declared output only proves the stage ran.
    """
    quality = _quality_verdicts(contract_stages, units)
    gaps = []
    for entry in quality:
        if entry["status"] in ("PASS", "NOT_DECLARED"):
            continue
        detail = entry.get("detail") or entry.get("overall") or ""
        gaps.append(
            f"data quality stage {entry['stage']} did not pass: {entry['status']}"
            + (f" — {detail}" if detail else "")
        )
    return {
        "state": "DATA_GATES_PASS",
        "value": "PASS" if not gaps else "FAIL",
        "evidence": quality,
        "gaps": gaps,
    }


def _train_config_frozen(domain: Domain, units: list[Any], derivations: dict[str, Any]) -> dict[str, Any]:
    """Do the configured values match the derived plan, and do the split fingerprints agree."""
    checks = _config_frozen_checks(domain, derivations)
    gaps = [
        f"{check['knob']}: derived {check['derived']!r} but config sets {check['configured']!r}"
        f" ({check.get('derivation')})"
        for check in checks if check["status"] == "DRIFTED"
    ]
    gaps += [
        f"{check['knob']}: {check['status']} — {check.get('detail')}"
        for check in checks if check["status"] in ("NO_CONFIG", "UNREADABLE")
    ]
    split = _split_status(domain, units)
    for entry in split:
        if not entry.get("exists"):
            gaps.append(f"split manifest missing for {entry.get('unit')}: {entry.get('split_manifest')}")
    if not derivations.get("ok", False):
        gaps.append(f"derivation table unavailable: {derivations.get('error')}")
    return {
        "state": "TRAIN_CONFIG_FROZEN",
        "value": "PASS" if not gaps else "FAIL",
        "evidence": [{"knob_reconciliation": checks, "split_status": split}],
        "gaps": gaps,
    }


def _stage_records_current_build(domain: Domain, readiness: list[dict[str, Any]]) -> dict[str, Any]:
    """Do the stage records on disk belong to the build the current recipe hashes to.

    Reads only; build_identity now covers the whole recipe read set, so a record written under a
    different id is evidence the recipe changed since that stage ran.
    """
    try:
        from run_stage import build_identity
        from execution_contract import CONTRACT_VERSION
        build_id, _ = build_identity(domain, CONTRACT_VERSION)
    except Exception as error:  # noqa: BLE001 — the ladder must still emit
        return {"build_id": None, "error": f"cannot compute build identity: {error}", "stale": []}
    records = C.DATA / domain.name / "builds" / build_id / "stages"
    present = {path.stem for path in records.glob("*.json")} if records.is_dir() else set()
    expected = [entry["stage"] for entry in readiness]
    return {
        "build_id": build_id,
        "records_dir": str(records),
        "present": sorted(present),
        "missing_for_current_build": [name for name in expected if name not in present],
    }


def _training_ready(rungs: list[dict[str, Any]], admitted: dict[str, Any],
                    records: dict[str, Any]) -> dict[str, Any]:
    """The conjunction. PASS requires every rung AND an operator attestation.

    Reported as a summary an operator can read at a glance, because the failure the review is
    guarding against is a human seeing one green token and authorising an unattended GPU run.
    """
    by_state = {rung["state"]: rung for rung in rungs}
    blocked_by = [name for name, rung in by_state.items() if rung["value"] != "PASS"]
    if admitted["value"] != "PASS":
        blocked_by.append("INPUTS_ADMITTED")
    summary = [
        "MECHANICALLY LAUNCHABLE" if by_state["MECHANICALLY_LAUNCHABLE"]["value"] == "PASS"
        else "NOT MECHANICALLY LAUNCHABLE",
        "DATA QUALITY PASSED" if by_state["DATA_GATES_PASS"]["value"] == "PASS" else "DATA QUALITY FAILED",
        "TRAIN CONFIG FROZEN" if by_state["TRAIN_CONFIG_FROZEN"]["value"] == "PASS" else "TRAIN CONFIG DRIFTED",
        "INPUTS ADMITTED" if admitted["value"] == "PASS" else "OPERATOR ATTESTATION REQUIRED",
        "TRAINING READY" if not blocked_by else "NOT TRAINING READY",
    ]
    gaps = [f"blocked by {name}" for name in blocked_by]
    stale = records.get("missing_for_current_build") or []
    if stale:
        gaps.append(f"stage records absent for the current build {records.get('build_id')}: {stale}")
    return {
        "state": "TRAINING_READY",
        "value": "PASS" if not gaps else "FAIL",
        "evidence": [{"summary": summary, "blocked_by": blocked_by, "stage_records": records}],
        "gaps": gaps,
    }


def _consent_gaps(effects: list[str], policy: dict[str, Any] | None) -> list[str]:
    if policy is None:
        return ["policy: not provided; consent state unknown"]
    egress = policy.get("egress", {}) if isinstance(policy.get("egress", {}), dict) else {}
    gaps = []
    for effect, key in (("egress:llm", "llm"), ("egress:upload", "upload")):
        if effect in effects and egress.get(key) is not True:
            gaps.append(f"{effect}: policy consent is not granted")
    return gaps


def _unavailable_plan(domain_name: str, manifest_state: dict[str, Any], error: Exception) -> dict[str, Any]:
    gap = f"manifest unavailable: {error}"
    unavailable = lambda state: {"state": state, "value": "FAIL", "evidence": [], "gaps": [gap]}
    return {
        "domain": domain_name,
        "editions": [],
        "ladder": [
            manifest_state,
            {
                "state": "INPUTS_ADMITTED",
                "value": "OPERATOR_ATTESTED_REQUIRED",
                "evidence": [],
                "gaps": [
                    "every source needs a run-record admission line naming basis, trust class, and PII/injection/poisoning review",
                    "operator attestation against data-governance.md is required",
                ],
            },
            unavailable("RECIPE_COMPLETE"),
            unavailable("MECHANICALLY_LAUNCHABLE"),
            unavailable("DATA_GATES_PASS"),
            unavailable("TRAIN_CONFIG_FROZEN"),
            unavailable("TRAINING_READY"),
        ],
        "plan": [],
    }


def build_plan(domain_name: str, editions: str = "", policy: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_state, _ = _manifest_state(domain_name, editions)
    try:
        domain = Domain(domain_name)
    except Exception as error:
        return _unavailable_plan(domain_name, manifest_state, error)

    unit_error = None
    try:
        units = list(iter_editions(domain_name, only=editions or None, with_raw=False))
    except Exception as error:
        units = []
        unit_error = f"cannot load registry units: {error}"

    recipe_state = _recipe_state(domain, editions)
    readiness = _stage_readiness(domain, units, editions, unit_error)
    # One contract build feeds both the data-quality rung (which reads the quality stages' declared
    # report outputs) and the ordered plan below.
    resolved_editions = ",".join(unit.slug for unit in units)
    contract_stages = build_contract(domain_name, resolved_editions)["stages"]
    admitted_state = {
        "state": "INPUTS_ADMITTED",
        "value": "OPERATOR_ATTESTED_REQUIRED",
        "evidence": [],
        "gaps": [
            "every source needs a run-record admission line naming basis, trust class, and PII/injection/poisoning review",
            "operator attestation against data-governance.md is required",
        ],
    }

    readiness_by_stage = {entry["stage"]: entry for entry in readiness}
    ordered_plan = []
    # Reuse P1's emitted stages so effects, approvals, providers, and manifest-owned GPU truth have one source.
    for stage in contract_stages:
        ready = readiness_by_stage[stage["id"]]
        planned = {
            "stage": stage["id"],
            "effects": stage["effects"],
            "approvals": stage["approvals"],
            "ready": ready["ready"],
            "missing": ready["missing"],
            "consent_gaps": _consent_gaps(stage["effects"], policy),
        }
        if "provider" in stage:
            planned["provider"] = stage["provider"]
        ordered_plan.append(planned)

    # Zero-knob derivation table (docs/cpt-overhaul/06) — every knob WITH its derivation line.
    try:
        from pipeline.derive_plan import derive_knobs
        # Helper diagnostics are useful in their own CLI, but this planner's stdout is a typed JSON
        # contract. Keep imported-helper chatter out of the machine-readable envelope.
        with contextlib.redirect_stdout(io.StringIO()):
            derivations = derive_knobs(
                domain_name,
                editions or None,
                device_only=bool((policy or {}).get("device_only")),
                domain_kind=str((policy or {}).get("domain_kind") or "facts"),
                overrides=(policy or {}).get("overrides") if isinstance((policy or {}).get("overrides"), dict) else None,
            )
    except Exception as error:  # noqa: BLE001 — plan must still emit without derivations
        derivations = {
            "ok": False,
            "error": f"derive_knobs unavailable: {error}",
            "knobs": [],
            "ladder": [],
            "plain_language": "",
        }

    # The rungs are built here, after the derivation table, because TRAIN_CONFIG_FROZEN reconciles
    # the derived plan against what the training config actually sets.
    mechanical_state = _mechanically_launchable(domain, readiness)
    data_gates_state = _data_gates_state(contract_stages, units)
    config_frozen_state = _train_config_frozen(domain, units, derivations)
    training_state = _training_ready(
        [mechanical_state, data_gates_state, config_frozen_state],
        admitted_state,
        _stage_records_current_build(domain, readiness),
    )

    result = {
        "domain": domain_name,
        "editions": [unit.slug for unit in units],
        "ladder": [manifest_state, admitted_state, recipe_state,
                   mechanical_state, data_gates_state, config_frozen_state, training_state],
        "plan": ordered_plan,
        "derivations": derivations,
    }
    assert [state["state"] for state in result["ladder"]] == LADDER_STATES
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="edc")
    parser.add_argument("--editions")
    parser.add_argument("--policy", help="path to the host machine-policy JSON")
    args = parser.parse_args()
    policy = None
    if args.policy:
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("--policy JSON must be an object")
    json.dump(build_plan(args.domain, args.editions or "", policy), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
