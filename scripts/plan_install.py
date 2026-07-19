#!/usr/bin/env python3
"""Compute a read-only install-readiness plan from a domain manifest and current disk state."""
from __future__ import annotations

import sys

# A read-only planner must not even leave import-cache files behind.
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import config as C
from execution_contract import build_contract
from pipeline.domain import Domain
from pipeline.editions import iter_editions
from run_pipeline import check_domain


LADDER_STATES = [
    "MANIFEST_VALID",
    "INPUTS_ADMITTED",
    "RECIPE_COMPLETE",
    "STAGE_READY",
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
    status = []
    for unit in units:
        split_path = unit.dir / "splits" / "split_v1.json"
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


def _training_state(domain: Domain, readiness: list[dict[str, Any]], units: list[Any]) -> dict[str, Any]:
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

    evidence = [{
        "data_stages_through_tokenize": data_stages,
        "training_stages": declared_train_stages,
        "train_configs": config_evidence,
        "split_status": _split_status(domain, units),
    }]
    return {
        "state": "TRAINING_READY",
        "value": "PASS" if not gaps else "FAIL",
        "evidence": evidence,
        "gaps": list(dict.fromkeys(gaps)),
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
            unavailable("STAGE_READY"),
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
    stage_state = {
        "state": "STAGE_READY",
        "value": "PASS" if readiness and all(entry["ready"] for entry in readiness) else "FAIL",
        "evidence": readiness,
        "gaps": [
            f"{entry['stage']}: {missing}"
            for entry in readiness
            for missing in entry["missing"]
        ],
    }
    training_state = _training_state(domain, readiness, units)
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
    resolved_editions = ",".join(unit.slug for unit in units)
    for stage in build_contract(domain_name, resolved_editions)["stages"]:
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

    result = {
        "domain": domain_name,
        "editions": [unit.slug for unit in units],
        "ladder": [manifest_state, admitted_state, recipe_state, stage_state, training_state],
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
