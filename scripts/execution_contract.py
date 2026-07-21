#!/usr/bin/env python3
"""Emit the read-only ExecutionContract advertised by a domain manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

import config as C
from pipeline.domain import Domain
from pipeline.editions import iter_editions, parse_only
from run_pipeline import editions_on_disk


CONTRACT_VERSION = "1"
RECIPE_VERSION = "1"

# GPU is listed here to document the intended effects for known stages, but the
# emitted GPU effect is always derived from the manifest's stage.gpu flag.  The
# manifest is the execution truth (notably, audit may run data_audit --no-gpu).
STAGE_EFFECTS = {
    "scrape": ["network", "write"],
    "extract": ["write"],
    "pii_scan": ["write"],
    # Teacher pass is provider-bound; egress:llm is also inferred from OPENROUTER_COMMAND_MARKERS.
    "score_educational": ["egress:llm", "write"],
    "style_fingerprint": ["write"],
    "structured": ["egress:llm", "write"],
    "resolve": ["gpu", "write"],
    "curate": ["gpu", "write"],
    "split": ["write"],
    "contrast": ["write"],
    "meta": ["write"],
    "hydrate": ["write"],
    "synth": ["egress:llm", "write"],
    "dedup_surfaces": ["write"],
    "probes": ["write"],
    "audit": ["gpu", "write"],
    "audit_recall": ["egress:llm", "write"],
    "tokenize": ["write"],
    "cpt": ["gpu", "write", "spend"],
    "build_sft": ["gpu", "write", "spend"],
    "sft": ["gpu", "write", "spend"],
}

STAGE_OUTPUTS = {
    "scrape": ["data/{domain}/**/00_raw/**"],
    "extract": ["data/{domain}/**/10_clean/**"],
    "pii_scan": [
        "data/_build/pii/quarantine_{domain}.json",
        "data/_build/pii/report_{domain}.json",
    ],
    "score_educational": [
        "data/_build/educational/scores_{domain}.json",
        "data/_build/educational/report_{domain}.json",
    ],
    "style_fingerprint": [
        "data/_build/style/{domain}/fingerprint.json",
        "data/_build/style/{domain}/style_card.md",
    ],
    "resolve": [
        "data/_build/resolve/{domain}/**/summary.json",
        "data/{domain}/**/15_structured/entities.resolved.jsonl",
        "data/{domain}/**/15_structured/triples.resolved.jsonl",
    ],
    "curate": [
        "data/_build/curation/keep_list_{domain}.json",
        "data/_build/curation/curation_report_{domain}.json",
        "data/_build/curation/curation_meta_{domain}.json",
    ],
    "structured": ["data/{domain}/**/15_structured/entities.jsonl", "data/{domain}/**/15_structured/triples.jsonl"],
    "split": ["data/{domain}/**/splits/split_v1.json"],
    "contrast": ["data/{domain}/**/25_hydrated/relations_contrast.jsonl"],
    "meta": ["data/{domain}/**/festival_meta.json"],
    "hydrate": ["data/{domain}/**/25_hydrated/relations.jsonl"],
    "synth": ["data/{domain}/**/25_hydrated/relations_llm.jsonl"],
    "dedup_surfaces": [
        "data/_build/dedup/{domain}/**/exclude_keys.json",
        "data/_build/dedup/{domain}/summary.json",
    ],
    "probes": ["data/{domain}/**/probes.jsonl"],
    "audit": ["data/_build/dissect/data_audit_results.jsonl"],
    "audit_recall": ["data/_build/recall/report_{domain}.json"],
    "tokenize": ["data/_build/30_tokenized/{corpus}.arrow", "data/_build/30_tokenized/{corpus}_stats.json"],
    "cpt": ["{cpt_save_dir}/**"],
    "build_sft": ["data/_build/30_tokenized/{sft_out}.arrow", "data/_build/30_tokenized/{sft_out}_stats.json"],
    "sft": ["{sft_save_dir}/**"],
}

OPENROUTER_COMMAND_MARKERS = (
    "--llm-rescue",
    "extract_structured.py",
    "generic_synth.py",
    "rehydrate_vc1.sh",
    "score_educational.py",
    "eval_generative.py",
    "audit_recall.py",
)

APPROVAL_EFFECTS = ["gpu", "egress:llm", "egress:upload", "spend", "delete"]

REQUIRED_PLAYBOOKS = {
    "cpt": "cpt-codebook.md",
    "pii_scan": "cpt/03a-pii-scan.md",
    "score_educational": "cpt/03b-score-educational.md",
    "resolve": "cpt/04b-resolve.md",
    "curate": "cpt/03-curate.md",
    "hydrate": "cpt/05-hydrate.md",
    "dedup_surfaces": "cpt/05b-dedup-surfaces.md",
    "audit_recall": "cpt/04c-audit-recall.md",
    "style_fingerprint": "cpt/09-style.md",
    "build_sft": "cpt/07-train.md",
    "sft": "cpt/07-train.md",
    "plan_install": "cpt/08-zero-knob.md",
}


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=C.ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=C.ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _uses_openrouter(command: list[str]) -> bool:
    return any(marker in argument for marker in OPENROUTER_COMMAND_MARKERS for argument in command)


def _effects(stage_name: str, command: list[str], gpu: bool) -> list[str]:
    configured = STAGE_EFFECTS.get(stage_name)
    if configured is None:
        raise ValueError(f"unknown execution-contract stage {stage_name!r}; declare its effects before execution")
    effects = [effect for effect in configured if effect not in {"gpu", "egress:llm"}]
    if _uses_openrouter(command):
        effects.insert(0, "egress:llm")
    if gpu:
        effects.insert(0, "gpu")
    return effects


def _rooted(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else C.ROOT / candidate


def _config_value(domain: Domain, key: str, *parts: str) -> str:
    ref = domain.configs.get(key)
    if not ref:
        return ""
    path = _rooted(str(ref))
    if not path.is_file():
        return ""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value: Any = loaded
    for part in parts:
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return str(value) if value else ""


def _command_option(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _outputs(
    domain: Domain,
    stage_name: str,
    editions: list[str],
    command: list[str],
) -> list[str]:
    outputs = STAGE_OUTPUTS.get(stage_name, [])
    if stage_name == "split":
        version = _command_option(command, "--version") or "v1"
        outputs = [f"data/{{domain}}/**/splits/split_{version}.json"]
    if stage_name == "curate" and domain.name == "edc":
        outputs = [
            "data/_build/curation/keep_list.json",
            "data/_build/curation/curation_report.json",
            "data/_build/curation/curation_meta.json",
        ]
    elif stage_name == "pii_scan" and domain.name == "edc":
        # Same paths for all domains (domain-suffixed); no EDC legacy rename.
        pass
    elif stage_name == "score_educational" and domain.name == "edc":
        pass
    substitutions = {
        "domain": domain.name,
        "corpus": domain.vars.get("corpus", "__contract_missing_corpus__"),
        "sft_out": domain.vars.get("sft_out", "__contract_missing_sft_out__"),
        "cpt_save_dir": _config_value(domain, "cpt", "run", "save_dir") or "__contract_missing_cpt_save_dir__",
        "sft_save_dir": _config_value(domain, "sft", "run", "save_dir") or "__contract_missing_sft_save_dir__",
    }
    rendered = [output.format(**substitutions) for output in outputs]
    marker = f"data/{domain.name}/**/"
    scoped = []
    for output in rendered:
        if editions and output.startswith(marker):
            scoped.extend(f"data/{domain.name}/{edition}/{output.removeprefix(marker)}" for edition in editions)
        else:
            scoped.append(output)
    return scoped


def _resolved_editions(domain_name: str, editions: str | None) -> str:
    if editions is None:
        return ",".join(editions_on_disk(domain_name))
    if not editions:
        return ""
    requested = parse_only(editions) or set()
    units = list(iter_editions(domain_name, only=editions, with_raw=False))
    found = {unit.slug for unit in units}
    unknown = sorted(requested - found)
    if unknown:
        raise ValueError(f"unknown unit(s) for {domain_name}: {', '.join(unknown)}")
    return ",".join(unit.slug for unit in units)


def build_contract(domain_name: str, editions: str | None = None) -> dict[str, Any]:
    domain = Domain(domain_name)
    resolved_editions = _resolved_editions(domain.name, editions)
    scoped_editions = [edition for edition in resolved_editions.split(",") if edition]
    commit, dirty = _git_identity()
    stages = []
    for stage in domain.stages:
        command = domain.resolve(stage.cmd, domain=domain.name, editions=resolved_editions)
        effects = _effects(stage.name, command, stage.gpu)
        outputs = _outputs(domain, stage.name, scoped_editions, command)
        emitted = {
            "id": stage.name,
            "editions": scoped_editions,
            "cmd_template": stage.cmd,
            "cmd": command,
            "gpu": stage.gpu,
            "effects": effects,
            "approvals": [effect for effect in APPROVAL_EFFECTS if effect in effects],
            "outputs": outputs,
            "postconditions": [
                {"check": f"output exists: {output}", "kind": "output_exists", "path": output}
                for output in outputs
            ],
        }
        if "egress:llm" in effects:
            emitted["provider"] = "openrouter"
        payload = json.dumps(emitted, sort_keys=True, separators=(",", ":")).encode("utf-8")
        emitted["stage_digest"] = hashlib.sha256(payload).hexdigest()
        stages.append(emitted)

    return {
        "contract_version": CONTRACT_VERSION,
        "engine": {
            "name": "kandi-cpt",
            "commit": commit,
            "dirty": dirty,
        },
        "recipe_version": RECIPE_VERSION,
        "capabilities": {"gemma4": {"cpt": "eager-pilot"}},
        "editions": [edition for edition in resolved_editions.split(",") if edition],
        "stages": stages,
        "required_playbooks": REQUIRED_PLAYBOOKS,
        "deviations": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="edc")
    parser.add_argument("--editions", help="comma-separated unit scope; default = all units with pipeline input")
    args = parser.parse_args()
    json.dump(build_contract(args.domain, args.editions), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
