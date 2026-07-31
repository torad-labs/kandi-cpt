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
    # Teacher pass is provider-bound; egress:llm is also inferred from LLM_COMMAND_MARKERS.
    # Provider policy: Google AI Studio plan A, OpenRouter plan B (pipeline.llm_provider).
    "score_educational": ["egress:llm", "write"],
    "style_fingerprint": ["write"],
    "approve_style": ["write"],
    "build_style": ["write"],
    "eval_style": ["gpu", "write"],
    "structured": ["egress:llm", "write"],
    "resolve": ["gpu", "write"],
    "curate": ["gpu", "write"],
    "split": ["write"],
    "contrast": ["write"],
    "reasoning": ["egress:llm", "write"],
    "meta": ["write"],
    "hydrate": ["write"],
    "synth": ["egress:llm", "write"],
    "dedup_surfaces": ["write"],
    "probes": ["write"],
    "probe_quarantine": ["write"],
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
    "approve_style": ["data/_build/style/{domain}/approval.json"],
    "build_style": [
        "data/_build/30_tokenized/{style_out}.arrow",
        "data/_build/30_tokenized/{style_out}_stats.json",
        "data/_build/style/{domain}/expansion_judgments.jsonl",
    ],
    "eval_style": [
        "data/_build/style/{domain}/eval_style.json",
        "data/_build/style/{domain}/voice_reveal.json",
        "data/_build/style/{domain}/certificate.json",
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
    "reasoning": ["data/{domain}/**/25_hydrated/relations_reasoning.jsonl"],
    "meta": ["data/{domain}/**/festival_meta.json"],
    "hydrate": ["data/{domain}/**/25_hydrated/relations.jsonl"],
    "synth": ["data/{domain}/**/25_hydrated/relations_llm.jsonl"],
    "dedup_surfaces": [
        "data/_build/dedup/{domain}/**/exclude_keys.json",
        "data/_build/dedup/{domain}/summary.json",
    ],
    "probes": ["data/{domain}/**/probes.jsonl"],
    "probe_quarantine": [
        "data/_build/probe_quarantine/{domain}/**.json",
        "data/_build/probe_quarantine/report_{domain}.json",
    ],
    "audit": ["data/_build/dissect/data_audit_results.jsonl"],
    "audit_recall": ["data/_build/recall/report_{domain}.json"],
    "tokenize": ["data/_build/30_tokenized/{corpus}.arrow", "data/_build/30_tokenized/{corpus}_stats.json"],
    "cpt": ["{cpt_save_dir}/**"],
    "build_sft": ["data/_build/30_tokenized/{sft_out}.arrow", "data/_build/30_tokenized/{sft_out}_stats.json"],
    "sft": ["{sft_save_dir}/**"],
}

# Historical name kept as an alias — markers identify any provider-bound stage.
OPENROUTER_COMMAND_MARKERS = (
    "--llm-rescue",
    "extract_structured.py",
    "generic_reasoning.py",
    "generic_synth.py",
    "rehydrate_vc1.sh",
    "score_educational.py",
    "eval_generative.py",
    "audit_recall.py",
)
LLM_COMMAND_MARKERS = OPENROUTER_COMMAND_MARKERS
# Declared provider on egress:llm stages. Runtime resolution still falls back to
# OpenRouter when Studio is unavailable; the contract advertises the preferred plan.
DEFAULT_LLM_PROVIDER = "googleai"
DEFAULT_LLM_PROVIDER_FALLBACK = "openrouter"

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
    "approve_style": "cpt/09-style.md",
    "build_style": "cpt/09-style.md",
    "eval_style": "cpt/09-style.md",
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


def _uses_llm(command: list[str]) -> bool:
    return any(marker in argument for marker in LLM_COMMAND_MARKERS for argument in command)


# Back-compat alias for anything still importing the old name.
_uses_openrouter = _uses_llm


def _effects(stage_name: str, command: list[str], gpu: bool) -> list[str]:
    configured = STAGE_EFFECTS.get(stage_name)
    if configured is None:
        raise ValueError(f"unknown execution-contract stage {stage_name!r}; declare its effects before execution")
    effects = [effect for effect in configured if effect not in {"gpu", "egress:llm"}]
    if _uses_llm(command):
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
        "style_out": domain.vars.get("style_out", "__contract_missing_style_out__"),
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


# Stages that RECORD a verdict, and how to read it. Existence of the report only proves the stage
# ran; each of these exits 0 whether it passed or failed, so without a semantic postcondition a
# FAILing audit still mints a PASS stage record. Keyed by stage, matched against that stage's OWN
# declared outputs, so a relocated report cannot silently detach the check from the artifact.
_SEMANTIC_POSTCONDITIONS: dict[str, dict[str, Any]] = {
    "audit_recall": {"suffix": ".json", "prefix": "report_",
                     "kind": "report_verdict", "field": "status", "expect": ["PASS"]},
    "pii_scan": {"suffix": ".json", "prefix": "report_",
                 "kind": "report_verdict", "field": "status", "expect": ["PASS"]},
    "audit": {"suffix": ".jsonl", "prefix": "",
              "kind": "jsonl_no_fail_verdict", "field": "verdicts"},
    "tokenize": {"suffix": "_stats.json", "prefix": "",
                 "kind": "tokenize_stats_sound", "field": "num_chunks"},
    "split": {"suffix": ".json", "prefix": "split_", "kind": "split_integrity", "field": None},
    # The fact layer's own shape. Every diversity and balance instrument in this engine reads the
    # HYDRATED layer — dominance_max_frac caps surface families, effective_forms() takes texts —
    # so a graph that is functionally one relation arrives at training wearing the costume of a
    # rich one and no gate downstream can tell. Measured on EDC: in_lineup 59.60% + plays_at
    # 35.93% = 95.53% of 19,229 facts, against pokemon-lore 28.54% and millbrook-hollow 30.00%.
    # Checked HERE because this is the last stage where the fix is cheap: after hydrate the
    # concentration is baked into ~43 surfaces per fact and rebalancing provably cannot undo it
    # (phase_h7b, phase_r1). Only triples.jsonl — entities.jsonl shares the suffix.
    "structured": {"suffix": "triples.jsonl", "prefix": "",
                   "kind": "fact_concentration_sound", "field": None},
    "probe_quarantine": {"suffix": ".json", "prefix": "report_",
                         "kind": "probe_quarantine_sound", "field": None},
    # The cpt output is a directory glob; the validator wants the directory itself.
    "cpt": {"suffix": "/**", "prefix": "", "kind": "checkpoint_integrity", "field": None,
            "strip_suffix": True},
    "sft": {"suffix": "/**", "prefix": "", "kind": "checkpoint_integrity", "field": None,
            "strip_suffix": True},
}


def _semantic_postconditions(stage_name: str, outputs: list[str],
                             scoped_editions: list[str], domain: Domain) -> list[dict[str, Any]]:
    rule = _SEMANTIC_POSTCONDITIONS.get(stage_name)
    if rule is None:
        return []
    emitted = []
    for output in outputs:
        if rule.get("strip_suffix"):
            if not output.endswith(rule["suffix"]):
                continue
            path = output[: -len(rule["suffix"])].rstrip("/")
        else:
            name = output.rsplit("/", 1)[-1]
            if not (name.endswith(rule["suffix"]) and name.startswith(rule["prefix"])):
                continue
            path = output
        check_text = f"{stage_name} produced a valid artifact at {path}"
        if rule["kind"] == "fact_concentration_sound":
            # A boolean tells the driver something is wrong; it cannot tell it what to DO. This
            # stage's failure has a specific, counter-intuitive remedy — breadth of relation TYPES,
            # never more rows — and the doctrine lives in the playbook the driver can read. Route
            # there in the check text, because the check text is what a failing record surfaces.
            check_text = (
                f"{stage_name}: the top-2 predicate share of {path} stays within the domain's "
                f"declared relations.dominance_max_frac. A graph that collapses onto a couple of "
                f"predicates installs those predicates no matter how much text is hydrated from "
                f"it, and every diversity gate downstream reads the hydrated layer, so nothing "
                f"else can see this. If it fails, the lever is extraction BREADTH — more relation "
                f"types — and never more rows: read cpt/04-structured.md before acting, and "
                f"cpt/06-improve.md for why rebalancing after hydrate cannot undo it."
            )
        condition = {"check": check_text, "kind": rule["kind"], "path": path}
        if rule.get("field"):
            condition["field"] = rule["field"]
        if rule["kind"] == "fact_concentration_sound":
            # The cap travels WITH the check, same as checkpoint_integrity's recipe: the domain
            # already declares relations.dominance_max_frac, which was only ever applied to
            # hydrated surface families. An undeclared cap is emitted as None and fails CLOSED
            # downstream — "I could not tell" is never "it passed".
            relations = getattr(domain, "spec", {}).get("relations") or {}
            condition["expect_max_top_share"] = relations.get("dominance_max_frac")
            condition["top_n_predicates"] = 2
        if rule["kind"] == "checkpoint_integrity":
            # The approved recipe travels WITH the check: the validator compares the trainer's own
            # run_contract against what the config declared, so "valid artifact" and "the run we
            # approved" stop being the same question.
            for key, path in (("expect_precision", ("train", "precision")),
                              ("expect_int8_master", ("train", "int8_master"))):
                value = _config_value(domain, "cpt" if stage_name == "cpt" else "sft", *path)
                if value:
                    condition[key] = value
        if "expect" in rule:
            condition["expect"] = rule["expect"]
        if rule["kind"] == "jsonl_no_fail_verdict":
            # The audit results file is shared across domains and keyed only by edition slug,
            # so the check must be scoped to the editions this stage actually ran.
            condition["editions"] = scoped_editions
        emitted.append(condition)
    return emitted


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
            ] + _semantic_postconditions(stage.name, outputs, scoped_editions, domain),
        }
        if "egress:llm" in effects:
            # Plan A advertised; runtime falls back to openrouter only if Studio is absent.
            emitted["provider"] = DEFAULT_LLM_PROVIDER
            emitted["provider_fallback"] = DEFAULT_LLM_PROVIDER_FALLBACK
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
