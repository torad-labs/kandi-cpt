#!/usr/bin/env python3
"""Generic forced-choice CPT comparison used by the canonical eval entrypoint.

The operator-facing command is scripts/eval.py. This module owns the generic-domain
base-vs-tuned scorer and its machine-readable receipt; it is imported by eval.py and
unit-tested without loading a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config as C
from eval_stats import DEFAULT_MIN_N, stamp_proportion
from pipeline.editions import iter_editions


@dataclass(frozen=True)
class ProbeCorpus:
    domain: str
    editions: list[str]
    probes: list[dict[str, Any]]
    probe_paths: list[Path]
    probe_hashes: list[str]
    probe_sha256: str
    split_paths: list[Path]
    split_hashes: list[str]
    split_sha256: str
    split_version: str
    token_stats_path: Path
    token_stats_sha256: str
    clean_tokens: int
    measurement_class: str
    chance: float


def _rooted(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(C.ROOT) / candidate


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(C.ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_display(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _token_stats_path(tokenized: str) -> Path:
    tokenized_path = _rooted(tokenized)
    name = tokenized_path.name.removesuffix(".arrow")
    return tokenized_path.with_name(f"{name}_stats.json")


def load_probe_corpus(domain: str, editions: str, tokenized: str) -> ProbeCorpus:
    requested = [value.strip() for value in editions.split(",") if value.strip()]
    if not requested:
        raise ValueError("generic probe eval requires at least one scoped edition")

    units = list(iter_editions(domain, only=editions, with_raw=False))
    found = [unit.slug for unit in units]
    if set(found) != set(requested):
        raise ValueError(f"probe edition scope mismatch: requested={requested}, found={found}")

    probes: list[dict[str, Any]] = []
    probe_paths: list[Path] = []
    split_paths: list[Path] = []
    split_versions: set[str] = set()
    choice_counts: set[int] = set()
    seen_source_ids: set[str] = set()
    for unit in sorted(units, key=lambda value: value.slug):
        probe_path = unit.dir / "probes.jsonl"
        if not probe_path.is_file():
            raise ValueError(f"probe file missing: {probe_path}")
        # Bind to the split version the probes declare (v2 series baseline), falling back to
        # pipeline.split.load_split preference (v2 > v1). Never hardcode split_v1 only — that
        # silently rejects a valid split_v2 probe set (Phase F measurement debt).
        from pipeline.split import load_split, split_path as _split_path
        probe_lines = [ln for ln in probe_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        declared_versions = {
            str(json.loads(ln).get("split_version") or "")
            for ln in probe_lines
        }
        declared_versions.discard("")
        if len(declared_versions) > 1:
            raise ValueError(f"probes declare mixed split_version values at {probe_path}: {sorted(declared_versions)}")
        preferred = next(iter(declared_versions)) if declared_versions else None
        split = load_split(unit.dir, version=preferred)
        if split is None:
            raise ValueError(f"split manifest missing under {unit.dir / 'splits'}")
        split_version = str(split.get("version") or "")
        split_path = _split_path(unit.dir, split_version)
        holdout_ids = {str(value) for value in split.get("holdout_ids") or []}
        train_ids = {str(value) for value in split.get("train_ids") or []}
        if not split_version or not holdout_ids:
            raise ValueError(f"split manifest has no versioned holdout: {split_path}")
        if train_ids & holdout_ids:
            raise ValueError(f"split manifest train/holdout overlap: {split_path}")
        split_versions.add(split_version)
        split_paths.append(split_path)
        probe_paths.append(probe_path)
        for line_number, line in enumerate(probe_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            probe = json.loads(line)
            choices = probe.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) < 2
                or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
                or len(set(choices)) != len(choices)
            ):
                raise ValueError(f"invalid or duplicate choices at {probe_path}:{line_number}")
            if probe.get("answer") not in choices:
                raise ValueError(f"answer is absent from choices at {probe_path}:{line_number}")
            if probe.get("split") != "held_out" or probe.get("split_version") != split_version:
                raise ValueError(f"probe/split mismatch at {probe_path}:{line_number}")
            source_id = str(probe.get("source_triple_id") or "")
            if source_id not in holdout_ids:
                raise ValueError(f"probe source is absent from split holdout_ids at {probe_path}:{line_number}")
            if source_id in seen_source_ids:
                raise ValueError(f"duplicate probe source_triple_id at {probe_path}:{line_number}")
            seen_source_ids.add(source_id)
            choice_counts.add(len(choices))
            probes.append(probe)

    if not probes:
        raise ValueError("no probes found — run the probes stage first")
    if len(split_versions) != 1:
        raise ValueError(f"probe units use different split versions: {sorted(split_versions)}")
    if len(choice_counts) != 1:
        raise ValueError(f"probe items use different choice counts: {sorted(choice_counts)}")

    token_stats_path = _token_stats_path(tokenized)
    if not token_stats_path.is_file():
        raise ValueError(f"tokenization stats missing: {token_stats_path}")
    token_stats = json.loads(token_stats_path.read_text(encoding="utf-8"))
    clean_tokens = int((token_stats.get("tokens_by_tag") or {}).get("clean") or 0)
    if clean_tokens <= 0:
        raise ValueError(
            "held-out probes with zero clean tokens need carved-holdout decontamination evidence; "
            "the generic comparator currently supports augmentation_holdout only"
        )

    return ProbeCorpus(
        domain=domain,
        editions=found,
        probes=probes,
        probe_paths=probe_paths,
        probe_hashes=[_sha256(path) for path in probe_paths],
        probe_sha256=_combined_hash(probe_paths),
        split_paths=split_paths,
        split_hashes=[_sha256(path) for path in split_paths],
        split_sha256=_combined_hash(split_paths),
        split_version=next(iter(split_versions)),
        token_stats_path=token_stats_path,
        token_stats_sha256=_sha256(token_stats_path),
        clean_tokens=clean_tokens,
        measurement_class="augmentation_holdout",
        chance=1.0 / next(iter(choice_counts)),
    )


def load_model(path: str, device: str):
    """Load base vanilla or an engine int8-master checkpoint through its dequantizing loader."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if (_rooted(path) / "quant_keys.json").exists():
        from pipeline.model_load import load_causal_ckpt_bf16
        model = load_causal_ckpt_bf16(path, device)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map=device)
    model.eval()
    return tokenizer, model


def choice_logprob(tokenizer, model, device: str, question: str, choice: str) -> float:
    import torch

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    choice_ids = tokenizer(
        " " + choice,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    if choice_ids.shape[1] <= 0:
        return float("-inf")
    full_ids = torch.cat((prompt_ids, choice_ids), dim=1)
    n_choice = choice_ids.shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits[0, :-1]
        targets = full_ids[0, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        picked = logprobs[torch.arange(len(targets)), targets][-n_choice:]
    return picked.mean().item()


def evaluate_model(tokenizer, model, device: str, probes: list[dict[str, Any]]) -> list[bool]:
    hits: list[bool] = []
    for index, probe in enumerate(probes):
        scores = [choice_logprob(tokenizer, model, device, probe["question"], choice)
                  for choice in probe["choices"]]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"non-finite choice score at probe {index}")
        predicted = probe["choices"][max(range(len(scores)), key=scores.__getitem__)]
        hits.append(predicted == probe["answer"])
    return hits


def summarize_hits(
    probes: list[dict[str, Any]],
    hits: list[bool],
    *,
    min_n: int = DEFAULT_MIN_N,
    bar: float | None = None,
    estimand: str | None = None,
) -> dict[str, Any]:
    """Per-slice forced-choice rates with Wilson intervals and mechanical n-floor labels."""
    if len(probes) != len(hits):
        raise ValueError(f"probe/hit length mismatch: {len(probes)} != {len(hits)}")
    per_predicate: dict[str, list[bool]] = {}
    for probe, hit in zip(probes, hits, strict=True):
        per_predicate.setdefault(str(probe.get("predicate") or "unknown"), []).append(hit)
    total_hits = sum(hits)
    n = len(hits)
    overall = stamp_proportion(
        total_hits,
        n,
        min_n=min_n,
        bar=bar,
        estimand=estimand,
        slice_name="all",
    )
    return {
        "acc": overall["acc"],
        "hits": total_hits,
        "n": n,
        "wilson": overall["wilson"],
        "rough_bound_half_over_sqrt_n": overall["rough_bound_half_over_sqrt_n"],
        "n_status": overall["n_status"],
        "min_n": min_n,
        **({"bar": bar, "bar_status": overall.get("bar_status"),
            "clears_bar": overall.get("clears_bar"),
            "within_noise_of_bar": overall.get("within_noise_of_bar")} if bar is not None else {}),
        "per_predicate": {
            predicate: stamp_proportion(
                sum(values),
                len(values),
                min_n=min_n,
                bar=bar,
                estimand=estimand,
                slice_name=predicate,
            )
            for predicate, values in sorted(per_predicate.items())
        },
    }


def build_report(
    *,
    run_name: str,
    config_path: Path,
    base_path: str,
    tuned_path: str,
    corpus: ProbeCorpus,
    base_hits: list[bool],
    tuned_hits: list[bool],
    min_n: int,
    min_delta_acc: float,
    config_fingerprint: str,
    tokenized: str,
) -> dict[str, Any]:
    # Chance is the floor a random forced-choice picker hits; stamp both sides so the
    # report carries Wilson + insufficient-n without hand arithmetic.
    base = summarize_hits(
        corpus.probes,
        base_hits,
        min_n=min_n,
        bar=corpus.chance,
        estimand=corpus.measurement_class,
    )
    tuned = summarize_hits(
        corpus.probes,
        tuned_hits,
        min_n=min_n,
        bar=corpus.chance,
        estimand=corpus.measurement_class,
    )
    rough_bound = tuned["rough_bound_half_over_sqrt_n"] if tuned["n"] else 0.0
    if rough_bound is None:
        rough_bound = 0.0
    delta_acc = tuned["acc"] - base["acc"]
    for result in (base, tuned):
        result["probe_sha256"] = corpus.probe_sha256

    checks = [
        {"check": "base and tuned used the identical probe set", "pass": base["probe_sha256"] == tuned["probe_sha256"]},
        {
            "check": f"probe denominator is at least {min_n}",
            "pass": base["n"] >= min_n and base["n_status"] != "insufficient-n",
        },
        {
            "check": "tuned accuracy clears chance by more than the rough bound",
            "pass": tuned["acc"] - corpus.chance > rough_bound and tuned.get("bar_status") != "within-noise",
        },
        {
            "check": f"accuracy delta clears {min_delta_acc:.4f} by more than the rough bound",
            "pass": delta_acc - min_delta_acc > rough_bound,
        },
    ]
    failed = [check["check"] for check in checks if not check["pass"]]

    config_path = config_path.resolve()
    tuned_root = _rooted(tuned_path)
    model_path = tuned_root / "model.safetensors"
    run_contract_path = tuned_root / "run_contract.json"
    if not model_path.is_file() or not run_contract_path.is_file():
        raise ValueError(f"tuned checkpoint is incomplete: {tuned_root}")
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    if run_contract.get("run_name") != run_name:
        raise ValueError(
            f"checkpoint run_name mismatch: {run_contract.get('run_name')!r} != {run_name!r}"
        )
    if run_contract.get("fingerprint") != config_fingerprint:
        raise ValueError("checkpoint fingerprint does not match the evaluated config")
    if run_contract.get("tokenized") != tokenized:
        raise ValueError(
            f"checkpoint tokenized shard mismatch: {run_contract.get('tokenized')!r} != {tokenized!r}"
        )

    base_root = _rooted(base_path)
    base_model_path = base_root / "model.safetensors"
    if not base_model_path.is_file():
        raise ValueError(f"base model weights are missing: {base_model_path}")

    return {
        "status": "PASS" if not failed else "FAIL",
        "pilot_class": True,
        "provisional": True,
        "run_name": run_name,
        "config_fingerprint": config_fingerprint,
        "measurement_class": corpus.measurement_class,
        "domain": corpus.domain,
        "editions": corpus.editions,
        "split_version": corpus.split_version,
        "chance": corpus.chance,
        "rough_bound_half_over_sqrt_n": rough_bound,
        "delta_acc": delta_acc,
        "axis": "forced_choice",
        "estimand": corpus.measurement_class,
        "thresholds": {"min_n": min_n, "min_delta_acc": min_delta_acc},
        "checks": checks,
        "failed_checks": failed,
        # Dual-axis column the generative merger fills in; present here so a forced-choice-only
        # report still names the axis shape install-strength will carry.
        "install_strength": {
            "forced_choice": {
                "tuned_acc": tuned["acc"],
                "base_acc": base["acc"],
                "n": tuned["n"],
                "wilson": tuned["wilson"],
                "n_status": tuned["n_status"],
                "delta_acc": delta_acc,
                "estimand": corpus.measurement_class,
            },
            "generative": None,
        },
        "inputs": {
            "config": _display(config_path),
            "config_sha256": _sha256(config_path),
            "probe_paths": [_display(path) for path in corpus.probe_paths],
            "probe_hashes": corpus.probe_hashes,
            "probe_sha256": corpus.probe_sha256,
            "split_paths": [_display(path) for path in corpus.split_paths],
            "split_hashes": corpus.split_hashes,
            "split_sha256": corpus.split_sha256,
            "token_stats_path": _display(corpus.token_stats_path),
            "token_stats_sha256": corpus.token_stats_sha256,
            "clean_tokens": corpus.clean_tokens,
        },
        "base": {**base, "path": base_path, "model_sha256": _sha256(base_model_path)},
        "tuned": {
            **tuned,
            "path": tuned_path,
            "model_sha256": _sha256(model_path),
            "run_contract_sha256": _sha256(run_contract_path),
            "run_contract": run_contract,
        },
    }


def run_probe_comparison(config, config_path: str | Path, checkpoint: str) -> tuple[Path, dict[str, Any]]:
    import torch

    eval_config = config.eval
    corpus = load_probe_corpus(eval_config.probe_domain, eval_config.probe_editions, config.data.tokenized)
    scored: dict[str, list[bool]] = {}
    for name, path in (("base", config.model.model_path), ("tuned", checkpoint)):
        tokenizer, model = load_model(path, eval_config.device)
        scored[name] = evaluate_model(tokenizer, model, eval_config.device, corpus.probes)
        del model
        torch.cuda.empty_cache()

    report = build_report(
        run_name=config.run.run_name,
        config_path=Path(config_path),
        base_path=config.model.model_path,
        tuned_path=checkpoint,
        corpus=corpus,
        base_hits=scored["base"],
        tuned_hits=scored["tuned"],
        min_n=eval_config.probe_min_n,
        min_delta_acc=eval_config.probe_min_delta_acc,
        config_fingerprint=config.fingerprint(),
        tokenized=config.data.tokenized,
    )
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.run.run_name).strip("._")
    if not safe_run_name:
        raise ValueError("run.run_name cannot produce an empty eval report name")
    report_dir = _rooted(eval_config.probe_report_root).resolve()
    engine_root = Path(C.ROOT).resolve()
    if not report_dir.is_relative_to(engine_root):
        raise ValueError(f"eval.probe_report_root escapes the engine repo: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{safe_run_name}_base_vs_tuned.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({
        "status": report["status"],
        "report": _display(report_path),
        "base_acc": report["base"]["acc"],
        "tuned_acc": report["tuned"]["acc"],
        "delta_acc": report["delta_acc"],
        "n": report["base"]["n"],
        "measurement_class": report["measurement_class"],
    }))
    return report_path, report
