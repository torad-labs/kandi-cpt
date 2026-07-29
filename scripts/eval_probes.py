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
from dataclasses import dataclass, field
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
    # Estimand-side decontamination (pipeline/probe_quarantine.py). Defaults keep every existing
    # construction site valid: no quarantine record on disk == score every probe, as before.
    quarantine_paths: list[Path] = field(default_factory=list)
    quarantine_hashes: list[str] = field(default_factory=list)
    quarantined_sources: dict[str, str] = field(default_factory=dict)   # source_triple_id → reason
    quarantine_instrument_blind: bool = False


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
    measurement_classes: set[str] = set()
    choice_counts: set[int] = set()
    seen_source_ids: set[str] = set()
    seen_surface_ids: set[str] = set()
    quarantine_paths: list[Path] = []
    quarantined_sources: dict[str, str] = {}
    quarantine_instrument_blind = False
    for unit in sorted(units, key=lambda value: value.slug):
        probe_path = unit.dir / "probes.jsonl"
        if not probe_path.is_file():
            raise ValueError(f"probe file missing: {probe_path}")
        # Bind to the split version the probes declare (v2 series baseline), falling back to
        # pipeline.split.load_split preference (v2 > v1). Never hardcode split_v1 only — that
        # silently rejects a valid split_v2 probe set (Phase F measurement debt).
        from pipeline.split import load_split, split_path as _split_path
        probe_lines = [ln for ln in probe_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed_probes = [json.loads(line) for line in probe_lines]
        declared_versions = {str(probe.get("split_version") or "") for probe in parsed_probes}
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
        split_labels = {str(probe.get("split") or "") for probe in parsed_probes}
        if len(split_labels) != 1 or not split_labels <= {"held_out", "trained"}:
            raise ValueError(f"probes have mixed or unsupported measurement classes at {probe_path}: {split_labels}")
        split_label = next(iter(split_labels))
        measurement_class = "augmentation_holdout" if split_label == "held_out" else "trained_fact_recall"
        if not split_version:
            raise ValueError(f"split manifest has no version: {split_path}")
        if split_label == "held_out" and not holdout_ids:
            raise ValueError(f"split manifest has no versioned holdout: {split_path}")
        if train_ids & holdout_ids:
            raise ValueError(f"split manifest train/holdout overlap: {split_path}")
        measurement_classes.add(measurement_class)
        split_versions.add(split_version)
        split_paths.append(split_path)
        probe_paths.append(probe_path)
        # Estimand-side decontamination, read the same way the split is: from the unit's own
        # artifact, non-destructively. Absent record == score everything (the pre-quarantine
        # behaviour); `instrument_blind` == this instrument could not read the unit's hydrate
        # floor, which is NOT the same claim as "measured clean".
        from pipeline.probe_quarantine import load_quarantine, quarantine_path, quarantined_reasons
        quarantine_record = load_quarantine(domain, unit.slug)
        if quarantine_record is not None:
            quarantine_paths.append(quarantine_path(domain, unit.slug))
            quarantine_instrument_blind = quarantine_instrument_blind or bool(
                quarantine_record.get("instrument_blind"))
            quarantined_sources.update(quarantined_reasons(quarantine_record))
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
            if probe.get("split") != split_label or probe.get("split_version") != split_version:
                raise ValueError(f"probe/split mismatch at {probe_path}:{line_number}")
            source_id = str(probe.get("source_triple_id") or "")
            source_ids = holdout_ids if split_label == "held_out" else train_ids
            if source_id not in source_ids:
                source_key = "holdout_ids" if split_label == "held_out" else "train_ids"
                raise ValueError(f"probe source is absent from split {source_key} at {probe_path}:{line_number}")
            surface_id = str(probe.get("probe_surface_id") or source_id)
            if not surface_id:
                raise ValueError(f"probe has no source/surface identity at {probe_path}:{line_number}")
            if surface_id in seen_surface_ids:
                raise ValueError(f"duplicate probe surface identity at {probe_path}:{line_number}")
            if source_id in seen_source_ids and not probe.get("probe_surface_id"):
                raise ValueError(
                    f"duplicate probe source_triple_id lacks probe_surface_id at {probe_path}:{line_number}"
                )
            seen_source_ids.add(source_id)
            seen_surface_ids.add(surface_id)
            choice_counts.add(len(choices))
            probes.append(probe)

    if not probes:
        raise ValueError("no probes found — run the probes stage first")
    scored = [
        probe for probe in probes
        if str(probe.get("source_triple_id") or "") not in quarantined_sources
    ]
    if not scored:
        raise ValueError(
            "the probe quarantine excludes every probe — an empty scored set is a refusal, "
            "never a 0/0 verdict; re-cut the estimand instead of scoring nothing"
        )
    if len(split_versions) != 1:
        raise ValueError(f"probe units use different split versions: {sorted(split_versions)}")
    if len(measurement_classes) != 1:
        raise ValueError(f"probe units mix measurement classes: {sorted(measurement_classes)}")
    if len(choice_counts) != 1:
        raise ValueError(f"probe items use different choice counts: {sorted(choice_counts)}")

    token_stats_path = _token_stats_path(tokenized)
    if not token_stats_path.is_file():
        raise ValueError(f"tokenization stats missing: {token_stats_path}")
    token_stats = json.loads(token_stats_path.read_text(encoding="utf-8"))
    clean_tokens = int((token_stats.get("tokens_by_tag") or {}).get("clean") or 0)
    measurement_class = next(iter(measurement_classes))
    if measurement_class == "augmentation_holdout" and clean_tokens <= 0:
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
        measurement_class=measurement_class,
        chance=1.0 / next(iter(choice_counts)),
        quarantine_paths=quarantine_paths,
        quarantine_hashes=[_sha256(path) for path in quarantine_paths],
        quarantined_sources=quarantined_sources,
        quarantine_instrument_blind=quarantine_instrument_blind,
    )


def release_eval_model() -> None:
    """Return host pages to the OS after a scored model is freed. CALLER MUST `del model`
    FIRST — deleting a binding inside this helper would release only the helper's own
    reference, leaving the caller's GPU tensor alive (that bug OOM'd the probe loop:
    12.56GiB still allocated at the candidate's .to(device), 2026-07-26).

    del + empty_cache alone releases VRAM, but glibc keeps the freed host weights in its
    arenas, so the next model load spikes RSS on top of the residue — the whale-reaper
    SIGTERM'd exactly such a load at rss=20GB / avail=6% (s3_r005 GA, 2026-07-26).
    malloc_trim(0) hands the arena pages back to the kernel."""
    import ctypes
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def load_model(path: str, device: str):
    """Load base vanilla or an engine int8-master checkpoint through its dequantizing loader.

    Base MUST go through the text-only remap (pipeline.model_load.load_text_model_bf16).
    AutoModelForCausalLM.from_pretrained on the multimodal gemma-4-E2B-it checkpoint
    materializes the full tower (~13GB on a 16GB card) and leaves no headroom for the
    subsequent tuned load — OOM'd every FC fill attempt on 2026-07-26 (s2_reasoning).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if (_rooted(path) / "quant_keys.json").exists():
        from pipeline.model_load import load_causal_ckpt_quartet
        model = load_causal_ckpt_quartet(path, device)
    else:
        from pipeline.model_load import load_text_model_bf16
        model = load_text_model_bf16(device)
    model.eval()
    return tokenizer, model


def scoring_prompt(probe: dict[str, Any]) -> str:
    """Resolve the measured prompt shape explicitly.

    CPT recall is a continuation estimand, so a probe may carry a verbatim
    cloze prefix. Domains without it retain the historical instruction prompt.
    """
    explicit = str(probe.get("prompt") or "").strip()
    if explicit:
        return explicit
    return f"Question: {probe['question']}\nAnswer:"


def choice_logprob(tokenizer, model, device: str, prompt: str, choice: str) -> float:
    import torch

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
    attention_mask = None
    if getattr(model, "_quartet_eval", False):
        # The Quartet fp4 GEMM requires the flattened token dim to be a multiple of 128
        # (training always runs seq_len=512). Left-pad so the last position stays the real
        # last token; the mask keeps pad tokens out of attention.
        pad = (-full_ids.shape[1]) % 128
        if pad:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            full_ids = torch.cat(
                [torch.full((1, pad), pad_id, dtype=full_ids.dtype, device=device), full_ids], dim=1)
            attention_mask = torch.cat(
                [torch.zeros((1, pad), dtype=torch.long, device=device),
                 torch.ones((1, full_ids.shape[1] - pad), dtype=torch.long, device=device)], dim=1)
    with torch.no_grad():
        logits = model(full_ids, attention_mask=attention_mask).logits[0, :-1]
        targets = full_ids[0, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        picked = logprobs[torch.arange(len(targets)), targets][-n_choice:]
    return picked.mean().item()


def evaluate_model(tokenizer, model, device: str, probes: list[dict[str, Any]]) -> list[bool]:
    hits: list[bool] = []
    for index, probe in enumerate(probes):
        prompt = scoring_prompt(probe)
        scores = [choice_logprob(tokenizer, model, device, prompt, choice)
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
    """Per-slice forced-choice rates with fact-clustered uncertainty and n-floor labels.

    Probe surfaces from one source triple are repeated measures, not independent observations.
    Collapse each fact cluster to one Bernoulli outcome ("the fact is known" only when every scored
    surface for it passes) for confidence intervals, rough bounds, and gate denominators. Item-level
    accuracy remains visible as the phrasing-robustness diagnostic.
    """
    if len(probes) != len(hits):
        raise ValueError(f"probe/hit length mismatch: {len(probes)} != {len(hits)}")
    per_predicate: dict[str, list[bool]] = {}
    per_predicate_clusters: dict[str, dict[str, list[bool]]] = {}
    for index, (probe, hit) in enumerate(zip(probes, hits, strict=True)):
        predicate = str(probe.get("predicate") or "unknown")
        per_predicate.setdefault(predicate, []).append(hit)
        cluster = str(probe.get("source_triple_id") or probe.get("subject") or index)
        per_predicate_clusters.setdefault(predicate, {}).setdefault(cluster, []).append(hit)
    total_hits = sum(hits)
    n = len(hits)
    cluster_hits: dict[str, list[bool]] = {}
    for index, (probe, hit) in enumerate(zip(probes, hits, strict=True)):
        key = str(probe.get("source_triple_id") or probe.get("subject") or index)
        cluster_hits.setdefault(key, []).append(hit)
    fact_passes = [all(values) for values in cluster_hits.values()]
    fact_hits = sum(fact_passes)
    n_clusters = len(fact_passes)
    overall = stamp_proportion(
        total_hits,
        n,
        min_n=min_n,
        bar=bar,
        estimand=estimand,
        slice_name="all",
    )
    fact_overall = stamp_proportion(
        fact_hits,
        n_clusters,
        min_n=min_n,
        bar=bar,
        estimand=estimand,
        slice_name="independent_facts",
    )
    # CLUSTERING. Probe items are NOT independent: generic_probes emits several surfaces per gold
    # triple (kanto: exactly 4 for each of 75), so n counts paraphrases while the model either knows
    # a fact or does not. Treating n as independent understates every interval built from it — the
    # same unpinned-denominator failure this pipeline keeps paying for. Reported so a reader can
    # deflate rather than having to notice.
    per_cluster = (n / n_clusters) if n_clusters else 1.0
    return {
        "acc": overall["acc"],
        "hits": total_hits,
        "n": n,
        "independent_fact_hits": fact_hits,
        "n_independent_facts": n_clusters,
        "independent_fact_acc": fact_overall["acc"],
        "independent_fact_wilson": fact_overall["wilson"],
        "independent_fact_n_status": fact_overall["n_status"],
        "independent_fact_rough_bound": fact_overall["rough_bound_half_over_sqrt_n"],
        **({
            "independent_fact_bar_status": fact_overall.get("bar_status"),
            "independent_facts_clear_bar": fact_overall.get("clears_bar"),
        } if bar is not None else {}),
        "items_per_fact": round(per_cluster, 3),
        "fact_pass_rule": "all scored surfaces for the source fact must pass",
        "clustering_note": (
            "n counts probe ITEMS; n_independent_facts counts the gold facts they came from. "
            "With m items per fact the design effect is 1+(m-1)*rho, so the effective sample size "
            "is n/DEFF — never n — for any comparison across runs."
        ),
        "wilson": overall["wilson"],
        "rough_bound_half_over_sqrt_n": overall["rough_bound_half_over_sqrt_n"],
        "n_status": overall["n_status"],
        "min_n": min_n,
        **({"bar": bar, "bar_status": overall.get("bar_status"),
            "clears_bar": overall.get("clears_bar"),
            "within_noise_of_bar": overall.get("within_noise_of_bar")} if bar is not None else {}),
        "per_predicate": {
            predicate: {
                **stamp_proportion(
                    sum(values),
                    len(values),
                    min_n=min_n,
                    bar=bar,
                    estimand=estimand,
                    slice_name=predicate,
                ),
                **(
                    lambda fact_values, fact_stamp: {
                        "independent_fact_hits": sum(fact_values),
                        "n_independent_facts": len(fact_values),
                        "independent_fact_acc": fact_stamp["acc"],
                        "independent_fact_wilson": fact_stamp["wilson"],
                        "independent_fact_n_status": fact_stamp["n_status"],
                        "independent_fact_bar_status": fact_stamp.get("bar_status"),
                    }
                )(
                    [
                        all(cluster_hits)
                        for cluster_hits in per_predicate_clusters[predicate].values()
                    ],
                    stamp_proportion(
                        sum(
                            all(cluster_hits)
                            for cluster_hits in per_predicate_clusters[predicate].values()
                        ),
                        len(per_predicate_clusters[predicate]),
                        min_n=min_n,
                        bar=bar,
                        estimand=estimand,
                        slice_name=f"{predicate}:independent_facts",
                    ),
                ),
            }
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
    checkpoint_fingerprint: str | None = None,
    tokenized: str,
) -> dict[str, Any]:
    # Scoring by INDEX below replaces a `zip(..., strict=True)`; keep that zip's length invariant
    # explicitly rather than letting a short hit list slice silently.
    if not len(base_hits) == len(tuned_hits) == len(corpus.probes):
        raise ValueError(
            f"probe/hit length mismatch: probes={len(corpus.probes)} "
            f"base={len(base_hits)} tuned={len(tuned_hits)}"
        )

    # ESTIMAND-SIDE DECONTAMINATION. A quarantined probe is still SCORED — its base/tuned outcome
    # is the evidence that the quarantine was warranted — but it is excluded from every reported
    # number, with a counted reason. The training data is untouched: un-teaching a train fact to
    # protect a probe would invert the estimand (pipeline/probe_quarantine.py).
    def _quarantine_reason(probe: dict[str, Any]) -> str | None:
        return corpus.quarantined_sources.get(str(probe.get("source_triple_id") or ""))

    kept = [
        index for index, probe in enumerate(corpus.probes)
        if _quarantine_reason(probe) is None
    ]
    excluded = [
        index for index, probe in enumerate(corpus.probes)
        if _quarantine_reason(probe) is not None
    ]
    quarantine_reason_counts: dict[str, int] = {}
    for index in excluded:
        reason = _quarantine_reason(corpus.probes[index]) or "unspecified"
        quarantine_reason_counts[reason] = quarantine_reason_counts.get(reason, 0) + 1
    scored_probes = [corpus.probes[index] for index in kept]
    scored_base_hits = [base_hits[index] for index in kept]
    scored_tuned_hits = [tuned_hits[index] for index in kept]

    # Chance is the floor a random forced-choice picker hits; stamp both sides so the
    # report carries Wilson + insufficient-n without hand arithmetic.
    base = summarize_hits(
        scored_probes,
        scored_base_hits,
        min_n=min_n,
        bar=corpus.chance,
        estimand=corpus.measurement_class,
    )
    tuned = summarize_hits(
        scored_probes,
        scored_tuned_hits,
        min_n=min_n,
        bar=corpus.chance,
        estimand=corpus.measurement_class,
    )
    rough_bound = tuned["independent_fact_rough_bound"] if tuned["n_independent_facts"] else 0.0
    if rough_bound is None:
        rough_bound = 0.0
    delta_acc = tuned["acc"] - base["acc"]
    delta_fact_acc = tuned["independent_fact_acc"] - base["independent_fact_acc"]
    for result in (base, tuned):
        result["probe_sha256"] = corpus.probe_sha256

    checks = [
        {"check": "base and tuned used the identical probe set", "pass": base["probe_sha256"] == tuned["probe_sha256"]},
        {
            "check": f"independent fact denominator is at least {min_n}",
            "pass": (
                base["n_independent_facts"] >= min_n
                and base["independent_fact_n_status"] != "insufficient-n"
            ),
        },
        {
            "check": "tuned accuracy clears chance by more than the rough bound",
            "pass": (
                tuned["independent_fact_acc"] - corpus.chance > rough_bound
                and tuned.get("independent_fact_bar_status") != "within-noise"
            ),
        },
        {
            "check": f"accuracy delta clears {min_delta_acc:.4f} by more than the rough bound",
            "pass": delta_fact_acc - min_delta_acc > rough_bound,
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
    expected_checkpoint_fingerprint = checkpoint_fingerprint or config_fingerprint
    if run_contract.get("fingerprint") != expected_checkpoint_fingerprint:
        raise ValueError("checkpoint fingerprint does not match the declared checkpoint lineage")
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
        "checkpoint_fingerprint": expected_checkpoint_fingerprint,
        "measurement_class": corpus.measurement_class,
        "domain": corpus.domain,
        "editions": corpus.editions,
        "split_version": corpus.split_version,
        "chance": corpus.chance,
        "rough_bound_half_over_sqrt_n": rough_bound,
        "delta_acc": delta_acc,
        "delta_independent_fact_acc": delta_fact_acc,
        "axis": "forced_choice",
        "estimand": corpus.measurement_class,
        "scored_n": len(kept),
        "quarantined_n": len(excluded),
        "quarantine_reasons": quarantine_reason_counts,
        "quarantine_instrument_blind": corpus.quarantine_instrument_blind,
        "thresholds": {"min_n": min_n, "min_delta_acc": min_delta_acc},
        "checks": checks,
        "failed_checks": failed,
        # Dual-axis column the generative merger fills in; present here so a forced-choice-only
        # report still names the axis shape install-strength will carry.
        "install_strength": {
            "forced_choice": {
                "tuned_acc": tuned["acc"],
                "tuned_independent_fact_acc": tuned["independent_fact_acc"],
                "base_acc": base["acc"],
                "base_independent_fact_acc": base["independent_fact_acc"],
                "n": tuned["n"],
                "n_independent_facts": tuned["n_independent_facts"],
                "items_per_fact": tuned["items_per_fact"],
                "wilson": tuned["wilson"],
                "independent_fact_wilson": tuned["independent_fact_wilson"],
                "n_status": tuned["n_status"],
                "independent_fact_n_status": tuned["independent_fact_n_status"],
                "delta_acc": delta_acc,
                "delta_independent_fact_acc": delta_fact_acc,
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
            "quarantine_paths": [_display(path) for path in corpus.quarantine_paths],
            "quarantine_hashes": corpus.quarantine_hashes,
            "quarantine_sha256": _combined_hash(corpus.quarantine_paths),
        },
        # The SCORED rows. Every downstream consumer of `probe_results` (certificate.py's booster
        # projection) therefore reads the decontaminated set without knowing this feature exists.
        "probe_results": [
            {
                "source_triple_id": corpus.probes[index].get("source_triple_id"),
                "probe_surface_id": corpus.probes[index].get("probe_surface_id"),
                "subject": corpus.probes[index].get("subject"),
                "predicate": corpus.probes[index].get("predicate"),
                "question": corpus.probes[index].get("question"),
                "answer": corpus.probes[index].get("answer"),
                "base_correct": bool(base_hits[index]),
                "tuned_correct": bool(tuned_hits[index]),
            }
            for index in kept
        ],
        # The EXCLUDED rows, kept with their outcomes: a quarantine whose excluded probes score
        # like the rest is a quarantine that needed re-examining, and that is only checkable if
        # the evidence ships with the receipt.
        "quarantined_probe_results": [
            {
                "source_triple_id": corpus.probes[index].get("source_triple_id"),
                "probe_surface_id": corpus.probes[index].get("probe_surface_id"),
                "subject": corpus.probes[index].get("subject"),
                "predicate": corpus.probes[index].get("predicate"),
                "question": corpus.probes[index].get("question"),
                "answer": corpus.probes[index].get("answer"),
                "base_correct": bool(base_hits[index]),
                "tuned_correct": bool(tuned_hits[index]),
                "quarantine_reason": _quarantine_reason(corpus.probes[index]),
            }
            for index in excluded
        ],
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
    # Tolerate a raw yaml.safe_load dict (ad-hoc drivers): re-load through the typed
    # config so attribute access below (config.eval/.data/.model/.run) cannot crash.
    if isinstance(config, dict):
        from cpt_config import CptConfig

        config = CptConfig.from_yaml(config_path)

    eval_config = config.eval
    corpus = load_probe_corpus(eval_config.probe_domain, eval_config.probe_editions, config.data.tokenized)
    scored: dict[str, list[bool]] = {}
    for name, path in (("base", config.model.model_path), ("tuned", checkpoint)):
        tokenizer, model = load_model(path, eval_config.device)
        scored[name] = evaluate_model(tokenizer, model, eval_config.device, corpus.probes)
        del model
        release_eval_model()

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
        checkpoint_fingerprint=getattr(config.eval, "checkpoint_fingerprint", None),
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
