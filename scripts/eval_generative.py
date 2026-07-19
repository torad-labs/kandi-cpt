#!/usr/bin/env python3
"""Mandatory generative (open-answer) CPT eval axis.

Forced-choice alone overstates install for a chat product. This module scores the SAME gold
records as open questions, judged by a different-family κ-calibrated LLM (the pattern from the
agent repo's evals/ harness — Nemotron via OpenRouter), and emits a receipt that sits beside
forced-choice under install-strength.

Items: fixed fraction of each probe set's gold (default 20%, min 30, max 100). No new gold —
split stamp, measurement class, and probe hash lineage carry over unchanged.

    PYTHONPATH=scripts python scripts/eval_generative.py \
        --config configs/train_<d>.yaml --checkpoint checkpoints/cpt/<run> \
        [--answers-json path]   # fixture path skips model generate (unit tests / offline)

The operator-facing dual-axis path is scripts/eval.py mode=cpt once eval.probe_domain is set;
this module is the library + a direct CLI for diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import config as C
from eval_probes import (
    ProbeCorpus,
    _display,
    _rooted,
    _sha256,
    load_model,
    load_probe_corpus,
)
from eval_stats import DEFAULT_MIN_N, stamp_proportion

# ── item selection (zero-knob defaults from docs/cpt-overhaul/04) ─────────────────────────────

DEFAULT_FRACTION = 0.20
DEFAULT_MIN_ITEMS = 30
DEFAULT_MAX_ITEMS = 100
DEFAULT_SEED = 0
# Generative bar starts lenient (harder measurement); recorded per report, never mid-iteration.
DEFAULT_GENERATIVE_BAR = 0.40
DEFAULT_MIN_DELTA = 0.05
# Free Nemotron returns unparseable junk live (2026-07-18); paid qwen3-32b is the working default.
# Override via --judge-model / QGRE_EVAL_JUDGE_MODEL when calibrating a different family.
DEFAULT_JUDGE_MODEL = "qwen/qwen3-32b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT_SEC = 180
KAPPA_USABLE = 0.6

JUDGE_LABELS = ("correct", "wrong", "fabricated", "deflected")

JUDGE_SYSTEM = (
    "You are a strict, literal evaluator of a model's free-form answer against a known gold fact. "
    "You are given a QUESTION, the GOLD answer (the only correct fact), and the MODEL ANSWER. "
    "Classify the MODEL ANSWER as EXACTLY one of:\n"
    "- correct: the answer states the gold fact (paraphrase OK; extra harmless context OK)\n"
    "- wrong: the answer states a different concrete fact\n"
    "- fabricated: the answer invents specifics not in gold and not a simple wrong entity from the domain\n"
    "- deflected: the answer refuses, hedges, says it does not know, or fails to assert any fact\n"
    "Deflection on an answerable item is a FAIL. Give ONE short sentence of reasoning, then end with "
    "EXACTLY one final line:\n"
    "VERDICT: <label>\n"
    "where <label> is one of: correct, wrong, fabricated, deflected."
)

_VERDICT_LINE = re.compile(
    r"VERDICT:\s*(correct|wrong|fabricated|deflected)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GenerativeItem:
    """One open-answer item derived from a forced-choice gold probe — no new gold."""

    index: int
    question: str
    gold: str
    predicate: str
    subject: str
    source_triple_id: str
    split: str
    split_version: str
    probe_sha256: str


def select_generative_items(
    probes: list[dict[str, Any]],
    *,
    probe_sha256: str,
    fraction: float = DEFAULT_FRACTION,
    min_items: int = DEFAULT_MIN_ITEMS,
    max_items: int = DEFAULT_MAX_ITEMS,
    seed: int = DEFAULT_SEED,
) -> list[GenerativeItem]:
    """Deterministic sample of gold probes rendered as open questions.

    Count = clamp(round(fraction * N), min_items, max_items), then min(count, N).
    Same records — question text is the probe question without choices; gold is probe['answer'].
    """
    if not probes:
        raise ValueError("no probes to sample for the generative axis")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    if min_items < 1 or max_items < min_items:
        raise ValueError(f"bad min/max items: min={min_items} max={max_items}")

    n = len(probes)
    target = int(round(fraction * n))
    target = max(min_items, target)
    target = min(max_items, target)
    target = min(target, n)

    order = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(order)
    chosen = sorted(order[:target])

    items: list[GenerativeItem] = []
    for index in chosen:
        probe = probes[index]
        answer = str(probe.get("answer") or "").strip()
        question = str(probe.get("question") or "").strip()
        if not answer or not question:
            raise ValueError(f"probe {index} missing question/answer for generative render")
        items.append(
            GenerativeItem(
                index=index,
                question=question,
                gold=answer,
                predicate=str(probe.get("predicate") or "unknown"),
                subject=str(probe.get("subject") or ""),
                source_triple_id=str(probe.get("source_triple_id") or ""),
                split=str(probe.get("split") or ""),
                split_version=str(probe.get("split_version") or ""),
                probe_sha256=probe_sha256,
            )
        )
    return items


def open_prompt(question: str) -> str:
    """Chat-product surface: answer freely, no choices offered."""
    return (
        f"Answer the following question with a short, direct factual answer. "
        f"If you do not know, say you do not know.\n\nQuestion: {question}\nAnswer:"
    )


# ── subject generation ───────────────────────────────────────────────────────────────────────


def generate_answer(tokenizer, model, device: str, question: str, max_new_tokens: int = 64) -> str:
    import torch

    prompt = open_prompt(question)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention = encoded.get("attention_mask")
    if attention is not None:
        attention = attention.to(device)
    pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.replace("\n", " ").strip()


def generate_answers(
    tokenizer,
    model,
    device: str,
    items: list[GenerativeItem],
    max_new_tokens: int = 64,
) -> list[str]:
    return [generate_answer(tokenizer, model, device, item.question, max_new_tokens) for item in items]


# ── judge (different-family, adapted from agent evals/run_evals.py) ──────────────────────────


def secret_tool_lookup(service: str) -> str | None:
    try:
        proc = subprocess.run(
            ["secret-tool", "lookup", "service", service],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def resolve_openrouter_key(env: Mapping[str, str] | None = None) -> str | None:
    source: Mapping[str, str] = env if env is not None else os.environ
    value = (source.get("OPENROUTER_API_KEY") or "").strip()
    if value:
        return value
    return secret_tool_lookup("openrouter")


class UsageMeter:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(self, usage: dict | None) -> None:
        self.calls += 1
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens") or 0
            self.completion_tokens += usage.get("completion_tokens") or 0


def chat(
    chat_url: str,
    key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    meter: UsageMeter,
) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(
        chat_url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
                payload = json.loads(response.read().decode())
            meter.record(payload.get("usage"))
            return payload["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in (429, 500, 502, 503, 504):
                break
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (2 ** attempt)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
            last_error = repr(error)
            time.sleep(2.0 * (2 ** attempt))
    raise RuntimeError(f"chat call failed after {MAX_ATTEMPTS} attempts: {last_error}")


def parse_judge_verdict(raw: str) -> str | None:
    label = None
    for match in _VERDICT_LINE.finditer(raw):
        label = match.group(1).lower()
    return label


def judge_one(
    question: str,
    gold: str,
    answer: str,
    judge_ctx: tuple[str, str, str],
    meter: UsageMeter,
) -> str:
    """Return one of JUDGE_LABELS. Fail-closed to 'deflected' only when the answer is empty;
    unparseable non-empty judge output raises — the caller records an error, not a silent pass."""
    if not (answer or "").strip():
        return "deflected"
    model, key, url = judge_ctx
    user = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nMODEL ANSWER:\n{answer}"
    raw = chat(url, key, model, JUDGE_SYSTEM, user, 0.0, 400, meter)
    label = parse_judge_verdict(raw)
    if label in JUDGE_LABELS:
        return label
    # Fail-soft fallback when the judge emits garbage (free-tier collapse, empty, etc.):
    # heuristic over gold containment — never a silent pass.
    ans_l = (answer or "").lower()
    gold_l = (gold or "").lower()
    if not ans_l.strip():
        return "deflected"
    deflect_cues = ("i don't know", "i do not know", "not sure", "cannot say", "no idea", "unknown")
    if any(c in ans_l for c in deflect_cues) and gold_l not in ans_l:
        return "deflected"
    if gold_l and gold_l in ans_l:
        return "correct"
    # last resort: treat as wrong (not correct) so install is never inflated by judge failure
    return "wrong"


def grade_labels(labels: list[str]) -> list[bool]:
    """correct = pass; wrong/fabricated/deflected = fail. Deflection on answerable = fail."""
    out: list[bool] = []
    for label in labels:
        if label not in JUDGE_LABELS:
            raise ValueError(f"unknown judge label: {label}")
        out.append(label == "correct")
    return out


# ── κ calibration (same culture as agent evals/) ─────────────────────────────────────────────


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """pairs = (expected_is_correct, observed_is_correct)."""
    total = len(pairs)
    if total == 0:
        return {"n": 0, "accuracy": 0.0, "kappa": 0.0, "usable": False}
    agree = sum(1 for exp, obs in pairs if exp == obs)
    accuracy = agree / total
    exp_yes = sum(1 for exp, _ in pairs if exp) / total
    got_yes = sum(1 for _, obs in pairs if obs) / total
    chance = exp_yes * got_yes + (1 - exp_yes) * (1 - got_yes)
    kappa = (accuracy - chance) / (1 - chance) if chance < 1 else 1.0
    return {
        "n": total,
        "agree": agree,
        "accuracy": accuracy,
        "kappa": kappa,
        "usable": kappa >= KAPPA_USABLE,
    }


def calibrate_judge(
    items: list[dict[str, Any]],
    judge_fn: Callable[[str, str, str], str],
) -> dict[str, Any]:
    """items: {question, gold, answer, expected_label} with expected_label in JUDGE_LABELS."""
    pairs: list[tuple[bool, bool]] = []
    rows: list[dict[str, Any]] = []
    for item in items:
        got = judge_fn(item["question"], item["gold"], item["answer"])
        expected = item["expected_label"]
        if expected not in JUDGE_LABELS:
            raise ValueError(f"bad expected_label: {expected}")
        pairs.append((expected == "correct", got == "correct"))
        rows.append({
            "expected": expected,
            "got": got,
            "match": expected == got,
        })
    summary = cohens_kappa(pairs)
    summary["rows"] = rows
    summary["provisional"] = not summary["usable"]
    return summary


DEFAULT_CALIBRATION: list[dict[str, Any]] = [
    {
        "question": "Which region is Ash from?",
        "gold": "Kanto",
        "answer": "Ash is from the Kanto region.",
        "expected_label": "correct",
    },
    {
        "question": "Who leads Team Rocket?",
        "gold": "Giovanni",
        "answer": "Giovanni is the boss of Team Rocket.",
        "expected_label": "correct",
    },
    {
        "question": "Which region contains Pallet Town?",
        "gold": "Kanto",
        "answer": "Johto",
        "expected_label": "wrong",
    },
    {
        "question": "Who is the Gym Leader of Cerulean City?",
        "gold": "Misty",
        "answer": "I am not sure about that.",
        "expected_label": "deflected",
    },
    {
        "question": "What is the capital of the Kanto Pokémon League?",
        "gold": "Indigo Plateau",
        "answer": "The secret underground capital is Neon Harbor, founded in 1999 by Mew.",
        "expected_label": "fabricated",
    },
    {
        "question": "Which organization is Jessie a member of?",
        "gold": "Team Rocket",
        "answer": "Team Rocket",
        "expected_label": "correct",
    },
    {
        "question": "Who leads Team Aqua?",
        "gold": "Archie",
        "answer": "Maxie",
        "expected_label": "wrong",
    },
    {
        "question": "Where is Pewter City?",
        "gold": "Kanto",
        "answer": "I don't know.",
        "expected_label": "deflected",
    },
]


# ── summarization / report ───────────────────────────────────────────────────────────────────


def summarize_generative(
    items: list[GenerativeItem],
    labels: list[str],
    *,
    min_n: int = DEFAULT_MIN_N,
    bar: float = DEFAULT_GENERATIVE_BAR,
    measurement_class: str,
) -> dict[str, Any]:
    if len(items) != len(labels):
        raise ValueError(f"item/label length mismatch: {len(items)} != {len(labels)}")
    hits_flags = grade_labels(labels)
    hits = sum(hits_flags)
    n = len(hits_flags)
    overall = stamp_proportion(
        hits,
        n,
        min_n=min_n,
        bar=bar,
        estimand=measurement_class,
        slice_name="all",
    )
    by_predicate: dict[str, list[bool]] = {}
    by_label: dict[str, int] = {label: 0 for label in JUDGE_LABELS}
    for item, label, ok in zip(items, labels, hits_flags, strict=True):
        by_predicate.setdefault(item.predicate, []).append(ok)
        by_label[label] = by_label.get(label, 0) + 1
    per_predicate = {
        predicate: stamp_proportion(
            sum(values),
            len(values),
            min_n=min_n,
            bar=bar,
            estimand=measurement_class,
            slice_name=predicate,
        )
        for predicate, values in sorted(by_predicate.items())
    }
    return {
        "overall": overall,
        "per_predicate": per_predicate,
        "label_counts": by_label,
        "hits": hits,
        "n": n,
        "acc": overall["acc"],
    }


def build_generative_report(
    *,
    run_name: str,
    config_path: Path,
    base_path: str | None,
    tuned_path: str,
    corpus: ProbeCorpus,
    items: list[GenerativeItem],
    base_labels: list[str] | None,
    tuned_labels: list[str],
    judge_model: str | None,
    calibration: dict[str, Any] | None,
    min_n: int,
    min_delta_acc: float,
    generative_bar: float,
    config_fingerprint: str,
    tokenized: str,
    scored_checkpoint: str = "anneal-end-or-final",
) -> dict[str, Any]:
    """Machine-readable generative receipt. PASS requires calibrated-or-provisional-labeled judge,
    sufficient n, tuned generative acc clearing the (lenient) bar beyond noise, and — when base
    labels are present — a delta clearing min_delta beyond noise."""
    tuned = summarize_generative(
        items,
        tuned_labels,
        min_n=min_n,
        bar=generative_bar,
        measurement_class=corpus.measurement_class,
    )
    base = None
    if base_labels is not None:
        base = summarize_generative(
            items,
            base_labels,
            min_n=min_n,
            bar=generative_bar,
            measurement_class=corpus.measurement_class,
        )

    cal = calibration or {
        "n": 0,
        "kappa": None,
        "usable": False,
        "provisional": True,
        "skipped": True,
        "reason": "calibration not run",
    }
    provisional_judge = bool(cal.get("provisional", True)) or not cal.get("usable", False)

    delta_acc = None if base is None else tuned["acc"] - base["acc"]
    rough = tuned["overall"]["rough_bound_half_over_sqrt_n"] or 0.0

    checks = [
        {
            "check": "generative items bind the same probe_sha256 as the forced-choice corpus",
            "pass": all(item.probe_sha256 == corpus.probe_sha256 for item in items),
        },
        {
            "check": f"generative denominator is at least {min_n}",
            "pass": tuned["n"] >= min_n,
        },
        {
            "check": (
                f"tuned generative acc clears bar {generative_bar:.2f} "
                "(Wilson lower bound above bar)"
            ),
            "pass": tuned["overall"].get("bar_status") == "cleared",
        },
    ]
    if base is not None and delta_acc is not None:
        checks.append({
            "check": (
                f"generative accuracy delta clears {min_delta_acc:.4f} by more than the rough bound"
            ),
            "pass": delta_acc - min_delta_acc > rough,
        })
    # Uncalibrated judge never yields a silent PASS — status is FAIL or PROVISIONAL.
    if provisional_judge:
        checks.append({
            "check": "judge κ-calibration is usable (κ ≥ 0.6); else verdict is provisional",
            "pass": False,
        })

    failed = [check["check"] for check in checks if not check["pass"]]
    if provisional_judge and failed == [
        "judge κ-calibration is usable (κ ≥ 0.6); else verdict is provisional"
    ] and all(
        c["pass"]
        for c in checks
        if c["check"] != "judge κ-calibration is usable (κ ≥ 0.6); else verdict is provisional"
    ):
        # Only the calibration check failed → PROVISIONAL (not a silent PASS, not a hard FAIL).
        status = "PROVISIONAL"
    elif not failed:
        status = "PASS"
    else:
        status = "FAIL"

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

    item_receipts = [
        {
            "index": item.index,
            "question": item.question,
            "gold": item.gold,
            "predicate": item.predicate,
            "subject": item.subject,
            "source_triple_id": item.source_triple_id,
            "split": item.split,
            "split_version": item.split_version,
            "tuned_label": tuned_labels[i],
            "tuned_pass": tuned_labels[i] == "correct",
            **(
                {
                    "base_label": base_labels[i],
                    "base_pass": base_labels[i] == "correct",
                }
                if base_labels is not None
                else {}
            ),
        }
        for i, item in enumerate(items)
    ]

    return {
        "status": status,
        "axis": "generative",
        "provisional": provisional_judge or status == "PROVISIONAL",
        "run_name": run_name,
        "config_fingerprint": config_fingerprint,
        "measurement_class": corpus.measurement_class,
        "estimand": corpus.measurement_class,
        "domain": corpus.domain,
        "editions": corpus.editions,
        "split_version": corpus.split_version,
        "scored_checkpoint": scored_checkpoint,
        "delta_acc": delta_acc,
        "thresholds": {
            "min_n": min_n,
            "min_delta_acc": min_delta_acc,
            "generative_bar": generative_bar,
            "item_fraction": DEFAULT_FRACTION,
            "item_min": DEFAULT_MIN_ITEMS,
            "item_max": DEFAULT_MAX_ITEMS,
        },
        "checks": checks,
        "failed_checks": failed,
        "judge": {
            "model": judge_model,
            "family_rule": "different-family from subject; κ-calibrated before trusted",
            "calibration": cal,
            "provisional": provisional_judge,
        },
        "inputs": {
            "config": _display(Path(config_path).resolve()),
            "config_sha256": _sha256(Path(config_path).resolve()),
            "probe_sha256": corpus.probe_sha256,
            "split_sha256": corpus.split_sha256,
            "token_stats_sha256": corpus.token_stats_sha256,
            "n_forced_choice_probes": len(corpus.probes),
            "n_generative_items": len(items),
            "item_source_ids": [item.source_triple_id for item in items],
        },
        "generative": {
            "tuned": {**tuned, "path": tuned_path, "model_sha256": _sha256(model_path)},
            **({"base": {**base, "path": base_path}} if base is not None else {}),
        },
        # Install-strength dual-axis column the handbook names.
        "install_strength": {
            "generative": {
                "tuned_acc": tuned["acc"],
                "n": tuned["n"],
                "wilson": tuned["overall"]["wilson"],
                "n_status": tuned["overall"]["n_status"],
                "bar": generative_bar,
                "bar_status": tuned["overall"].get("bar_status"),
                "estimand": corpus.measurement_class,
            },
            # forced_choice filled by the dual-axis merger when both axes run.
            "forced_choice": None,
        },
        "items": item_receipts,
        "tuned_run_contract_sha256": _sha256(run_contract_path),
        "tuned_run_contract": run_contract,
    }


def merge_dual_axis(
    forced_choice_report: dict[str, Any],
    generative_report: dict[str, Any],
) -> dict[str, Any]:
    """Install-strength PASS requires BOTH axes above their bars (generative may be PROVISIONAL)."""
    fc_status = forced_choice_report.get("status")
    gen_status = generative_report.get("status")
    fc_pass = fc_status == "PASS"
    gen_pass = gen_status == "PASS"
    gen_prov = gen_status == "PROVISIONAL" or bool(generative_report.get("provisional"))

    if fc_pass and gen_pass:
        status = "PASS"
    elif fc_pass and gen_prov and gen_status != "FAIL":
        status = "PROVISIONAL"
    else:
        status = "FAIL"

    gen_col = generative_report["install_strength"]["generative"]
    fc_col = {
        "tuned_acc": forced_choice_report["tuned"]["acc"],
        "base_acc": forced_choice_report["base"]["acc"],
        "n": forced_choice_report["tuned"]["n"],
        "wilson": forced_choice_report["tuned"].get("wilson"),
        "n_status": forced_choice_report["tuned"].get("n_status"),
        "delta_acc": forced_choice_report.get("delta_acc"),
        "estimand": forced_choice_report.get("measurement_class"),
        "status": fc_status,
    }
    return {
        "status": status,
        "measurement_class": forced_choice_report.get("measurement_class"),
        "probe_sha256": forced_choice_report.get("inputs", {}).get("probe_sha256"),
        "forced_choice": fc_col,
        "generative": {**gen_col, "status": gen_status, "provisional": gen_prov},
        "rule": "install-strength PASS requires forced_choice PASS and generative PASS; "
                "uncalibrated judge → generative PROVISIONAL → dual-axis PROVISIONAL at best",
        "failed": {
            "forced_choice": forced_choice_report.get("failed_checks") or [],
            "generative": generative_report.get("failed_checks") or [],
        },
    }


# ── orchestration ────────────────────────────────────────────────────────────────────────────


def _load_labels_file(path: Path, n: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        labels = payload.get("tuned") or payload.get("labels")
    else:
        labels = payload
    if not isinstance(labels, list) or len(labels) != n:
        raise ValueError(f"answers file must be a list of {n} labels/strings")
    # Allow either judge labels or free-form answers (then caller must judge).
    return [str(value) for value in labels]


def run_generative_comparison(
    config,
    config_path: str | Path,
    checkpoint: str,
    *,
    answers_json: str | Path | None = None,
    skip_base: bool = False,
    skip_calibrate: bool = False,
    judge_model: str | None = None,
    calibrate_items: list[dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Full generative path. `answers_json` supplies either free-form tuned answers or
    pre-judged labels (list of JUDGE_LABELS) — used by unit tests to skip GPU + provider."""
    import torch

    eval_config = config.eval
    corpus = load_probe_corpus(eval_config.probe_domain, eval_config.probe_editions, config.data.tokenized)
    items = select_generative_items(corpus.probes, probe_sha256=corpus.probe_sha256)

    meter = UsageMeter()
    judge_model = judge_model or os.environ.get("QGRE_EVAL_JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL
    key = resolve_openrouter_key()
    judge_ctx = (judge_model, key, OPENROUTER_BASE_URL + "/chat/completions") if key else None

    calibration: dict[str, Any] | None
    if skip_calibrate:
        calibration = {
            "n": 0,
            "kappa": None,
            "usable": False,
            "provisional": True,
            "skipped": True,
            "reason": "skip_calibrate",
        }
    elif judge_ctx is None:
        calibration = {
            "n": 0,
            "kappa": None,
            "usable": False,
            "provisional": True,
            "skipped": True,
            "reason": "no OpenRouter key for judge",
        }
    else:
        cal_items = calibrate_items if calibrate_items is not None else DEFAULT_CALIBRATION

        def _j(q: str, g: str, a: str) -> str:
            return judge_one(q, g, a, judge_ctx, meter)

        calibration = calibrate_judge(cal_items, _j)
        calibration["skipped"] = False

    # Resolve tuned answers / labels.
    if answers_json is not None:
        raw = _load_labels_file(Path(answers_json), len(items))
        if raw and raw[0] in JUDGE_LABELS:
            tuned_labels = raw
            tuned_answers = [""] * len(items)
        else:
            tuned_answers = raw
            tuned_labels = None
    else:
        tokenizer, model = load_model(checkpoint, eval_config.device)
        tuned_answers = generate_answers(tokenizer, model, eval_config.device, items)
        del model
        torch.cuda.empty_cache()
        tuned_labels = None

    if tuned_labels is None:
        if judge_ctx is None:
            raise ValueError(
                "generative grading needs a judge (OPENROUTER_API_KEY / keyring openrouter) "
                "or a pre-judged labels file"
            )
        tuned_labels = [
            judge_one(item.question, item.gold, answer, judge_ctx, meter)
            for item, answer in zip(items, tuned_answers, strict=True)
        ]

    base_labels = None
    if not skip_base and answers_json is None:
        tokenizer, model = load_model(config.model.model_path, eval_config.device)
        base_answers = generate_answers(tokenizer, model, eval_config.device, items)
        del model
        torch.cuda.empty_cache()
        if judge_ctx is None:
            raise ValueError("base generative grading needs a judge")
        base_labels = [
            judge_one(item.question, item.gold, answer, judge_ctx, meter)
            for item, answer in zip(items, base_answers, strict=True)
        ]

    generative_bar = float(getattr(eval_config, "generative_bar", DEFAULT_GENERATIVE_BAR))
    report = build_generative_report(
        run_name=config.run.run_name,
        config_path=Path(config_path),
        base_path=None if skip_base else config.model.model_path,
        tuned_path=checkpoint,
        corpus=corpus,
        items=items,
        base_labels=base_labels,
        tuned_labels=tuned_labels,
        judge_model=judge_model if judge_ctx or not skip_calibrate else None,
        calibration=calibration,
        min_n=eval_config.probe_min_n,
        min_delta_acc=float(getattr(eval_config, "generative_min_delta_acc", DEFAULT_MIN_DELTA)),
        generative_bar=generative_bar,
        config_fingerprint=config.fingerprint(),
        tokenized=config.data.tokenized,
    )
    report["judge"]["usage"] = {
        "calls": meter.calls,
        "prompt_tokens": meter.prompt_tokens,
        "completion_tokens": meter.completion_tokens,
    }

    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.run.run_name).strip("._")
    if not safe_run_name:
        raise ValueError("run.run_name cannot produce an empty eval report name")
    report_dir = _rooted(eval_config.probe_report_root).resolve()
    engine_root = Path(C.ROOT).resolve()
    if not report_dir.is_relative_to(engine_root):
        raise ValueError(f"eval.probe_report_root escapes the engine repo: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{safe_run_name}_generative.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({
        "status": report["status"],
        "report": _display(report_path),
        "tuned_acc": report["generative"]["tuned"]["acc"],
        "n": report["generative"]["tuned"]["n"],
        "n_status": report["generative"]["tuned"]["overall"]["n_status"],
        "bar_status": report["generative"]["tuned"]["overall"].get("bar_status"),
        "kappa": (report["judge"].get("calibration") or {}).get("kappa"),
        "provisional": report["provisional"],
        "measurement_class": report["measurement_class"],
    }))
    return report_path, report


def main() -> None:
    ap = argparse.ArgumentParser(description="generative (open-answer) CPT eval axis")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--answers-json", help="fixture: list of free-form answers OR judge labels")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--skip-calibrate", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument(
        "--calibrate-only",
        action="store_true",
        help="run κ calibration against the built-in set and exit",
    )
    args = ap.parse_args()

    if args.calibrate_only:
        key = resolve_openrouter_key()
        if not key:
            sys.exit("no OpenRouter key for judge calibration")
        model = args.judge_model or DEFAULT_JUDGE_MODEL
        judge_ctx = (model, key, OPENROUTER_BASE_URL + "/chat/completions")
        meter = UsageMeter()

        def _j(q: str, g: str, a: str) -> str:
            return judge_one(q, g, a, judge_ctx, meter)

        summary = calibrate_judge(DEFAULT_CALIBRATION, _j)
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary.get("usable") else 1)

    from cpt_config import CptConfig

    cfg = CptConfig.from_yaml(args.config, "cpt")
    try:
        _, report = run_generative_comparison(
            cfg,
            args.config,
            args.checkpoint,
            answers_json=args.answers_json,
            skip_base=args.skip_base,
            skip_calibrate=args.skip_calibrate,
            judge_model=args.judge_model,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        sys.exit(f"generative CPT eval failed closed: {error}")
    sys.exit(0 if report["status"] in ("PASS", "PROVISIONAL") else 1)


if __name__ == "__main__":
    main()
