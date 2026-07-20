#!/usr/bin/env python3
"""κ-calibrate the generative judge on a labeled set; write a phase-F receipt.

    PYTHONPATH=scripts python scripts/pipeline/run_judge_calibration.py \
        --calibration-json configs/evals/calibration_pokemon-lore.json \
        --judge-model qwen/qwen3-32b \
        --out logs/phase_f_judge_calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import config as C
from eval_generative import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_LABELS,
    JUDGE_SYSTEM,
    KAPPA_USABLE,
    OPENROUTER_BASE_URL,
    UsageMeter,
    calibrate_judge,
    chat,
    judge_one,
    load_calibration_items,
    parse_judge_verdict,
    resolve_openrouter_key,
)


def multi_class_kappa(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> float:
    """Cohen's κ over the full label set (not the binary correct/not collapse)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in idx or p not in idx:
            continue
        cm[idx[t]][idx[p]] += 1
    agree = sum(cm[i][i] for i in range(k)) / n
    # marginals
    p_true = [sum(cm[i][j] for j in range(k)) / n for i in range(k)]
    p_pred = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]
    chance = sum(a * b for a, b in zip(p_true, p_pred, strict=True))
    if chance >= 1.0:
        return 1.0
    return (agree - chance) / (1.0 - chance)


def confusion(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> dict:
    mat: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        tt = t if t in labels else "unknown"
        pp = p if p in labels else "unknown"
        if tt not in mat:
            mat[tt] = {lab: 0 for lab in labels}
        if pp not in mat[tt]:
            mat[tt][pp] = 0
        mat[tt][pp] = mat[tt].get(pp, 0) + 1
    return mat


def set_hash(items: list[dict]) -> str:
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def run_once(
    items: list[dict],
    *,
    judge_model: str,
    key: str,
    system_prompt: str,
) -> dict:
    """Calibrate with an injectable system prompt (for one tighten iteration)."""
    meter = UsageMeter()
    url = OPENROUTER_BASE_URL + "/chat/completions"
    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict] = []

    # Local judge that uses the (possibly tightened) system prompt.
    def _judge(question: str, gold: str, answer: str) -> str:
        if not (answer or "").strip():
            return "deflected"
        user = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nMODEL ANSWER:\n{answer}"
        raw = chat(url, key, judge_model, system_prompt, user, 0.0, 400, meter)
        label = parse_judge_verdict(raw)
        if label in JUDGE_LABELS:
            return label
        # same fail-soft as judge_one
        ans_l = (answer or "").lower()
        gold_l = (gold or "").lower()
        deflect_cues = ("i don't know", "i do not know", "not sure", "cannot say", "no idea", "unknown")
        if any(c in ans_l for c in deflect_cues) and gold_l not in ans_l:
            return "deflected"
        if gold_l and gold_l in ans_l:
            return "correct"
        return "wrong"

    for i, item in enumerate(items):
        got = _judge(item["question"], item["gold"], item["answer"])
        exp = item["expected_label"]
        y_true.append(exp)
        y_pred.append(got)
        rows.append({
            "i": i,
            "question": item["question"],
            "expected": exp,
            "got": got,
            "match": exp == got,
            "case": item.get("case"),
        })
        print(f"  [{i+1}/{len(items)}] exp={exp} got={got} match={exp == got}", flush=True)

    # Binary κ on correct vs not-correct — the usable gate (KAPPA_USABLE).
    # Computed from already-collected labels (no second API pass).
    from eval_generative import cohens_kappa
    pairs = [(t == "correct", p == "correct") for t, p in zip(y_true, y_pred, strict=True)]
    binary = cohens_kappa(pairs)
    binary["provisional"] = not binary["usable"]
    # Multi-class metrics for the receipt (honest 4-way picture).
    label_acc = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b) / len(items)
    mc_kappa = multi_class_kappa(y_true, y_pred, JUDGE_LABELS)
    conf = confusion(y_true, y_pred, JUDGE_LABELS)
    per_label = {}
    for lab in JUDGE_LABELS:
        idxs = [i for i, t in enumerate(y_true) if t == lab]
        if not idxs:
            per_label[lab] = {"n": 0, "recall": None, "precision": None}
            continue
        tp = sum(1 for i in idxs if y_pred[i] == lab)
        pred_n = sum(1 for p in y_pred if p == lab)
        per_label[lab] = {
            "n": len(idxs),
            "recall": tp / len(idxs),
            "precision": (tp / pred_n) if pred_n else None,
        }
    return {
        "binary_correct_vs_rest": {
            "n": binary["n"],
            "accuracy": binary["accuracy"],
            "kappa": binary["kappa"],
            "usable": binary["usable"],
            "provisional": binary["provisional"],
            "agree": binary["agree"],
        },
        "multiclass": {
            "n": len(items),
            "accuracy": label_acc,
            "kappa": mc_kappa,
            "per_label": per_label,
            "confusion": conf,
        },
        "rows": rows,
        "usage": {
            "calls": meter.calls,
            "prompt_tokens": meter.prompt_tokens,
            "completion_tokens": meter.completion_tokens,
        },
    }


# One tighten iteration allowed by the measurement-debt brief when κ < 0.6.
TIGHTENED_JUDGE_SYSTEM = (
    "You are a strict, literal evaluator of a model's free-form answer against a known gold fact. "
    "You are given a QUESTION, the GOLD answer (the only correct fact), and the MODEL ANSWER. "
    "Classify the MODEL ANSWER as EXACTLY one of:\n"
    "- correct: the answer states the gold fact (paraphrase OK; extra harmless context OK). "
    "Do NOT mark correct if a different concrete entity is named.\n"
    "- wrong: the answer states a different concrete fact from the same domain "
    "(entity swap, wrong region/leader/gym, confusable rival). A simple wrong entity is WRONG, not fabricated.\n"
    "- fabricated: the answer invents specifics that are NOT a simple wrong entity from the domain "
    "(made-up places, dates, organizations, types, secret capitals). If it sounds like fanfic detail, fabricated.\n"
    "- deflected: the answer refuses, hedges, says it does not know, or fails to assert any fact. "
    "Phrases like 'not sure', 'I don't know', 'cannot answer', 'no idea' are deflected even if polite.\n"
    "Priority when ambiguous: deflected beats wrong if no concrete fact is asserted; "
    "wrong beats fabricated when the answer is a confusable in-domain entity; "
    "fabricated only for invented specifics beyond a simple entity swap.\n"
    "Deflection on an answerable item is a FAIL. Give ONE short sentence of reasoning, then end with "
    "EXACTLY one final line:\n"
    "VERDICT: <label>\n"
    "where <label> is one of: correct, wrong, fabricated, deflected."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-json", default="configs/evals/calibration_pokemon-lore.json")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--out", default="logs/phase_f_judge_calibration.json")
    ap.add_argument("--domain", default="pokemon-lore")
    args = ap.parse_args()

    items = load_calibration_items(args.calibration_json, domain=args.domain)
    key = resolve_openrouter_key()
    if not key:
        sys.exit("OPENROUTER_API_KEY missing (env or libsecret service=openrouter)")

    cal_path = Path(args.calibration_json)
    sh = set_hash(items)
    label_dist = dict(Counter(i["expected_label"] for i in items))

    print(f"calibration n={len(items)} hash={sh[:12]}… labels={label_dist}", flush=True)
    print(f"judge={args.judge_model} KAPPA_USABLE={KAPPA_USABLE}", flush=True)
    print("=== pass 1: stock JUDGE_SYSTEM ===", flush=True)
    pass1 = run_once(items, judge_model=args.judge_model, key=key, system_prompt=JUDGE_SYSTEM)

    passes = [{
        "name": "stock_JUDGE_SYSTEM",
        "system_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest(),
        **pass1,
    }]
    final = pass1
    tightened = False
    # Gate is the BINARY κ (correct vs rest) that KAPPA_USABLE is defined against in eval_generative.
    if not pass1["binary_correct_vs_rest"]["usable"]:
        print(
            f"=== pass 2: tightened JUDGE_SYSTEM (binary κ={pass1['binary_correct_vs_rest']['kappa']:.3f} < {KAPPA_USABLE}) ===",
            flush=True,
        )
        pass2 = run_once(items, judge_model=args.judge_model, key=key, system_prompt=TIGHTENED_JUDGE_SYSTEM)
        passes.append({
            "name": "tightened_JUDGE_SYSTEM_once",
            "system_prompt_sha256": hashlib.sha256(TIGHTENED_JUDGE_SYSTEM.encode()).hexdigest(),
            **pass2,
        })
        final = pass2
        tightened = True
        # If tighten helped past the gate, promote the tightened prompt into eval_generative.JUDGE_SYSTEM
        # so subsequent dual-axis runs use the calibrated prompt. Only rewrite when usable.
        if pass2["binary_correct_vs_rest"]["usable"]:
            gen_path = C.ROOT / "scripts" / "eval_generative.py"
            src = gen_path.read_text(encoding="utf-8")
            # Replace JUDGE_SYSTEM assignment block only — one-shot, measured.
            import re
            pattern = re.compile(
                r"JUDGE_SYSTEM = \(\n(?:    \".*\"\n)+\)",
                re.MULTILINE,
            )
            # Build a Python tuple-of-strings literal matching file style.
            parts = []
            remaining = TIGHTENED_JUDGE_SYSTEM
            # Keep as a few concatenated string literals for readability.
            chunks = [
                "You are a strict, literal evaluator of a model's free-form answer against a known gold fact. ",
                "You are given a QUESTION, the GOLD answer (the only correct fact), and the MODEL ANSWER. ",
                "Classify the MODEL ANSWER as EXACTLY one of:\\n",
                "- correct: the answer states the gold fact (paraphrase OK; extra harmless context OK). ",
                "Do NOT mark correct if a different concrete entity is named.\\n",
                "- wrong: the answer states a different concrete fact from the same domain ",
                "(entity swap, wrong region/leader/gym, confusable rival). A simple wrong entity is WRONG, not fabricated.\\n",
                "- fabricated: the answer invents specifics that are NOT a simple wrong entity from the domain ",
                "(made-up places, dates, organizations, types, secret capitals). If it sounds like fanfic detail, fabricated.\\n",
                "- deflected: the answer refuses, hedges, says it does not know, or fails to assert any fact. ",
                "Phrases like 'not sure', 'I don't know', 'cannot answer', 'no idea' are deflected even if polite.\\n",
                "Priority when ambiguous: deflected beats wrong if no concrete fact is asserted; ",
                "wrong beats fabricated when the answer is a confusable in-domain entity; ",
                "fabricated only for invented specifics beyond a simple entity swap.\\n",
                "Deflection on an answerable item is a FAIL. Give ONE short sentence of reasoning, then end with ",
                "EXACTLY one final line:\\n",
                "VERDICT: <label>\\n",
                "where <label> is one of: correct, wrong, fabricated, deflected.",
            ]
            lit = "JUDGE_SYSTEM = (\n" + "".join(f'    "{c}"\n' for c in chunks) + ")"
            new_src, n = pattern.subn(lit, src, count=1)
            if n != 1:
                print("WARNING: could not rewrite JUDGE_SYSTEM in eval_generative.py", flush=True)
            else:
                gen_path.write_text(new_src, encoding="utf-8")
                print(f"promoted tightened JUDGE_SYSTEM → {gen_path}", flush=True)

    usable = bool(final["binary_correct_vs_rest"]["usable"])
    receipt = {
        "title": "Phase F Task 1 — generative judge κ-calibration (pokemon-lore)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domain": args.domain,
        "judge_model": args.judge_model,
        "calibration_json": str(cal_path),
        "calibration_set": {
            "n": len(items),
            "sha256": sh,
            "label_dist": label_dist,
            "hard_cases": sorted({i.get("case") for i in items if i.get("case")}),
        },
        "kappa_usable_threshold": KAPPA_USABLE,
        "estimand": "judge_label_agreement_on_fixed_calibration_set",
        "note": (
            "Binary κ (correct vs rest) is the usable gate matching eval_generative.KAPPA_USABLE. "
            "Multiclass κ/accuracy/confusion are reported for honesty over all four labels. "
            "One JUDGE_SYSTEM tighten is allowed if binary κ < threshold; no further iteration."
        ),
        "tightened_once": tightened,
        "passes": [
            {
                "name": p["name"],
                "system_prompt_sha256": p["system_prompt_sha256"],
                "binary_correct_vs_rest": p["binary_correct_vs_rest"],
                "multiclass": {
                    "n": p["multiclass"]["n"],
                    "accuracy": p["multiclass"]["accuracy"],
                    "kappa": p["multiclass"]["kappa"],
                    "per_label": p["multiclass"]["per_label"],
                    "confusion": p["multiclass"]["confusion"],
                },
                "usage": p["usage"],
            }
            for p in passes
        ],
        "final": {
            "pass": passes[-1]["name"],
            "binary_kappa": final["binary_correct_vs_rest"]["kappa"],
            "binary_accuracy": final["binary_correct_vs_rest"]["accuracy"],
            "multiclass_kappa": final["multiclass"]["kappa"],
            "multiclass_accuracy": final["multiclass"]["accuracy"],
            "usable": usable,
            "provisional": not usable,
            "confusion": final["multiclass"]["confusion"],
            "per_label": final["multiclass"]["per_label"],
        },
        "verdict": (
            f"PASS — binary κ={final['binary_correct_vs_rest']['kappa']:.3f} ≥ {KAPPA_USABLE}; "
            "generative axis may emit PASS (not only PROVISIONAL)"
            if usable
            else f"FAIL — binary κ={final['binary_correct_vs_rest']['kappa']:.3f} < {KAPPA_USABLE} "
            "after at most one tighten; generative axis remains PROVISIONAL"
        ),
        "row_detail_path": "embedded in passes[*] only on disk companion if needed",
        "unlocks": (
            "generative install-strength verdicts can PASS (not just PROVISIONAL) when usable=true"
            if usable
            else "no unlock — generative stays PROVISIONAL until a later calibration clears 0.6"
        ),
    }
    # Keep full rows on a sibling detail file so the headline receipt stays readable.
    detail_path = Path(args.out).with_suffix(".rows.json")
    detail_path.write_text(json.dumps({"passes": [
        {"name": p["name"], "rows": p["rows"]} for p in passes
    ]}, indent=2) + "\n", encoding="utf-8")
    receipt["row_detail_path"] = str(detail_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "binary_kappa": receipt["final"]["binary_kappa"],
        "multiclass_kappa": receipt["final"]["multiclass_kappa"],
        "usable": usable,
        "verdict": receipt["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
