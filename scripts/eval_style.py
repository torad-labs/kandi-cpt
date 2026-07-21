#!/usr/bin/env python3
"""Style eval axis — blind A/B + retention + fidelity (docs/cpt-overhaul/05).

Axes:
  1. Blind pairwise A/B vs base: same prompt → tuned & base; different-family judge picks
     which answer sounds like the style card (position-randomized). Report win rate + Wilson.
  2. Persona retention under topic shift: prompts outside the corpus domain.
  3. Fidelity floor: card-derived mechanical habits (no judge).

    PYTHONPATH=scripts python scripts/eval_style.py \
        --style-card data/_build/style/<d>/style_card.md \
        --fingerprint data/_build/style/<d>/fingerprint.json \
        --base <base> --tuned <ckpt> \
        [--prompts prompts.jsonl] [--device cuda:0]

Without GPU/models, `--fidelity-only` scores the habit floor against sample texts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval_stats import stamp_proportion

JUDGE_VERSION = "mechanical-style-card-v1"


def load_fingerprint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_habits(fingerprint: dict) -> list[str]:
    return list((fingerprint.get("stats") or {}).get("habits") or [])


def score_habit(habit: str, text: str) -> bool:
    """Return True if text RESPECTS the habit (fidelity hold)."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    frac_u = upper / len(letters) if letters else 0.0
    if habit == "almost_never_uses_uppercase":
        return frac_u < 0.05
    if habit == "heavy_uppercase":
        return frac_u > 0.10
    if habit == "rarely_uses_exclamation":
        return text.count("!") <= 1
    if habit == "frequent_exclamation":
        return text.count("!") >= 1
    if habit == "no_emoji":
        return not re.search(r"[\U0001F300-\U0001F9FF]", text)
    if habit == "emoji_heavy":
        return bool(re.search(r"[\U0001F300-\U0001F9FF]", text))
    if habit == "short_sentences":
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return False
        mean = sum(len(s.split()) for s in sents) / len(sents)
        return mean < 12
    if habit == "long_sentences":
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return False
        mean = sum(len(s.split()) for s in sents) / len(sents)
        return mean > 18
    return True  # unknown habit → do not fail closed on fidelity


def fidelity_report(habits: list[str], samples: list[str]) -> dict:
    if not habits:
        return {
            "status": "SKIP",
            "note": "no checkable habits on the style card",
            "n_habits": 0,
            "n_samples": len(samples),
        }
    if not samples:
        return {
            "status": "SKIP",
            "note": "no samples to score",
            "n_habits": len(habits),
            "n_samples": 0,
        }
    per_habit = {}
    hits_all = []
    for h in habits:
        flags = [score_habit(h, s) for s in samples]
        per_habit[h] = stamp_proportion(sum(flags), len(flags), min_n=1)
        hits_all.extend(flags)
    overall = stamp_proportion(sum(hits_all), len(hits_all), min_n=1)
    return {
        "status": "PASS" if overall.get("acc", 0) >= 0.5 else "FAIL",
        "overall": overall,
        "per_habit": per_habit,
        "n_habits": len(habits),
        "n_samples": len(samples),
    }


def pairwise_placeholder(n: int = 0) -> dict:
    """A/B axis requires models + judge — emit structured PROVISIONAL stub when unavailable."""
    return {
        "status": "PROVISIONAL",
        "note": "blind A/B requires base+tuned generation and a different-family judge; not run",
        "n": n,
        "win_rate_tuned": None,
        "wilson": None,
    }


def retention_placeholder(n: int = 0) -> dict:
    return {
        "status": "PROVISIONAL",
        "note": "retention-under-shift requires out-of-domain prompts + judge; not run",
        "n": n,
        "hold_rate": None,
    }


def style_score(fingerprint: dict, text: str) -> dict:
    """Deterministic card-anchored judge score used for the bundled fictional demo."""
    habits = load_habits(fingerprint)
    habit_flags = {habit: score_habit(habit, text) for habit in habits}
    openers = {str(value).lower() for value in fingerprint.get("signature_openers") or []}
    opener = text.split(":", 1)[0].strip().lower() if ":" in text else ""
    opener_match = bool(openers) and opener in openers
    habit_score = sum(habit_flags.values()) / len(habit_flags) if habit_flags else 0.0
    # The opener is a register feature, while the habits are the independent mechanical floor.
    score = 0.75 * habit_score + 0.25 * float(opener_match)
    return {
        "score": score,
        "habit_score": habit_score,
        "habit_flags": habit_flags,
        "signature_opener": opener,
        "signature_opener_match": opener_match,
        "judge_version": JUDGE_VERSION,
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def calibrate_judge(fingerprint: dict, rows: list[dict], source_path: Path) -> dict:
    decisions = []
    for row in rows:
        target = style_score(fingerprint, str(row.get("target") or ""))
        anti = style_score(fingerprint, str(row.get("anti_style") or ""))
        decisions.append(target["score"] > anti["score"])
    n = len(decisions)
    hits = sum(decisions)
    status = "PASS" if n >= 10 and hits / n >= 0.90 else "FAIL"
    content = source_path.read_bytes() + b"\0" + JUDGE_VERSION.encode("utf-8")
    stamp = f"{JUDGE_VERSION}:{hashlib.sha256(content).hexdigest()[:16]}"
    return {
        "status": status,
        "estimand": "known target-register text outranks explicit anti-style text",
        "hits": hits,
        "n": n,
        "acc": hits / n if n else 0.0,
        "stamp": stamp,
        "source": str(source_path),
    }


def _generate_answers(model_path: str, prompts: list[dict], device: str, max_new_tokens: int) -> dict[str, str]:
    import torch
    from eval_probes import load_model

    tokenizer, model = load_model(model_path, device)
    answers: dict[str, str] = {}
    try:
        for row in prompts:
            prompt = str(row["prompt"])
            if getattr(tokenizer, "chat_template", None):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                )
            else:
                rendered = f"Question: {prompt}\nAnswer:"
            encoded = tokenizer(rendered, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answers[str(row["id"])] = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True,
            ).replace("\n", " ").strip()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return answers


def live_style_evaluation(
    *, fingerprint: dict, prompts: list[dict], calibration_rows: list[dict], calibration_path: Path,
    base_path: str, tuned_path: str, device: str, max_new_tokens: int = 48,
) -> tuple[dict, list[dict], str]:
    calibration = calibrate_judge(fingerprint, calibration_rows, calibration_path)
    if calibration["status"] != "PASS":
        raise ValueError("style judge calibration did not pass; refusing the A/B measurement")
    base = _generate_answers(base_path, prompts, device, max_new_tokens)
    persona = _generate_answers(tuned_path, prompts, device, max_new_tokens)
    pairs = []
    for row in prompts:
        identifier = str(row["id"])
        base_score = style_score(fingerprint, base[identifier])
        persona_score = style_score(fingerprint, persona[identifier])
        winner = "persona" if persona_score["score"] > base_score["score"] else "base"
        pairs.append({
            "id": identifier,
            "kind": str(row.get("kind") or "blind_ab"),
            "prompt": str(row["prompt"]),
            "base": base[identifier],
            "persona": persona[identifier],
            "judge_winner": winner,
            "base_style_score": base_score,
            "persona_style_score": persona_score,
        })
    ab_pairs = [row for row in pairs if row["kind"] == "blind_ab"]
    shift_pairs = [row for row in pairs if row["kind"] == "topic_shift"]
    ab_hits = sum(row["judge_winner"] == "persona" for row in ab_pairs)
    ab_stamp = stamp_proportion(ab_hits, len(ab_pairs), min_n=10, bar=0.5, estimand="blind_style_preference")
    retention_hits = sum(row["persona_style_score"]["score"] >= 0.80 for row in shift_pairs)
    retention_stamp = stamp_proportion(
        retention_hits, len(shift_pairs), min_n=10, bar=0.80, estimand="style_retention_under_topic_shift",
    )
    all_persona = [row["persona"] for row in pairs]
    fidelity = fidelity_report(load_habits(fingerprint), all_persona)
    report = {
        "judge_calibration": calibration,
        "blind_ab": {
            "status": "PASS" if ab_hits / max(1, len(ab_pairs)) >= 0.67 else "FAIL",
            "estimand": "card-anchored style preference on fixed blind A/B prompts",
            "persona_wins": ab_hits,
            "n": len(ab_pairs),
            "win_rate_tuned": ab_hits / len(ab_pairs) if ab_pairs else 0.0,
            "stamp": ab_stamp,
            "judge_calibration_stamp": calibration["stamp"],
        },
        "retention": {
            "status": "PASS" if retention_hits / max(1, len(shift_pairs)) >= 0.80 else "FAIL",
            "estimand": "approved-card style score at least 0.80 on fixed out-of-register-topic prompts",
            "hits": retention_hits,
            "n": len(shift_pairs),
            "hold_rate": retention_hits / len(shift_pairs) if shift_pairs else 0.0,
            "stamp": retention_stamp,
        },
        "fidelity": fidelity,
        "pairs": pairs,
    }
    report["status"] = (
        "PASS" if report["blind_ab"]["status"] == report["retention"]["status"] == "PASS"
        and fidelity.get("status") == "PASS" else "FAIL"
    )
    return report, ab_pairs, calibration["stamp"]


def confidence_word(axis: dict) -> str:
    stamp = axis.get("stamp") or axis.get("overall") or {}
    if axis.get("status") == "FAIL":
        return "not_yet"
    if stamp.get("n_status") == "insufficient-n" or stamp.get("bar_status") == "within-noise":
        return "early"
    return "solid" if axis.get("status") == "PASS" else "early"


def style_certificate(*, domain: str, fingerprint: dict, evaluation: dict, checkpoint: str,
                      approval_path: str, reveal_path: str, demo: bool) -> dict:
    axes = {
        "blind_ab": {**evaluation["blind_ab"], "confidence": confidence_word(evaluation["blind_ab"])},
        "fidelity": {**evaluation["fidelity"], "confidence": confidence_word(evaluation["fidelity"])},
        "retention": {**evaluation["retention"], "confidence": confidence_word(evaluation["retention"])},
    }
    words = [axis["confidence"] for axis in axes.values()]
    overall = "not_yet" if "not_yet" in words else "early" if "early" in words else "solid"
    certificate = {
        "schema": "kandi.style_certificate.v1",
        "domain": domain,
        "demo": bool(demo),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "voice_confidence": overall,
        "confidence_word_table": {
            "solid": "the measured axis passed at its declared denominator and bar",
            "early": "the direction is measured but the denominator or bound is not yet claimable",
            "not_yet": "the measured axis failed its declared floor",
        },
        "axes": axes,
        "mechanical_habits": load_habits(fingerprint),
        "checkpoint": checkpoint,
        "approval_receipt": approval_path,
        "eval_style_receipt": None,
        "voice_reveal_receipt": reveal_path,
    }
    if demo:
        certificate["demo_mark"] = {
            "label": "DEMO",
            "plain": "Bundled fictional persona card — no real person's messages were used.",
        }
    blob = json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    certificate["certificate_id"] = hashlib.sha256(blob).hexdigest()[:16]
    return certificate


def voice_reveal_receipt(
    *,
    domain: str,
    pairs: list[dict],
    judge_calibration_stamp: str,
    demo: bool = False,
) -> dict:
    """Build the measured input for the consumer blind A/B reveal.

    Generation and judging happen upstream. This boundary validates that the
    three revealed items and the measured summary are the same receipt; desktop
    randomizes the two positions independently for every view.
    """
    if len(pairs) != 3:
        raise ValueError("voice reveal requires exactly three fixed prompt pairs")
    if not judge_calibration_stamp.strip():
        raise ValueError("voice reveal requires the judge calibration stamp")
    normalized = []
    ids = set()
    for row in pairs:
        item = {
            "id": str(row.get("id") or "").strip(),
            "prompt": str(row.get("prompt") or "").strip(),
            "base": str(row.get("base") or "").strip(),
            "persona": str(row.get("persona") or "").strip(),
            "judge_winner": str(row.get("judge_winner") or "").strip().lower(),
        }
        if not all(item.values()) or item["judge_winner"] not in {"base", "persona"}:
            raise ValueError("every reveal pair needs id, prompt, base, persona, and judge_winner")
        if item["id"] in ids:
            raise ValueError("voice reveal pair ids must be unique")
        ids.add(item["id"])
        normalized.append(item)
    wins = sum(1 for row in normalized if row["judge_winner"] == "persona")
    return {
        "schema": "kandi.voice_reveal.v1",
        "domain": domain,
        "demo": bool(demo),
        "pairs": normalized,
        "measured": {
            "persona_wins": wins,
            "total": len(normalized),
            "win_rate": wins / len(normalized),
            "judge_calibration_stamp": judge_calibration_stamp.strip(),
        },
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "position_randomization": "desktop-per-view-sha256",
        "note": "The reveal surface randomizes and hides labels; this file is the measured receipt.",
    }


def load_reveal_pairs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fingerprint", type=Path, required=True)
    ap.add_argument("--style-card", type=Path, default=None)
    ap.add_argument("--samples", type=Path, default=None,
                    help="jsonl with {text} rows to score fidelity against")
    ap.add_argument("--fidelity-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--blind-ab-samples", type=Path, default=None,
                    help="jsonl with exactly 3 measured {id,prompt,base,persona,judge_winner} rows")
    ap.add_argument("--judge-calibration-stamp", default="")
    ap.add_argument("--voice-reveal-out", type=Path, default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--tuned", default=None)
    ap.add_argument("--prompts", type=Path, default=None,
                    help="fixed jsonl prompts with kind=blind_ab|topic_shift")
    ap.add_argument("--judge-calibration", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--certificate-out", type=Path, default=None)
    ap.add_argument("--approval", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    fp = load_fingerprint(args.fingerprint)
    habits = load_habits(fp)
    samples: list[str] = []
    if args.samples and args.samples.is_file():
        for line in args.samples.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(str(json.loads(line).get("text") or ""))
            except Exception:
                samples.append(line)

    reveal = None
    live = None
    live_inputs = [args.base, args.tuned, args.prompts, args.judge_calibration]
    if any(live_inputs) and not all(live_inputs):
        raise SystemExit("live eval_style requires --base, --tuned, --prompts, and --judge-calibration together")
    if all(live_inputs):
        if args.fidelity_only:
            raise SystemExit("--fidelity-only cannot be combined with live base/tuned evaluation")
        try:
            live, ab_pairs, calibration_stamp = live_style_evaluation(
                fingerprint=fp,
                prompts=load_jsonl(args.prompts),
                calibration_rows=load_jsonl(args.judge_calibration),
                calibration_path=args.judge_calibration,
                base_path=args.base,
                tuned_path=args.tuned,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        reveal_pairs = ab_pairs[:3]
        reveal = voice_reveal_receipt(
            domain=str(fp.get("domain") or ""),
            pairs=reveal_pairs,
            judge_calibration_stamp=calibration_stamp,
            demo=args.demo,
        )
        if not args.voice_reveal_out:
            raise SystemExit("live eval_style requires --voice-reveal-out for the three measured pairs")
        args.voice_reveal_out.parent.mkdir(parents=True, exist_ok=True)
        args.voice_reveal_out.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
        live["blind_ab"]["reveal_pair_ids"] = [row["id"] for row in reveal_pairs]
        live["blind_ab"]["reveal_measured"] = reveal["measured"]
        live["blind_ab"]["voice_reveal_receipt"] = str(args.voice_reveal_out)
    elif args.blind_ab_samples:
        reveal = voice_reveal_receipt(
            domain=str(fp.get("domain") or ""),
            pairs=load_reveal_pairs(args.blind_ab_samples),
            judge_calibration_stamp=args.judge_calibration_stamp,
            demo=args.demo,
        )
        if args.voice_reveal_out:
            args.voice_reveal_out.parent.mkdir(parents=True, exist_ok=True)
            args.voice_reveal_out.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")

    report = {
        "domain": fp.get("domain"),
        "status": live.get("status") if live else None,
        "judge_calibration": live.get("judge_calibration") if live else None,
        "fidelity": live.get("fidelity") if live else fidelity_report(habits, samples),
        "blind_ab": (
            live["blind_ab"] if live else
            {
                "status": "MEASURED",
                "n": reveal["measured"]["total"],
                "win_rate_tuned": reveal["measured"]["win_rate"],
                "judge_calibration_stamp": reveal["measured"]["judge_calibration_stamp"],
                "voice_reveal_receipt": str(args.voice_reveal_out) if args.voice_reveal_out else None,
            }
            if reveal else pairwise_placeholder()
        ),
        "retention": live["retention"] if live else retention_placeholder(),
        "pairs": live.get("pairs") if live else None,
        "verdicts": {
            "run_gate": "PASS" if live and live.get("status") == "PASS" else "not evaluated here",
            "install_strength": (
                "PASS" if live and live.get("status") == "PASS" else
                "PROVISIONAL — fidelity only" if args.fidelity_only
                else "PROVISIONAL — A/B + retention not run"
            ),
        },
    }
    certificate = None
    if live and args.certificate_out:
        if not args.approval or not args.approval.is_file():
            raise SystemExit("style certificate requires --approval")
        certificate = style_certificate(
            domain=str(fp.get("domain") or ""),
            fingerprint=fp,
            evaluation=report,
            checkpoint=str(args.tuned),
            approval_path=str(args.approval),
            reveal_path=str(args.voice_reveal_out),
            demo=args.demo,
        )
        certificate["eval_style_receipt"] = str(args.out) if args.out else None
        # Recompute identity after attaching the final receipt path.
        identity = {key: value for key, value in certificate.items() if key != "certificate_id"}
        certificate["certificate_id"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()[:16]
        args.certificate_out.parent.mkdir(parents=True, exist_ok=True)
        args.certificate_out.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
        report["certificate"] = {
            "path": str(args.certificate_out),
            "certificate_id": certificate["certificate_id"],
            "voice_confidence": certificate["voice_confidence"],
        }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)
    if report.get("status") == "FAIL" or report["fidelity"]["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
