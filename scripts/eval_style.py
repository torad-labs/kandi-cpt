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
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval_stats import stamp_proportion


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
    if args.blind_ab_samples:
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
        "fidelity": fidelity_report(habits, samples),
        "blind_ab": (
            {
                "status": "MEASURED",
                "n": reveal["measured"]["total"],
                "win_rate_tuned": reveal["measured"]["win_rate"],
                "judge_calibration_stamp": reveal["measured"]["judge_calibration_stamp"],
                "voice_reveal_receipt": str(args.voice_reveal_out) if args.voice_reveal_out else None,
            }
            if reveal else pairwise_placeholder()
        ),
        "retention": retention_placeholder(),
        "verdicts": {
            "run_gate": "not evaluated here",
            "install_strength": (
                "PROVISIONAL — fidelity only" if args.fidelity_only
                else "PROVISIONAL — A/B + retention not run"
            ),
        },
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)
    if report["fidelity"]["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
