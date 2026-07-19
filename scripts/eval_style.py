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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fingerprint", type=Path, required=True)
    ap.add_argument("--style-card", type=Path, default=None)
    ap.add_argument("--samples", type=Path, default=None,
                    help="jsonl with {text} rows to score fidelity against")
    ap.add_argument("--fidelity-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
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

    report = {
        "domain": fp.get("domain"),
        "fidelity": fidelity_report(habits, samples),
        "blind_ab": pairwise_placeholder(),
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
