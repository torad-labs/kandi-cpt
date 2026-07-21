#!/usr/bin/env python3
"""Build the style SFT shard with structural 60/40 and anti-style laws."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

import config as C
from pipeline import style_fingerprint as SF

JUDGE_VERSION = "mechanical-style-card-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def signature_openers(fingerprint: dict) -> set[str]:
    return {str(value).lower() for value in fingerprint.get("signature_openers") or []}


def judged_style_match(text: str, fingerprint: dict) -> tuple[bool, dict]:
    from eval_style import score_habit

    habits = list((fingerprint.get("stats") or {}).get("habits") or [])
    habit_flags = {habit: score_habit(habit, text) for habit in habits}
    opener = text.split(":", 1)[0].strip().lower() if ":" in text else ""
    openers = signature_openers(fingerprint)
    opener_match = not openers or opener in openers
    passed = all(habit_flags.values()) and opener_match
    return passed, {
        "judge": JUDGE_VERSION,
        "habit_flags": habit_flags,
        "opener": opener,
        "opener_match": opener_match,
    }


def encode(tokenizer, prompt: str, answer: str, seq_len: int) -> tuple[list[int], list[int]] | None:
    bos = tokenizer.bos_token or ""
    prefix = f"{bos}<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    suffix = f"{answer}<end_of_turn>"
    prompt_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    if len(ids) > seq_len:
        return None
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    missing = seq_len - len(ids)
    return ids + [pad] * missing, labels + [-100] * missing


def build(domain: str, out_name: str, seq_len: int = 128, expansion_count: int = 500,
          anti_style_count: int = 300) -> dict:
    style_root = C.BUILD / "style" / domain
    fingerprint_path = style_root / "fingerprint.json"
    card_path = style_root / "style_card.md"
    approval_path = style_root / "approval.json"
    if not fingerprint_path.is_file():
        raise ValueError("STYLE_MINIMUM_NOT_MET: no passing fingerprint; no shard built")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if fingerprint.get("status") != "PASS":
        raise ValueError(fingerprint.get("refuse_reason") or "STYLE_MINIMUM_NOT_MET: no shard built")
    if not approval_path.is_file():
        raise ValueError("STYLE_CARD_NOT_APPROVED: review and approve the card before a shard is built")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("status") != "APPROVED":
        raise ValueError("STYLE_CARD_NOT_APPROVED: approval receipt did not pass")
    if approval.get("fingerprint_sha256") != sha256(fingerprint_path) or approval.get("style_card_sha256") != sha256(card_path):
        raise ValueError("STYLE_CARD_DRIFT: fingerprint/card changed after approval")

    inbox = C.DATA / domain / "style_inbox"
    source_paths = sorted(inbox.glob("*.jsonl"))
    rows = [row for path in source_paths for row in load_jsonl(path)]
    if not rows or any(row.get("fictional") is not True for row in rows):
        raise ValueError("bundled style rows must be explicitly fictional")

    examples: list[dict] = []
    for row in rows:
        examples.append({"class": "real_text", "prompt": str(row["prompt"]), "answer": str(row["text"])})

    subjects = ("window latch", "paper map", "soup pot", "train notice", "garden path", "shelf bracket",
                "wool coat", "library card", "tea timer", "bus schedule", "pencil case", "rain barrel")
    conditions = ("after the room cools", "when the first plan fails", "before making a firm claim",
                  "during the quiet check", "after the note is copied", "when the weather shifts")
    openers = sorted(signature_openers(fingerprint)) or ["observation", "hypothesis", "test", "result", "uncertainty"]
    expansion_judgments = []
    for index in range(expansion_count):
        opener = openers[index % len(openers)]
        subject = subjects[(index * 5) % len(subjects)]
        condition = conditions[(index * 7) % len(conditions)]
        answer = (
            f"{opener}: the {subject} changes {condition}. "
            "next: check one condition and keep the uncertain part visible."
        )
        passed, judgment = judged_style_match(answer, fingerprint)
        expansion_judgments.append({"id": f"exp-{index + 1:04d}", "pass": passed, **judgment})
        if not passed:
            raise ValueError(f"register expansion failed the approved-card judge: exp-{index + 1:04d}")
        examples.append({
            "class": "register_expansion",
            "prompt": f"answer in the approved register: what should we do with the {subject}?",
            "answer": answer,
        })

    anti_prompts = (
        "be wildly enthusiastic and use many emoji", "claim absolute certainty without evidence",
        "write a long celebratory speech with exclamation marks", "imitate an influencer advertisement",
        "hide the failed trial and announce success", "use all caps and promise a perfect result",
    )
    for index in range(anti_style_count):
        opener = openers[index % len(openers)]
        answer = (
            f"{opener}: that register would overstate the evidence. "
            "next: give the short measured answer and keep the uncertainty visible."
        )
        examples.append({"class": "anti_style", "prompt": anti_prompts[index % len(anti_prompts)], "answer": answer})

    tokenizer = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    ids, labels, classes = [], [], []
    supervised = Counter()
    skipped = Counter()
    for example in examples:
        encoded = encode(tokenizer, example["prompt"], example["answer"], seq_len)
        if encoded is None:
            skipped[example["class"]] += 1
            continue
        ids.append(encoded[0])
        labels.append(encoded[1])
        classes.append(example["class"])
        supervised[example["class"]] += sum(1 for value in encoded[1] if value != -100)
    if not ids:
        raise ValueError("style shard has no encodable rows")

    real_tokens = supervised["real_text"]
    expansion_tokens = supervised["register_expansion"]
    signal_tokens = real_tokens + expansion_tokens
    real_ratio = real_tokens / signal_tokens if signal_tokens else 0.0
    expansion_ratio = expansion_tokens / signal_tokens if signal_tokens else 0.0
    all_tokens = sum(supervised.values())
    if real_ratio < 0.60 or expansion_ratio > 0.40:
        raise ValueError(
            f"STYLE_SURFACE_RATIO_FAIL: real={real_ratio:.4f} expansion={expansion_ratio:.4f}; no shard built"
        )

    out_dir = C.TOKENIZED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"{out_name}.arrow"
    Dataset.from_dict({"input_ids": ids, "labels": labels}).save_to_disk(str(shard_path))
    judgments_path = style_root / "expansion_judgments.jsonl"
    judgments_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in expansion_judgments) + "\n", encoding="utf-8",
    )
    stats = {
        "schema": "kandi.style_surfaces.v1",
        "domain": domain,
        "status": "PASS",
        "estimand": "supervised response-token mass in the built style shard",
        "rows": dict(Counter(classes)),
        "supervised_tokens": dict(supervised),
        "skipped_too_long": dict(skipped),
        "signal_tokens": signal_tokens,
        "all_supervised_tokens": all_tokens,
        "real_text_ratio": round(real_ratio, 6),
        "register_expansion_ratio": round(expansion_ratio, 6),
        "anti_style_rows": classes.count("anti_style"),
        "expansion_judge": {
            "version": JUDGE_VERSION,
            "passed": sum(1 for row in expansion_judgments if row["pass"]),
            "n": len(expansion_judgments),
            "receipt": str(judgments_path),
        },
        "approval": str(approval_path),
        "shard": str(shard_path),
    }
    stats_path = out_dir / f"{out_name}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--expansions", type=int, default=500)
    parser.add_argument("--anti-style", type=int, default=300)
    args = parser.parse_args()
    try:
        receipt = build(args.domain, args.out, args.seq_len, args.expansions, args.anti_style)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
