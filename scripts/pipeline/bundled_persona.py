#!/usr/bin/env python3
"""Generate the bundled, explicitly fictional Mara Venn sample-persona corpus.

The output is deterministic and contains no donated or real-person data.  It is
checked in so the voice path can be demonstrated without asking an operator for
private exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as C

PERSONA = "Dr. Mara Venn"
OPENERS = ("observation", "hypothesis", "test", "result", "uncertainty")
SUBJECTS = (
    "copper sensor", "glass chamber", "field notebook", "north antenna", "sealed reference",
    "river sample", "thermal plate", "quiet motor", "blue reagent", "calibration clock",
    "garden soil", "weather mast", "spectral lamp", "control vial", "archive drawer",
    "battery pack", "rain gauge", "ceramic probe", "wind tunnel", "laser mount",
    "coffee timer", "door hinge", "desk fan", "paper filter", "sample tray",
    "signal cable", "pressure valve", "cooling loop", "light meter", "brass scale",
)
CONDITIONS = (
    "after dusk", "under a steady load", "when the room warms", "during the second trial",
    "near the open window", "after a careful rinse", "when the battery falls", "before sunrise",
    "under blue light", "during a quiet interval", "after three minutes", "when humidity rises",
    "beside the control sample", "after the motor stops", "during the baseline run",
    "when the table vibrates", "after the lid closes", "under a weak signal", "before calibration",
    "after the first reading", "when the air cools", "during the repeat test", "under low pressure",
    "after the cable moves", "when the filter dries", "beside the north wall", "during light rain",
    "after the reference shifts", "when the valve opens", "before the final note",
    "during the noon check", "after the tray settles", "under constant current", "when dust gathers",
    "after the clock resets", "during the blank run", "when the fan slows", "before the sample warms",
    "after the second rinse", "under the same geometry", "when the lamp flickers", "during the control pass",
    "after the hinge turns", "when the gauge steadies", "before the next claim", "under a clean cover",
    "during the short pause", "after the probe cools", "when the signal returns", "before changing the setup",
    "after the note is copied", "under a sealed lid", "when the wind changes", "during the third reading",
    "after the scale zeros", "when the loop warms", "before the data review", "under a known weight",
    "during the evening check", "after the baseline repeats", "when the plate cools", "before the next test",
    "under a dry cloth", "during the slow sweep", "after the vial clears", "when the motor hums",
    "before the room opens", "under a measured distance", "during the final control", "after the rain ends",
)
ACTIONS = (
    "compare it with the sealed reference", "repeat the measurement before changing anything",
    "record the raw value and its uncertainty", "hold one variable fixed", "run the blank control first",
    "check the cable before blaming the sensor", "keep the baseline beside the claim",
    "separate the observation from the inference", "measure twice with the same geometry",
    "write the failed trial into the notebook", "reduce the claim to what the data supports",
    "inspect the control before tuning the model", "leave the outlier visible",
    "change one condition and repeat", "mark the missing reading as missing",
)
PROMPTS = (
    "what should we check next?", "can we trust this reading?", "why did the result move?",
    "how would you test that?", "what belongs in the note?", "is the claim ready?",
    "what should stay fixed?", "how do we handle the outlier?", "what failed here?",
    "what is the smallest useful next step?", "should we change the setup?", "how certain are we?",
)


def response(index: int, subject: str, condition: str) -> str:
    opener = OPENERS[index % len(OPENERS)]
    action = ACTIONS[(index * 7) % len(ACTIONS)]
    return f"{opener}: the {subject} changes {condition}. next: {action}."


def repo_path(path: Path) -> str:
    """Keep the checked-in bundle manifest portable across engine roots."""
    return str(path.relative_to(C.ROOT))


def generate(domain: str, count: int = 2100) -> dict:
    if count < 2000:
        raise ValueError("the bundled flagship corpus must contain at least 2,000 fictional messages")
    inbox = C.DATA / domain / "style_inbox"
    eval_dir = C.DATA / domain / "eval"
    inbox.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = inbox / "mara_venn_fictional_chat.jsonl"
    rows = []
    for index in range(count):
        subject = SUBJECTS[index % len(SUBJECTS)]
        condition = CONDITIONS[(index * 11) % len(CONDITIONS)]
        rows.append({
            "message_id": f"mara-{index + 1:04d}",
            "persona": PERSONA,
            "fictional": True,
            "admission_basis": "bundled-fictional-own-data",
            "prompt": PROMPTS[(index * 5) % len(PROMPTS)],
            "text": response(index, subject, condition),
        })
    corpus_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    # Everyday, deliberately non-scientific topics: the persona is a terse scientist, so these shift
    # AWAY from the trained domain — that is the whole point of the retention estimand. The set is
    # authored blind (never selected by how the tuned model scored on them); it was widened from 12 to
    # 32 on 2026-07-24 because retention at n=12 could not clear its own 0.80 bar by Wilson lower
    # bound (12/12 → 0.758) — a sample-size limit, not a measured weakness.
    eval_topics = (
        "plan a quiet weekend", "decline a meeting", "explain a delayed train", "choose a soup",
        "ask a neighbor for help", "summarize a mystery novel", "pack for a cold walk", "repair a loose shelf",
        "compare two pencils", "describe a rainy morning", "organize a small kitchen", "review a board game",
        "return a wrong parcel", "pick a birthday card", "water a neighbour's plants", "wait for a late bus",
        "fold a fitted sheet", "sort a drawer of cables", "queue at a post office", "rearrange two chairs",
        "label a spice jar", "hang a small picture", "buy stamps", "brew tea for two",
        "find a lost glove", "wrap an awkward gift", "sweep a narrow hallway", "choose between two coats",
        "replace a light bulb", "dry boots by a door", "tidy a bookshelf", "share an umbrella",
    )
    prompts = []
    for index, topic in enumerate(eval_topics):
        prompts.append({"id": f"ab-{index + 1:02d}", "kind": "blind_ab", "prompt": f"in your usual voice, {topic}."})
        prompts.append({"id": f"shift-{index + 1:02d}", "kind": "topic_shift", "prompt": f"answer briefly: {topic}."})
    prompt_path = eval_dir / "prompts.jsonl"
    prompt_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in prompts) + "\n", encoding="utf-8")

    calibration = []
    for index in range(20):
        target = rows[(index * 97) % len(rows)]["text"]
        anti = (
            "AMAZING NEWS!!! I absolutely love this incredible result 😀😀 "
            "Everything is definitely perfect and we should celebrate immediately!"
        )
        calibration.append({"id": f"cal-{index + 1:02d}", "target": target, "anti_style": anti})
    calibration_path = eval_dir / "judge_calibration.jsonl"
    calibration_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in calibration) + "\n", encoding="utf-8",
    )
    manifest = {
        "schema": "kandi.bundled_persona.v1",
        "domain": domain,
        "persona": PERSONA,
        "fictional": True,
        "real_person_data": False,
        "message_count": len(rows),
        "corpus": repo_path(corpus_path),
        "eval_prompts": repo_path(prompt_path),
        "judge_calibration": repo_path(calibration_path),
    }
    manifest_path = C.DATA / domain / "bundled_persona.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="mara-venn-sample")
    parser.add_argument("--count", type=int, default=2100)
    args = parser.parse_args()
    print(json.dumps(generate(args.domain, args.count), indent=2))


if __name__ == "__main__":
    main()
