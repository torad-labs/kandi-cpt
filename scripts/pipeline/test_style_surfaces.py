#!/usr/bin/env python3
"""Unit proofs for style-surface card judging and response masking."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import style_surfaces as SS  # noqa: E402


def fingerprint() -> dict:
    return {
        "signature_openers": ["observation", "test"],
        "stats": {"habits": ["almost_never_uses_uppercase", "rarely_uses_exclamation", "no_emoji", "short_sentences"]},
    }


def test_register_judge_accepts_card_voice_and_rejects_anti_style():
    accepted, receipt = SS.judged_style_match(
        "observation: the value moved. next: repeat the control.", fingerprint(),
    )
    refused, _ = SS.judged_style_match("AMAZING!!! PERFECT RESULT 😀", fingerprint())
    assert accepted is True
    assert receipt["opener_match"] is True
    assert refused is False


def test_response_masking_uses_only_answer_tokens():
    class Encoded:
        def __init__(self, values): self.input_ids = values
    class Tokenizer:
        bos_token = "<bos>"
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, value, add_special_tokens=False):
            return Encoded(list(range(2, 2 + len(value.split()))))
    ids, labels = SS.encode(Tokenizer(), "a prompt", "observation: a result.", 32)
    first_supervised = next(index for index, value in enumerate(labels) if value != -100)
    assert first_supervised > 0
    assert all(value == -100 for value in labels[:first_supervised])
    assert len(ids) == len(labels) == 32


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} STYLE-SURFACE TESTS PASS")
