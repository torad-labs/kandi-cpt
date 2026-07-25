#!/usr/bin/env python3
"""Red-green proofs for educational-density scoring (no provider)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.score_educational as S  # noqa: E402


def _write_corpus(root: Path, docs: list[tuple[str, str]]) -> None:
    """docs = [(name, body), …] under data/dom/unit/10_clean/."""
    unit = root / "data" / "dom" / "unit" / "10_clean"
    unit.mkdir(parents=True)
    for name, body in docs:
        (unit / f"{name}.md").write_text(f"---\ntitle: {name}\n---\n{body}\n", encoding="utf-8")


def test_parse_score():
    assert S.parse_score("looks dense\nSCORE: 4\n") == 4
    assert S.parse_score("SCORE: 1") == 1
    assert S.parse_score("nope") is None


def test_heuristic_score_monotonic_in_length():
    short = S.heuristic_score("word " * 10, "facts")
    long = S.heuristic_score("word " * 900, "facts")
    assert short < long


def test_run_score_fixture_writes_sidecars_and_distribution():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        C.BUILD.mkdir(parents=True)
        _write_corpus(root, [
            ("a", "alpha " * 50),
            ("b", "beta " * 200),
            ("c", "gamma " * 10),
        ])
        # Patch iter_editions
        unit_dir = root / "data" / "dom" / "unit"
        original = S.iter_editions
        S.iter_editions = lambda *a, **k: [SimpleNamespace(slug="unit", dir=unit_dir)]
        try:
            fixture = {
                str((unit_dir / "10_clean" / "a.md").relative_to(root)): 5,
                str((unit_dir / "10_clean" / "b.md").relative_to(root)): 3,
                str((unit_dir / "10_clean" / "c.md").relative_to(root)): 1,
            }
            fix_path = root / "fix.json"
            fix_path.write_text(json.dumps(fixture), encoding="utf-8")
            out = S.run_score("dom", None, "facts", scores_json=fix_path)
            assert out["report"]["n_docs"] == 3
            assert out["report"]["score_distribution"]["5"] == 1
            assert out["report"]["score_distribution"]["1"] == 1
            loaded = S.load_educational_scores("dom")
            assert loaded is not None
            assert set(loaded.values()) == {5.0, 3.0, 1.0}
            # Idempotent cache hit on re-run
            out2 = S.run_score("dom", None, "facts", scores_json=fix_path)
            assert out2["report"]["cache_hits"] == 3
        finally:
            S.iter_editions = original


def test_load_educational_scores_missing():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        assert S.load_educational_scores("nope") is None


def test_style_mode_accepted():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        C.BUILD.mkdir(parents=True)
        _write_corpus(root, [("chat", "lol ok sure\nyeah\nnice")])
        unit_dir = root / "data" / "dom" / "unit"
        original = S.iter_editions
        S.iter_editions = lambda *a, **k: [SimpleNamespace(slug="unit", dir=unit_dir)]
        try:
            out = S.run_score("dom", None, "style", allow_heuristic=True)
            assert out["report"]["mode"] == "style"
            assert out["report"]["n_docs"] == 1
        finally:
            S.iter_editions = original




def _payload(content, reasoning=None):
    message = {"content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {"choices": [{"message": message}]}


def _stub_chat(payloads: list[dict]):
    """Drive S.chat over a scripted sequence of provider payloads; returns (result, calls)."""
    calls = {"n": 0}

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(_request, timeout=None):
        payload = payloads[min(calls["n"], len(payloads) - 1)]
        calls["n"] += 1
        return _Response(payload)

    original_urlopen = S.urllib.request.urlopen
    original_sleep = S.time.sleep
    S.urllib.request.urlopen = _urlopen
    S.time.sleep = lambda _seconds: None
    try:
        return S.chat("m", "k", "sys", "user"), calls["n"]
    finally:
        S.urllib.request.urlopen = original_urlopen
        S.time.sleep = original_sleep


def test_message_text_falls_back_to_reasoning():
    """The EDC/H3 failure: providers that answer only in the reasoning channel."""
    assert S.message_text(_payload("SCORE: 4")) == "SCORE: 4"
    assert S.message_text(_payload("", reasoning="SCORE: 2")) == "SCORE: 2"
    assert S.message_text(_payload(None, reasoning="SCORE: 5")) == "SCORE: 5"
    assert S.message_text(_payload("")) == ""


def test_chat_retries_an_empty_completion_instead_of_returning_it():
    """Red: `content or ""` returned "" as success -> `unparseable teacher score: ''`."""
    result, calls = _stub_chat([_payload(""), _payload("SCORE: 3")])
    assert result == "SCORE: 3", result
    assert calls == 2, f"the empty completion must be retried, saw {calls} call(s)"


def test_chat_reads_the_reasoning_channel_without_retrying():
    result, calls = _stub_chat([_payload("", reasoning="SCORE: 1")])
    assert result == "SCORE: 1", result
    assert calls == 1, "a reasoning-only answer is a real answer, not a retry"


def test_chat_surfaces_persistent_emptiness_as_a_call_failure():
    """An all-empty provider fails as a teacher-call error, never as a bogus grade."""
    try:
        _stub_chat([_payload("")])
    except RuntimeError as error:
        assert "empty completion" in str(error), str(error)
    else:
        raise AssertionError("persistent empty completions must raise, not return")


def test_score_one_still_rejects_unparseable_teacher_output():
    """The guard stays: a non-empty answer with no SCORE line is still a hard error."""
    try:
        S.score_one("facts", [], "text", teacher_fn=lambda _s, _u: "I decline to grade.")
    except ValueError as error:
        assert "unparseable" in str(error), str(error)
    else:
        raise AssertionError("unparseable teacher output must raise")


def test_rubric_prompt_carries_no_think_in_both_modes():
    """qwen3 spends the whole 200-token budget reasoning and never emits `SCORE: n` without it.

    Measured live on edc/zhuhai/2018 (2026-07-24): pure deliberation, no verdict, stage failed
    closed. audit_recall.py carries the same directive against the same model.
    """
    facts = S.rubric_prompt("facts", ["festival", "venue"], "some document text")
    style = S.rubric_prompt("style", [], "some chat text")
    assert facts.rstrip().endswith("/no_think"), facts[-80:]
    assert style.rstrip().endswith("/no_think"), style[-80:]


def test_default_judge_model_is_paid_qwen():
    """Free Nemotron was unusable live; default must match eval_generative (qwen3-32b)."""
    assert S.DEFAULT_JUDGE_MODEL == "qwen/qwen3-32b"
    assert "nemotron" not in S.DEFAULT_JUDGE_MODEL.lower()
    assert ":free" not in S.DEFAULT_JUDGE_MODEL


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} SCORE-EDUCATIONAL TESTS PASS")
