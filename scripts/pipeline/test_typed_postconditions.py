#!/usr/bin/env python3
"""A stage that RAN is not a stage that PASSED — the run-gate half.

run_stage's generic checks are exit-zero + declared outputs exist + a nonzero output delta +
recipe identity held. None of that can see semantic failure: audit_recall writes
report_<domain>.json with status FAIL, exits 0, and the file it was supposed to produce is
right there and changed, so the stage records PASS. The readiness ladder was fixed to read
those verdicts; this is the same hole one layer down, where the stage record is minted.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_typed_postconditions.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stage
from execution_contract import build_contract


def _kinds(stage_id: str, domain: str = "pokemon-lore", editions: str = "kanto") -> list[str]:
    stage = next(s for s in build_contract(domain, editions)["stages"] if s["id"] == stage_id)
    return [str(p.get("kind")) for p in stage["postconditions"]]


def test_contract_emits_semantic_postconditions_for_verdict_stages() -> None:
    assert "report_verdict" in _kinds("audit_recall"), _kinds("audit_recall")
    assert "report_verdict" in _kinds("pii_scan"), _kinds("pii_scan")
    assert "jsonl_no_fail_verdict" in _kinds("audit"), _kinds("audit")
    assert "json_field_positive" in _kinds("tokenize"), _kinds("tokenize")
    assert "split_integrity" in _kinds("split"), _kinds("split")
    assert "checkpoint_integrity" in _kinds("cpt"), _kinds("cpt")

    # Contrast: a stage with no recorded verdict must NOT acquire one, or the emission rule is
    # just decorating everything and the assertions above prove nothing.
    assert _kinds("extract") == ["output_exists"] * len(_kinds("extract")), _kinds("extract")


def test_report_verdict_rejects_a_failing_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = run_stage.C.ROOT
        run_stage.C.ROOT = root
        try:
            report = root / "report.json"
            spec = {"check": "verdict", "kind": "report_verdict", "path": "report.json",
                    "field": "status", "expect": ["PASS"]}

            report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is True

            # The whole point: the file EXISTS and CHANGED, which is all the generic checks see.
            report.write_text(json.dumps({"status": "FAIL", "overall": {"ratio": 0.35}}), encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False

            # Fail closed on every degenerate shape rather than treating "unreadable" as "fine".
            report.write_text(json.dumps({"overall": {}}), encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "missing field must not pass"
            report.write_text("{not json", encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "unparseable must not pass"
            report.unlink()
            assert run_stage._postcondition(spec)["pass"] is False, "absent must not pass"
        finally:
            run_stage.C.ROOT = original


def test_jsonl_no_fail_verdict_and_json_field_positive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = run_stage.C.ROOT
        run_stage.C.ROOT = root
        try:
            rows = root / "rows.jsonl"
            spec = {"check": "audit", "kind": "jsonl_no_fail_verdict", "path": "rows.jsonl",
                    "field": "verdicts"}
            rows.write_text(
                json.dumps({"editions": "a", "verdicts": {"balance": "pass", "phrasing": "warn"}}) + "\n",
                encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is True, "warn is not fail"
            rows.write_text(
                json.dumps({"editions": "a", "verdicts": {"balance": "pass"}}) + "\n"
                + json.dumps({"editions": "b", "verdicts": {"coverage": "fail"}}) + "\n",
                encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "a fail in ANY row must fail"
            rows.write_text("", encoding="utf-8")
            assert run_stage._postcondition(spec)["pass"] is False, "no rows is not a pass"

            # The audit file is shared across domains and keyed only by edition slug, so the check
            # must be scoped to the editions the stage ran. Measured, not hypothetical: unscoped,
            # millbrook/main failed because a booster_1 row in the same file had coverage=fail.
            rows.write_text(
                json.dumps({"editions": "main", "verdicts": {"balance": "pass"}}) + "\n"
                + json.dumps({"editions": "booster_1", "verdicts": {"coverage": "fail"}}) + "\n",
                encoding="utf-8")
            assert run_stage._postcondition({**spec, "editions": ["main"]})["pass"] is True
            assert run_stage._postcondition({**spec, "editions": ["booster_1"]})["pass"] is False
            assert run_stage._postcondition({**spec, "editions": ["absent"]})["pass"] is False, (
                "no row for the scoped editions means no verdict, which is not a pass"
            )

            stats = root / "stats.json"
            positive = {"check": "blocks", "kind": "json_field_positive", "path": "stats.json",
                        "field": "num_chunks"}
            stats.write_text(json.dumps({"num_chunks": 1480}), encoding="utf-8")
            assert run_stage._postcondition(positive)["pass"] is True
            stats.write_text(json.dumps({"num_chunks": 0}), encoding="utf-8")
            assert run_stage._postcondition(positive)["pass"] is False, "zero blocks is a failed tokenize"
        finally:
            run_stage.C.ROOT = original


def _write_safetensors(path: Path, names: list[str]) -> None:
    """A minimal real safetensors file: 8-byte LE header length, JSON header, then the data block."""
    header = {name: {"dtype": "I8", "shape": [1], "data_offsets": [i, i + 1]}
              for i, name in enumerate(names)}
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + bytes(len(names)))


def test_split_integrity_rejects_leaking_and_degenerate_splits() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = {"version": "v2", "holdout_frac": 0.1,
                "train_ids": [f"t{i}" for i in range(90)],
                "holdout_ids": [f"h{i}" for i in range(10)],
                "decontamination": {"n_alias_leaks": 0, "zero_alias_leaks": True}}
        path = root / "split_v2.json"

        path.write_text(json.dumps(good), encoding="utf-8")
        assert run_stage._split_integrity_ok(path) is True, "the healthy split must pass"

        for label, mutate in (
            ("holdout leaks into train", lambda d: d.update(train_ids=d["train_ids"] + ["h0"])),
            ("empty holdout", lambda d: d.update(holdout_ids=[])),
            ("alias leaks present", lambda d: d["decontamination"].update(n_alias_leaks=3, zero_alias_leaks=False)),
            ("decontamination block absent", lambda d: d.pop("decontamination")),
            ("realized fraction far from declared", lambda d: d.update(holdout_frac=0.5)),
        ):
            broken = json.loads(json.dumps(good))
            mutate(broken)
            path.write_text(json.dumps(broken), encoding="utf-8")
            assert run_stage._split_integrity_ok(path) is False, f"accepted a split with: {label}"


def test_checkpoint_integrity_rejects_an_unloadable_checkpoint() -> None:
    """The generic delta check sees "the directory changed" — a touched log satisfies that.

    This is the guard against the failure that actually happened here: 22 checkpoints converted and
    byte-verified, 181GB of originals deleted, four commits shipped — and the format was unloadable.
    Bytes moving is not an artifact working.
    """
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "ckpt"
        directory.mkdir()

        def rebuild(tensors: list[str], keys: list[str]) -> None:
            _write_safetensors(directory / "model.safetensors", tensors)
            (directory / "quant_keys.json").write_text(
                json.dumps({"method": "int8_rowwise_v2", "keys": keys, "embedding_keys": []}),
                encoding="utf-8")
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "tokenizer.json").write_text("{}", encoding="utf-8")

        rebuild(["a.weight", "a.weight.weight_scale"], ["a.weight"])
        assert run_stage._checkpoint_integrity_ok(directory) is True, "the healthy checkpoint must pass"

        # quant_keys names a tensor the file does not hold
        rebuild(["a.weight", "a.weight.weight_scale"], ["a.weight", "b.weight"])
        assert run_stage._checkpoint_integrity_ok(directory) is False, "accepted a manifest/tensor mismatch"

        # the tensor is there but its scale is not — half-converted
        rebuild(["a.weight"], ["a.weight"])
        assert run_stage._checkpoint_integrity_ok(directory) is False, "accepted a weight with no scale"

        rebuild(["a.weight", "a.weight.weight_scale"], ["a.weight"])
        (directory / "model.safetensors.partial").write_text("x", encoding="utf-8")
        assert run_stage._checkpoint_integrity_ok(directory) is False, "accepted a half-written checkpoint"
        (directory / "model.safetensors.partial").unlink()

        (directory / "tokenizer.json").unlink()
        assert run_stage._checkpoint_integrity_ok(directory) is False, "accepted a checkpoint with no tokenizer"
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")

        (directory / "model.safetensors").write_bytes(b"\xff" * 4)
        assert run_stage._checkpoint_integrity_ok(directory) is False, "accepted an unparseable header"


if __name__ == "__main__":
    test_contract_emits_semantic_postconditions_for_verdict_stages()
    test_report_verdict_rejects_a_failing_report()
    test_jsonl_no_fail_verdict_and_json_field_positive()
    test_split_integrity_rejects_leaking_and_degenerate_splits()
    test_checkpoint_integrity_rejects_an_unloadable_checkpoint()
    print("OK — a FAIL verdict now fails its stage, and degenerate reports fail closed")
