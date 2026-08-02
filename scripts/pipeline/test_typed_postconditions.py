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
    assert "tokenize_stats_sound" in _kinds("tokenize"), _kinds("tokenize")
    assert "split_integrity" in _kinds("split"), _kinds("split")
    assert "probe_quarantine_sound" in _kinds("probe_quarantine"), _kinds("probe_quarantine")
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

        good["n_unique_triple_ids"] = 100                      # every id assigned
        path.write_text(json.dumps(good), encoding="utf-8")
        assert run_stage._split_integrity_ok(path) is True, "the healthy split must pass"

        # A legacy split predating the denominator must not be retro-failed.
        legacy = {k: v for k, v in good.items() if k != "n_unique_triple_ids"}
        path.write_text(json.dumps(legacy), encoding="utf-8")
        assert run_stage._split_integrity_ok(path) is True

        for label, mutate in (
            ("holdout leaks into train", lambda d: d.update(train_ids=d["train_ids"] + ["h0"])),
            ("empty holdout", lambda d: d.update(holdout_ids=[])),
            ("alias leaks present", lambda d: d["decontamination"].update(n_alias_leaks=3, zero_alias_leaks=False)),
            ("decontamination block absent", lambda d: d.pop("decontamination")),
            ("realized fraction far from declared", lambda d: d.update(holdout_frac=0.5)),
            # Population coverage, newly measurable now that split records its denominator: facts
            # the split never classified are invisible to evaluation entirely.
            ("facts left unclassified", lambda d: d.update(n_unique_triple_ids=140)),
        ):
            broken = json.loads(json.dumps(good))
            mutate(broken)
            path.write_text(json.dumps(broken), encoding="utf-8")
            assert run_stage._split_integrity_ok(path) is False, f"accepted a split with: {label}"


def test_probe_quarantine_sound_requires_a_measured_nonblind_report() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "report_domain.json"
        path.write_text(json.dumps({
            "schema": "probe_quarantine/v1",
            "units_scanned": 1,
            "units_measured": 1,
            "units_instrument_blind": 0,
            "quarantined_sources": 2,
        }), encoding="utf-8")
        assert run_stage._probe_quarantine_ok(path)

        blind = json.loads(path.read_text(encoding="utf-8"))
        blind["units_measured"] = 0
        blind["units_instrument_blind"] = 1
        path.write_text(json.dumps(blind), encoding="utf-8")
        assert not run_stage._probe_quarantine_ok(path)


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


def test_checkpoint_recipe_catches_a_short_run_and_a_wrong_precision() -> None:
    """A checkpoint that stopped early is byte-identical in SHAPE to one that finished.

    Same files, same manifest, same tensor set — so "the artifact is valid" cannot answer "is this
    the run we approved". run_contract.json is the trainer's own record of what it did.
    """
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "ckpt"
        directory.mkdir()
        spec = {"expect_precision": "fp4", "expect_int8_master": "storage"}

        def write(**contract):
            (directory / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")

        write(final_step=450, max_steps=450, precision="fp4", int8_master="storage")
        assert run_stage._checkpoint_recipe_ok(directory, spec) is True, "the healthy run must pass"

        # Stopped early with NO recorded reason — the case this exists for.
        write(final_step=120, max_steps=450, precision="fp4", int8_master="storage")
        assert run_stage._checkpoint_recipe_ok(directory, spec) is False

        # Stopped early WITH a recorded reason is legitimate.
        write(final_step=120, max_steps=450, precision="fp4", int8_master="storage",
              scored_checkpoint="anneal-end")
        assert run_stage._checkpoint_recipe_ok(directory, spec) is True

        # Ran at a precision other than the approved one.
        write(final_step=450, max_steps=450, precision="bf16", int8_master="storage")
        assert run_stage._checkpoint_recipe_ok(directory, spec) is False
        write(final_step=450, max_steps=450, precision="fp4", int8_master="off")
        assert run_stage._checkpoint_recipe_ok(directory, spec) is False

        # A run that cannot say where it stopped is not verifiable.
        write(max_steps=450, precision="fp4", int8_master="storage")
        assert run_stage._checkpoint_recipe_ok(directory, spec) is False


def test_tokenize_stats_reconciles_requested_against_realized_replay() -> None:
    """"replay_frac was requested" is not "replay_frac was delivered".

    The packer truncates its replay pool to a token target, so a shard can silently under-deliver
    while its config still claims the requested fraction.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = run_stage.C.ROOT
        run_stage.C.ROOT = root
        try:
            stats = root / "s_stats.json"
            spec = {"check": "tok", "kind": "tokenize_stats_sound", "path": "s_stats.json",
                    "require_lever_metadata": True}

            def write(**fields):
                stats.write_text(json.dumps(fields), encoding="utf-8")

            write(num_chunks=1480, lever_metadata=True,
                  requested_final_replay_fraction=0.0909, realized_final_replay_fraction=0.0907)
            assert run_stage._postcondition(spec)["pass"] is True

            # Silently under-delivered replay — the defect.
            write(num_chunks=1480, lever_metadata=True,
                  requested_final_replay_fraction=0.0909, realized_final_replay_fraction=0.0400)
            assert run_stage._postcondition(spec)["pass"] is False

            # Stamped only on one side cannot be reconciled.
            write(num_chunks=1480, lever_metadata=True, requested_final_replay_fraction=0.0909)
            assert run_stage._postcondition(spec)["pass"] is False

            # No replay requested and none delivered is fine; none requested but tokens present is not.
            write(num_chunks=1480, lever_metadata=True)
            assert run_stage._postcondition(spec)["pass"] is True
            write(num_chunks=1480, lever_metadata=True, replay_tokens=34542)
            assert run_stage._postcondition(spec)["pass"] is False

            # Zero blocks and missing lever metadata each fail on their own.
            write(num_chunks=0, lever_metadata=True)
            assert run_stage._postcondition(spec)["pass"] is False
            write(num_chunks=1480)
            assert run_stage._postcondition(spec)["pass"] is False
        finally:
            run_stage.C.ROOT = original


def test_fact_concentration_catches_a_one_relation_graph():
    """A graph that collapses onto two predicates must fail at STRUCTURED, not sail to training.

    Every other balance instrument here reads the hydrated layer, where ~43 surfaces per fact
    manufacture the diversity signal those instruments measure. Measured on the real tree: EDC is
    95.53% two predicates (in_lineup 59.60 + plays_at 35.93) and still produced 347,797 distinct
    six-token openings. This is the only check that can see the difference.
    """
    import json as _json
    from pathlib import Path as _Path
    import run_stage as R

    with tempfile.TemporaryDirectory() as tmp:
        root = _Path(tmp)

        def write(name, rows):
            f = root / name
            f.write_text("\n".join(_json.dumps({"predicate": p}) for p in rows) + "\n")
            return f

        # 95% of rows in two predicates — the EDC shape, scaled down.
        degenerate = write("degenerate.jsonl", ["in_lineup"] * 60 + ["plays_at"] * 35
                           + [f"other_{i}" for i in range(5)])
        # A spread graph: no two predicates dominate.
        healthy = write("healthy.jsonl", [f"rel_{i % 10}" for i in range(100)])

        cap = {"expect_max_top_share": 0.85, "top_n_predicates": 2}
        assert not R._fact_concentration_ok([degenerate], cap), "a 95% two-predicate graph passed"
        assert R._fact_concentration_ok([healthy], cap), "a spread graph was wrongly rejected"

        # It must be able to go red on the healthy corpus too, or the threshold is decorative.
        assert not R._fact_concentration_ok([healthy], {"expect_max_top_share": 0.10,
                                                        "top_n_predicates": 2})

        # Fails CLOSED on every "I could not tell": no files, undeclared cap, nonsense cap,
        # and a file with no parseable predicate rows.
        assert not R._fact_concentration_ok([], cap), "no files read as a pass"
        assert not R._fact_concentration_ok([healthy], {"expect_max_top_share": None}), \
            "an undeclared cap silently exempted the domain"
        assert not R._fact_concentration_ok([healthy], {"expect_max_top_share": 1.5})
        empty = write("empty.jsonl", [])
        assert not R._fact_concentration_ok([empty], cap), "a corpus with no predicates passed"


if __name__ == "__main__":
    test_contract_emits_semantic_postconditions_for_verdict_stages()
    test_report_verdict_rejects_a_failing_report()
    test_jsonl_no_fail_verdict_and_json_field_positive()
    test_split_integrity_rejects_leaking_and_degenerate_splits()
    test_probe_quarantine_sound_requires_a_measured_nonblind_report()
    test_checkpoint_integrity_rejects_an_unloadable_checkpoint()
    test_checkpoint_recipe_catches_a_short_run_and_a_wrong_precision()
    test_tokenize_stats_reconciles_requested_against_realized_replay()
    test_fact_concentration_catches_a_one_relation_graph()
    print("OK — a FAIL verdict now fails its stage, and degenerate reports fail closed")
