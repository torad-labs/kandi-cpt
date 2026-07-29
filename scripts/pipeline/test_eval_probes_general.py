#!/usr/bin/env python3
"""Fast regression tests for the generic base-vs-tuned CPT receipt."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_probes as E  # noqa: E402
from pipeline import probe_quarantine as Q  # noqa: E402


def _write_quarantine(root: Path, source_ids, *, instrument_blind: bool = False,
                      domain: str = "pokemon-eval", unit: str = "kanto") -> Path:
    """A quarantine record in the shape `probe_quarantine.run_scan` writes."""
    E.C.ROOT = root
    path = Q.quarantine_path(domain, unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": Q.SCHEMA,
        "domain": domain,
        "unit": unit,
        "split_version": "v1",
        "instrument_blind": instrument_blind,
        "floor_channel": Q.FLOOR_CHANNEL_UNREADABLE if instrument_blind else Q.FLOOR_CHANNEL_GENERIC,
        "counts": {"quarantined_sources": len(source_ids)},
        "quarantined": [
            {
                "source_triple_id": source_id,
                "reason": Q.REASON_FLOOR_PAIR,
                "entity_class_pair": ["a", "b"],
                "train_witness_triple_ids": ["train-witness"],
            }
            for source_id in source_ids
        ],
    }, indent=2) + "\n", encoding="utf-8")
    return path


def test_scoring_prompt_uses_explicit_cpt_cloze_without_instruction_wrapper():
    probe = {"question": "Where?", "prompt": "Elowen Dask teaches at"}
    assert E.scoring_prompt(probe) == "Elowen Dask teaches at"
    assert E.scoring_prompt({"question": "Where?"}) == "Question: Where?\nAnswer:"


def _sha(path: Path) -> str:
    return E._sha256(path)


def _corpus(root: Path, n: int = 72) -> E.ProbeCorpus:
    unit = root / "data" / "pokemon-eval" / "kanto"
    probe_path = unit / "probes.jsonl"
    split_path = unit / "splits" / "split_v1.json"
    stats_path = root / "data" / "_build" / "30_tokenized" / "pokemon_eval_stats.json"
    probe_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    stats_path.parent.mkdir(parents=True)
    probes = [
        {
            "question": f"Where is subject {index}?",
            "choices": ["a", "b", "c", "d"],
            "answer": "a",
            "predicate": "located_in" if index % 2 else "member_of",
            "subject": f"subject-{index}",
            "source_triple_id": f"id-{index}",
            "split": "held_out",
            "split_version": "v1",
        }
        for index in range(n)
    ]
    probe_path.write_text("\n".join(json.dumps(probe) for probe in probes) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps({
        "version": "v1",
        "train_ids": [],
        "holdout_ids": [f"id-{index}" for index in range(n)],
    }), encoding="utf-8")
    stats_path.write_text(json.dumps({"tokens_by_tag": {"clean": 1000, "hydrated": 2000}}), encoding="utf-8")
    return E.ProbeCorpus(
        domain="pokemon-eval",
        editions=["kanto"],
        probes=probes,
        probe_paths=[probe_path],
        probe_hashes=[_sha(probe_path)],
        probe_sha256=E._combined_hash([probe_path]),
        split_paths=[split_path],
        split_hashes=[_sha(split_path)],
        split_sha256=E._combined_hash([split_path]),
        split_version="v1",
        token_stats_path=stats_path,
        token_stats_sha256=_sha(stats_path),
        clean_tokens=1000,
        measurement_class="augmentation_holdout",
        chance=0.25,
    )


def _load_written_corpus(root: Path) -> E.ProbeCorpus:
    unit_dir = root / "data" / "pokemon-eval" / "kanto"
    original_iter_editions = E.iter_editions
    E.iter_editions = lambda *_args, **_kwargs: [SimpleNamespace(slug="kanto", dir=unit_dir)]
    try:
        return E.load_probe_corpus(
            "pokemon-eval",
            "kanto",
            "data/_build/30_tokenized/pokemon_eval.arrow",
        )
    finally:
        E.iter_editions = original_iter_editions


def _checkpoint(root: Path, run_name: str = "fresh") -> Path:
    base = root / "models" / "base"
    base.mkdir(parents=True, exist_ok=True)
    base.joinpath("model.safetensors").write_bytes(b"base-model-receipt")
    checkpoint = root / "checkpoints" / "cpt" / run_name
    checkpoint.mkdir(parents=True)
    checkpoint.joinpath("model.safetensors").write_bytes(b"real-model-receipt")
    checkpoint.joinpath("run_contract.json").write_text(json.dumps({
        "run_name": run_name,
        "fingerprint": "fixture-fingerprint",
        "tokenized": "data/_build/30_tokenized/pokemon_eval.arrow",
    }), encoding="utf-8")
    return checkpoint


def _config(root: Path) -> Path:
    path = root / "configs" / "eval.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("mode: cpt\n", encoding="utf-8")
    return path


def test_pilot_gate_passes_only_with_a_clear_delta():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root)
        checkpoint = _checkpoint(root)
        report = E.build_report(
            run_name="fresh",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[index < 19 for index in range(72)],
            tuned_hits=[index < 45 for index in range(72)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["status"] == "PASS"
        assert report["measurement_class"] == "augmentation_holdout"
        assert report["base"]["hits"] == 19 and report["tuned"]["hits"] == 45
        assert report["delta_acc"] > 0.36
        assert all(check["pass"] for check in report["checks"])
        assert report["inputs"]["probe_sha256"] == report["base"]["probe_sha256"]
        assert report["base"]["probe_sha256"] == report["tuned"]["probe_sha256"]
        assert len(report["probe_results"]) == 72
        assert report["probe_results"][0] == {
            "source_triple_id": "id-0",
            "probe_surface_id": None,
            "subject": "subject-0",
            "predicate": "member_of",
            "question": "Where is subject 0?",
            "answer": "a",
            "base_correct": True,
            "tuned_correct": True,
        }


def test_pilot_gate_fails_closed_on_small_n_and_weak_delta():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=20)
        checkpoint = _checkpoint(root, "small")
        report = E.build_report(
            run_name="small",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[index < 5 for index in range(20)],
            tuned_hits=[index < 8 for index in range(20)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["status"] == "FAIL"
        assert any("denominator" in failure for failure in report["failed_checks"])
        assert any("accuracy delta" in failure for failure in report["failed_checks"])


def test_gate_uses_independent_facts_not_repeated_probe_surfaces():
    """Sixty prompt rows over fifteen facts are still an underpowered fifteen-fact eval."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=60)
        for index, probe in enumerate(corpus.probes):
            probe["source_triple_id"] = f"fact-{index // 4}"
        checkpoint = _checkpoint(root, "clustered")
        report = E.build_report(
            run_name="clustered",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[False] * 60,
            tuned_hits=[True] * 60,
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["status"] == "FAIL"
        assert report["tuned"]["n"] == 60
        assert report["tuned"]["n_independent_facts"] == 15
        assert report["tuned"]["n_status"] == "ok"
        assert report["tuned"]["independent_fact_n_status"] == "insufficient-n"
        assert any("independent fact denominator" in failure for failure in report["failed_checks"])


def test_probe_loader_binds_every_item_to_the_hashed_holdout():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        _corpus(root)
        corpus = _load_written_corpus(root)
        assert len(corpus.probes) == 72
        assert {probe["source_triple_id"] for probe in corpus.probes} == {
            f"id-{index}" for index in range(72)
        }


def test_probe_loader_names_trained_fact_recall_without_clean_tokens():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=4)
        for probe in corpus.probes:
            probe["split"] = "trained"
            probe["split_version"] = "booster_v1"
        corpus.probe_paths[0].write_text(
            "\n".join(json.dumps(probe) for probe in corpus.probes) + "\n",
            encoding="utf-8",
        )
        booster_split = corpus.split_paths[0].with_name("split_booster_v1.json")
        booster_split.write_text(json.dumps({
            "version": "booster_v1",
            "train_ids": [f"id-{index}" for index in range(4)],
            "holdout_ids": [],
        }), encoding="utf-8")
        corpus.token_stats_path.write_text(
            json.dumps({"tokens_by_tag": {"clean": 0, "hydrated": 1000}}),
            encoding="utf-8",
        )

        loaded = _load_written_corpus(root)
        assert loaded.measurement_class == "trained_fact_recall"
        assert loaded.clean_tokens == 0
        assert len(loaded.probes) == 4


def test_probe_loader_rejects_unbound_and_duplicate_sources():
    for source_ids in (["not-held-out", "id-1"], ["id-1", "id-1"]):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            E.C.ROOT = root
            corpus = _corpus(root, n=2)
            probes = corpus.probes
            for probe, source_id in zip(probes, source_ids, strict=True):
                probe["source_triple_id"] = source_id
            corpus.probe_paths[0].write_text(
                "\n".join(json.dumps(probe) for probe in probes) + "\n",
                encoding="utf-8",
            )
            try:
                _load_written_corpus(root)
            except ValueError as error:
                assert "probe source" in str(error) or "duplicate probe" in str(error)
            else:
                raise AssertionError(f"invalid source IDs passed: {source_ids}")


def test_probe_loader_allows_distinct_surfaces_for_one_heldout_source():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=2)
        first = corpus.probes[0]
        second = dict(first)
        first["probe_surface_id"] = "surface-a"
        second["probe_surface_id"] = "surface-b"
        second["question"] = "Answer briefly: " + second["question"]
        corpus.probe_paths[0].write_text(
            "\n".join(json.dumps(probe) for probe in (first, second)) + "\n",
            encoding="utf-8",
        )
        loaded = _load_written_corpus(root)
        assert len(loaded.probes) == 2
        assert len({probe["source_triple_id"] for probe in loaded.probes}) == 1


def test_probe_loader_rejects_duplicate_choices():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=2)
        corpus.probes[0]["choices"] = ["a", "a", "b", "c"]
        corpus.probe_paths[0].write_text(
            "\n".join(json.dumps(probe) for probe in corpus.probes) + "\n",
            encoding="utf-8",
        )
        try:
            _load_written_corpus(root)
        except ValueError as error:
            assert "duplicate choices" in str(error)
        else:
            raise AssertionError("duplicate choices passed the probe loader")


def test_evaluator_rejects_non_finite_choice_scores():
    original_choice_logprob = E.choice_logprob
    E.choice_logprob = lambda *_args, **_kwargs: float("nan")
    try:
        E.evaluate_model(
            tokenizer=None,
            model=None,
            device="cpu",
            probes=[{"question": "Where?", "choices": ["a", "b"], "answer": "a"}],
        )
    except ValueError as error:
        assert "non-finite choice score" in str(error)
    else:
        raise AssertionError("non-finite model scores produced a forced-choice prediction")
    finally:
        E.choice_logprob = original_choice_logprob


def test_report_binds_hashes_and_per_predicate_denominators():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root)
        checkpoint = _checkpoint(root, "bound")
        config = _config(root)
        report = E.build_report(
            run_name="bound",
            config_path=config,
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[index % 4 == 0 for index in range(72)],
            tuned_hits=[index % 4 != 0 for index in range(72)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["inputs"]["config_sha256"] == _sha(config)
        assert report["base"]["model_sha256"] == _sha(root / "models" / "base" / "model.safetensors")
        assert report["tuned"]["model_sha256"] == _sha(checkpoint / "model.safetensors")
        assert sum(row["n"] for row in report["tuned"]["per_predicate"].values()) == 72
        assert report["editions"] == ["kanto"]


def test_report_can_bind_an_explicit_historical_checkpoint_fingerprint():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root)
        checkpoint = _checkpoint(root, "historical")
        report = E.build_report(
            run_name="historical",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[False] * 72,
            tuned_hits=[True] * 72,
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="current-eval-recipe",
            checkpoint_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["config_fingerprint"] == "current-eval-recipe"
        assert report["checkpoint_fingerprint"] == "fixture-fingerprint"


def test_report_excludes_quarantined_sources_with_a_counted_reason():
    """The estimand close: a probe whose class pair the hydrate floor already states is scored
    OUT with a counted reason. Behavioural — the denominator and the accuracy both move."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        _corpus(root)
        _write_quarantine(root, [f"id-{index}" for index in range(4)])
        corpus = _load_written_corpus(root)
        checkpoint = _checkpoint(root, "quarantined")
        report = E.build_report(
            run_name="quarantined",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[index < 19 for index in range(72)],
            tuned_hits=[index < 45 for index in range(72)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        # Behavioural first: the scored denominator and BOTH accuracies move off the unfiltered run.
        assert report["base"]["n"] == 68, report["base"]["n"]
        assert report["tuned"]["n"] == 68
        assert report["tuned"]["hits"] == 41 and report["base"]["hits"] == 15
        assert report["quarantined_n"] == 4
        assert report["scored_n"] == 68
        assert report["quarantine_reasons"] == {Q.REASON_FLOOR_PAIR: 4}
        assert report["quarantine_instrument_blind"] is False
        assert len(report["probe_results"]) == 68
        assert all(row["source_triple_id"] not in {f"id-{i}" for i in range(4)}
                   for row in report["probe_results"])
        excluded = report["quarantined_probe_results"]
        assert [row["source_triple_id"] for row in excluded] == [f"id-{i}" for i in range(4)]
        assert all(row["quarantine_reason"] == Q.REASON_FLOOR_PAIR for row in excluded)
        # The exclusion is an INPUT to the number, so its file is hashed into the receipt.
        assert len(report["inputs"]["quarantine_paths"]) == 1
        assert len(report["inputs"]["quarantine_hashes"]) == 1
        assert report["inputs"]["quarantine_sha256"]


def test_report_without_a_quarantine_record_scores_every_probe():
    """Calibration control: absent artifact must leave the denominator and accuracy untouched."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        _corpus(root)
        corpus = _load_written_corpus(root)
        checkpoint = _checkpoint(root, "unquarantined")
        report = E.build_report(
            run_name="unquarantined",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[index < 19 for index in range(72)],
            tuned_hits=[index < 45 for index in range(72)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["base"]["n"] == 72 and report["tuned"]["hits"] == 45
        assert report["quarantined_n"] == 0
        assert report["quarantine_reasons"] == {}
        assert report["quarantined_probe_results"] == []
        assert report["inputs"]["quarantine_paths"] == []


def test_probe_loader_carries_the_instrument_blind_flag_rather_than_a_clean_read():
    """A domain whose floor this instrument cannot read is BLIND, never measured-clean."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        _corpus(root, n=4)
        _write_quarantine(root, [], instrument_blind=True)
        corpus = _load_written_corpus(root)
        assert corpus.quarantine_instrument_blind is True
        assert corpus.quarantined_sources == {}
        assert len(corpus.quarantine_hashes) == 1


def test_probe_loader_refuses_a_quarantine_that_excludes_every_probe():
    """Fail closed: an empty scored set is never a 0/0 verdict."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        _corpus(root, n=4)
        _write_quarantine(root, [f"id-{index}" for index in range(4)])
        try:
            _load_written_corpus(root)
        except ValueError as error:
            assert "quarantine" in str(error)
        else:
            raise AssertionError("a quarantine covering every probe produced a scorable corpus")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} GENERIC-PROBE TESTS PASS")
