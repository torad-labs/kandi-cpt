#!/usr/bin/env python3
"""Red-green proofs for the generative CPT axis (no GPU, no provider)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_generative as G  # noqa: E402
import eval_probes as E  # noqa: E402
import eval_stats as S  # noqa: E402


def _probes(n: int = 100) -> list[dict]:
    return [
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


def _corpus(root: Path, n: int = 100) -> E.ProbeCorpus:
    unit = root / "data" / "pokemon-eval" / "kanto"
    probe_path = unit / "probes.jsonl"
    split_path = unit / "splits" / "split_v1.json"
    stats_path = root / "data" / "_build" / "30_tokenized" / "pokemon_eval_stats.json"
    probe_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    stats_path.parent.mkdir(parents=True)
    probes = _probes(n)
    probe_path.write_text("\n".join(json.dumps(p) for p in probes) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps({
        "version": "v1",
        "holdout_ids": [f"id-{i}" for i in range(n)],
        "train_ids": [f"train-{i}" for i in range(n)],
    }), encoding="utf-8")
    stats_path.write_text(json.dumps({"tokens_by_tag": {"clean": 1000}}), encoding="utf-8")
    return E.ProbeCorpus(
        domain="pokemon-eval",
        editions=["kanto"],
        probes=probes,
        probe_paths=[probe_path],
        probe_hashes=[E._sha256(probe_path)],
        probe_sha256=E._combined_hash([probe_path]),
        split_paths=[split_path],
        split_hashes=[E._sha256(split_path)],
        split_sha256=E._combined_hash([split_path]),
        split_version="v1",
        token_stats_path=stats_path,
        token_stats_sha256=E._sha256(stats_path),
        clean_tokens=1000,
        measurement_class="augmentation_holdout",
        chance=0.25,
    )


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("mode: cpt\n", encoding="utf-8")
    return path


def test_select_items_fraction_min_max_and_same_gold():
    probes = _probes(100)
    items = G.select_generative_items(probes, probe_sha256="abc", fraction=0.20, min_items=30, max_items=100)
    # 20% of 100 = 20 → clamped up to min 30
    assert len(items) == 30
    # Deterministic
    again = G.select_generative_items(probes, probe_sha256="abc", fraction=0.20, min_items=30, max_items=100)
    assert [i.source_triple_id for i in items] == [i.source_triple_id for i in again]
    # Same gold records — answers and questions come from probes, split stamps carry
    by_id = {p["source_triple_id"]: p for p in probes}
    for item in items:
        src = by_id[item.source_triple_id]
        assert item.gold == src["answer"]
        assert item.question == src["question"]
        assert item.split == "held_out"
        assert item.split_version == "v1"
        assert item.probe_sha256 == "abc"


def test_select_items_caps_at_max_and_at_n():
    probes = _probes(200)
    items = G.select_generative_items(probes, probe_sha256="x", fraction=0.20, min_items=30, max_items=100)
    assert len(items) == 40  # 20% of 200
    tiny = G.select_generative_items(_probes(10), probe_sha256="x", fraction=0.20, min_items=30, max_items=100)
    assert len(tiny) == 10  # cannot exceed N


def test_deflection_and_wrong_are_fails():
    assert G.grade_labels(["correct", "wrong", "fabricated", "deflected"]) == [True, False, False, False]


def test_parse_judge_verdict():
    assert G.parse_judge_verdict("reasoning\nVERDICT: correct\n") == "correct"
    assert G.parse_judge_verdict("VERDICT: deflected") == "deflected"
    assert G.parse_judge_verdict("no verdict here") is None


def test_kappa_perfect_and_chance():
    perfect = G.cohens_kappa([(True, True), (False, False), (True, True), (False, False)])
    assert perfect["kappa"] == 1.0
    assert perfect["usable"] is True
    # All predicted True regardless of expected → low/zero kappa
    junk = G.cohens_kappa([(True, True), (False, True), (True, True), (False, True)])
    assert junk["kappa"] < 0.6


def test_calibrate_judge_with_stub():
    def stub(q, g, a):
        # Mirror expected from DEFAULT_CALIBRATION by answer text
        for item in G.DEFAULT_CALIBRATION:
            if item["answer"] == a:
                return item["expected_label"]
        return "wrong"

    summary = G.calibrate_judge(G.DEFAULT_CALIBRATION, stub)
    assert summary["kappa"] == 1.0
    assert summary["usable"] is True
    assert summary["provisional"] is False


def test_summarize_emits_wilson_and_insufficient_n():
    probes = _probes(40)
    items = G.select_generative_items(probes, probe_sha256="h", fraction=1.0, min_items=1, max_items=100)
    assert len(items) == 40
    # 32/40 = 0.80 against bar 0.80 → within-noise when min_n lowered; insufficient-n at 50
    labels = ["correct"] * 32 + ["wrong"] * 8
    summary = G.summarize_generative(
        items, labels, min_n=50, bar=0.80, measurement_class="augmentation_holdout",
    )
    assert summary["overall"]["n_status"] == "insufficient-n"
    assert summary["overall"]["wilson"] is not None
    assert summary["overall"]["bar_status"] == "insufficient-n"
    assert summary["label_counts"]["correct"] == 32
    assert summary["label_counts"]["wrong"] == 8

    summary_ok_n = G.summarize_generative(
        items, labels, min_n=40, bar=0.80, measurement_class="augmentation_holdout",
    )
    assert summary_ok_n["overall"]["bar_status"] == "within-noise"
    assert summary_ok_n["overall"]["clears_bar"] is False


def test_build_report_provisional_when_uncalibrated():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=60)
        checkpoint = _checkpoint(root)
        config = _config(root)
        items = G.select_generative_items(corpus.probes, probe_sha256=corpus.probe_sha256,
                                          fraction=1.0, min_items=1, max_items=100)
        # Strong generative scores
        labels = ["correct"] * 50 + ["wrong"] * 10
        report = G.build_generative_report(
            run_name="fresh",
            config_path=config,
            base_path=None,
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            items=items,
            base_labels=None,
            tuned_labels=labels,
            judge_model="nvidia/nemotron-test",
            calibration={"n": 0, "kappa": None, "usable": False, "provisional": True, "skipped": True},
            min_n=50,
            min_delta_acc=0.05,
            generative_bar=0.40,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["status"] == "PROVISIONAL"
        assert report["provisional"] is True
        assert report["axis"] == "generative"
        assert report["estimand"] == "augmentation_holdout"
        assert report["generative"]["tuned"]["overall"]["wilson"] is not None
        assert report["install_strength"]["generative"]["n_status"] == "ok"
        assert all(item["split"] == "held_out" for item in report["items"])


def test_build_report_pass_when_calibrated_and_clears_bar():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=60)
        checkpoint = _checkpoint(root)
        config = _config(root)
        items = G.select_generative_items(corpus.probes, probe_sha256=corpus.probe_sha256,
                                          fraction=1.0, min_items=1, max_items=100)
        labels = ["correct"] * 50 + ["wrong"] * 10
        report = G.build_generative_report(
            run_name="fresh",
            config_path=config,
            base_path=None,
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            items=items,
            base_labels=None,
            tuned_labels=labels,
            judge_model="nvidia/nemotron-test",
            calibration={"n": 8, "kappa": 0.85, "usable": True, "provisional": False, "skipped": False},
            min_n=50,
            min_delta_acc=0.05,
            generative_bar=0.40,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["status"] == "PASS"
        assert report["provisional"] is False
        assert report["generative"]["tuned"]["overall"]["bar_status"] == "cleared"


def test_merge_dual_axis_requires_both():
    fc = {
        "status": "PASS",
        "measurement_class": "augmentation_holdout",
        "delta_acc": 0.3,
        "base": {"acc": 0.3},
        "tuned": {"acc": 0.6, "n": 72, "wilson": {"low": 0.5, "high": 0.7}, "n_status": "ok"},
        "inputs": {"probe_sha256": "abc"},
        "failed_checks": [],
    }
    gen_pass = {
        "status": "PASS",
        "provisional": False,
        "install_strength": {
            "generative": {
                "tuned_acc": 0.55, "n": 30, "wilson": {}, "n_status": "insufficient-n",
                "bar": 0.4, "bar_status": "insufficient-n", "estimand": "augmentation_holdout",
            }
        },
        "failed_checks": [],
    }
    dual = G.merge_dual_axis(fc, gen_pass)
    assert dual["status"] == "PASS"
    assert dual["forced_choice"]["tuned_acc"] == 0.6
    assert dual["generative"]["tuned_acc"] == 0.55

    gen_fail = {**gen_pass, "status": "FAIL", "failed_checks": ["bar"]}
    assert G.merge_dual_axis(fc, gen_fail)["status"] == "FAIL"

    gen_prov = {**gen_pass, "status": "PROVISIONAL", "provisional": True}
    assert G.merge_dual_axis(fc, gen_prov)["status"] == "PROVISIONAL"

    fc_fail = {**fc, "status": "FAIL", "failed_checks": ["delta"]}
    assert G.merge_dual_axis(fc_fail, gen_pass)["status"] == "FAIL"


def test_forced_choice_report_carries_wilson():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=72)
        checkpoint = _checkpoint(root)
        report = E.build_report(
            run_name="fresh",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[i < 19 for i in range(72)],
            tuned_hits=[i < 45 for i in range(72)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["tuned"]["wilson"] is not None
        assert report["tuned"]["n_status"] == "ok"
        assert "low" in report["tuned"]["wilson"]
        assert report["install_strength"]["forced_choice"]["wilson"] is not None
        # per-predicate stamps carry wilson too
        for row in report["tuned"]["per_predicate"].values():
            assert "wilson" in row
            assert "n_status" in row


def test_forced_choice_insufficient_n_label():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        E.C.ROOT = root
        corpus = _corpus(root, n=40)
        checkpoint = _checkpoint(root, "small")
        # 32/40 = 0.80 — the handbook red-green: within-noise of a 0.80 bar is not the forced-choice
        # bar (chance is), but n_status must be insufficient-n at min_n=50.
        report = E.build_report(
            run_name="small",
            config_path=_config(root),
            base_path="models/base",
            tuned_path=str(checkpoint.relative_to(root)),
            corpus=corpus,
            base_hits=[i < 10 for i in range(40)],
            tuned_hits=[i < 32 for i in range(40)],
            min_n=50,
            min_delta_acc=0.10,
            config_fingerprint="fixture-fingerprint",
            tokenized="data/_build/30_tokenized/pokemon_eval.arrow",
        )
        assert report["status"] == "FAIL"
        assert report["tuned"]["n_status"] == "insufficient-n"
        assert report["tuned"]["wilson"] is not None
        # Direct stamp red-green for the 0.80-vs-0.80 case
        stamp = S.stamp_proportion(32, 40, min_n=40, bar=0.80)
        assert stamp["bar_status"] == "within-noise"




def test_judge_one_unparseable_falls_back_not_raise():
    """Unparseable judge output must not raise — heuristic toward wrong/deflected/correct."""
    class _Meter:
        def add(self, *a, **k):
            return None
    # Patch chat to return garbage
    real_chat = G.chat
    def _junk(*a, **k):
        return "I am a free-tier model and cannot follow the format lol"
    G.chat = _junk
    try:
        # gold contained → correct
        label = G.judge_one("Q?", "Kanto", "Ash is from Kanto.", ("m", "k", "u"), _Meter())
        assert label == "correct", label
        # deflection cue → deflected
        label = G.judge_one("Q?", "Kanto", "I don't know.", ("m", "k", "u"), _Meter())
        assert label == "deflected", label
        # neither → wrong (never silent pass / never raise)
        label = G.judge_one("Q?", "Kanto", "Johto maybe.", ("m", "k", "u"), _Meter())
        assert label == "wrong", label
        # empty answer short-circuits without chat
        label = G.judge_one("Q?", "Kanto", "   ", ("m", "k", "u"), _Meter())
        assert label == "deflected", label
    finally:
        G.chat = real_chat


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} GENERATIVE-EVAL TESTS PASS")
