#!/usr/bin/env python3
"""Red-green proofs for the automatic post-CPT trust chain."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.sentinel as Sent  # noqa: E402
from pipeline.auto_chain import run_auto_chain  # noqa: E402


def _fixture(root: Path, *, event: str, booster_n: int) -> Path:
    C.ROOT = root
    C.BUILD = root / "data" / "_build"
    domain, unit = "dom", "u"
    unit_dir = root / "data" / domain / unit
    unit_dir.mkdir(parents=True)
    probes = []
    for predicate in ("leads", "located_in"):
        for index in range(2):
            probes.append({
                "question": f"Q {predicate} {index}?",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "predicate": predicate,
                "subject": f"s{index}",
                "source_triple_id": f"{predicate}{index}",
                "split": "held_out",
                "split_version": "v2",
            })
    (unit_dir / "probes.jsonl").write_text(
        "\n".join(json.dumps(probe) for probe in probes) + "\n",
        encoding="utf-8",
    )
    general = root / "configs" / "domains" / f"{domain}.general_ability.txt"
    general.parent.mkdir(parents=True)
    general.write_text("What is 2 + 2?\nName a color.\n", encoding="utf-8")
    sentinel = Sent.build_sentinel(
        domain=domain,
        unit=unit,
        general_ability_path=general,
        per_predicate=2,
    )
    sentinel_path = unit_dir.parent / "sentinel" / "sentinel_v1.json"
    sentinel_path.parent.mkdir(parents=True)
    sentinel_path.write_text(json.dumps(sentinel) + "\n", encoding="utf-8")
    questions = {
        "domain": domain,
        "questions": [
            {"id": "q1", "text": "Who leads?", "predicate": "leads"},
            {"id": "q2", "text": "Where is it?", "predicate": "located_in"},
            {"id": "q3", "text": "Who else leads?", "predicate": "leads"},
        ],
    }
    (unit_dir.parent / "questions.json").write_text(json.dumps(questions) + "\n", encoding="utf-8")
    candidate = root / "checkpoints" / f"candidate_b{booster_n}"
    candidate.mkdir(parents=True)
    reference = root / "checkpoints" / "prior_b0"
    reference.mkdir(parents=True)
    config_path = root / "configs" / "train.yaml"
    config_path.write_text(yaml.safe_dump({
        "mode": "cpt",
        "model": {"model_path": str(reference)},
        "run": {"run_name": "fixture", "save_dir": str(candidate)},
        "eval": {
            "probe_domain": domain,
            "probe_editions": unit,
            "probe_min_n": 4,
            "generative_enabled": False,
        },
        "install": {
            "enabled": True,
            "domain": domain,
            "unit": unit,
            "event": event,
            "reference_artifact": str(reference),
            "booster_n": booster_n,
            "demo": True,
        },
    }), encoding="utf-8")
    return config_path


def _probe_runner(root: Path, tuned_hits: int):
    def run(_cfg, _config_path, _candidate):
        path = root / "logs" / "probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "estimand": "augmentation_holdout",
            "base": {
                "hits": 4, "n": 4, "acc": 1.0,
                "per_predicate": {
                    "leads": {"hits": 2, "n": 2, "acc": 1.0},
                    "located_in": {"hits": 2, "n": 2, "acc": 1.0},
                },
            },
            "tuned": {
                "hits": tuned_hits, "n": 4, "acc": tuned_hits / 4,
                "per_predicate": {
                    "leads": {"hits": tuned_hits // 2, "n": 2, "acc": (tuned_hits // 2) / 2},
                    "located_in": {"hits": tuned_hits - tuned_hits // 2, "n": 2,
                                   "acc": (tuned_hits - tuned_hits // 2) / 2},
                },
            },
        }
        path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return path, report
    return run


def _install_scorer(candidate_hits: int):
    def score(*, model_path: str, device: str, probes: list[dict]):
        del device
        hits = 4 if "prior_b0" in model_path else candidate_hits
        return {"hits": hits, "n": len(probes), "acc": hits / len(probes), "model_path": model_path}
    return score


def _general_scorer(*, model_path: str, device: str, items: list[dict], domain: str):
    del device, domain
    return {"hits": len(items), "n": len(items), "acc": 1.0, "model_path": model_path}


def test_cpt_completion_runs_the_whole_chain_without_manual_invocation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = _fixture(root, event="cpt", booster_n=0)
        chain = run_auto_chain(
            config_path,
            probe_runner=_probe_runner(root, tuned_hits=4),
            install_scorer=_install_scorer(candidate_hits=4),
            general_scorer=_general_scorer,
        )
        assert chain["status"] == "CLEAR"
        assert Path(chain["sentinel_score_receipt"]).is_file()
        assert Path(chain["install_event_path"]).is_file()
        certificate_path = Path(chain["certificate_path"])
        assert certificate_path.is_file()
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        assert certificate["demo_mark"]["label"] == "DEMO"
        assert certificate["inspector"]["probe_receipt"] == str(root / "logs" / "probe.json")
        assert (root / "data" / "_build" / "install_chains" / "dom" / "latest.json").is_file()
        assert (root / "checkpoints" / "candidate_b0" / "auto_chain.json").is_file()


def test_halt_keeps_prior_and_certificate_measures_the_kept_artifact() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = _fixture(root, event="booster", booster_n=1)
        chain = run_auto_chain(
            config_path,
            probe_runner=_probe_runner(root, tuned_hits=0),
            install_scorer=_install_scorer(candidate_hits=0),
            general_scorer=_general_scorer,
        )
        assert chain["status"] == "HALTED"
        assert chain["artifact_kept"].endswith("prior_b0")
        certificate = json.loads(Path(chain["certificate_path"]).read_text(encoding="utf-8"))
        assert certificate["booster_n"] == 0
        assert certificate["freshness"]["halted_boosters"] == [1]
        assert certificate["inspector"]["run_record"].endswith("prior_b0")
        assert all(row["stamp"]["acc"] == 1.0 for row in certificate["what_it_knows"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} AUTO-CHAIN TESTS PASS")
