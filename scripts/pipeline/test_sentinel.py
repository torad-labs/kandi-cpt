#!/usr/bin/env python3
"""Red-green proofs for the install sentinel (trust-loop Phase 2)."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import eval_stats as ES  # noqa: E402
import pipeline.sentinel as S  # noqa: E402


def _fixture_domain(root: Path) -> tuple[str, str, Path]:
    domain, unit = "dom", "u"
    C.ROOT = root
    udir = root / "data" / domain / unit
    udir.mkdir(parents=True)
    probes = []
    for pred in ("leads", "located_in", "member_of"):
        for i in range(4):
            probes.append({
                "question": f"Q {pred} {i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "predicate": pred,
                "subject": f"{pred}_{i}",
                "source_triple_id": f"{pred}_{i}",
                "split": "held_out",
                "split_version": "v2",
            })
    (udir / "probes.jsonl").write_text(
        "\n".join(json.dumps(p) for p in probes) + "\n", encoding="utf-8",
    )
    ga = root / "configs" / "domains" / f"{domain}.general_ability.txt"
    ga.parent.mkdir(parents=True, exist_ok=True)
    ga.write_text(
        "# general ability frozen prompts\n"
        "What is 2 + 2?\n"
        "Name a primary color.\n"
        "What day comes after Monday?\n",
        encoding="utf-8",
    )
    return domain, unit, ga


def test_build_sentinel_hash_stamped():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, unit, ga = _fixture_domain(root)
        sent = S.build_sentinel(
            domain=domain, unit=unit, general_ability_path=ga,
            per_predicate=2, seed=0, version="v1",
        )
        assert sent["schema"] == S.SENTINEL_SCHEMA
        assert sent["n_install"] == 6  # 3 preds × 2
        assert sent["n_general_ability"] == 3
        assert len(sent["content_sha256"]) == 64
        # Re-build identical → same hash
        sent2 = S.build_sentinel(
            domain=domain, unit=unit, general_ability_path=ga,
            per_predicate=2, seed=0, version="v1",
        )
        assert sent["content_sha256"] == sent2["content_sha256"]
        # Version bump → different identity hash body includes version
        sent3 = S.build_sentinel(
            domain=domain, unit=unit, general_ability_path=ga,
            per_predicate=2, seed=0, version="v2",
        )
        assert sent3["content_sha256"] != sent["content_sha256"]


def test_degraded_checkpoint_halted_mechanically():
    """Red-green: synthetic degraded install is halted; prior artifact remains."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, unit, ga = _fixture_domain(root)
        sent = S.build_sentinel(domain=domain, unit=unit, general_ability_path=ga)

        # n=100 → rough_bound = 0.05. Reference 0.80, candidate 0.70 → delta -0.10 < -0.05 → HALT
        n = 100
        bound = ES.rough_bound(n)
        assert bound is not None and abs(bound - 0.5 / math.sqrt(n)) < 1e-12

        result = S.evaluate_sentinel(
            sent,
            install_reference=S.scores_from_hits(80, n),
            install_candidate=S.scores_from_hits(70, n),
            general_base=S.scores_from_hits(90, n),
            general_candidate=S.scores_from_hits(90, n),  # GA holds
            event="booster",
            prior_artifact="ckpt_prior",
            candidate_artifact="ckpt_bad",
        )
        assert result["halted"] is True
        assert result["status"] == "FAILED"
        assert result["artifact_kept"] == "ckpt_prior"
        assert result["install"]["halt"] is True
        assert result["install"]["delta"] < -result["install"]["rough_bound"]
        assert "HALT" in result["plain"]
        assert "ckpt_prior" in result["plain"]


def test_healthy_checkpoint_passes():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, unit, ga = _fixture_domain(root)
        sent = S.build_sentinel(domain=domain, unit=unit, general_ability_path=ga)
        n = 100
        result = S.evaluate_sentinel(
            sent,
            install_reference=S.scores_from_hits(80, n),
            install_candidate=S.scores_from_hits(82, n),  # slight up
            general_base=S.scores_from_hits(90, n),
            general_candidate=S.scores_from_hits(89, n),  # delta -0.01 > -0.05
            event="cpt",
            prior_artifact="ckpt_prior",
            candidate_artifact="ckpt_new",
        )
        assert result["halted"] is False
        assert result["status"] == "PASS"
        assert result["artifact_kept"] == "ckpt_new"
        assert "PASS" in result["plain"]


def test_general_ability_drop_halts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, unit, ga = _fixture_domain(root)
        sent = S.build_sentinel(domain=domain, unit=unit, general_ability_path=ga)
        n = 100
        result = S.evaluate_sentinel(
            sent,
            install_reference=S.scores_from_hits(80, n),
            install_candidate=S.scores_from_hits(81, n),
            general_base=S.scores_from_hits(90, n),
            general_candidate=S.scores_from_hits(70, n),  # -0.20 << -0.05
            event="sft",
            prior_artifact="ckpt_prior",
            candidate_artifact="ckpt_forget",
        )
        assert result["halted"] is True
        assert result["general_ability"]["halt"] is True
        assert result["artifact_kept"] == "ckpt_prior"


def test_missing_general_ability_file_fails_build():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, unit, ga = _fixture_domain(root)
        ga.unlink()
        try:
            S.build_sentinel(domain=domain, unit=unit, general_ability_path=ga)
            raise AssertionError("expected SentinelError")
        except S.SentinelError as error:
            assert "general-ability" in str(error)


def test_missing_ga_scores_on_evaluate_fails():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, unit, ga = _fixture_domain(root)
        sent = S.build_sentinel(domain=domain, unit=unit, general_ability_path=ga)
        try:
            S.evaluate_sentinel(
                sent,
                install_reference=S.scores_from_hits(80, 100),
                install_candidate=S.scores_from_hits(80, 100),
                general_base=None,
                general_candidate=None,
            )
            raise AssertionError("expected SentinelError")
        except S.SentinelError as error:
            assert "general-ability" in str(error)


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} SENTINEL TESTS PASS")
