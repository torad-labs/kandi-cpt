#!/usr/bin/env python3
"""Red-green proofs for the install certificate generator (trust-loop Phase 1)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import eval_stats as ES  # noqa: E402
import pipeline.certificate as Cert  # noqa: E402


def _questions(n: int = 4) -> dict:
    return {
        "domain": "dom",
        "version": 1,
        "questions": [
            {"id": f"q{i}", "text": f"Question {i} about pred_{i % 3}?", "predicate": f"pred_{i % 3}"}
            for i in range(n)
        ],
    }


def test_confidence_word_table_fixed():
    # solid: large n, high acc clears bar 0.5
    solid = Cert.stamp_slice(90, 100, bar=0.5, min_n=50)
    assert solid["confidence"] == "solid"
    assert solid["clears_bar"] is True

    early = Cert.stamp_slice(8, 10, bar=0.5, min_n=50)  # n < min_n
    assert early["confidence"] == "early"
    assert early["n_status"] == "insufficient-n"

    failing = Cert.stamp_slice(10, 100, bar=0.5, min_n=50)
    assert failing["confidence"] == "failing"
    assert failing["bar_status"] == "below-bar"


def test_solid_over_insufficient_n_fails_generation():
    """Red: injecting solid over insufficient-n must refuse."""
    stamp = ES.stamp_proportion(8, 10, min_n=50, bar=0.5)
    assert stamp["n_status"] == "insufficient-n"
    try:
        Cert.assert_confidence_legal("solid", stamp)
        raise AssertionError("expected CertificateError")
    except Cert.CertificateError as error:
        assert "insufficient-n" in str(error)


def test_unmapped_confidence_word_fails():
    stamp = ES.stamp_proportion(90, 100, min_n=50, bar=0.5)
    try:
        Cert.assert_confidence_legal("excellent", stamp)
        raise AssertionError("expected CertificateError")
    except Cert.CertificateError as error:
        assert "unmapped" in str(error)


def test_domain_with_below_bar_emits_failing_word():
    """A domain with any below-bar sufficient-n slice emits ≥1 failing word."""
    questions = _questions(3)
    # Force predicates pred_0/1/2 with clear below-bar on pred_0
    probe = {
        "receipt_path": "fixture://probe",
        "tuned": {
            "acc": 0.2,
            "hits": 20,
            "n": 100,
            "per_predicate": {
                "pred_0": {"hits": 5, "n": 50, "acc": 0.1},   # below 0.5
                "pred_1": {"hits": 45, "n": 50, "acc": 0.9},  # solid-ish
                "pred_2": {"hits": 5, "n": 10, "acc": 0.5},   # early (n<50)
            },
        },
    }
    cert = Cert.generate_certificate(
        domain="dom",
        unit="u",
        questions_doc=questions,
        probe_receipt=probe,
        bar=0.5,
        min_n=50,
        demo=False,
    )
    words = {row["confidence"] for row in cert["what_it_knows"]}
    assert "failing" in words
    assert "early" in words  # pred_2 insufficient-n
    # solid may or may not appear depending on wilson vs bar 0.5 at n=50 p=0.9
    assert all(w in Cert.CONFIDENCE_WORDS for w in words)
    assert cert["schema"] == "kandi.certificate.v1"
    assert cert.get("demo") is False


def test_demo_mark_supported():
    questions = {
        "domain": "demo-dom",
        "questions": [
            {"id": "q1", "text": "Demo Q1?", "predicate": "p"},
            {"id": "q2", "text": "Demo Q2?", "predicate": "p"},
        ],  # <3 — only legal with demo=True
    }
    try:
        Cert.generate_certificate(domain="demo-dom", questions_doc=questions, demo=False)
        raise AssertionError("expected fail without demo")
    except Cert.CertificateError:
        pass
    cert = Cert.generate_certificate(domain="demo-dom", questions_doc=questions, demo=True)
    assert cert["demo"] is True
    assert cert["demo_mark"]["label"] == "DEMO"


def test_coverage_line_names_estimand_and_fail():
    recall = {
        "estimand": "clean",
        "min_coverage": 0.6,
        "status": "FAIL",
        "clean": {"ratio": 0.3103, "hits": 36, "n": 116},
        "raw": {"ratio": 0.1053, "hits": 44, "n": 418},
        "receipt_path": "logs/phase_f2_coverage_metric.json",
    }
    questions = _questions(3)
    cert = Cert.generate_certificate(
        domain="dom",
        questions_doc=questions,
        recall_report=recall,
        probe_receipt={
            "tuned": {
                "hits": 10, "n": 10, "acc": 1.0,
                "per_predicate": {
                    "pred_0": {"hits": 60, "n": 60},
                    "pred_1": {"hits": 60, "n": 60},
                    "pred_2": {"hits": 60, "n": 60},
                },
            }
        },
        min_n=50,
        bar=0.5,
    )
    cov = cert["coverage"]
    assert cov is not None
    assert cov["estimand"] == "clean"
    assert cov["clean"]["status"] == "FAIL"
    assert "0.310" in cov["plain"]
    assert "clean" in cov["plain"]


def test_every_claim_has_receipt_path():
    questions = _questions(3)
    cert = Cert.generate_certificate(
        domain="dom",
        questions_doc=questions,
        probe_receipt={
            "receipt_path": "logs/probe.json",
            "tuned": {
                "hits": 0, "n": 60, "acc": 0.0,
                "per_predicate": {
                    "pred_0": {"hits": 0, "n": 60},
                    "pred_1": {"hits": 0, "n": 60},
                    "pred_2": {"hits": 0, "n": 60},
                },
            },
        },
        min_n=50,
        bar=0.5,
    )
    for row in cert["what_it_knows"]:
        assert row["receipt"]["path"]
        assert row["confidence"] == "failing"
    assert cert["what_it_ate"]["sources"]
    assert all(s.get("admission_basis") for s in cert["what_it_ate"]["sources"])


def test_booster_certificate_projects_named_facts_for_kept_artifact():
    questions = _questions(3)
    probe = {
        "receipt_path": "logs/booster_probe.json",
        "measurement_class": "trained_fact_recall",
        "base": {"hits": 1, "n": 4, "acc": 0.25, "per_predicate": {}},
        "tuned": {"hits": 4, "n": 4, "acc": 1.0, "per_predicate": {}},
        "probe_results": [
            {
                "source_triple_id": f"fact-{i}",
                "question": f"Where does trainer {i} teach?",
                "answer": f"Room {i}",
                "subject": f"Trainer {i}",
                "predicate": "teaches_at",
                "base_correct": i == 0,
                "tuned_correct": True,
            }
            for i in range(4)
        ],
    }
    clear = Cert.generate_certificate(
        domain="dom", questions_doc=questions, probe_receipt=probe,
        column="tuned", demo=True,
    )
    halted = Cert.generate_certificate(
        domain="dom", questions_doc=questions, probe_receipt=probe,
        column="base", demo=True,
    )
    assert clear["booster_learning"]["hits"] == 4
    assert clear["booster_learning"]["n"] == 4
    assert clear["booster_learning"]["artifact_column"] == "tuned"
    assert all(row["installed_artifact_correct"] for row in clear["booster_learning"]["facts"])
    assert halted["booster_learning"]["hits"] == 1
    assert halted["booster_learning"]["artifact_column"] == "base"
    assert halted["booster_learning"]["receipt"]["path"] == "logs/booster_probe.json"


_STABLE_ROWS = [
    {"predicate": "leads", "tier": "stable", "declared": True},
    {"predicate": "located_in", "tier": "stable", "declared": True},
]
_HOT_ROWS = _STABLE_ROWS + [
    {"predicate": "current_season", "tier": "hot", "declared": True},
]


def test_freshness_without_hot_predicates_claims_no_live_lane():
    """The old default line claimed live retrieval for every domain, backed by nothing."""
    fresh = Cert.build_freshness(
        verified_at="2026-07-24T00:00:00Z", booster_n=1,
        domain="dom", tier_rows=_STABLE_ROWS,
    )
    assert fresh["tiers"]["hot"]["n_predicates"] == 0
    assert fresh["tiers"]["hot"]["receipt"] is None
    assert "nothing is retrieved live" in fresh["tiers"]["hot"]["line"]
    assert fresh["tiers"]["stable"]["n_predicates"] == 2
    assert "booster #1" in fresh["tiers"]["stable"]["line"]


def test_hot_tier_without_an_index_receipt_fails_generation():
    """A live-lane claim must point at the index that serves it (the campaign's one law)."""
    try:
        Cert.build_freshness(
            verified_at="2026-07-24T00:00:00Z", domain="dom", tier_rows=_HOT_ROWS,
        )
    except Cert.CertificateError as error:
        assert "hot-index receipt" in str(error), str(error)
    else:
        raise AssertionError("a hot claim with no index receipt must fail generation")


def test_hot_tier_freshness_is_earned_from_the_index_receipt():
    fresh = Cert.build_freshness(
        verified_at="2026-07-24T00:00:00Z", booster_n=0,
        domain="dom", tier_rows=_HOT_ROWS,
        hot_index={"n_docs": 12, "built_at": "2026-07-24T09:00:00Z", "path": "data/_build/hot_index/dom/index.json"},
    )
    hot = fresh["tiers"]["hot"]
    assert hot["n_predicates"] == 1
    assert hot["predicates"] == ["current_season"]
    assert hot["receipt"] == "data/_build/hot_index/dom/index.json"
    assert "12 source document(s)" in hot["line"]
    assert "2026-07-24T09:00:00Z" in hot["line"]
    # The legacy flat key stays in sync with the tier block rather than drifting from it.
    assert fresh["hot"] == hot["line"]


def test_freshness_without_a_domain_makes_no_tier_claims():
    """Callers that pass no domain get the pre-tier shape, not fabricated tier rows."""
    fresh = Cert.build_freshness(verified_at="2026-07-24T00:00:00Z", booster_n=2)
    assert "tiers" not in fresh
    assert "hot" not in fresh
    assert fresh["booster_n"] == 2


def test_seed_questions_from_probes_fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        unit = root / "data" / "dom" / "u"
        unit.mkdir(parents=True)
        probes = [
            {
                "question": f"Who leads org {i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "predicate": "leads",
                "subject": f"S{i}",
                "source_triple_id": f"id{i}",
                "split": "held_out",
                "split_version": "v2",
            }
            for i in range(5)
        ]
        (unit / "probes.jsonl").write_text(
            "\n".join(json.dumps(p) for p in probes) + "\n", encoding="utf-8",
        )
        doc = Cert.seed_questions_from_probes("dom", "u", max_questions=20)
        assert doc["n"] >= 3
        assert all("text" in q and "predicate" in q for q in doc["questions"])


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} CERTIFICATE TESTS PASS")
