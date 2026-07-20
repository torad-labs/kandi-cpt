#!/usr/bin/env python3
"""Red-green proofs for the post-install event hook."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import pipeline.install_event as IE  # noqa: E402
import pipeline.sentinel as Sent  # noqa: E402


def _setup(root: Path) -> tuple[str, str]:
    C.ROOT = root
    C.BUILD = root / "data" / "_build"
    domain, unit = "dom", "u"
    # probes + ga + sentinel
    udir = root / "data" / domain / unit
    udir.mkdir(parents=True)
    probes = []
    for pred in ("leads", "located_in"):
        for i in range(3):
            probes.append({
                "question": f"Q {pred} {i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "predicate": pred,
                "subject": f"s{i}",
                "source_triple_id": f"{pred}{i}",
                "split": "held_out",
                "split_version": "v2",
            })
    (udir / "probes.jsonl").write_text("\n".join(json.dumps(p) for p in probes) + "\n")
    ga = root / "configs" / "domains" / f"{domain}.general_ability.txt"
    ga.parent.mkdir(parents=True)
    ga.write_text("What is 2 + 2?\nName a color.\n", encoding="utf-8")
    sent = Sent.build_sentinel(domain=domain, unit=unit, general_ability_path=ga, per_predicate=2)
    sp = root / "data" / domain / "sentinel" / "sentinel_v1.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(sent) + "\n")
    # questions
    qs = {
        "domain": domain,
        "questions": [
            {"id": "q1", "text": "Who leads?", "predicate": "leads"},
            {"id": "q2", "text": "Where located?", "predicate": "located_in"},
            {"id": "q3", "text": "Spare?", "predicate": "leads"},
        ],
    }
    (root / "data" / domain / "questions.json").write_text(json.dumps(qs) + "\n")
    # checkpoint dir
    ckpt = root / "checkpoints" / "cpt" / "run1"
    ckpt.mkdir(parents=True)
    return domain, str(ckpt)


def test_install_event_pass_writes_sidecars():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, ckpt = _setup(root)
        n = 4
        rec = IE.run_install_event(
            domain=domain,
            unit="u",
            event="cpt",
            candidate_artifact=ckpt,
            install_reference_scores=Sent.scores_from_hits(3, n),
            install_candidate_scores=Sent.scores_from_hits(3, n),
            general_base_scores=Sent.scores_from_hits(2, 2),
            general_candidate_scores=Sent.scores_from_hits(2, 2),
            booster_n=0,
        )
        assert rec["halted"] is False
        assert rec["status"] == "INSTALLED_CANDIDATE"
        assert Path(ckpt, "install_event.json").is_file()
        assert Path(ckpt, "certificate.json").is_file()
        assert Path(ckpt, "sentinel_result.json").is_file()


def test_install_event_halt_keeps_prior():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        domain, ckpt = _setup(root)
        n = 4
        rec = IE.run_install_event(
            domain=domain,
            unit="u",
            event="booster",
            candidate_artifact=ckpt,
            reference_artifact="ckpt_prior",
            install_reference_scores=Sent.scores_from_hits(4, n),
            install_candidate_scores=Sent.scores_from_hits(2, n),
            general_base_scores=Sent.scores_from_hits(2, 2),
            general_candidate_scores=Sent.scores_from_hits(2, 2),
            booster_n=5,
        )
        assert rec["halted"] is True
        assert rec["artifact_kept"] == "ckpt_prior"
        assert rec["status"] == "KEPT_PRIOR"


def test_missing_sentinel_fails_closed():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        C.ROOT = root
        C.BUILD = root / "data" / "_build"
        try:
            IE.run_install_event(
                domain="missing",
                unit="u",
                event="cpt",
                candidate_artifact=str(root / "ckpt"),
                install_candidate_scores=Sent.scores_from_hits(1, 1),
                install_reference_scores=Sent.scores_from_hits(1, 1),
                general_base_scores=Sent.scores_from_hits(1, 1),
                general_candidate_scores=Sent.scores_from_hits(1, 1),
            )
            raise AssertionError("expected fail")
        except IE.InstallEventError as error:
            assert "sentinel" in str(error).lower()


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} INSTALL-EVENT TESTS PASS")
