"""CPU-only read-only regression for plan_install.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_plan_install.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

import config as C
from plan_install import build_plan


def _by_state(plan: dict) -> dict[str, dict]:
    return {entry["state"]: entry for entry in plan["ladder"]}


def _listing(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _fake_domain(root: Path) -> None:
    (root / "configs/domains").mkdir(parents=True)
    (root / "configs/templates").mkdir(parents=True)
    (root / "schema").mkdir()
    (root / "model").mkdir()
    (root / "scripts/pipeline").mkdir(parents=True)
    (root / "scripts/pipeline/generic_expand.py").write_text("# fixture\n", encoding="utf-8")
    for schema in ("entity", "triple", "discourse", "metadata"):
        (root / f"schema/{schema}.schema.json").write_text("{}\n", encoding="utf-8")
    (root / "configs/editions_fake.json").write_text(
        json.dumps({"editions": [{"unit": "one"}]}),
        encoding="utf-8",
    )
    (root / "configs/templates/fake.yaml").write_text("hydrate_templates: {}\n", encoding="utf-8")
    manifest = {
        "name": "fake",
        "hierarchy": ["unit"],
        "registry": "configs/editions_fake.json",
        "templates": "configs/templates/fake.yaml",
        "ontology": {"schema_dir": "schema"},
        "model_path": str(root / "model"),
        "configs": {"cpt": None, "sft": None},
        "stages": [{
            "name": "hydrate",
            "cmd": ["{py}", "scripts/pipeline/generic_expand.py", "--domain", "fake", "--editions", "{editions}"],
            "gpu": False,
        }],
        "entity_types": ["thing"],
        "predicates": {"related_to": {"object": ["thing"]}},
    }
    (root / "configs/domains/fake.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / "data/fake/one/15_structured").mkdir(parents=True)


def test_plan_install_is_read_only_and_computed() -> None:
    result = subprocess.run(
        [sys.executable, str(C.ROOT / "scripts/plan_install.py"), "--domain", "pokemon-lore", "--editions", "kanto"],
        cwd=C.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert [entry["state"] for entry in plan["ladder"]] == [
        "MANIFEST_VALID", "INPUTS_ADMITTED", "RECIPE_COMPLETE", "STAGE_READY", "TRAINING_READY",
    ]
    states = _by_state(plan)
    assert states["MANIFEST_VALID"]["value"] == "PASS"
    assert states["INPUTS_ADMITTED"]["value"] == "OPERATOR_ATTESTED_REQUIRED"
    readiness = {entry["stage"]: entry for entry in states["STAGE_READY"]["evidence"]}
    for stage in (
        "extract", "pii_scan", "score_educational", "structured", "resolve",
        "hydrate", "synth", "dedup_surfaces", "probes", "audit", "audit_recall",
        "tokenize", "cpt",
    ):
        assert readiness[stage]["ready"], (stage, readiness[stage]["missing"])
    # Mechanically launchable — every stage's outputs exist — but NOT training ready: audit_recall
    # judged the extraction FAIL. This assertion read PASS until 2026-07-27; see
    # test_data_quality_verdicts_gate_training_ready for why that was a green light on bad data.
    assert states["TRAINING_READY"]["value"] == "FAIL"
    split = states["TRAINING_READY"]["evidence"][0]["split_status"]
    assert split[0]["unit"] == "kanto"
    assert split[0]["split_manifest"].endswith("/splits/split_v2.json")
    assert split[0]["exists"] and split[0]["measurement"] == "augmentation-holdout only"

    millbrook = build_plan("millbrook-hollow", "main")
    millbrook_states = _by_state(millbrook)
    millbrook_split = millbrook_states["TRAINING_READY"]["evidence"][0]["split_status"]
    assert millbrook_split[0]["split_manifest"].endswith("/splits/split_v2.json")
    assert millbrook_split[0]["exists"]

    # Phase D zero-knob derivation table rides the same plan payload.
    derivations = plan["derivations"]
    assert derivations["ok"] is True
    assert derivations["knobs"]
    assert all(k.get("derivation") for k in derivations["knobs"])
    assert derivations["plain_language"]
    assert derivations["measured"]["contrast_authorable"] is True
    assert any(k["knob"] == "drop_frac" for k in derivations["knobs"])

    denied = build_plan(
        "pokemon-lore",
        "kanto",
        {"devices": {"allowed": "all"}, "egress": {"llm": False, "upload": True}},
    )
    denied_stages = {entry["stage"]: entry for entry in denied["plan"]}
    assert denied_stages["structured"]["consent_gaps"] == ["egress:llm: policy consent is not granted"]
    assert denied_stages["synth"]["consent_gaps"] == ["egress:llm: policy consent is not granted"]
    assert denied_stages["extract"]["consent_gaps"] == []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fake_domain(root)
        original = C.ROOT, C.CONFIGS, C.DATA
        C.ROOT = root
        C.CONFIGS = root / "configs"
        C.DATA = root / "data"
        try:
            before = _listing(root)
            fake_plan = build_plan("fake", "one")
            after = _listing(root)
        finally:
            C.ROOT, C.CONFIGS, C.DATA = original
        fake_states = _by_state(fake_plan)
        assert fake_states["MANIFEST_VALID"]["value"] == "PASS"
        assert fake_states["INPUTS_ADMITTED"]["value"] == "OPERATOR_ATTESTED_REQUIRED"
        assert fake_states["RECIPE_COMPLETE"]["value"] == "FAIL"
        assert any("predicate_templates" in gap for gap in fake_states["RECIPE_COMPLETE"]["gaps"])
        assert before == after, "plan_install created a file or directory"


def test_data_quality_verdicts_gate_training_ready() -> None:
    """A quality stage that RAN is not a quality stage that PASSED.

    audit_recall writes report_<domain>.json and exits 0 whether its verdict is PASS or FAIL,
    so a readiness ladder that only stats the declared output reports TRAINING_READY on data
    the pipeline itself judged unusable. Kanto is the live case: extraction recall 0.3529
    against the 0.6 gate, status FAIL, quotable false — and the ladder said PASS, which is a
    green light for an unattended multi-hour GPU run on data that already failed its own audit.
    """
    report = C.BUILD / "recall" / "report_pokemon-lore.json"
    assert report.is_file(), f"fixture gone: {report}"
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "FAIL", (
        "this test is only meaningful while the recall report is FAILing"
    )

    training = _by_state(build_plan("pokemon-lore", "kanto"))["TRAINING_READY"]
    assert training["value"] == "FAIL", (
        "audit_recall reported FAIL but the ladder still declared the domain trainable"
    )
    assert any("audit_recall" in gap for gap in training["gaps"]), training["gaps"]
    quality = {q["stage"]: q for q in training["evidence"][0]["data_quality"]}
    assert quality["audit_recall"]["status"] == "FAIL"

    # The load-bearing contrast, in the SAME domain: pii_scan is declared, ran, and returned PASS.
    # It must appear in the evidence and produce no gap — otherwise this gate is just failing
    # everything and the FAIL above would prove nothing.
    assert quality["pii_scan"]["status"] == "PASS"
    assert not [g for g in training["gaps"] if "pii_scan" in g], training["gaps"]

    # A domain declaring neither quality stage records NOT_DECLARED — visible, deliberately not
    # gapped (see _quality_verdicts). "No gate" must not read the same as "gate passed".
    millbrook = _by_state(build_plan("millbrook-hollow", "main"))["TRAINING_READY"]
    undeclared = {q["stage"]: q["status"] for q in millbrook["evidence"][0]["data_quality"]}
    assert undeclared == {"pii_scan": "NOT_DECLARED", "audit_recall": "NOT_DECLARED"}, undeclared
    assert not [g for g in millbrook["gaps"] if "data quality" in g], millbrook["gaps"]


if __name__ == "__main__":
    test_plan_install_is_read_only_and_computed()
    test_data_quality_verdicts_gate_training_ready()
    print("OK — computed ladder, data-quality gating, Kanto reality, and zero planner writes")
