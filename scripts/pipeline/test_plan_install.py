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
    assert states["TRAINING_READY"]["value"] == "PASS"
    split = states["TRAINING_READY"]["evidence"][0]["split_status"]
    assert split[0]["unit"] == "kanto"
    assert split[0]["exists"] and split[0]["measurement"] == "augmentation-holdout only"

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


if __name__ == "__main__":
    test_plan_install_is_read_only_and_computed()
    print("OK — computed ladder, missing-template gap, Kanto reality, and zero planner writes")
