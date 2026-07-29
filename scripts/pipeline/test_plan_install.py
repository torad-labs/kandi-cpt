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
        "MANIFEST_VALID", "INPUTS_ADMITTED", "RECIPE_COMPLETE",
        "MECHANICALLY_LAUNCHABLE", "DATA_GATES_PASS", "TRAIN_CONFIG_FROZEN", "TRAINING_READY",
    ]
    states = _by_state(plan)
    assert states["MANIFEST_VALID"]["value"] == "PASS"
    assert states["INPUTS_ADMITTED"]["value"] == "OPERATOR_ATTESTED_REQUIRED"
    readiness = {
        entry["stage"]: entry
        for entry in states["MECHANICALLY_LAUNCHABLE"]["evidence"][0]["stage_readiness"]
    }
    for stage in (
        "extract", "pii_scan", "score_educational", "structured", "resolve",
        "hydrate", "synth", "dedup_surfaces", "probes", "audit", "audit_recall",
        "tokenize", "cpt",
    ):
        assert readiness[stage]["ready"], (stage, readiness[stage]["missing"])

    # The whole point of the split: every input a stage would read EXISTS, and the domain is still
    # not ready. A single state cannot say both, and the one that existed said only the first.
    assert states["MECHANICALLY_LAUNCHABLE"]["value"] == "PASS"
    assert states["TRAINING_READY"]["value"] == "FAIL"
    assert states["TRAINING_READY"]["evidence"][0]["summary"][:2] == [
        "MECHANICALLY LAUNCHABLE", "DATA QUALITY FAILED",
    ]
    assert states["TRAINING_READY"]["evidence"][0]["summary"][-1] == "NOT TRAINING READY"

    split = states["TRAIN_CONFIG_FROZEN"]["evidence"][0]["split_status"]
    assert split[0]["unit"] == "kanto"
    assert split[0]["split_manifest"].endswith("/splits/split_v2.json")
    assert split[0]["exists"] and split[0]["measurement"] == "augmentation-holdout only"

    millbrook = build_plan("millbrook-hollow", "main")
    millbrook_states = _by_state(millbrook)
    millbrook_split = millbrook_states["TRAIN_CONFIG_FROZEN"]["evidence"][0]["split_status"]
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

    states = _by_state(build_plan("pokemon-lore", "kanto"))
    gates = states["DATA_GATES_PASS"]
    assert gates["value"] == "FAIL", (
        "audit_recall reported FAIL but the ladder still declared the data gates clear"
    )
    assert any("audit_recall" in gap for gap in gates["gaps"]), gates["gaps"]
    quality = {q["stage"]: q for q in gates["evidence"]}
    assert quality["audit_recall"]["status"] == "FAIL"

    # The load-bearing contrast, in the SAME domain: pii_scan is declared, ran, and returned PASS.
    # It must appear in the evidence and produce no gap — otherwise this rung is just failing
    # everything and the FAIL above would prove nothing.
    assert quality["pii_scan"]["status"] == "PASS"
    assert not [g for g in gates["gaps"] if "pii_scan" in g], gates["gaps"]

    # ...and the failure must reach TRAINING_READY rather than staying a leaf verdict nothing reads.
    assert "DATA_GATES_PASS" in states["TRAINING_READY"]["evidence"][0]["blocked_by"]

    # A domain declaring no quality stage records NOT_DECLARED — visible, deliberately not gapped
    # (see _quality_verdicts). "No gate" must not read the same as "gate passed".
    millbrook = _by_state(build_plan("millbrook-hollow", "main"))["DATA_GATES_PASS"]
    verdicts = {q["stage"]: q["status"] for q in millbrook["evidence"]}
    assert verdicts == {"pii_scan": "NOT_DECLARED", "audit_recall": "NOT_DECLARED",
                        "audit": "PASS"}, verdicts

    # `audit` records a different shape — JSONL, one row per edition, dimension -> pass/fail — in a
    # file SHARED across domains and keyed only by edition slug. Red-green on that reader: the same
    # file holds a booster_1 row with coverage=fail, so selecting that edition must flip the verdict.
    # Without this the reader could match nothing and report PASS for everything.
    booster = _by_state(build_plan("millbrook-hollow", "booster_1"))["DATA_GATES_PASS"]
    audit_row = next(q for q in booster["evidence"] if q["stage"] == "audit")
    assert audit_row["status"] == "FAIL", audit_row
    assert "coverage" in str(audit_row["detail"]), audit_row


def test_train_config_frozen_catches_derived_vs_configured_drift() -> None:
    """A derived step count that disagrees with the referenced config must block readiness.

    The planner derives train.max_steps from corpus size and the optimizer window; the training
    config sets its own. Nothing reconciled them, so an operator could approve a plan describing
    277 steps and launch a run configured for 450.
    """
    frozen = _by_state(build_plan("pokemon-lore", "kanto"))["TRAIN_CONFIG_FROZEN"]
    checks = {c["knob"]: c for c in frozen["evidence"][0]["knob_reconciliation"]}

    # Non-vacuous anchor: this knob must actually RESOLVE against the config, not fall into
    # NOT_IN_CONFIG. A resolver that finds nothing would make every check vacuously clean.
    assert checks["train.max_steps"]["status"] == "DRIFTED", checks["train.max_steps"]
    assert checks["train.max_steps"]["derived"] != checks["train.max_steps"]["configured"]
    assert frozen["value"] == "FAIL"
    assert any("train.max_steps" in gap for gap in frozen["gaps"]), frozen["gaps"]

    # Knobs that name no config path are not drift — the config legitimately relies on the default.
    assert checks["anneal.enabled"]["status"] == "NOT_IN_CONFIG"


def test_admission_can_be_attested_and_is_bound_to_the_recipe() -> None:
    """INPUTS_ADMITTED was hardcoded to OPERATOR_ATTESTED_REQUIRED with nothing able to satisfy it.

    TRAINING_READY therefore could never be PASS: the top rung was unreachable by construction. A
    gate that cannot open is a wall, and a permanently-failing check is one people learn to route
    around — the exact failure this ladder exists to prevent.

    The attestation is bound to the BUILD ID, so attesting to one recipe cannot authorize another.
    """
    import shutil
    from attest_admission import SCHEMA, attestation_path
    from execution_contract import CONTRACT_VERSION
    from pipeline.domain import Domain
    from run_stage import build_identity

    build_id, _ = build_identity(Domain("pokemon-lore"), CONTRACT_VERSION)
    path = attestation_path("pokemon-lore", build_id)
    assert not path.exists(), f"a real attestation exists at {path}; refusing to overwrite it"

    def admitted():
        return _by_state(build_plan("pokemon-lore", "kanto"))["INPUTS_ADMITTED"]

    assert admitted()["value"] == "OPERATOR_ATTESTED_REQUIRED"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": SCHEMA, "domain": "pokemon-lore", "build_id": build_id,
            "attested_by": "test", "sources_reviewed": ["data-governance.md"],
        }), encoding="utf-8")
        state = admitted()
        assert state["value"] == "PASS", state
        assert state["evidence"][0]["attestation"]["attested_by"] == "test"

        # A record naming a DIFFERENT build must not authorize this one.
        path.write_text(json.dumps({
            "schema": SCHEMA, "domain": "pokemon-lore", "build_id": "0" * 16,
            "attested_by": "test", "sources_reviewed": ["data-governance.md"],
        }), encoding="utf-8")
        assert admitted()["value"] == "OPERATOR_ATTESTED_REQUIRED", "a foreign build id was accepted"

        # An incomplete record is not an attestation.
        path.write_text(json.dumps({
            "schema": SCHEMA, "domain": "pokemon-lore", "build_id": build_id, "attested_by": "test",
        }), encoding="utf-8")
        assert admitted()["value"] == "OPERATOR_ATTESTED_REQUIRED", "a record with no sources passed"
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)

    assert admitted()["value"] == "OPERATOR_ATTESTED_REQUIRED", "cleanup left the domain attested"


# The agent's tool surface returns only the last RUN_REPORT_TAIL stdout lines (20 at the time of
# writing, core-agent RunReport.kt). Anything the Conductor agent must ACT on has to land inside
# that window or it does not exist as far as the agent is concerned.
MODEL_VISIBLE_TAIL_LINES = 20


def test_the_verdict_reaches_the_model_not_just_the_file() -> None:
    """A summary the reader never receives is not a summary.

    plan_install emits ~1340 lines; the ladder sits around line 670. The readiness verdict, written
    for a human to read at a glance, was outside the window the agent is handed — so the playbook
    instruction to read it was a dead instruction. plain_language is the last field emitted and is
    a single line, which is why the verdict leads it.
    """
    result = subprocess.run(
        [sys.executable, str(C.ROOT / "scripts/plan_install.py"),
         "--domain", "pokemon-lore", "--editions", "kanto"],
        cwd=C.ROOT, check=True, capture_output=True, text=True,
    )
    tail = "\n".join(result.stdout.splitlines()[-MODEL_VISIBLE_TAIL_LINES:])

    plan = json.loads(result.stdout)
    verdict = _by_state(plan)["TRAINING_READY"]["evidence"][0]["summary"]
    for line in verdict:
        assert line in tail, f"readiness line {line!r} never reaches the agent"

    # The two numbers that decide whether a study is worth running must arrive too.
    for knob in ("holdout_fact_pool", "resolvable_effect_size"):
        assert knob in tail, f"{knob} is derived but never reaches the agent"

    # Guard the guard: the file really is far longer than the window, so this is a real constraint
    # and not a coincidence of a short output.
    assert len(result.stdout.splitlines()) > 10 * MODEL_VISIBLE_TAIL_LINES


if __name__ == "__main__":
    test_plan_install_is_read_only_and_computed()
    test_data_quality_verdicts_gate_training_ready()
    test_train_config_frozen_catches_derived_vs_configured_drift()
    test_admission_can_be_attested_and_is_bound_to_the_recipe()
    test_the_verdict_reaches_the_model_not_just_the_file()
    print("OK — computed ladder, data-quality gating, Kanto reality, and zero planner writes")
