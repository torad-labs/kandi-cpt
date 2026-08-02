#!/usr/bin/env python3
"""The manifest is the single owner of stage order — nothing may hold a second copy.

Conductor carried a static `CptStage` enum listing 11 stages (including a festival-era `META`)
while the manifests declared 17, omitting pii_scan, score_educational, resolve, split, synth,
contrast, dedup_surfaces, probes and audit_recall. Its tests were green the whole time, because a
parallel definition only disagrees with reality, never with itself. That enum is deleted; this test
is what stops the next one being written.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_stage_lockstep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution_contract import build_contract
from pipeline.domain import Domain
from plan_install import build_plan

DOMAINS = (("pokemon-lore", "kanto"), ("millbrook-hollow", "main"), ("edc", "las_vegas/2011"))


def test_manifest_contract_and_plan_agree_on_stages() -> None:
    for name, editions in DOMAINS:
        manifest = [stage.name for stage in Domain(name).stages]
        contract = [stage["id"] for stage in build_contract(name, editions)["stages"]]
        plan = [entry["stage"] for entry in build_plan(name, editions)["plan"]]

        # Order, not just membership: the DAG is ordered, and a reordering is exactly the kind of
        # drift a set comparison would wave through.
        assert manifest == contract, (
            f"{name}: manifest and execution contract disagree\n"
            f"  manifest only: {[s for s in manifest if s not in contract]}\n"
            f"  contract only: {[s for s in contract if s not in manifest]}"
        )
        assert contract == plan, (
            f"{name}: execution contract and plan_install disagree\n"
            f"  contract only: {[s for s in contract if s not in plan]}\n"
            f"  plan only: {[s for s in plan if s not in contract]}"
        )
        assert len(manifest) == len(set(manifest)), f"{name}: duplicate stage id in manifest"


def test_the_comparison_can_actually_fail() -> None:
    """Guard the guard: prove the assertion above is capable of detecting a mismatch.

    A lockstep test that compares two lists which are derived from one another can be vacuously
    green. This mutates one list and confirms the same comparison rejects it.
    """
    manifest = [stage.name for stage in Domain("pokemon-lore").stages]
    assert len(manifest) > 3, manifest
    for mutation in (manifest[:-1], [*manifest[1:], manifest[0]], [*manifest, "invented_stage"]):
        assert mutation != manifest, "mutation did not change the list — the check proves nothing"


def test_a_declared_semantic_postcondition_is_satisfiable_on_the_artifact_on_disk() -> None:
    """A manifest may not declare a stage whose own verdict-checker rejects what is already there.

    probe_quarantine was proposed for EDC while generic_expand.predicate_templates("edc") == {},
    so all 97 units report instrument_blind and probe_quarantine_sound is red forever — a stage
    that is FAIL by construction, not by data. plan_install._QUALITY_STAGES does not include
    probe_quarantine, so the readiness ladder is blind to it; this is the instrument that is not.
    An ABSENT report is fine (not yet run); an EXISTING report the checker rejects is not.

    SCOPE. execution_contract._SEMANTIC_POSTCONDITIONS declares SEVEN distinct kinds, SIX of which
    resolve to a file (checkpoint_integrity is the one that does not — its rules set strip_suffix,
    so it resolves to a directory, and it is out of scope here for the separate reason that every
    EDC and pokemon-lore checkpoint was purged for disk). Two of the six are read; four are not, and
    each one has its own reason:
      * report_verdict, jsonl_no_fail_verdict, tokenize_stats_sound — red TODAY on artifacts already
        on disk (pokemon-lore's audit_recall report, the shared audit jsonl, every tokenize stats
        file) for DATA reasons the readiness ladder already reports as DATA_GATES_PASS FAIL.
      * fact_concentration_sound — NOT the same situation, and it was unaccounted for in the first
        version of this docstring, which said "five ... that resolve to a file" and named only three
        exclusions. It is red for EDC SPECIFICALLY, and only for EDC: the artifact this fixture
        would read, data/edc/las_vegas/2011/15_structured/triples.jsonl, has a top-2 predicate share
        of 0.8832 over 137 rows against a declared relations.dominance_max_frac of 0.85, while
        pokemon-lore's reads 0.2854 and passes. (The 0.9553 the manifest quotes is the DOMAIN-WIDE
        figure over 19,229 rows in 98 editions — a different scope, and quoting it for this one file
        would be the same class of error as the "five" above.) So it is not a domain-wide data
        verdict the ladder already reports: plan_install._QUALITY_STAGES is ("pii_scan",
        "audit_recall") only, so unlike the other three NOTHING surfaces it.
    Reading any of the four here would convert a data verdict into a build-gate failure and make
    this guard un-landable, which is a different check with a different owner.

    MECHANICAL NOTE, so the omission is not read as a judgement it is not: fact_concentration_sound
    could not simply be added to `checkers` even if the verdict were green. Its checker in run_stage
    takes (paths: list[Path], spec) — a LIST of artifacts plus the declared cap — and does not fit
    the checker(report) convention this dict is built on. Widening the dict is an interface change
    and a data-verdict decision at the same time, and it is measurably not free: adding this kind
    reds this fixture immediately on the EDC artifact named above.

    The arithmetic above is pinned by test_the_file_resolving_postcondition_kinds_are_all_accounted_for
    rather than left as prose, because "five" was wrong here for exactly as long as nobody counted.
    """
    import config as C
    import run_stage

    checkers = {
        "probe_quarantine_sound": run_stage._probe_quarantine_ok,
        "split_integrity": run_stage._split_integrity_ok,
    }
    seen = 0
    for name, editions in DOMAINS:
        for stage in build_contract(name, editions)["stages"]:
            for condition in stage.get("postconditions", []):
                checker = checkers.get(condition.get("kind"))
                if checker is None:
                    continue
                report = Path(C.ROOT) / condition["path"]
                if not report.is_file():
                    continue
                seen += 1
                assert checker(report), (name, stage["id"], condition["kind"], condition["path"])
    assert seen, "no declared semantic postcondition had an artifact on disk — guard is vacuous"


def test_the_file_resolving_postcondition_kinds_are_all_accounted_for() -> None:
    """The scope claim above is a CHECK, not prose — it was wrong as prose and nothing noticed.

    The docstring it pins shipped saying "five semantic postcondition kinds ... that resolve to a
    file" when there are six, in a file authored from scratch by the same change that added a guard
    for stale file:line citations, inside the blind spot that guard names in its own docstring: it
    polices pointers, never measured claims. This is the measured claim, re-derived on every run.

    A kind resolves to a DIRECTORY iff any of its rules sets strip_suffix (execution_contract strips
    the glob and hands the validator the directory); otherwise it resolves to a file. Adding a new
    kind, or dropping one from the accounting below, reds here — which is the only thing that makes
    the exclusions above a decision rather than a list somebody stopped updating.
    """
    from execution_contract import _SEMANTIC_POSTCONDITIONS

    # Read on disk by the fixture above. EXCLUDED is not a wish list: widening it into `checkers` is
    # caught immediately, because each of the four has at least one artifact ON DISK that its own
    # checker rejects. Measured 2026-07-31 by driving the four checkers over every postcondition
    # build_contract emits for DOMAINS, so these reasons are re-derivable, not remembered.
    READ = {"probe_quarantine_sound", "split_integrity"}
    EXCLUDED = {
        "report_verdict": "red on pokemon-lore's and edc's audit_recall reports (pii_scan passes) — "
                          "a DATA verdict the ladder already reports as DATA_GATES_PASS FAIL",
        "jsonl_no_fail_verdict": "red for pokemon-lore and edc on the shared audit jsonl, which "
                                 "millbrook-hollow passes on the same file — same, a data verdict",
        "tokenize_stats_sound": "red on all three tokenize stats files — same, a data verdict",
        "fact_concentration_sound": "red for EDC alone — the artifact above reads a top-2 predicate "
                                    "share of 0.8832 against a declared cap of 0.85 while "
                                    "pokemon-lore reads 0.2854 and passes; NOT surfaced by the "
                                    "ladder, and its checker takes (paths, spec) rather than "
                                    "(report), so it does not fit the checkers dict either",
    }
    assert not (READ & set(EXCLUDED)), sorted(READ & set(EXCLUDED))

    directory_resolving: dict[str, bool] = {}
    for rule in _SEMANTIC_POSTCONDITIONS.values():
        kind = rule["kind"]
        directory_resolving[kind] = directory_resolving.get(kind, False) or bool(rule.get("strip_suffix"))
    file_kinds = {kind for kind, directory in directory_resolving.items() if not directory}

    # The equality pins the docstring's SIX (READ and EXCLUDED are disjoint and hold 2 + 4), so no
    # separate count assertion is written for it — it could not fail. The total is a different
    # statement and does fail: a new DIRECTORY-resolving kind leaves file_kinds untouched and would
    # slip past the equality, so the docstring's SEVEN is pinned on its own.
    accounted = READ | set(EXCLUDED)
    assert file_kinds == accounted, sorted(file_kinds ^ accounted)
    assert len(directory_resolving) == 7, sorted(directory_resolving)


def test_a_declared_stage_has_the_floor_instrument_its_verdict_needs() -> None:
    """Satisfiability that does not depend on the stage having run yet.

    The check above only fires while the report exists, so deleting the report would launder the
    defect back in — absence reads as "not yet run". This one is independent of any artifact: it
    asks whether the declared postcondition COULD ever pass. probe_quarantine_sound requires
    units_measured == units_scanned with units_instrument_blind == 0, and a unit is measurable only
    where the floor channel generic_expand.predicate_templates(domain) is non-empty, so declaring
    the stage for a domain with no templates pins the verdict at FAIL by construction.

    Measured 2026-07-31: edc 0, pokemon-lore 18, millbrook-hollow 16.
    """
    from pipeline.generic_expand import predicate_templates

    # kind -> (probe, what the probe reads) — one entry today, and the shape is the point: a new
    # semantic postcondition earns a satisfiability probe here or it is unowned.
    instruments = {"probe_quarantine_sound": (predicate_templates, "generic_expand.predicate_templates")}
    seen = 0
    for name, _editions in DOMAINS:
        for stage in build_contract(name, _editions)["stages"]:
            for condition in stage.get("postconditions", []):
                probe = instruments.get(condition.get("kind"))
                # There is deliberately NO `stage["id"] not in declared` filter beside this.
                # build_contract derives its ids from the same Domain(name).stages that the first
                # test in this file pins against the plan, so that disjunct could never be true
                # (measured: 52 contract stages across DOMAINS, 0 outside the declared set) and a
                # conditional that can never be true reads as a guard forever. Dropping it also
                # means an undeclared contract stage would get CHECKED rather than skipped, which
                # is the direction this guard should fail in.
                if probe is None:
                    continue
                seen += 1
                read, label = probe
                assert read(name), (
                    f"{name} declares stage {stage['id']!r}, whose {condition['kind']} postcondition "
                    f"needs a non-empty floor channel, but {label}({name!r}) is empty — the verdict "
                    f"is FAIL by construction, not by data"
                )
    assert seen, "no declared stage carried a satisfiability-probed postcondition — guard is vacuous"


if __name__ == "__main__":
    test_manifest_contract_and_plan_agree_on_stages()
    test_the_comparison_can_actually_fail()
    test_a_declared_semantic_postcondition_is_satisfiable_on_the_artifact_on_disk()
    test_the_file_resolving_postcondition_kinds_are_all_accounted_for()
    test_a_declared_stage_has_the_floor_instrument_its_verdict_needs()
    print("OK — manifest == execution contract == plan_install, in order, for every domain")
