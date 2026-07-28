#!/usr/bin/env python3
"""Record the operator's source-admission attestation — the only thing that can satisfy INPUTS_ADMITTED.

Until now `INPUTS_ADMITTED` was hardcoded to OPERATOR_ATTESTED_REQUIRED with no code path anywhere
that could set it otherwise, so TRAINING_READY could never be PASS. A top rung that cannot be
reached is not a gate, it is a wall — and a permanently-failing check is one people learn to route
around, which is the failure this ladder exists to prevent.

THE ATTESTATION IS BOUND TO A BUILD ID. The build id is the hash of the complete recipe read set
(manifest, templates, registry, ontology, train configs, stage code), so attesting to one recipe
cannot authorize a different one: change any of those and the id moves, and the attestation stops
applying. That is the same rule run_stage enforces for approved digests, applied to the human
approval rather than the machine one.

    python scripts/attest_admission.py --domain pokemon-lore --editions kanto \
        --attested-by marcos --source-review data-governance.md

A recorded attestation is evidence, not a formality: it names who attested, when, to which recipe,
and which sources they reviewed. It is never written by the planner — plan_install is read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import config as C
from execution_contract import CONTRACT_VERSION
from pipeline.domain import Domain
from run_stage import build_identity

SCHEMA = "kandi.admission_attestation.v1"


def attestation_path(domain: str, build_id: str) -> Path:
    return C.DATA / domain / "builds" / build_id / "admission.json"


def load_attestation(domain: str, build_id: str) -> dict | None:
    """The attestation for THIS recipe, or None. A record for another build does not count."""
    path = attestation_path(domain, build_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("schema") != SCHEMA or record.get("build_id") != build_id:
        return None
    if not record.get("attested_by") or not record.get("sources_reviewed"):
        return None
    return record


def admission_state(domain_name: str) -> dict:
    """What plan_install reports for INPUTS_ADMITTED, computed from disk. Read-only."""
    try:
        build_id, _ = build_identity(Domain(domain_name), CONTRACT_VERSION)
    except Exception as error:  # noqa: BLE001 — the ladder must still emit
        return {"value": "OPERATOR_ATTESTED_REQUIRED", "evidence": [],
                "gaps": [f"cannot compute build identity: {error}"]}

    record = load_attestation(domain_name, build_id)
    if record is None:
        return {
            "value": "OPERATOR_ATTESTED_REQUIRED",
            "evidence": [{"build_id": build_id, "attestation": None}],
            "gaps": [
                "every source needs a run-record admission line naming basis, trust class, and PII/injection/poisoning review",
                f"operator attestation against data-governance.md is required for build {build_id} "
                f"(record it with scripts/attest_admission.py)",
            ],
        }
    return {
        "value": "PASS",
        "evidence": [{"build_id": build_id, "attestation": {
            "attested_by": record["attested_by"],
            "attested_at": record.get("attested_at"),
            "sources_reviewed": record["sources_reviewed"],
            "note": record.get("note"),
        }}],
        "gaps": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--attested-by", required=True, help="who is attesting; recorded verbatim")
    parser.add_argument("--source-review", required=True, nargs="+",
                        help="the source-admission review(s) performed, e.g. data-governance.md")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    build_id, inputs = build_identity(Domain(args.domain), CONTRACT_VERSION)
    path = attestation_path(args.domain, build_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": SCHEMA,
        "domain": args.domain,
        "build_id": build_id,
        "recipe_sha256": inputs["recipe_sha256"],
        "attested_by": args.attested_by,
        "attested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources_reviewed": list(args.source_review),
        "note": args.note,
        "binding": ("This attestation applies ONLY to build "
                    f"{build_id}. Any change to the recipe read set moves the build id and this "
                    "record stops applying."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"attested build {build_id} for {args.domain} -> {path}", file=sys.stderr)
    print(build_id)


if __name__ == "__main__":
    main()
