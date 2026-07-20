#!/usr/bin/env python3
"""Install sentinel — trust-loop Phase 2 (docs/trust-loop/02-sentinel.md).

Frozen per-domain probe subset + operator-named general-ability file. Runs on every
install event (cpt/sft/anneal/booster). Halt rule is mechanical from eval_stats.rough_bound:

  install slice:  (candidate_acc - reference_acc) < -rough_bound(n)  → HALT
  general slice:  (candidate_acc - base_acc)      < -rough_bound(n)  → HALT

A halt keeps the prior artifact. Sentinel is a regression tripwire, never an
install-quality verdict: it reports CLEAR or HALTED, while the certificate owns
the user-facing verdict.
The set is hash-stamped; re-cutting is a new version (goalpost law).

    PYTHONPATH=scripts python scripts/pipeline/sentinel.py build \
        --domain pokemon-lore --unit kanto \
        --general-ability configs/domains/pokemon-lore.general_ability.txt

    PYTHONPATH=scripts python scripts/pipeline/sentinel.py evaluate \
        --sentinel data/<d>/sentinel/sentinel_v1.json \
        --reference-scores ... --candidate-scores ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as C
import eval_stats as ES

SENTINEL_SCHEMA = "kandi.sentinel.v1"
SENTINEL_RESULT_SCHEMA = "kandi.sentinel_result.v2"
DEFAULT_PER_PRED = 2  # minimal per-predicate; minutes-class on GPU 1
CERTIFIED_INSTALL_N = 50


class SentinelError(ValueError):
    """Sentinel build/evaluate refused."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_probes(domain: str, unit: str) -> list[dict[str, Any]]:
    path = C.ROOT / "data" / domain / unit / "probes.jsonl"
    if not path.is_file():
        raise SentinelError(f"probes.jsonl missing: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_general_ability(path: Path) -> list[dict[str, Any]]:
    """Operator-named frozen prompt file: one prompt per non-empty non-# line."""
    if not path.is_file():
        raise SentinelError(
            f"general-ability file missing: {path} — operator must name this BEFORE first booster "
            "(closes cpt/06-improve.md harness gap)"
        )
    raw = path.read_text(encoding="utf-8")
    items = []
    for i, line in enumerate(raw.splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        items.append({"id": f"ga_{i:03d}", "prompt": text, "slice": "general_ability"})
    if not items:
        raise SentinelError(f"general-ability file has no prompts: {path}")
    return items


def select_install_slice(
    probes: list[dict[str, Any]],
    *,
    per_predicate: int = DEFAULT_PER_PRED,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Fixed minimal subset: up to `per_predicate` items per predicate, deterministic order."""
    import random
    rng = random.Random(seed)
    by_pred: dict[str, list[dict]] = defaultdict(list)
    for probe in probes:
        pred = str(probe.get("predicate") or "")
        if pred:
            by_pred[pred].append(probe)
    selected: list[dict[str, Any]] = []
    for pred in sorted(by_pred):
        bucket = list(by_pred[pred])
        # stable order then sample
        bucket.sort(key=lambda p: str(p.get("source_triple_id") or p.get("question") or ""))
        take = bucket if len(bucket) <= per_predicate else rng.sample(bucket, per_predicate)
        take = sorted(take, key=lambda p: str(p.get("source_triple_id") or p.get("question") or ""))
        for probe in take:
            selected.append({
                "id": f"inst_{pred}_{probe.get('source_triple_id') or len(selected)}",
                "slice": "install",
                "predicate": pred,
                "question": probe.get("question"),
                "choices": probe.get("choices"),
                "answer": probe.get("answer"),
                "subject": probe.get("subject"),
                "source_triple_id": probe.get("source_triple_id"),
                "split": probe.get("split"),
                "split_version": probe.get("split_version"),
            })
    if not selected:
        raise SentinelError("install slice empty — probes produced no selectable items")
    return selected


def build_sentinel(
    *,
    domain: str,
    unit: str,
    general_ability_path: Path,
    per_predicate: int = DEFAULT_PER_PRED,
    seed: int = 0,
    version: str = "v1",
    minimum_install_n: int = 0,
) -> dict[str, Any]:
    """Freeze install + general-ability slices; hash-stamp the set."""
    probes = load_probes(domain, unit)
    install = select_install_slice(probes, per_predicate=per_predicate, seed=seed)
    if minimum_install_n < 0:
        raise SentinelError("minimum install n cannot be negative")
    if minimum_install_n and len(install) < minimum_install_n:
        raise SentinelError(
            f"install slice n={len(install)} is below declared minimum n={minimum_install_n}; "
            "enlarge the frozen set instead of loosening the tripwire"
        )
    ga_path = Path(general_ability_path)
    general = load_general_ability(ga_path)
    ga_text = ga_path.read_text(encoding="utf-8")
    body = {
        "schema": SENTINEL_SCHEMA,
        "domain": domain,
        "unit": unit,
        "version": version,
        "seed": seed,
        "per_predicate": per_predicate,
        "install_slice": install,
        "general_ability_slice": general,
        "general_ability_file": str(
            ga_path.relative_to(C.ROOT) if ga_path.is_relative_to(C.ROOT) else ga_path
        ),
        "general_ability_sha256": _sha256_text(ga_text),
        "n_install": len(install),
        "minimum_install_n": minimum_install_n,
        "n_general_ability": len(general),
        "predicates": sorted({i["predicate"] for i in install}),
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Frozen sentinel set. Re-cutting requires a new version string and restarts the series "
            "(goalpost law). Do not edit items in place to clear a halt."
        ),
    }
    # Hash over the frozen content excluding frozen_at / content_sha (identity of the set)
    identity = {
        "schema": body["schema"],
        "domain": domain,
        "unit": unit,
        "version": version,
        "seed": seed,
        "per_predicate": per_predicate,
        "install_slice": install,
        "general_ability_slice": general,
        "general_ability_sha256": body["general_ability_sha256"],
        "minimum_install_n": minimum_install_n,
    }
    body["content_sha256"] = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return body


def scores_from_hits(hits: int, n: int) -> dict[str, Any]:
    """Fixture helper: build an acc block."""
    if n < 0 or hits < 0 or hits > n:
        raise SentinelError(f"invalid hits/n: {hits}/{n}")
    return {"hits": hits, "n": n, "acc": (hits / n) if n else 0.0}


def halt_decision(
    *,
    reference_acc: float,
    candidate_acc: float,
    n: int,
    slice_name: str,
) -> dict[str, Any]:
    """Mechanical halt: candidate drop vs reference beyond rough_bound(n)."""
    bound = ES.rough_bound(n)
    if bound is None:
        raise SentinelError(f"cannot evaluate halt on empty slice {slice_name!r}")
    delta = candidate_acc - reference_acc
    halted = delta < -bound
    return {
        "slice": slice_name,
        "reference_acc": reference_acc,
        "candidate_acc": candidate_acc,
        "delta": delta,
        "n": n,
        "rough_bound": bound,
        "halt": halted,
        "rule": "halt if delta < -rough_bound(n) where rough_bound = 0.5/sqrt(n)",
    }


def evaluate_sentinel(
    sentinel: dict[str, Any],
    *,
    install_reference: dict[str, Any],
    install_candidate: dict[str, Any],
    general_base: dict[str, Any] | None = None,
    general_candidate: dict[str, Any] | None = None,
    event: str = "booster",
    prior_artifact: str = "prior_checkpoint",
    candidate_artifact: str = "candidate_checkpoint",
) -> dict[str, Any]:
    """Compare candidate to reference; HALT keeps prior artifact.

    Score dicts: {hits, n, acc}. General-ability is optional only when the sentinel
    was built without it (should not happen); missing GA scores on a GA-bearing
    sentinel is a hard error (evidence contract).
    """
    if sentinel.get("schema") != SENTINEL_SCHEMA:
        raise SentinelError(f"unknown sentinel schema {sentinel.get('schema')!r}")

    def _acc(block: dict[str, Any], name: str) -> tuple[float, int, int]:
        n = int(block.get("n") or 0)
        hits = block.get("hits")
        if hits is None and block.get("acc") is not None and n:
            hits = int(round(float(block["acc"]) * n))
        hits = int(hits or 0)
        acc = float(block["acc"]) if block.get("acc") is not None else ((hits / n) if n else 0.0)
        if n <= 0:
            raise SentinelError(f"{name} scores have n=0")
        return acc, hits, n

    ref_acc, ref_hits, ref_n = _acc(install_reference, "install_reference")
    cand_acc, cand_hits, cand_n = _acc(install_candidate, "install_candidate")
    frozen_n = len(sentinel.get("install_slice") or [])
    if ref_n != cand_n or ref_n != frozen_n:
        raise SentinelError(
            "install scores must cover the exact same frozen set: "
            f"reference n={ref_n}, candidate n={cand_n}, frozen n={frozen_n}"
        )
    n_bound = frozen_n

    install_dec = halt_decision(
        reference_acc=ref_acc, candidate_acc=cand_acc, n=n_bound, slice_name="install",
    )

    decisions = [install_dec]
    general_dec = None
    if sentinel.get("general_ability_slice"):
        if general_base is None or general_candidate is None:
            raise SentinelError(
                "sentinel carries a general-ability slice but scores were not provided — "
                "every install event must run both slices"
            )
        g_ref, g_ref_hits, g_ref_n = _acc(general_base, "general_base")
        g_cand, g_cand_hits, g_cand_n = _acc(general_candidate, "general_candidate")
        g_n = min(g_ref_n, g_cand_n)
        general_dec = halt_decision(
            reference_acc=g_ref, candidate_acc=g_cand, n=g_n, slice_name="general_ability",
        )
        decisions.append(general_dec)

    halted = any(d["halt"] for d in decisions)
    status = "HALTED" if halted else "CLEAR"
    declared_minimum = int(sentinel.get("minimum_install_n") or 0)
    comparison_certified = (
        declared_minimum >= CERTIFIED_INSTALL_N
        and frozen_n >= declared_minimum
    )

    # Plain-language report line (digest-ready)
    if halted:
        plain = (
            f"I kept {prior_artifact}. On the frozen check, {candidate_artifact} "
            "fell beyond the expected wiggle room. This is a safety stop, not a strength verdict."
        )
    else:
        plain = (
            f"The frozen check for {candidate_artifact} stayed inside the expected wiggle room. "
            "This only clears the safety stop; read the certificate for how strong the install is."
        )

    return {
        "schema": SENTINEL_RESULT_SCHEMA,
        "role": "regression_tripwire",
        "not_an_install_verdict": True,
        "sentinel_version": sentinel.get("version"),
        "sentinel_content_sha256": sentinel.get("content_sha256"),
        "domain": sentinel.get("domain"),
        "unit": sentinel.get("unit"),
        "event": event,
        "status": status,
        "halted": halted,
        "tripwire_action": "KEEP_PRIOR" if halted else "CONTINUE_TO_CERTIFICATE",
        "strength_verdict": "NOT_OWNED_BY_SENTINEL",
        "candidate_cleared_for_certificate": not halted,
        "certification": {
            "standard_minimum_n": CERTIFIED_INSTALL_N,
            "declared_minimum_n": declared_minimum,
            "observed_n": frozen_n,
            "comparison_certified": comparison_certified,
            "label": "certified_comparison" if comparison_certified else "tripwire_only",
        },
        "protected_artifact": prior_artifact if halted else None,
        "prior_artifact": prior_artifact,
        "candidate_artifact": candidate_artifact,
        "decisions": decisions,
        "install": {
            "reference": {"hits": ref_hits, "n": ref_n, "acc": ref_acc},
            "candidate": {"hits": cand_hits, "n": cand_n, "acc": cand_acc},
            **install_dec,
        },
        "general_ability": general_dec,
        "plain": plain,
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def default_paths(domain: str, unit: str, version: str = "v1") -> dict[str, Path]:
    root = C.ROOT / "data" / domain / "sentinel"
    return {
        "dir": root,
        "sentinel": root / f"sentinel_{version}.json",
        "general_ability": C.ROOT / "configs" / "domains" / f"{domain}.general_ability.txt",
    }


def latest_sentinel_path(domain: str) -> Path:
    """Resolve the newest frozen series version without editing an older set in place."""
    root = C.ROOT / "data" / domain / "sentinel"
    candidates = list(root.glob("sentinel_v*.json")) if root.is_dir() else []

    def version_number(path: Path) -> int:
        match = path.stem.removeprefix("sentinel_v")
        return int(match) if match.isdigit() else -1

    return max(candidates, key=version_number) if candidates else root / "sentinel_v1.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="install sentinel build/evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="freeze sentinel set")
    b.add_argument("--domain", required=True)
    b.add_argument("--unit", required=True)
    b.add_argument("--general-ability")
    b.add_argument("--per-predicate", type=int, default=DEFAULT_PER_PRED)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--version", default="v1")
    b.add_argument(
        "--minimum-install-n",
        type=int,
        default=0,
        help="declared certification floor; build refuses when the frozen install slice is smaller",
    )
    b.add_argument("--out")

    e = sub.add_parser("evaluate", help="halt decision from score blocks")
    e.add_argument("--sentinel", required=True)
    e.add_argument("--install-reference", required=True, help="JSON {hits,n,acc}")
    e.add_argument("--install-candidate", required=True)
    e.add_argument("--general-base")
    e.add_argument("--general-candidate")
    e.add_argument("--event", default="booster")
    e.add_argument("--prior-artifact", default="prior_checkpoint")
    e.add_argument("--candidate-artifact", default="candidate_checkpoint")
    e.add_argument("--out")

    args = ap.parse_args()
    try:
        if args.cmd == "build":
            paths = default_paths(args.domain, args.unit, args.version)
            ga = Path(args.general_ability) if args.general_ability else paths["general_ability"]
            sent = build_sentinel(
                domain=args.domain,
                unit=args.unit,
                general_ability_path=ga,
                per_predicate=args.per_predicate,
                seed=args.seed,
                version=args.version,
                minimum_install_n=args.minimum_install_n,
            )
            out = Path(args.out) if args.out else paths["sentinel"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(sent, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "wrote": str(out),
                "content_sha256": sent["content_sha256"],
                "n_install": sent["n_install"],
                "n_general_ability": sent["n_general_ability"],
                "predicates": sent["predicates"],
            }))
            return

        sent = json.loads(Path(args.sentinel).read_text(encoding="utf-8"))
        def _load(p: str | None) -> dict | None:
            if not p:
                return None
            return json.loads(Path(p).read_text(encoding="utf-8"))
        result = evaluate_sentinel(
            sent,
            install_reference=_load(args.install_reference),  # type: ignore[arg-type]
            install_candidate=_load(args.install_candidate),  # type: ignore[arg-type]
            general_base=_load(args.general_base),
            general_candidate=_load(args.general_candidate),
            event=args.event,
            prior_artifact=args.prior_artifact,
            candidate_artifact=args.candidate_artifact,
        )
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": result["status"],
            "halted": result["halted"],
            "tripwire_action": result["tripwire_action"],
            "plain": result["plain"],
            "wrote": args.out,
        }))
        sys.exit(0 if not result["halted"] else 2)
    except (SentinelError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        sys.exit(f"sentinel failed closed: {error}")


if __name__ == "__main__":
    main()
