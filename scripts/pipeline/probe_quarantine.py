#!/usr/bin/env python3
"""Probe quarantine — the ESTIMAND-side close on a holdout class pair the hydrate floor states.

WHAT THE LEAK IS. `split` carves holdout by triple id and `generic_expand` hydrates TRAIN ids only,
so no held-out triple is ever rendered. That is an ID-level guarantee and it cannot see the FACT
level: a TRAIN triple and a HOLDOUT triple may share the same canonical endpoint CLASS PAIR under a
different predicate or in the reverse direction. Training the train fact then teaches exactly the
connection the holdout probe measures. Measured on pokemon-lore/kanto: TRAIN (Lt. Surge|opposes|Red)
against HOLDOUT (Red|opposes|Lt. Surge), and TRAIN (Giovanni|leads|Team Rocket) x2 +
(Giovanni|member_of|Team Rocket) against HOLDOUT (Team Rocket|allied_with|Giovanni).

WHY THIS IS SCORED, NOT UNTAUGHT (the arm this module deliberately does NOT take). The floor renders
ONE TRIPLE PER LINE and has no composition surface, so "drop the floor surfaces whose constituent
edges include a holdout edge" matches nothing — every such surface's single edge IS a train edge.
The only floor-side move available is dropping the LINES that state the pair, which un-teaches 4
real train triples to protect 2 probes and inverts the estimand: the probe exists to measure the
training, not the reverse. The exact mirror of that arm was landed and REVERTED at commit 9248ea1d
with a measured delta of zero on its own instrument (`test_generic_reasoning.py`
::test_holdout_stated_as_the_walks_own_adjacent_edge_is_NOT_dropped_and_why keeps that record and
fails if anyone re-adds it). So the contaminated PROBE is quarantined with a counted reason and the
TRAINING DATA is left alone — the same non-destructive, keep-list shape as `pii_scan.load_quarantine`.

A ZERO HERE IS NOT AUTOMATICALLY A MEASUREMENT. The floor channel this module can read is the
PYTHON one, `generic_expand.predicate_templates`. EDC's floor is the Rust festival-factory and
`predicate_templates("edc")` is `{}` — on all 97 EDC units the floor-renderable test is VACUOUSLY
false and a bare `leaks: 0` would be an uncalibrated zero published 97 times. Those units are
reported `instrument_blind: true` and carry NO leak count at all. The WIDER pool
(`pairs_with_any_train_witness`) needs no template map, is measured everywhere, and is reported
separately so the non-floor overlaps stay counted instead of being filtered into invisibility.

    PYTHONPATH=scripts python scripts/pipeline/probe_quarantine.py --domain <d> [--editions …] [--dry-run]

Outputs (non-destructive — probes.jsonl is never edited and never regenerated):
  data/_build/probe_quarantine/<domain>/<unit>.json    per-unit record ('/' in a unit slug → '_')
  data/_build/probe_quarantine/report_<domain>.json    summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config as C
from pipeline.generic_reasoning import canonical_endpoints

SCHEMA = "probe_quarantine/v1"

# The one reason this module quarantines on. A counted reason, never a silent drop.
REASON_FLOOR_PAIR = "floor_states_holdout_class_pair"

FLOOR_CHANNEL_GENERIC = "generic_expand.predicate_templates"
FLOOR_CHANNEL_UNREADABLE = "unreadable_by_this_instrument"


def _build_root() -> Path:
    """`data/_build` derived from C.ROOT AT CALL TIME, not the import-time `C.BUILD` constant.

    `C.BUILD` is bound when `config` is imported, so a caller that repoints `C.ROOT` at a temporary
    tree (every eval test here does) would still read and write the real repo's build directory.
    Same value as `C.BUILD` in production; the difference only shows up under a repointed root.
    """
    return Path(C.ROOT) / "data" / "_build"


def quarantine_path(domain: str, unit_slug: str) -> Path:
    """Per-unit record path. A unit slug may be a path (EDC `city/year`) — flattened with '_',
    the same convention `audit_recall.id_map_path` uses for resolve artifacts."""
    return _build_root() / "probe_quarantine" / domain / f"{unit_slug.replace('/', '_')}.json"


def report_path(domain: str) -> Path:
    return _build_root() / "probe_quarantine" / f"report_{domain}.json"


def _triple_view(t: dict) -> dict:
    return {
        "subject": t.get("subject"),
        "predicate": t.get("predicate"),
        "object": t.get("object"),
    }


def floor_stated_holdout_pairs(
    uniq: dict[str, dict],
    train_ids: set[str],
    holdout_ids: set[str],
    classes: dict[str, str],
    floor_predicates: set[str],
) -> dict:
    """Holdout triples whose endpoint CLASS PAIR a FLOOR-RENDERABLE train triple also states.

    Mirrors `split.alias_leak_report`: pure, deterministic, and it reports the pool it examined
    rather than only its hits. `classes` is `generic_reasoning.entity_classes` (resolve-merged +
    same-display-name union-find) so an entity under two ids is ONE endpoint, and `floor_predicates`
    is the set of predicates the hydrate floor has a template for.

    An empty `floor_predicates` means this instrument cannot see the unit's floor — the result is
    `instrument_blind: true` with `pairs_floor_stated: None`, NEVER a zero. The wider
    `pairs_with_any_train_witness` pool is measured either way.
    """
    blind = not floor_predicates
    holdout_pairs: dict[frozenset, list[str]] = {}
    for tid in sorted(holdout_ids & set(uniq)):
        a, b = canonical_endpoints(uniq[tid], classes)
        if a and b and a != b:
            holdout_pairs.setdefault(frozenset((a, b)), []).append(tid)

    any_witness: dict[frozenset, list[str]] = {}
    floor_witness: dict[frozenset, list[str]] = {}
    for tid in sorted(train_ids & set(uniq)):
        t = uniq[tid]
        a, b = canonical_endpoints(t, classes)
        if not (a and b and a != b):
            continue
        pair = frozenset((a, b))
        if pair not in holdout_pairs:
            continue
        any_witness.setdefault(pair, []).append(tid)
        if str(t.get("predicate") or "") in floor_predicates:
            floor_witness.setdefault(pair, []).append(tid)

    quarantined: list[dict] = []
    if not blind:
        for pair in sorted(floor_witness, key=sorted):
            for tid in holdout_pairs[pair]:
                quarantined.append({
                    "source_triple_id": tid,
                    "reason": REASON_FLOOR_PAIR,
                    "entity_class_pair": sorted(pair),
                    "holdout_triple": _triple_view(uniq[tid]),
                    "train_witness_triple_ids": sorted(floor_witness[pair]),
                    "train_witness_triples": [_triple_view(uniq[w]) for w in sorted(floor_witness[pair])],
                })

    witnessed = [
        {
            "entity_class_pair": sorted(pair),
            "floor_stated": pair in floor_witness,
            "holdout_triple_ids": sorted(holdout_pairs[pair]),
            "holdout_triples": [_triple_view(uniq[h]) for h in sorted(holdout_pairs[pair])],
            "train_witness_triple_ids": sorted(any_witness[pair]),
            "train_witness_triples": [_triple_view(uniq[w]) for w in sorted(any_witness[pair])],
        }
        for pair in sorted(any_witness, key=sorted)
    ]

    return {
        "instrument_blind": blind,
        "floor_channel": FLOOR_CHANNEL_UNREADABLE if blind else FLOOR_CHANNEL_GENERIC,
        "counts": {
            "holdout_class_pairs": len(holdout_pairs),
            "pairs_with_any_train_witness": len(any_witness),
            "pairs_floor_stated": None if blind else len(floor_witness),
            "pairs_train_witness_not_floor_stated": (
                None if blind else len(any_witness) - len(floor_witness)
            ),
            "quarantined_sources": len(quarantined),
        },
        "pairs_with_any_train_witness": witnessed,
        "quarantined": quarantined,
    }


def load_quarantine(domain: str, unit_slug: str) -> dict | None:
    """Eval consumer: the unit's quarantine record, or None when the scan has not run."""
    path = quarantine_path(domain, unit_slug)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return None
    return record


def quarantined_reasons(record: dict | None) -> dict[str, str]:
    """source_triple_id → reason for one loaded record ({} when absent or empty)."""
    if not record:
        return {}
    rows = record.get("quarantined")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["source_triple_id"]): str(row.get("reason") or REASON_FLOOR_PAIR)
        for row in rows
        if isinstance(row, dict) and row.get("source_triple_id")
    }


def scan_unit(domain: str, ed, floor_predicates: set[str]) -> dict | None:
    """One unit's record, or None when it carries no split (nothing to decontaminate)."""
    from pipeline.audit_recall import load_id_map
    from pipeline.generic_expand import load_all_triples
    from pipeline.generic_reasoning import entity_classes
    from pipeline.split import load_split, triple_id

    split = load_split(ed.dir)
    if not split:
        return None
    triples = load_all_triples(ed)
    if not triples:
        return None
    uniq: dict[str, dict] = {}
    for t in triples:
        uniq.setdefault(triple_id(t), t)
    classes = entity_classes(list(uniq.values()), load_id_map(domain, ed.slug))
    detection = floor_stated_holdout_pairs(
        uniq,
        set(split["train_ids"]),
        set(split["holdout_ids"]),
        classes,
        floor_predicates,
    )
    return {
        "schema": SCHEMA,
        "domain": domain,
        "unit": ed.slug,
        "split_version": str(split.get("version") or ""),
        "n_unique_triples": len(uniq),
        "n_train_ids": len(split["train_ids"]),
        "n_holdout_ids": len(split["holdout_ids"]),
        **detection,
    }


def run_scan(domain: str, editions: str | None = None, *, dry_run: bool = False) -> dict:
    from pipeline.editions import iter_editions
    from pipeline.generic_expand import predicate_templates

    floor_predicates = set(predicate_templates(domain))
    records: list[dict] = []
    for ed in iter_editions(domain, only=editions, with_raw=False):
        record = scan_unit(domain, ed, floor_predicates)
        if record is None:
            continue
        records.append(record)
        if not dry_run:
            out = quarantine_path(domain, record["unit"])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            tmp.replace(out)

    blind = [r["unit"] for r in records if r["instrument_blind"]]
    measured = [r for r in records if not r["instrument_blind"]]
    report = {
        "schema": SCHEMA,
        "domain": domain,
        "floor_channel": FLOOR_CHANNEL_GENERIC if floor_predicates else FLOOR_CHANNEL_UNREADABLE,
        "n_floor_predicates": len(floor_predicates),
        "units_scanned": len(records),
        "units_measured": len(measured),
        "units_instrument_blind": len(blind),
        "instrument_blind_units": sorted(blind),
        "holdout_class_pairs": sum(r["counts"]["holdout_class_pairs"] for r in records),
        "pairs_with_any_train_witness": sum(
            r["counts"]["pairs_with_any_train_witness"] for r in records),
        # A SUM OVER ZERO MEASURED UNITS IS NOT A ZERO. When every scanned unit is blind this stays
        # None, so a domain whose floor this instrument cannot read can never publish a clean read.
        "pairs_floor_stated": (
            sum(r["counts"]["pairs_floor_stated"] for r in measured) if measured else None),
        "quarantined_sources": sum(r["counts"]["quarantined_sources"] for r in records),
        "per_unit": [
            {"unit": r["unit"], "instrument_blind": r["instrument_blind"], **r["counts"]}
            for r in records
        ],
    }
    if not dry_run:
        path = report_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "domain": domain,
        "units_scanned": report["units_scanned"],
        "units_measured": report["units_measured"],
        "units_instrument_blind": report["units_instrument_blind"],
        "holdout_class_pairs": report["holdout_class_pairs"],
        "pairs_with_any_train_witness": report["pairs_with_any_train_witness"],
        "pairs_floor_stated": report["pairs_floor_stated"],
        "quarantined_sources": report["quarantined_sources"],
    }))
    return {"report": report, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quarantine holdout probes whose class pair the hydrate floor already states")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = ap.parse_args()
    try:
        run_scan(args.domain, args.editions, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        sys.exit(f"probe_quarantine failed closed: {error}")


if __name__ == "__main__":
    main()
