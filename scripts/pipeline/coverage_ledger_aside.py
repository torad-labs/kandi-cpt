#!/usr/bin/env python3
"""Prepare a reversible, document-scoped extraction retry from a recall receipt.

The raw extraction ledger is SHA based.  This helper copies every touched file to a named
backup, removes only triples whose pipeline-owned ``provenance.doc_id`` names a selected
clean document, and removes only those documents' SHAs from the ledger.  Entity/discourse
rows have no document id and are deliberately left alone rather than guessed at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def parse_predicates(value: str) -> list[str]:
    predicates = sorted({part.strip() for part in value.split(",") if part.strip()})
    if not predicates:
        raise ValueError("at least one predicate is required")
    return predicates


def selected_docs(report: dict, predicates: list[str]) -> list[str]:
    selected = set(predicates)
    return sorted({
        str(doc["path"])
        for doc in report.get("docs", [])
        if any(str(row.get("predicate") or "") in selected for row in doc.get("misses", []))
    })


def prepare(domain: str, unit: str, predicate: str, tag: str) -> dict:
    report_path = C.BUILD / "recall" / domain / f"{unit.replace('/', '_')}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    predicates = parse_predicates(predicate)
    names = selected_docs(report, predicates)
    if not names:
        raise ValueError(f"no clean misses for predicates {predicates!r}")

    unit_root = C.DATA / domain / unit
    clean_root = unit_root / "10_clean"
    structured = unit_root / "15_structured"
    triples_path = structured / "triples.jsonl"
    ledger_path = structured / ".extracted.json"
    backup = C.BUILD / "recall" / domain / f"{tag}_backup"
    backup.mkdir(parents=True, exist_ok=False)
    for source in (triples_path, ledger_path, report_path):
        shutil.copy2(source, backup / source.name)

    clean_paths: list[Path] = []
    for name in names:
        matches = sorted(clean_root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"expected one clean document named {name!r}, found {len(matches)}")
        clean_paths.append(matches[0])
    doc_ids = {path.relative_to(clean_root).as_posix() for path in clean_paths}
    shas = {hashlib.sha256(path.read_bytes()).hexdigest() for path in clean_paths}

    rows = load_jsonl(triples_path)
    kept = [
        row for row in rows
        if not (
            str((row.get("provenance") or {}).get("doc_id") or "") in doc_ids
            and str(row.get("predicate") or "") in predicates
        )
    ]
    ledger = set(json.loads(ledger_path.read_text(encoding="utf-8")))
    write_jsonl_atomic(triples_path, kept)
    ledger_path.write_text(json.dumps(sorted(ledger - shas)), encoding="utf-8")

    receipt = {
        "schema": "kandi.coverage_ledger_aside.v1",
        "domain": domain,
        "unit": unit,
        "predicate": predicate,
        "predicates": predicates,
        "documents": names,
        "doc_ids": sorted(doc_ids),
        "document_shas": sorted(shas),
        "triples_before": len(rows),
        "triples_removed": len(rows) - len(kept),
        "triples_after": len(kept),
        "ledger_before": len(ledger),
        "ledger_removed": len(ledger & shas),
        "ledger_after": len(ledger - shas),
        "backup": str(backup.relative_to(C.ROOT)),
        "note": "Only selected predicate rows with pipeline-owned doc_id were removed. Entity/discourse and healthy-predicate rows were preserved.",
    }
    receipt_path = C.ROOT / "logs" / f"{tag}_ledger_aside.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def park_unselected_pending(domain: str, unit: str, scope_receipt: Path) -> dict:
    """Temporarily mark older failed documents done so a retry stays inside its declared scope."""
    scope = json.loads(scope_receipt.read_text(encoding="utf-8"))
    selected_shas = set(scope.get("document_shas") or [])
    unit_root = C.DATA / domain / unit
    clean_root = unit_root / "10_clean"
    ledger_path = unit_root / "15_structured" / ".extracted.json"
    ledger = set(json.loads(ledger_path.read_text(encoding="utf-8")))
    candidates = []
    for path in sorted(clean_root.rglob("*.md")):
        if path.name == "README.md":
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in ledger and sha not in selected_shas:
            candidates.append({"doc_id": path.relative_to(clean_root).as_posix(), "sha256": sha})
    ledger_path.write_text(json.dumps(sorted(ledger | {row["sha256"] for row in candidates})), encoding="utf-8")
    receipt = {
        "schema": "kandi.coverage_scope_repair.v1",
        "domain": domain,
        "unit": unit,
        "scope_receipt": str(scope_receipt.relative_to(C.ROOT)),
        "parked_preexisting_pending": candidates,
        "parked_n": len(candidates),
        "note": "These documents were pending before the scoped retry and were not extracted in this iteration.",
    }
    output = scope_receipt.with_name(scope_receipt.stem.replace("_ledger_aside", "_scope_repair") + ".json")
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def restore(domain: str, unit: str, tag: str) -> dict:
    """Restore every file touched by ``prepare`` from its immutable named backup."""
    unit_root = C.DATA / domain / unit
    structured = unit_root / "15_structured"
    report_path = C.BUILD / "recall" / domain / f"{unit.replace('/', '_')}.json"
    backup = C.BUILD / "recall" / domain / f"{tag}_backup"
    targets = (
        (backup / "triples.jsonl", structured / "triples.jsonl"),
        (backup / ".extracted.json", structured / ".extracted.json"),
        (backup / report_path.name, report_path),
    )
    missing = [str(source) for source, _ in targets if not source.is_file()]
    if missing:
        raise ValueError(f"backup is incomplete: {missing}")
    restored = []
    for source, target in targets:
        temporary = target.with_suffix(target.suffix + ".restore")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        restored.append(str(target.relative_to(C.ROOT)))
    return {
        "schema": "kandi.coverage_ledger_restore.v1",
        "domain": domain,
        "unit": unit,
        "tag": tag,
        "backup": str(backup.relative_to(C.ROOT)),
        "restored": restored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--predicate")
    parser.add_argument("--tag")
    parser.add_argument("--repair-scope")
    parser.add_argument("--restore-backup", action="store_true")
    args = parser.parse_args()
    try:
        if args.restore_backup:
            if not args.tag:
                raise ValueError("--tag is required with --restore-backup")
            receipt = restore(args.domain, args.unit, args.tag)
        elif args.repair_scope:
            receipt = park_unselected_pending(args.domain, args.unit, C.ROOT / args.repair_scope)
        else:
            if not args.predicate or not args.tag:
                raise ValueError("--predicate and --tag are required unless a repair/restore mode is used")
            receipt = prepare(args.domain, args.unit, args.predicate, args.tag)
        print(json.dumps(receipt, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"coverage ledger-aside refused: {error}") from error


if __name__ == "__main__":
    main()
