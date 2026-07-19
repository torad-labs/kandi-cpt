#!/usr/bin/env python3
"""Generic contrast-surface generator for hydrated relation corpora.

The domain owns the confusable values and the sentence shapes in a ``contrast`` block in
``configs/templates/<domain>.yaml``.  This driver supplies the engine seam: it renders only
verified TRAIN triples, validates every denied attractor against the complete unit graph, and
writes the same relation records consumed by ``tokenize_corpus.py``.

    PYTHONPATH=scripts python scripts/pipeline/generic_contrast.py --domain <d>
        [--editions unit,…] [--dry-run]
Output: <edition>/25_hydrated/relations_contrast.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import string
import sys
from collections import defaultdict

from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples, source_doc_ids
from pipeline.schema_validate import validate
from pipeline.split import load_split, triple_id


_ALLOWED_FIELDS = {"subject", "object", "unit", "attractor"}


def _norm(value) -> str:
    """Normalize display values for membership/truth checks without changing emitted text."""
    return str(value).strip().casefold()


def _template_fields(template: str) -> list[str]:
    """Return simple field names from a format string (nested/indexed fields are not authorable)."""
    fields: list[str] = []
    try:
        parsed = string.Formatter().parse(template)
        for _, field, _, _ in parsed:
            if field is not None:
                fields.append(field)
    except ValueError as exc:
        raise SystemExit(f"[generic_contrast] malformed format template {template!r}: {exc}") from exc
    return fields


def load_contrast(domain: str) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Load and fail-closed validate the domain's machine-readable contrast block.

    A missing block is a supported, reported state: class 2 is unavailable for that domain and no
    stale output is touched.  A malformed authored block is an operator error and stops the run.
    """
    block = (Domain(domain).templates or {}).get("contrast")
    if block is None:
        print(f"[generic_contrast] domain '{domain}': contrast unavailable — no contrast block authored")
        return {}, {}
    if not isinstance(block, dict):
        raise SystemExit(f"[generic_contrast] domain '{domain}': contrast must be a mapping with groups/predicates")

    raw_groups = block.get("groups")
    raw_preds = block.get("predicates")
    if not isinstance(raw_groups, dict) or not isinstance(raw_preds, dict):
        raise SystemExit(f"[generic_contrast] domain '{domain}': contrast requires mapping keys groups and predicates")

    groups: dict[str, list[str]] = {}
    for name, values in raw_groups.items():
        if not isinstance(values, list):
            raise SystemExit(f"[generic_contrast] group '{name}' must be a list of display values")
        vals: list[str] = []
        seen: set[str] = set()
        for value in values:
            value = str(value)
            if not value.strip() or _norm(value) in seen:
                continue
            vals.append(value)
            seen.add(_norm(value))
        if len(vals) < 2:
            raise SystemExit(f"[generic_contrast] group '{name}' needs at least two distinct display values")
        groups[str(name)] = vals

    predicates: dict[str, dict] = {}
    for predicate, spec in raw_preds.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"[generic_contrast] predicate '{predicate}' contrast entry must be a mapping")
        group = str(spec.get("group", ""))
        if group not in groups:
            raise SystemExit(f"[generic_contrast] predicate '{predicate}' names missing/unauthorable group '{group}'")
        templates = spec.get("templates")
        if not isinstance(templates, list) or not templates:
            raise SystemExit(f"[generic_contrast] predicate '{predicate}' needs a non-empty templates list")
        checked: list[str] = []
        for i, template in enumerate(templates):
            if not isinstance(template, str):
                raise SystemExit(f"[generic_contrast] predicate '{predicate}' template #{i} is not a string")
            fields = _template_fields(template)
            bad = [f for f in fields if f != "attractor" and f not in _ALLOWED_FIELDS
                   and not f.startswith("attr_")]
            if fields.count("attractor") != 1 or bad:
                raise SystemExit(
                    f"[generic_contrast] predicate '{predicate}' template #{i} must contain exactly one "
                    f"{{attractor}} and only {{subject}}/{{object}}/{{unit}}/{{attr_*}} fields; bad={bad or 'none'}"
                )
            checked.append(template)
        predicates[str(predicate)] = {"group": group, "templates": checked}
    return groups, predicates


def _unit_name(ed) -> str:
    meta = ed.dir / "festival_meta.json"
    if meta.exists():
        try:
            return (json.loads(meta.read_text(encoding="utf-8")).get("name") or ed.slug)
        except (json.JSONDecodeError, OSError):
            pass
    return ed.slug


def _line_count(path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def contrast_records(triples: list[dict], groups: dict[str, list[str]], predicates: dict[str, dict],
                     unit_name: str, domain: str, truth_triples: list[dict] | None = None
                     ) -> tuple[list[dict], int, int]:
    """Render at most two validated contrast surfaces per TRAIN triple.

    Returns records, the number of truth-validation drops, and the number of cap-suppressed valid
    candidates. Truth validation is against ``truth_triples`` (the complete unit graph), not just
    TRAIN; callers that have no split can omit it.
    """
    truth_triples = triples if truth_triples is None else truth_triples
    true_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in truth_triples:
        key = (str(t.get("subject_id", "") or t.get("subject", "")), str(t.get("predicate", "")))
        true_values[key].add(_norm(t.get("object", "")))

    out: list[dict] = []
    truth_drops = 0
    cap_suppressed = 0
    seen_source_ids: set[str] = set()
    for t in triples:
        source_id = triple_id(t)
        if source_id in seen_source_ids:
            continue                                      # duplicate source row = one semantic triple id
        seen_source_ids.add(source_id)
        predicate = t.get("predicate", "")
        spec = predicates.get(predicate)
        if not spec:
            continue
        group_values = groups[spec["group"]]
        object_value = str(t.get("object", ""))
        object_norm = _norm(object_value)
        if object_norm not in {_norm(v) for v in group_values}:
            continue

        subject_key = str(t.get("subject_id", "") or t.get("subject", ""))
        true_for_subject = true_values[(subject_key, str(predicate))]
        candidates: list[str] = []
        for attractor in group_values:
            if _norm(attractor) == object_norm:
                continue
            if _norm(attractor) in true_for_subject:
                truth_drops += 1
                print(f"  truth-drop: {subject_key} {predicate} — '{attractor}' is also true", file=sys.stderr)
                continue
            candidates.append(attractor)

        kw = {"subject": t.get("subject", ""), "object": object_value, "unit": unit_name}
        kw.update({f"attr_{k}": v for k, v in (t.get("attrs") or {}).items()})
        made_for_triple = 0
        for template in spec["templates"]:
            for attractor in candidates:
                if made_for_triple >= 2:
                    cap_suppressed += 1
                    continue
                try:
                    text = template.format(**kw, attractor=attractor).strip()
                except (KeyError, IndexError, ValueError) as exc:
                    raise SystemExit(
                        f"[generic_contrast] template render failed for predicate '{predicate}': {exc}; "
                        f"template={template!r} triple={t}"
                    )
                if not text:
                    continue
                rec = {"text": text, "relation_type": predicate, "direction": "forward",
                       "surface_form": "contrast", "source_triple_ids": [source_id],
                       "entity_ids": [t.get("subject", ""), object_value], "split": "cpt"}
                if doc_ids := source_doc_ids(t):
                    rec["source_doc_ids"] = doc_ids
                errors = validate("relation", rec, domain=domain)
                if errors:
                    raise SystemExit(
                        f"[generic_contrast] 25_hydrated relation record fails schema for domain '{domain}': "
                        f"{errors[:3]} | rec={rec}"
                    )
                out.append(rec)
                made_for_triple += 1
    return out, truth_drops, cap_suppressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions", help="restrict to 'unit,…' (default: all on disk)")
    ap.add_argument("--max-fraction", type=float, default=0.10,
                    help="soft cap on contrast lines as a fraction of relations.jsonl "
                         "(Phase C default 0.10; was warn-only at 0.20). Emit still writes all "
                         "validated surfaces; exceeding this is a hard WARNING + non-zero note.")
    ap.add_argument("--dry-run", action="store_true", help="print per-unit counts, write nothing")
    args = ap.parse_args()

    groups, predicates = load_contrast(args.domain)
    if not groups or not predicates:
        if groups or predicates:
            print(f"[generic_contrast] domain '{args.domain}': contrast unavailable — no usable groups/predicates")
        return

    wrote = 0
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        all_triples = load_all_triples(ed)
        if not all_triples:
            continue
        split = load_split(ed.dir)
        if split:
            train_triples = [t for t in all_triples if triple_id(t) in split["train_ids"]]
            mode = f"split {split['version']} TRAIN"
        else:
            train_triples = all_triples
            mode = "no split manifest (ALL triples)"
        unit_name = _unit_name(ed)
        records, truth_drops, cap_suppressed = contrast_records(
            train_triples, groups, predicates, unit_name, args.domain, truth_triples=all_triples
        )
        base_path = ed.dir / "25_hydrated" / "relations.jsonl"
        base_lines = _line_count(base_path)
        fraction = (len(records) / base_lines) if base_lines else None
        frac_text = f"{len(records)}/{base_lines} = {fraction:.1%}" if fraction is not None else \
            f"{len(records)}/0 = n/a"
        print(f"  {ed.slug}: {mode} — {len(train_triples)}/{len(all_triples)} triples; "
              f"emitted {len(records)}, truth-drops {truth_drops}, cap-suppressed {cap_suppressed}; "
              f"contrast fraction {frac_text} (soft-cap {args.max_fraction:.0%})")
        if fraction is not None and fraction > args.max_fraction:
            print(f"  WARNING: {ed.slug} contrast fraction {fraction:.1%} exceeds "
                  f"max-fraction {args.max_fraction:.0%} of relations.jsonl — "
                  f"prefer raising SFT hard-negatives over growing CPT contrast "
                  f"(docs/cpt-overhaul/03)",
                  file=sys.stderr)
        if args.dry_run:
            continue
        out_dir = ed.dir / "25_hydrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "relations_contrast.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) +
                       ("\n" if records else ""), encoding="utf-8")
        os.replace(tmp, out_path)
        wrote += 1
        print(f"    wrote {out_path}")
    if not args.dry_run:
        print(f"\nwrote {wrote} relations_contrast.jsonl file(s)")


if __name__ == "__main__":
    main()
