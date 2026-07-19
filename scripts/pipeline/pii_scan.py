#!/usr/bin/env python3
"""PII scan — quarantine-not-sanitize for dense sources; mask incidental (docs/cpt-overhaul/01).

Presidio-style regex (+ optional NER when available) per 10_clean doc → per-doc report +
quarantine list (keep-list semantics, non-destructive). Dense-PII docs are quarantined, not
scrubbed. Sparse-PII docs get a masked curate-side copy with the masking recorded in the sidecar.

Mandatory for style corpora and own-data admission sources. Retires PII_SCAN_UNAVAILABLE from
new run records when the stage runs.

    PYTHONPATH=scripts python scripts/pipeline/pii_scan.py \
        --domain <d> [--editions …] [--dense-threshold 3] [--mandatory]

Outputs (non-destructive — never deletes 10_clean):
  data/_build/pii/<domain>/<unit>.json          per-doc findings
  data/_build/pii/quarantine_<domain>.json      quarantine list (paths)
  data/_build/pii/report_<domain>.json          summary
  data/_build/pii/masked/<rel-path>.md          masked copies for sparse-PII docs (curate may prefer)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import config as C
from pipeline.editions import iter_editions

# Dense = this many distinct PII hits (or more) → quarantine, do not mask.
DEFAULT_DENSE_THRESHOLD = 3

# Regex patterns — Presidio-style, intentionally simple and high-precision. NER is optional.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    )),
    ("phone", re.compile(
        r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)",
    )),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    # @handles but not emails (no domain after)
    ("handle", re.compile(r"(?<![A-Za-z0-9._%+\-])@[A-Za-z0-9_]{2,30}\b")),
    ("ip_address", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    )),
]

_MASK = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ssn": "[SSN]",
    "credit_card": "[CARD]",
    "handle": "[HANDLE]",
    "ip_address": "[IP]",
}


def body_of(md_text: str) -> str:
    return md_text.split("\n---\n", 1)[-1].strip() if "\n---\n" in md_text else md_text.strip()


def find_pii(text: str) -> list[dict]:
    """Return list of {type, start, end, value} (value redacted in reports if needed)."""
    hits: list[dict] = []
    for pii_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            # credit_card: reject if too few digits
            if pii_type == "credit_card":
                digits = re.sub(r"\D", "", match.group(0))
                if not (13 <= len(digits) <= 19):
                    continue
            hits.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "value": match.group(0),
            })
    # Optional NER (spaCy) — best-effort; absence is fine.
    hits.extend(_ner_hits(text))
    # De-dupe overlapping spans (prefer longer)
    hits.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
    cleaned: list[dict] = []
    last_end = -1
    for hit in hits:
        if hit["start"] < last_end:
            continue
        cleaned.append(hit)
        last_end = hit["end"]
    return cleaned


def _ner_hits(text: str) -> list[dict]:
    try:
        import spacy  # type: ignore
    except ImportError:
        return []
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []
    doc = nlp(text[:100_000])
    out = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON",):
            out.append({
                "type": "person_ner",
                "start": ent.start_char,
                "end": ent.end_char,
                "value": ent.text,
            })
    return out


def mask_text(text: str, hits: list[dict]) -> str:
    """Replace hit spans with type masks; apply from end so offsets stay valid."""
    if not hits:
        return text
    chars = list(text)
    for hit in sorted(hits, key=lambda h: h["start"], reverse=True):
        token = _MASK.get(hit["type"], f"[{hit['type'].upper()}]")
        chars[hit["start"]:hit["end"]] = list(token)
    return "".join(chars)


def collect_docs(domain: str, editions_only: str | None) -> list[dict]:
    docs = []
    for ed in iter_editions(domain, only=editions_only, with_raw=False):
        clean_root = ed.dir / "10_clean"
        if not clean_root.is_dir():
            continue
        for md in sorted(clean_root.rglob("*.md")):
            if md.name == "README.md":
                continue
            raw = md.read_text(encoding="utf-8", errors="replace")
            text = body_of(raw)
            rel = str(md.relative_to(C.ROOT))
            docs.append({
                "path": rel,
                "md": md,
                "edition": ed.slug,
                "text": text,
                "raw": raw,
            })
    return docs


def out_paths(domain: str) -> dict[str, Path]:
    root = C.BUILD / "pii"
    return {
        "root": root,
        "per_doc": root / domain,
        "quarantine": root / f"quarantine_{domain}.json",
        "report": root / f"report_{domain}.json",
        "masked_root": root / "masked",
    }


def run_scan(
    domain: str,
    editions: str | None,
    *,
    dense_threshold: int = DEFAULT_DENSE_THRESHOLD,
    mandatory: bool = False,
    dry_run: bool = False,
) -> dict:
    docs = collect_docs(domain, editions)
    if not docs:
        if mandatory:
            raise ValueError(
                f"pii_scan mandatory but no 10_clean docs for domain={domain} — refuse plainly"
            )
        report = {
            "domain": domain, "n_docs": 0, "quarantined": 0, "masked": 0, "clean": 0,
            "status": "EMPTY", "mandatory": mandatory,
        }
        print(json.dumps(report))
        return report

    paths = out_paths(domain)
    quarantine: list[str] = []
    masked_paths: list[str] = []
    per_doc_records: list[dict] = []
    type_counts: Counter = Counter()

    for doc in docs:
        hits = find_pii(doc["text"])
        for hit in hits:
            type_counts[hit["type"]] += 1
        n_hits = len(hits)
        if n_hits >= dense_threshold:
            action = "quarantine"
            quarantine.append(doc["path"])
            masked_rel = None
        elif n_hits > 0:
            action = "mask"
            masked_body = mask_text(doc["text"], hits)
            # Preserve header if present
            if "\n---\n" in doc["raw"]:
                header, _ = doc["raw"].split("\n---\n", 1)
                masked_full = header + "\n---\n" + masked_body
            else:
                masked_full = masked_body
            masked_rel = str(Path("data/_build/pii/masked") / doc["path"])
            if not dry_run:
                out_file = C.ROOT / masked_rel
                out_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_file.with_suffix(out_file.suffix + ".tmp")
                tmp.write_text(masked_full, encoding="utf-8")
                os.replace(tmp, out_file)
            masked_paths.append(doc["path"])
        else:
            action = "clean"
            masked_rel = None

        record = {
            "path": doc["path"],
            "edition": doc["edition"],
            "n_hits": n_hits,
            "types": sorted({h["type"] for h in hits}),
            "action": action,
            "masked_copy": masked_rel,
            # values omitted from durable report — only types + counts (governance)
            "hits": [
                {"type": h["type"], "start": h["start"], "end": h["end"]}
                for h in hits
            ],
        }
        per_doc_records.append(record)
        if not dry_run:
            paths["per_doc"].mkdir(parents=True, exist_ok=True)
            # unit key = path with slashes → underscores
            unit_key = doc["path"].replace("/", "__")
            unit_path = paths["per_doc"] / f"{unit_key}.json"
            unit_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    report = {
        "domain": domain,
        "n_docs": len(docs),
        "quarantined": len(quarantine),
        "masked": len(masked_paths),
        "clean": sum(1 for r in per_doc_records if r["action"] == "clean"),
        "dense_threshold": dense_threshold,
        "type_counts": dict(type_counts),
        "mandatory": mandatory,
        "status": "PASS",
        "quarantine_path": str(paths["quarantine"].relative_to(C.ROOT))
            if paths["quarantine"].is_relative_to(C.ROOT) else str(paths["quarantine"]),
        # Explicitly retire the standing deviation when this stage ran.
        "PII_SCAN_UNAVAILABLE": False,
    }
    if mandatory and len(docs) == 0:
        report["status"] = "REFUSED"

    if not dry_run:
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["quarantine"].write_text(
            json.dumps(sorted(quarantine), indent=2) + "\n", encoding="utf-8",
        )
        paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "domain": domain,
        "n": report["n_docs"],
        "quarantined": report["quarantined"],
        "masked": report["masked"],
        "clean": report["clean"],
        "PII_SCAN_UNAVAILABLE": False,
    }))
    return {"report": report, "quarantine": quarantine, "records": per_doc_records}


def load_quarantine(domain: str) -> set[str] | None:
    """Curate/tokenize consumer: set of quarantined paths, or None if scan not run."""
    path = out_paths(domain)["quarantine"]
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, list):
        return None
    return {str(p) for p in raw}


def main() -> None:
    ap = argparse.ArgumentParser(description="PII scan — quarantine dense, mask incidental")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--dense-threshold", type=int, default=DEFAULT_DENSE_THRESHOLD)
    ap.add_argument("--mandatory", action="store_true",
                    help="refuse empty input (style / own-data corpora)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dense_threshold < 1:
        sys.exit("dense-threshold must be >= 1")
    try:
        run_scan(
            args.domain, args.editions,
            dense_threshold=args.dense_threshold,
            mandatory=args.mandatory,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        sys.exit(f"pii_scan failed closed: {error}")


if __name__ == "__main__":
    main()
