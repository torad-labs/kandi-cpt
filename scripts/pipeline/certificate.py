#!/usr/bin/env python3
"""Install certificate generator — trust-loop Phase 1 (docs/trust-loop/01-certificate.md).

Consumes run records + probe receipts and emits one legible certificate JSON per install
event. Confidence words are CODE-FIXED from eval_stats stamps — never free adjectives:

  solid   — slice clears its bar (Wilson lower bound strictly above bar)
  early   — insufficient-n OR within-noise (not yet claimable)
  failing — below-bar

Generation FAILS (raises CertificateError) if a claim would be labeled `solid` over an
insufficient-n slice, or if any unmapped confidence word is requested.

    PYTHONPATH=scripts python scripts/pipeline/certificate.py \
        --domain pokemon-lore --unit kanto \
        --probe-receipt logs/probe_eval_pokemon_lore_splitv1_base_vs_tuned.json \
        --out data/_build/certificates/pokemon-lore/kanto_cert.json

Coverage line cites extraction coverage with its estimand (clean / raw) when a recall
report is present — measurement debt is named honestly, never omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as C
import eval_stats as ES

# ── confidence-word table (FIXED — do not extend without a design change) ─────────
CONFIDENCE_SOLID = "solid"
CONFIDENCE_EARLY = "early"
CONFIDENCE_FAILING = "failing"
CONFIDENCE_WORDS = frozenset({CONFIDENCE_SOLID, CONFIDENCE_EARLY, CONFIDENCE_FAILING})

DEFAULT_BAR = 0.50  # chance-ish floor for forced-choice; callers may override per receipt


class CertificateError(ValueError):
    """Certificate generation refused — schema / confidence-word law violation."""


def confidence_word(stamp: dict[str, Any]) -> str:
    """Map an eval_stats stamp to the fixed confidence word. Never invents labels."""
    if not isinstance(stamp, dict):
        raise CertificateError(f"stamp must be a dict, got {type(stamp).__name__}")
    n_status = stamp.get("n_status")
    bar_status = stamp.get("bar_status")
    if n_status == "insufficient-n" or bar_status == "insufficient-n":
        return CONFIDENCE_EARLY
    if bar_status == "cleared" and stamp.get("clears_bar") is True:
        # Guard: solid is illegal on insufficient-n even if a caller forged clears_bar.
        if n_status == "insufficient-n":
            raise CertificateError("solid is illegal over insufficient-n (confidence-word law)")
        return CONFIDENCE_SOLID
    if bar_status == "below-bar":
        return CONFIDENCE_FAILING
    if bar_status == "within-noise":
        return CONFIDENCE_EARLY
    # No bar attached — report early (cannot claim solid without a bar).
    return CONFIDENCE_EARLY


def assert_confidence_legal(word: str, stamp: dict[str, Any]) -> None:
    """Fail generation on illegal word/stamp pairs (red-green: solid + insufficient-n)."""
    if word not in CONFIDENCE_WORDS:
        raise CertificateError(f"unmapped confidence word {word!r} — only {sorted(CONFIDENCE_WORDS)}")
    if word == CONFIDENCE_SOLID and stamp.get("n_status") == "insufficient-n":
        raise CertificateError(
            "confidence-word law violation: solid over insufficient-n slice "
            f"(hits={stamp.get('hits')} n={stamp.get('n')} min_n={stamp.get('min_n')})"
        )
    if word == CONFIDENCE_SOLID and not stamp.get("clears_bar"):
        raise CertificateError(
            "confidence-word law violation: solid requires clears_bar=true "
            f"(bar_status={stamp.get('bar_status')!r})"
        )


def stamp_slice(
    hits: int,
    n: int,
    *,
    bar: float = DEFAULT_BAR,
    min_n: int = ES.DEFAULT_MIN_N,
    estimand: str | None = None,
    slice_name: str | None = None,
) -> dict[str, Any]:
    """Build a stamped proportion and attach the fixed confidence word."""
    stamp = ES.stamp_proportion(
        hits, n, min_n=min_n, bar=bar, estimand=estimand, slice_name=slice_name,
    )
    word = confidence_word(stamp)
    assert_confidence_legal(word, stamp)
    stamp["confidence"] = word
    return stamp


def load_questions(domain: str, path: Path | None = None) -> dict[str, Any]:
    """Load domain questions.json (the certificate's per-question spine)."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([
        C.ROOT / "data" / domain / "questions.json",
        C.ROOT / "configs" / "domains" / f"{domain}.questions.json",
    ])
    for candidate in candidates:
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "questions" not in data:
                raise CertificateError(f"questions.json malformed at {candidate}")
            return data
    raise CertificateError(
        f"questions.json missing for domain={domain!r} — certificate requires the question spine"
    )


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _probe_column(receipt: dict[str, Any], column: str = "tuned") -> dict[str, Any]:
    """Normalize base-vs-tuned and single-column probe receipts into {acc,hits,n,per_predicate}."""
    if column in receipt and isinstance(receipt[column], dict):
        col = receipt[column]
        # nested held_out form
        if "held_out" in col and isinstance(col["held_out"], dict) and "per_predicate" not in col:
            held = col["held_out"]
            return {
                "acc": held.get("acc"),
                "hits": held.get("hits"),
                "n": held.get("n") or held.get("total"),
                "per_predicate": held.get("per_predicate") or {},
            }
        return col
    if "per_predicate" in receipt or "acc" in receipt:
        return receipt
    raise CertificateError(f"probe receipt missing column {column!r} and is not a flat report")


def _slice_from_pred_row(row: dict[str, Any], *, bar: float, min_n: int, estimand: str, name: str) -> dict:
    if row.get("n_independent_facts") is not None:
        fact_hits = int(row.get("independent_fact_hits") or 0)
        fact_n = int(row.get("n_independent_facts") or 0)
        return stamp_slice(
            fact_hits,
            fact_n,
            bar=bar,
            min_n=min_n,
            estimand=estimand,
            slice_name=f"{name}:independent_facts",
        )
    hits = int(row.get("hits") or 0)
    n = int(row.get("n") or 0)
    if "hits" not in row and "acc" in row and n:
        hits = int(round(float(row["acc"]) * n))
    return stamp_slice(hits, n, bar=bar, min_n=min_n, estimand=estimand, slice_name=name)


def build_what_it_knows(
    questions: list[dict[str, Any]],
    probe_receipt: dict[str, Any] | None,
    *,
    column: str = "tuned",
    bar: float = DEFAULT_BAR,
    min_n: int = ES.DEFAULT_MIN_N,
    estimand: str = "augmentation_holdout",
) -> list[dict[str, Any]]:
    """Per-question certificate rows in the user's words, each with confidence + receipt path."""
    per_pred: dict[str, Any] = {}
    overall = None
    receipt_id = None
    if probe_receipt is not None:
        col = _probe_column(probe_receipt, column)
        per_pred = dict(col.get("per_predicate") or {})
        if col.get("n") is not None:
            hits = col.get("hits")
            n = int(col["n"])
            if hits is None and col.get("acc") is not None:
                hits = int(round(float(col["acc"]) * n))
            overall = stamp_slice(int(hits or 0), n, bar=bar, min_n=min_n, estimand=estimand, slice_name="overall")
        receipt_id = probe_receipt.get("receipt_path") or probe_receipt.get("source") or "probe_receipt"

    rows = []
    for q in questions:
        qid = str(q.get("id") or q.get("text") or "")
        text = str(q.get("text") or "")
        pred = str(q.get("predicate") or "")
        pred_row = per_pred.get(pred) if pred else None
        if pred_row is None:
            # No probe coverage for this question yet → early (insufficient evidence).
            stamp = stamp_slice(0, 0, bar=bar, min_n=min_n, estimand=estimand, slice_name=qid or pred or "q")
            plain = f"You asked: {text} — not yet measured on the probe set (early)."
        else:
            stamp = _slice_from_pred_row(
                pred_row, bar=bar, min_n=min_n, estimand=estimand, name=pred or qid,
            )
            word = stamp["confidence"]
            if word == CONFIDENCE_SOLID:
                plain = f"You asked: {text} — it gets this right (solid)."
            elif word == CONFIDENCE_FAILING:
                plain = f"You asked: {text} — still below bar on this slice (failing)."
            else:
                plain = f"You asked: {text} — evidence still early (n={stamp['n']}, insufficient to claim)."
        rows.append({
            "question_id": qid,
            "question": text,
            "predicate": pred or None,
            "plain": plain,
            "confidence": stamp["confidence"],
            "stamp": {
                "hits": stamp["hits"],
                "n": stamp["n"],
                "acc": stamp["acc"],
                "bar": stamp.get("bar"),
                "bar_status": stamp.get("bar_status"),
                "n_status": stamp["n_status"],
                "clears_bar": stamp.get("clears_bar"),
                "wilson": stamp.get("wilson"),
                "estimand": stamp.get("estimand"),
            },
            "receipt": {
                "kind": "probe_slice",
                "path": receipt_id,
                "slice": pred or qid,
            },
        })

    # Domain-level invariant: if any measured slice is below bar, ≥1 failing word must appear
    # when n is sufficient; if all measured are insufficient-n, ≥1 early must appear.
    words = {r["confidence"] for r in rows}
    measured_failing = any(
        r["stamp"].get("bar_status") == "below-bar" for r in rows
    )
    if measured_failing and CONFIDENCE_FAILING not in words:
        raise CertificateError("certificate bug: below-bar slice present but no failing word emitted")
    if rows and words <= {CONFIDENCE_SOLID} and any(r["stamp"]["n_status"] == "insufficient-n" for r in rows):
        raise CertificateError("certificate bug: solid-only card over insufficient-n rows")

    return rows


def build_booster_learning(
    probe_receipt: dict[str, Any] | None,
    *,
    column: str = "tuned",
) -> dict[str, Any] | None:
    """Project named booster probes onto the artifact this certificate measures.

    ``probe_results`` is optional because older/domain-level receipts do not carry
    per-probe rows.  A HALT certificate selects ``base`` while a CLEAR certificate
    selects ``tuned``; this keeps the card about the artifact that actually remains
    installed instead of accidentally crediting a rejected candidate.
    """
    if probe_receipt is None:
        return None
    results = probe_receipt.get("probe_results")
    if not isinstance(results, list) or not results:
        return None
    correctness_key = f"{column}_correct"
    facts: list[dict[str, Any]] = []
    for index, raw in enumerate(results, start=1):
        if not isinstance(raw, dict) or correctness_key not in raw:
            continue
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            continue
        facts.append({
            "probe_id": str(raw.get("source_triple_id") or f"probe_{index}"),
            "question": question,
            "answer": answer,
            "subject": raw.get("subject"),
            "predicate": raw.get("predicate"),
            "installed_artifact_correct": bool(raw[correctness_key]),
            "base_correct": raw.get("base_correct"),
            "candidate_correct": raw.get("tuned_correct"),
        })
    if not facts:
        return None
    hits = sum(1 for row in facts if row["installed_artifact_correct"])
    n = len(facts)
    measurement_class = str(
        probe_receipt.get("measurement_class")
        or probe_receipt.get("estimand")
        or "trained_fact_recall"
    )
    return {
        "estimand": measurement_class,
        "artifact_column": column,
        "hits": hits,
        "n": n,
        "acc": hits / n,
        "pilot_class": bool(probe_receipt.get("pilot_class", n < ES.DEFAULT_MIN_N)),
        "provisional": bool(probe_receipt.get("provisional", n < ES.DEFAULT_MIN_N)),
        "plain": (
            f"On the {n} named booster facts, the installed artifact answered "
            f"{hits}/{n} correctly. This is the {measurement_class} estimand only."
        ),
        "facts": facts,
        "receipt": {
            "kind": "named_booster_probes",
            "path": probe_receipt.get("receipt_path") or probe_receipt.get("source") or "probe_receipt",
        },
    }


def build_what_it_ate(
    *,
    domain: str,
    sources: list[dict[str, Any]] | None = None,
    pii_summary: str | None = None,
    admission_default: str = "public-permitted",
) -> dict[str, Any]:
    """Sources + admission basis + PII line. Every source needs an admission basis."""
    if sources is None:
        # Derive a minimal honest default from the domain corpus layout when present.
        raw = C.ROOT / "data" / domain
        sources = []
        if raw.is_dir():
            sources.append({
                "name": f"{domain} corpus",
                "path": str(raw.relative_to(C.ROOT)) if raw.is_relative_to(C.ROOT) else str(raw),
                "admission_basis": admission_default,
            })
        else:
            sources.append({
                "name": f"{domain} (path not materialized in this run)",
                "path": f"data/{domain}",
                "admission_basis": admission_default,
            })
    cleaned = []
    for src in sources:
        basis = src.get("admission_basis") or admission_default
        if basis not in {"own-data", "public-permitted", "operator-asserted-license"}:
            raise CertificateError(f"unknown admission_basis {basis!r}")
        cleaned.append({
            "name": src.get("name") or src.get("path") or "source",
            "path": src.get("path"),
            "admission_basis": basis,
            "receipt": src.get("receipt") or {"kind": "source_list", "path": src.get("path")},
        })
    return {
        "sources": cleaned,
        "pii_summary": pii_summary or "PII scan: see pipeline pii_scan stage receipt (not inlined).",
        "admission_bases_present": sorted({s["admission_basis"] for s in cleaned}),
    }


def build_coverage_line(recall_report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extraction-coverage line with estimand named. Honest FAIL when clean < gate."""
    if not recall_report:
        return None
    clean = recall_report.get("clean") or recall_report.get("overall") or {}
    raw = recall_report.get("raw") or {}
    # Phase F2 receipt shape
    if "clean_ratio" in recall_report.get("clean", {}):
        pass
    ratio = clean.get("ratio")
    if ratio is None and "ratio" in (recall_report.get("overall") or {}):
        ratio = recall_report["overall"]["ratio"]
        clean = recall_report["overall"]
    gate = float(recall_report.get("min_coverage") or 0.6)
    status = recall_report.get("status")
    if status is None and ratio is not None:
        status = "PASS" if ratio >= gate else "FAIL"
    return {
        "estimand": recall_report.get("estimand") or "clean",
        "clean": {
            "ratio": ratio,
            "hits": clean.get("hits"),
            "n": clean.get("n"),
            "status": status,
            "gate": gate,
        },
        "raw": {
            "ratio": raw.get("ratio"),
            "hits": raw.get("hits"),
            "n": raw.get("n"),
        } if raw else None,
        "plain": (
            f"Extraction coverage ({recall_report.get('estimand') or 'clean'} estimand): "
            f"{ratio:.3f} ({clean.get('hits')}/{clean.get('n')}) — {status} vs gate {gate}."
            if ratio is not None else
            "Extraction coverage: not measured."
        ),
        "receipt": {
            "kind": "recall_report",
            "path": recall_report.get("receipt_path") or "data/_build/recall/report_<domain>.json",
        },
    }


def build_tier_freshness(
    domain: str,
    *,
    verified_at: str,
    booster_n: int = 0,
    hot_index: dict[str, Any] | None = None,
    tier_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-tier freshness rows (trust-loop/04) — each one earned from a receipt.

    ``tier_rows`` is ``tier_routing.derivation_rows(domain)``; ``hot_index`` is the built hot-index
    receipt. A domain with no hot predicates gets an explicit "nothing is retrieved live" row rather
    than the old unconditional "hot facts retrieved live" line, which was a claim with nothing behind
    it. Claiming a live lane while no index exists is a CertificateError, not a nicer sentence.
    """
    rows: list[dict[str, Any]] = list(tier_rows) if tier_rows is not None else []
    if tier_rows is None:
        try:
            from pipeline.tier_routing import derivation_rows
            rows = list(derivation_rows(domain))
        except Exception:  # manifest unreadable → no tier claims, not a fabricated one
            rows = []
    by_tier: dict[str, list[str]] = {}
    for row in rows:
        by_tier.setdefault(str(row.get("tier")), []).append(str(row.get("predicate")))

    hot_predicates = sorted(by_tier.get("hot", []))
    if not hot_predicates:
        hot = {
            "line": "nothing is retrieved live — every fact in this domain is installed in the model",
            "n_predicates": 0,
            "receipt": None,
        }
    elif hot_index is None:
        raise CertificateError(
            f"{len(hot_predicates)} hot predicate(s) routed to retrieval but no hot-index receipt "
            "was supplied — a live-lane claim needs the index that serves it"
        )
    else:
        hot = {
            "line": (
                f"{len(hot_predicates)} kind(s) of fact answered live from "
                f"{int(hot_index.get('n_docs') or 0)} source document(s), indexed "
                f"{hot_index.get('built_at')}"
            ),
            "n_predicates": len(hot_predicates),
            "predicates": hot_predicates,
            "n_docs": hot_index.get("n_docs"),
            "built_at": hot_index.get("built_at"),
            "receipt": hot_index.get("path"),
        }

    stable_predicates = sorted(by_tier.get("stable", []))
    proven_predicates = sorted(by_tier.get("proven", []))
    return {
        "hot": hot,
        "stable": {
            "line": (
                f"{len(stable_predicates)} kind(s) of fact trained into the model, "
                f"verified {verified_at} at booster #{booster_n}"
            ),
            "n_predicates": len(stable_predicates),
            "predicates": stable_predicates,
        },
        "proven": {
            "line": (
                f"{len(proven_predicates)} kind(s) of fact promoted from use "
                "and installed only through an approved digest"
            ),
            "n_predicates": len(proven_predicates),
            "predicates": proven_predicates,
        },
    }


def build_freshness(
    *,
    verified_at: str | None = None,
    booster_n: int = 0,
    halted_boosters: list[int] | None = None,
    domain: str | None = None,
    hot_index: dict[str, Any] | None = None,
    tier_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    verified_at = verified_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    halted_boosters = list(halted_boosters or [])
    installed = f"verified {verified_at}, booster #{booster_n}"
    if halted_boosters:
        installed += (
            f" (halted boosters: {', '.join(f'#{h}' for h in halted_boosters)} — "
            f"#{booster_n} still installed)"
        )
    freshness = {
        "installed": installed,
        "verified_at": verified_at,
        "booster_n": booster_n,
        "halted_boosters": halted_boosters,
    }
    if domain is not None:
        tiers = build_tier_freshness(
            domain,
            verified_at=verified_at,
            booster_n=booster_n,
            hot_index=hot_index,
            tier_rows=tier_rows,
        )
        freshness["tiers"] = tiers
        freshness["hot"] = tiers["hot"]["line"]
    return freshness


def generate_certificate(
    *,
    domain: str,
    unit: str | None = None,
    questions_doc: dict[str, Any],
    probe_receipt: dict[str, Any] | None = None,
    recall_report: dict[str, Any] | None = None,
    calibration_line: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    pii_summary: str | None = None,
    run_record_path: str | None = None,
    column: str = "tuned",
    bar: float = DEFAULT_BAR,
    min_n: int = ES.DEFAULT_MIN_N,
    estimand: str = "augmentation_holdout",
    booster_n: int = 0,
    halted_boosters: list[int] | None = None,
    verified_at: str | None = None,
    demo: bool = False,
    hot_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit one certificate object. Raises CertificateError on confidence-word law violations.

    ``hot_index`` is the built hot-index receipt (``tier_routing.build_hot_index``). It is required
    only when the domain actually routes predicates to the hot tier: the per-tier freshness block
    refuses to claim a live retrieval lane it cannot point at.
    """
    questions = list(questions_doc.get("questions") or [])
    if len(questions) < 3 and not demo:
        raise CertificateError(
            f"questions.json must carry ≥3 user questions (got {len(questions)}); "
            "demo domains may pass demo=True"
        )

    knows = build_what_it_knows(
        questions, probe_receipt,
        column=column, bar=bar, min_n=min_n, estimand=estimand,
    )
    # Marketing-drift guard: a non-empty measured card that never says failing OR early
    # when every slice is below-bar / insufficient is a bug. solid-only is only legal when
    # every measured slice truly clears.
    words = {k["confidence"] for k in knows}
    if knows and CONFIDENCE_SOLID in words:
        for row in knows:
            if row["confidence"] == CONFIDENCE_SOLID:
                assert_confidence_legal(CONFIDENCE_SOLID, {
                    **row["stamp"],
                    "clears_bar": row["stamp"].get("clears_bar"),
                    "n_status": row["stamp"].get("n_status"),
                })

    ate = build_what_it_ate(domain=domain, sources=sources, pii_summary=pii_summary)
    coverage = build_coverage_line(recall_report)
    freshness = build_freshness(
        verified_at=verified_at, booster_n=booster_n, halted_boosters=halted_boosters,
        domain=domain, hot_index=hot_index,
    )
    booster_learning = build_booster_learning(probe_receipt, column=column)

    cert = {
        "schema": "kandi.certificate.v1",
        "domain": domain,
        "unit": unit,
        "demo": bool(demo),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "booster_n": booster_n,
        "what_it_ate": ate,
        "what_it_knows": knows,
        "booster_learning": booster_learning,
        "what_it_refuses": {
            "plain": calibration_line
            or "Abstention / calibration: see SFT calibration receipt; unmapped claims are refused.",
            "receipt": {"kind": "calibration", "path": "logs/*calibration*"},
        },
        "freshness": freshness,
        "coverage": coverage,
        "inspector": {
            "run_record": run_record_path,
            "probe_receipt": (probe_receipt or {}).get("receipt_path"),
            "questions": questions_doc.get("path") or f"data/{domain}/questions.json",
        },
        "confidence_word_table": {
            CONFIDENCE_SOLID: "Wilson lower bound strictly above bar; n ≥ min_n",
            CONFIDENCE_EARLY: "insufficient-n or within-noise — not claimable yet",
            CONFIDENCE_FAILING: "below-bar on a sufficient-n slice",
        },
    }
    if demo:
        cert["demo_mark"] = {
            "label": "DEMO",
            "plain": "Demo-domain card — not a production install certificate.",
        }

    # Final law check: no unmapped words escaped
    for row in knows:
        if row["confidence"] not in CONFIDENCE_WORDS:
            raise CertificateError(f"unmapped confidence word in output: {row['confidence']!r}")

    # Content hash for archival identity (booster N archives; never edits prior)
    blob = json.dumps(cert, sort_keys=True, separators=(",", ":")).encode()
    cert["certificate_id"] = hashlib.sha256(blob).hexdigest()[:16]
    return cert


def seed_questions_from_probes(
    domain: str,
    unit: str,
    *,
    max_questions: int = 20,
    min_questions: int = 3,
) -> dict[str, Any]:
    """Build a questions.json spine from the unit probe set's question shapes."""
    probes_path = C.ROOT / "data" / domain / unit / "probes.jsonl"
    if not probes_path.is_file():
        raise CertificateError(f"probes.jsonl missing at {probes_path}")
    probes = [
        json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    seen: set[str] = set()
    questions: list[dict[str, Any]] = []
    for probe in probes:
        text = str(probe.get("question") or "").strip()
        pred = str(probe.get("predicate") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        questions.append({
            "id": f"q{len(questions)+1:02d}",
            "text": text,
            "predicate": pred,
            "subject": probe.get("subject"),
            "source": "probe_shape",
            "source_triple_id": probe.get("source_triple_id"),
        })
        if len(questions) >= max_questions:
            break
    if len(questions) < min_questions:
        raise CertificateError(
            f"could not seed ≥{min_questions} questions from {probes_path} (got {len(questions)})"
        )
    return {
        "domain": domain,
        "unit": unit,
        "version": 1,
        "seeded_from": str(probes_path.relative_to(C.ROOT)) if probes_path.is_relative_to(C.ROOT) else str(probes_path),
        "n": len(questions),
        "questions": questions,
        "note": (
            "Seeded from existing probe question shapes for the certificate spine. "
            "Consumer-flow interview UI comes later; generator must not depend on it."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="generate install certificate")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--unit")
    ap.add_argument("--questions-json")
    ap.add_argument("--probe-receipt")
    ap.add_argument("--recall-report")
    ap.add_argument("--run-record")
    ap.add_argument("--column", default="tuned")
    ap.add_argument("--bar", type=float, default=DEFAULT_BAR)
    ap.add_argument("--min-n", type=int, default=ES.DEFAULT_MIN_N)
    ap.add_argument("--booster-n", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed-questions", action="store_true",
                    help="write data/<domain>/questions.json from unit probes and exit")
    ap.add_argument("--out")
    args = ap.parse_args()

    try:
        if args.seed_questions:
            if not args.unit:
                sys.exit("--unit required with --seed-questions")
            doc = seed_questions_from_probes(args.domain, args.unit)
            out = Path(args.out) if args.out else C.ROOT / "data" / args.domain / "questions.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"wrote": str(out), "n": doc["n"]}))
            return

        qpath = Path(args.questions_json) if args.questions_json else None
        questions_doc = load_questions(args.domain, qpath)
        probe = load_json(Path(args.probe_receipt) if args.probe_receipt else None)
        if probe is not None and args.probe_receipt:
            probe = dict(probe)
            probe["receipt_path"] = args.probe_receipt
        recall = load_json(Path(args.recall_report) if args.recall_report else None)
        if recall is None:
            # LIVE STAGE REPORT FIRST. This chain used to prefer logs/phase_f2_coverage_metric.json,
            # a DATED CAMPAIGN LOG — so the published certificate quoted 0.1391 (12) while the live
            # pinned instrument read 0.1923 (15/78). A campaign log does not move when the graph
            # moves, carries no ruler pin, and has no `quotable` flag, which meant the entire
            # coverage-ruler apparatus was bypassed at publish time: the number that actually ships
            # was neither pinned nor falsifiable. The stage report is the only artifact that tracks
            # the corpus. An explicit --recall-report still wins over both.
            for candidate in (
                C.ROOT / "data" / "_build" / "recall" / f"report_{args.domain}.json",
            ):
                recall = load_json(candidate)
                if recall is not None:
                    recall = dict(recall)
                    recall["receipt_path"] = str(candidate)
                    # Normalize F2 receipt into clean/raw shape
                    if "clean" in recall and isinstance(recall["clean"], dict) and "ratio" in recall["clean"]:
                        pass
                    elif "clean" in recall and "ratio" in recall.get("clean", {}):
                        pass
                    if recall.get("clean") and "ratio" not in recall["clean"] and "clean_ratio" in recall.get("eval", {}).get("after", {}):
                        pass
                    # phase_f2 shape
                    if "clean" in recall and isinstance(recall["clean"], dict) and "ratio" in recall["clean"]:
                        recall["estimand"] = "clean"
                    elif "clean" in recall and isinstance(recall["clean"], dict) and "ratio" not in recall["clean"]:
                        # phase_f2: clean: {hits,n,ratio,...} already
                        pass
                    if "clean" in recall and "ratio" in recall["clean"]:
                        recall["status"] = recall.get("verdict") or recall.get("status")
                        if recall["status"] is None and recall["clean"]["ratio"] is not None:
                            gate = float(recall.get("gate", {}).get("quality.extraction_recall.min") or 0.6)
                            recall["min_coverage"] = gate
                            recall["status"] = "PASS" if recall["clean"]["ratio"] >= gate else "FAIL"
                    break

        # Normalize phase_f2 receipt
        if recall and "clean" in recall and isinstance(recall["clean"], dict):
            if "ratio" not in recall["clean"] and "clean_ratio" in str(recall):
                pass
        if recall and recall.get("clean") and "ratio" in recall["clean"]:
            recall.setdefault("estimand", "clean")
            if "status" not in recall and "verdict" in recall:
                recall["status"] = recall["verdict"]
            # phase_f2 has gate nested
            if "min_coverage" not in recall:
                gate = recall.get("gate", {})
                if isinstance(gate, dict):
                    recall["min_coverage"] = gate.get("quality.extraction_recall.min", 0.6)

        cert = generate_certificate(
            domain=args.domain,
            unit=args.unit,
            questions_doc=questions_doc,
            probe_receipt=probe,
            recall_report=recall,
            run_record_path=args.run_record,
            column=args.column,
            bar=args.bar,
            min_n=args.min_n,
            booster_n=args.booster_n,
            demo=args.demo,
        )
    except CertificateError as error:
        sys.exit(f"certificate failed closed: {error}")

    out = Path(args.out) if args.out else (
        C.ROOT / "data" / "_build" / "certificates" / args.domain /
        f"{args.unit or 'domain'}_b{args.booster_n}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "certificate_id": cert["certificate_id"],
        "n_questions": len(cert["what_it_knows"]),
        "confidence_words": sorted({r["confidence"] for r in cert["what_it_knows"]}),
        "demo": cert.get("demo"),
        "coverage": (cert.get("coverage") or {}).get("plain"),
    }))


if __name__ == "__main__":
    main()
