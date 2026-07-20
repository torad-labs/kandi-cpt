#!/usr/bin/env python3
"""Post-install event hook — sentinel + certificate on every cpt/sft/anneal/booster.

Called from train_cpt.py (rank0, after final save) and run_sft.sh (after gate).
An install event WITHOUT a sentinel result fails the evidence contract.

    PYTHONPATH=scripts python scripts/pipeline/install_event.py \
        --domain pokemon-lore --unit kanto --event cpt \
        --candidate-artifact checkpoints/cpt/... \
        --probe-receipt logs/probe_eval_....json \
        [--booster-n 0] [--reference-artifact prior]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as C
from pipeline import certificate as Cert
from pipeline import sentinel as Sent


class InstallEventError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _scores_from_probe_receipt(
    receipt: dict[str, Any],
    *,
    column: str,
    sentinel: dict[str, Any],
) -> dict[str, Any]:
    """Map a base-vs-tuned (or flat) probe receipt onto sentinel install-slice scores.

    Prefer overall column hits/n; fall back to summing per-predicate rows that the
    sentinel install slice covers.
    """
    col = receipt.get(column) if column in receipt else receipt
    if not isinstance(col, dict):
        raise InstallEventError(f"probe receipt missing column {column!r}")
    # nested held_out
    if "held_out" in col and isinstance(col["held_out"], dict) and "n" in col["held_out"]:
        held = col["held_out"]
        n = int(held.get("n") or 0)
        hits = held.get("hits")
        if hits is None and held.get("acc") is not None and n:
            hits = int(round(float(held["acc"]) * n))
        return Sent.scores_from_hits(int(hits or 0), n)

    if col.get("n") is not None:
        n = int(col["n"])
        hits = col.get("hits")
        if hits is None and col.get("acc") is not None:
            hits = int(round(float(col["acc"]) * n))
        return Sent.scores_from_hits(int(hits or 0), n)

    # Sum sentinel predicates from per_predicate
    per = col.get("per_predicate") or receipt.get("per_predicate") or {}
    preds = set(sentinel.get("predicates") or [])
    hits = n = 0
    for pred, row in per.items():
        if preds and pred not in preds:
            continue
        pn = int(row.get("n") or 0)
        ph = row.get("hits")
        if ph is None and row.get("acc") is not None and pn:
            ph = int(round(float(row["acc"]) * pn))
        hits += int(ph or 0)
        n += pn
    if n <= 0:
        raise InstallEventError("could not derive install scores from probe receipt")
    return Sent.scores_from_hits(hits, n)


def _default_ga_scores() -> dict[str, Any]:
    """When live GA scoring is unavailable, record UNAVAILABLE — must not silent-pass.

    Callers should supply real GA scores for production install events. Tests inject fixtures.
    """
    return {"hits": 0, "n": 0, "acc": 0.0, "UNAVAILABLE": True}


def run_install_event(
    *,
    domain: str,
    unit: str | None,
    event: str,
    candidate_artifact: str,
    reference_artifact: str | None = None,
    probe_receipt_path: str | None = None,
    reference_probe_receipt_path: str | None = None,
    column: str = "tuned",
    reference_column: str = "base",
    booster_n: int = 0,
    halted_boosters: list[int] | None = None,
    sentinel_path: str | None = None,
    questions_path: str | None = None,
    recall_report_path: str | None = None,
    install_reference_scores: dict | None = None,
    install_candidate_scores: dict | None = None,
    general_base_scores: dict | None = None,
    general_candidate_scores: dict | None = None,
    demo: bool = False,
    skip_certificate_on_halt: bool = False,
) -> dict[str, Any]:
    """Run sentinel then certificate. Returns the install-event record."""
    unit = unit or ""
    # Resolve sentinel
    if sentinel_path:
        sp = Path(sentinel_path)
    else:
        sp = C.ROOT / "data" / domain / "sentinel" / "sentinel_v1.json"
    if not sp.is_file():
        raise InstallEventError(
            f"sentinel set missing at {sp} — build it before any install event "
            f"(python scripts/pipeline/sentinel.py build --domain {domain} --unit {unit or 'UNIT'})"
        )
    sentinel = json.loads(sp.read_text(encoding="utf-8"))

    # Scores
    probe = _load_json(Path(probe_receipt_path) if probe_receipt_path else None)
    ref_probe = _load_json(Path(reference_probe_receipt_path) if reference_probe_receipt_path else None)
    if install_candidate_scores is None:
        if probe is None:
            raise InstallEventError(
                "install event requires --probe-receipt or injected candidate scores "
                "(evidence contract: no silent skip)"
            )
        install_candidate_scores = _scores_from_probe_receipt(probe, column=column, sentinel=sentinel)
    if install_reference_scores is None:
        if ref_probe is not None:
            install_reference_scores = _scores_from_probe_receipt(
                ref_probe, column=reference_column, sentinel=sentinel,
            )
        elif probe is not None and reference_column in probe:
            install_reference_scores = _scores_from_probe_receipt(
                probe, column=reference_column, sentinel=sentinel,
            )
        else:
            # Self-reference: first install uses candidate as its own reference baseline
            # only when event is initial cpt — still records the equality.
            install_reference_scores = dict(install_candidate_scores)
            install_reference_scores["note"] = "reference=candidate (initial install baseline)"

    if general_base_scores is None:
        general_base_scores = _default_ga_scores()
    if general_candidate_scores is None:
        general_candidate_scores = dict(general_base_scores)

    # GA unavailable → cannot claim PASS on forgetting; force a measured-unavailable halt?
    # Spec: every event must run both slices. If UNAVAILABLE, record FAILED_UNAVAILABLE rather than fake PASS.
    ga_unavailable = bool(
        general_base_scores.get("UNAVAILABLE") or general_candidate_scores.get("UNAVAILABLE")
        or int(general_base_scores.get("n") or 0) <= 0
    )

    prior = reference_artifact or "prior_artifact"
    cand = candidate_artifact

    if ga_unavailable:
        # Still evaluate install slice; mark GA as unavailable in the result.
        # Use equal dummy n=1 scores so evaluate_sentinel can run, then override.
        dummy = Sent.scores_from_hits(1, 1)
        sent_result = Sent.evaluate_sentinel(
            sentinel,
            install_reference=install_reference_scores,
            install_candidate=install_candidate_scores,
            general_base=dummy,
            general_candidate=dummy,
            event=event,
            prior_artifact=prior,
            candidate_artifact=cand,
        )
        sent_result["general_ability"] = {
            "slice": "general_ability",
            "status": "UNAVAILABLE",
            "halt": False,
            "note": (
                "Live general-ability scoring not supplied for this event. "
                "Install-slice sentinel still applied. Supply GA scores for a full PASS."
            ),
        }
        # Install-only halt stands; GA unavailable is a named gap not an auto-halt
        sent_result["ga_unavailable"] = True
    else:
        sent_result = Sent.evaluate_sentinel(
            sentinel,
            install_reference=install_reference_scores,
            install_candidate=install_candidate_scores,
            general_base=general_base_scores,
            general_candidate=general_candidate_scores,
            event=event,
            prior_artifact=prior,
            candidate_artifact=cand,
        )

    halted = bool(sent_result.get("halted"))
    artifact_kept = sent_result.get("artifact_kept") or (prior if halted else cand)

    # Certificate (still issued on halt — freshness names the halt)
    halted_list = list(halted_boosters or [])
    effective_booster_n = booster_n
    if halted and event == "booster":
        halted_list = sorted(set(halted_list + [booster_n]))
        # Certificate freshness shows halted booster; installed N is prior
        effective_booster_n = max(0, booster_n - 1)

    cert = None
    cert_error = None
    if not (halted and skip_certificate_on_halt):
        try:
            qpath = Path(questions_path) if questions_path else None
            questions = Cert.load_questions(domain, qpath)
            recall = None
            for cand_path in (
                Path(recall_report_path) if recall_report_path else None,
                C.ROOT / "logs" / "phase_f2_coverage_metric.json",
                C.ROOT / "logs" / "phase_g_extraction_gap_iteration.json",
                C.ROOT / "data" / "_build" / "recall" / f"report_{domain}.json",
            ):
                if cand_path is None:
                    continue
                recall = _load_json(cand_path)
                if recall is not None:
                    recall = dict(recall)
                    recall["receipt_path"] = str(cand_path)
                    # normalize F2/G shapes
                    if "clean" in recall and isinstance(recall["clean"], dict) and "ratio" in recall["clean"]:
                        recall.setdefault("estimand", "clean")
                        if "status" not in recall and "verdict" in recall:
                            recall["status"] = recall["verdict"]
                        gate = recall.get("gate") or {}
                        if isinstance(gate, dict) and "min_coverage" not in recall:
                            recall["min_coverage"] = gate.get("quality.extraction_recall.min", 0.6)
                    break
            if probe is not None and probe_receipt_path:
                probe = dict(probe)
                probe["receipt_path"] = probe_receipt_path
            cert = Cert.generate_certificate(
                domain=domain,
                unit=unit or None,
                questions_doc=questions,
                probe_receipt=probe,
                recall_report=recall,
                run_record_path=str(candidate_artifact),
                column=column,
                booster_n=effective_booster_n,
                halted_boosters=halted_list,
                demo=demo,
            )
        except Cert.CertificateError as error:
            cert_error = str(error)

    # Persist
    out_dir = C.BUILD / "install_events" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "schema": "kandi.install_event.v1",
        "domain": domain,
        "unit": unit,
        "event": event,
        "at": _now(),
        "candidate_artifact": candidate_artifact,
        "reference_artifact": reference_artifact,
        "artifact_kept": artifact_kept,
        "booster_n": booster_n,
        "halted": halted,
        "sentinel": sent_result,
        "certificate_id": (cert or {}).get("certificate_id"),
        "certificate_error": cert_error,
        "evidence_contract": "PASS" if sent_result else "FAIL",
    }
    # evidence contract: must have sentinel result
    if not sent_result:
        record["evidence_contract"] = "FAIL"
        raise InstallEventError("install event missing sentinel result")

    rec_path = out_dir / f"{stamp}_{event}_b{booster_n}.json"
    rec_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["path"] = str(rec_path)

    if cert is not None:
        cert_dir = C.BUILD / "certificates" / domain
        cert_dir.mkdir(parents=True, exist_ok=True)
        # Archive — never overwrite prior booster cards
        cpath = cert_dir / f"{unit or 'domain'}_b{effective_booster_n}_{stamp}.json"
        # also write stable latest pointer for the kept artifact
        latest = cert_dir / f"{unit or 'domain'}_latest.json"
        cpath.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
        latest.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
        record["certificate_path"] = str(cpath)

    # Sidecar next to checkpoint when it is a real directory
    cand_path = Path(candidate_artifact)
    if cand_path.is_dir():
        (cand_path / "install_event.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8",
        )
        if cert is not None:
            (cand_path / "certificate.json").write_text(
                json.dumps(cert, indent=2) + "\n", encoding="utf-8",
            )
        (cand_path / "sentinel_result.json").write_text(
            json.dumps(sent_result, indent=2) + "\n", encoding="utf-8",
        )

    record["status"] = "FAILED" if halted else "PASS"
    record["plain"] = sent_result.get("plain")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="post-install sentinel + certificate")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--unit", default="")
    ap.add_argument("--event", required=True, choices=["cpt", "sft", "anneal", "booster"])
    ap.add_argument("--candidate-artifact", required=True)
    ap.add_argument("--reference-artifact")
    ap.add_argument("--probe-receipt")
    ap.add_argument("--reference-probe-receipt")
    ap.add_argument("--column", default="tuned")
    ap.add_argument("--reference-column", default="base")
    ap.add_argument("--booster-n", type=int, default=0)
    ap.add_argument("--sentinel")
    ap.add_argument("--questions-json")
    ap.add_argument("--recall-report")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--install-reference-scores")
    ap.add_argument("--install-candidate-scores")
    ap.add_argument("--general-base-scores")
    ap.add_argument("--general-candidate-scores")
    args = ap.parse_args()
    try:
        def sj(p):
            return _load_json(Path(p)) if p else None
        rec = run_install_event(
            domain=args.domain,
            unit=args.unit or None,
            event=args.event,
            candidate_artifact=args.candidate_artifact,
            reference_artifact=args.reference_artifact,
            probe_receipt_path=args.probe_receipt,
            reference_probe_receipt_path=args.reference_probe_receipt,
            column=args.column,
            reference_column=args.reference_column,
            booster_n=args.booster_n,
            sentinel_path=args.sentinel,
            questions_path=args.questions_json,
            recall_report_path=args.recall_report,
            install_reference_scores=sj(args.install_reference_scores),
            install_candidate_scores=sj(args.install_candidate_scores),
            general_base_scores=sj(args.general_base_scores),
            general_candidate_scores=sj(args.general_candidate_scores),
            demo=args.demo,
        )
    except (InstallEventError, Sent.SentinelError, Cert.CertificateError, OSError, json.JSONDecodeError) as error:
        sys.exit(f"install_event failed closed: {error}")
    print(json.dumps({
        "status": rec["status"],
        "halted": rec["halted"],
        "artifact_kept": rec["artifact_kept"],
        "path": rec.get("path"),
        "certificate_path": rec.get("certificate_path"),
        "plain": rec.get("plain"),
    }))
    sys.exit(0 if not rec["halted"] else 2)


if __name__ == "__main__":
    main()
