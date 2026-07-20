"""Automatic post-training trust chain: live eval → sentinel → install event → certificate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config as C
from cpt_config import CptConfig
from pipeline import install_event as IE
from pipeline import sentinel as Sent


class AutoChainError(RuntimeError):
    """The automatic install chain failed its evidence contract."""


@dataclass
class ProbeEvidence:
    certificate_path: Path
    certificate_report: dict[str, Any]
    install_strength: dict[str, Any]
    receipts: dict[str, str]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rooted(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else C.ROOT / path


def _default_probe_runner(cfg: CptConfig, config_path: str | Path, candidate: str) -> ProbeEvidence:
    """Run the real configured CPT eval axes and retain forced-choice for the certificate spine."""
    from eval_probes import run_probe_comparison

    fc_path, fc_report = run_probe_comparison(cfg, str(config_path), candidate)
    receipts = {"forced_choice": str(fc_path)}
    install_strength: dict[str, Any] = {
        "status": fc_report.get("status", "FAIL"),
        "estimand": fc_report.get("estimand") or "augmentation_holdout",
        "forced_choice": fc_report,
        "generative": {"status": "DISABLED", "reason": "eval.generative_enabled=false"},
    }
    if cfg.eval.generative_enabled:
        from eval_generative import merge_dual_axis, run_generative_comparison

        gen_path, gen_report = run_generative_comparison(cfg, str(config_path), candidate)
        dual = merge_dual_axis(fc_report, gen_report)
        dual_dir = _rooted(cfg.eval.probe_report_root)
        dual_dir.mkdir(parents=True, exist_ok=True)
        dual_path = dual_dir / f"{cfg.run.run_name}_install_strength.json"
        dual_path.write_text(json.dumps(dual, indent=2) + "\n", encoding="utf-8")
        receipts.update({"generative": str(gen_path), "dual_axis": str(dual_path)})
        install_strength = {
            "status": dual.get("status", "FAIL"),
            "estimand": fc_report.get("estimand") or "augmentation_holdout",
            "forced_choice": dual.get("forced_choice"),
            "generative": dual.get("generative"),
            "rule": dual.get("rule"),
        }
    return ProbeEvidence(Path(fc_path), fc_report, install_strength, receipts)


def _normalize_probe_evidence(value: Any) -> ProbeEvidence:
    if isinstance(value, ProbeEvidence):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        path, report = value
        report = dict(report)
        return ProbeEvidence(
            certificate_path=Path(path),
            certificate_report=report,
            install_strength={
                "status": report.get("status", "MEASURED"),
                "estimand": report.get("estimand") or "augmentation_holdout",
                "forced_choice": report,
                "generative": {"status": "NOT_IN_FIXTURE"},
            },
            receipts={"forced_choice": str(path)},
        )
    raise AutoChainError("probe runner returned an unknown evidence shape")


def _score_receipt_path(domain: str, event: str, booster_n: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return C.BUILD / "install_chains" / domain / f"{stamp}_{event}_b{booster_n}_sentinel_scores.json"


def _write_chain(chain: dict[str, Any]) -> Path:
    out_dir = C.BUILD / "install_chains" / chain["domain"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}_{chain['event']}_b{chain['booster_n']}.json"
    path.write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
    return path


def run_auto_chain(
    config_path: str | Path,
    *,
    mode: str | None = None,
    probe_runner: Callable[..., Any] | None = None,
    install_scorer: Callable[..., dict[str, Any]] | None = None,
    general_scorer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the whole post-training chain from one typed config; no manual receipt stitching."""
    cfg = CptConfig.from_yaml(config_path, mode)
    install = cfg.install
    if not install.enabled:
        return {"schema": "kandi.auto_chain.v1", "status": "DISABLED", "config": str(config_path)}

    candidate = str(_rooted(cfg.run.save_dir))
    reference = str(_rooted(install.reference_artifact or cfg.model.model_path))
    if not Path(candidate).is_dir():
        raise AutoChainError(f"candidate checkpoint missing after training: {candidate}")
    sentinel_path = _rooted(install.sentinel_path) if install.sentinel_path else Sent.latest_sentinel_path(install.domain)
    if not sentinel_path.is_file():
        raise AutoChainError(f"sentinel set missing: {sentinel_path}")
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))

    evidence = _normalize_probe_evidence(
        (probe_runner or _default_probe_runner)(cfg, config_path, candidate),
    )
    if not evidence.certificate_path.is_file():
        raise AutoChainError(f"real probe receipt missing: {evidence.certificate_path}")

    if install_scorer is None or general_scorer is None:
        from pipeline.score_sentinel_live import score_general_ability, score_install_slice

        install_scorer = install_scorer or score_install_slice
        general_scorer = general_scorer or score_general_ability
    install_items = list(sentinel.get("install_slice") or [])
    general_items = list(sentinel.get("general_ability_slice") or [])
    reference_install = install_scorer(model_path=reference, device=cfg.eval.device, probes=install_items)
    candidate_install = install_scorer(model_path=candidate, device=cfg.eval.device, probes=install_items)
    reference_general = general_scorer(
        model_path=reference, device=cfg.eval.device, items=general_items, domain=install.domain,
    )
    candidate_general = general_scorer(
        model_path=candidate, device=cfg.eval.device, items=general_items, domain=install.domain,
    )

    score_receipt = {
        "schema": "kandi.sentinel_live_scores.v1",
        "domain": install.domain,
        "unit": install.unit,
        "event": install.event,
        "booster_n": install.booster_n,
        "sentinel_content_sha256": sentinel.get("content_sha256"),
        "device": cfg.eval.device,
        "estimand": "frozen_sentinel_install_slice_and_general_ability",
        "scored_at": _now(),
        "reference": {"artifact": reference, "install": reference_install, "general_ability": reference_general},
        "candidate": {"artifact": candidate, "install": candidate_install, "general_ability": candidate_general},
    }
    score_path = _score_receipt_path(install.domain, install.event, install.booster_n)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score_receipt, indent=2) + "\n", encoding="utf-8")

    event = IE.run_install_event(
        domain=install.domain,
        unit=install.unit,
        event=install.event,
        candidate_artifact=candidate,
        reference_artifact=reference,
        probe_receipt_path=str(evidence.certificate_path),
        reference_probe_receipt_path=str(evidence.certificate_path),
        install_reference_scores=reference_install,
        install_candidate_scores=candidate_install,
        general_base_scores=reference_general,
        general_candidate_scores=candidate_general,
        booster_n=install.booster_n,
        sentinel_path=str(sentinel_path),
        questions_path=install.questions_path or None,
        recall_report_path=install.recall_report_path or None,
        demo=install.demo,
    )
    if not event.get("certificate_path"):
        raise AutoChainError(f"certificate generation failed closed: {event.get('certificate_error')}")

    chain = {
        "schema": "kandi.auto_chain.v1",
        "status": "HALTED" if event["halted"] else "CLEAR",
        "run_gate": {"status": "PASS", "estimand": "completed configured training run"},
        "install_strength": evidence.install_strength,
        "domain": install.domain,
        "unit": install.unit,
        "event": install.event,
        "booster_n": install.booster_n,
        "candidate_artifact": candidate,
        "reference_artifact": reference,
        "artifact_kept": event["artifact_kept"],
        "sentinel": {
            "status": event["sentinel"]["status"],
            "role": event["sentinel"]["role"],
            "tripwire_action": event["sentinel"]["tripwire_action"],
            "estimand": score_receipt["estimand"],
        },
        "sentinel_score_receipt": str(score_path),
        "probe_receipts": evidence.receipts,
        "install_event_path": event["path"],
        "certificate_path": event["certificate_path"],
        "certificate_id": event.get("certificate_id"),
        "demo": install.demo,
        "generated_at": _now(),
        "reveal_handoff": {
            "certificate_path": event["certificate_path"],
            "plain": event.get("plain"),
        },
    }
    chain_path = _write_chain(chain)
    chain["path"] = str(chain_path)
    # Keep the persisted record self-contained, including its own archive path.
    chain_path.write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
    (chain_path.parent / "latest.json").write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
    (Path(candidate) / "auto_chain.json").write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
    return chain
