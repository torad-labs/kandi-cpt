#!/usr/bin/env python3
"""Zero-knob derivation table (docs/cpt-overhaul/06).

Measures corpus/disk state and emits every product knob WITH its derivation line.
Nothing here is a training decision frozen into weights — it is the plan the operator
approves once. Operator overrides always win and are recorded beside the derived default.

    PYTHONPATH=scripts python scripts/pipeline/derive_plan.py --domain <d> [--editions …]

Wired into plan_install.build_plan as the `derivations` section.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config as C
from curate_corpus import derive_drop_frac
from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_contrast import load_contrast


# User-facing failure sentences (gate FAIL → one action). Living next to traps in playbooks too.
FAILURE_SENTENCES = {
    "diversity": {
        "sentence": "The paraphrases came out too same-y; I'm regenerating with more variety.",
        "action": "re-run synth (persona variation ON) then dedup_surfaces; do not train yet",
        "playbook": "cpt/09-style.md",
    },
    "audit_recall": {
        "sentence": "The graph missed too many facts from the source docs; I'm re-extracting, not re-training.",
        "action": "second extraction pass on weak predicates named by the report",
        "playbook": "cpt/04c-audit-recall.md",
    },
    "pii": {
        "sentence": "Some of your files look like they contain personal info; those are quarantined and won't be trained on.",
        "action": "review data/_build/pii/quarantine_*.json; re-run pii_scan after redaction if needed",
        "playbook": "cpt/03a-pii-scan.md",
    },
    "exposure_floor": {
        "sentence": "This material only supports installing about N facts well — I'll target that ceiling, not fake volume.",
        "action": "accept the plan ceiling or add source docs; never hydrate 40× without forms",
        "playbook": "cpt/05-hydrate.md",
    },
    "style_minimum": {
        "sentence": "Not enough of your writing to learn your voice — add exports or notes and try again.",
        "action": "refuse train; require ≥2k messages or ≥20k tokens for style domains",
        "playbook": "cpt/09-style.md",
    },
    "hard_negatives": {
        "sentence": "I can't build hard-negative drills for this domain yet (no confusable groups authored).",
        "action": "author a contrast block or accept DOMAIN_HARD_NEGATIVES_UNAVAILABLE",
        "playbook": "cpt/09-style.md",
    },
}


def _count_clean_docs(domain: str, editions: str | None) -> int:
    n = 0
    for ed in iter_editions(domain, only=editions, with_raw=False):
        root = ed.dir / "10_clean"
        if not root.is_dir():
            continue
        n += sum(1 for p in root.rglob("*.md") if p.name != "README.md")
    return n


def _count_triples(domain: str, editions: str | None) -> int:
    n = 0
    for ed in iter_editions(domain, only=editions, with_raw=False):
        for name in ("triples.resolved.jsonl", "triples.jsonl"):
            path = ed.dir / "15_structured" / name
            if path.is_file():
                n += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                break
    return n


def _count_hydrated_surfaces(domain: str, editions: str | None) -> int:
    n = 0
    for ed in iter_editions(domain, only=editions, with_raw=False):
        hyd = ed.dir / "25_hydrated"
        if not hyd.is_dir():
            continue
        for jf in hyd.glob("*.jsonl"):
            n += sum(1 for line in jf.read_text(encoding="utf-8").splitlines() if line.strip())
    return n


def _effective_forms_estimate(domain: str, editions: str | None) -> int | None:
    """Prefer dedup report total forms; else None (not yet measured)."""
    summary = C.BUILD / "dedup" / domain / "summary.json"
    if summary.is_file():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            # sum per-unit after counts as a lower bound on forms if forms not present
            units = data.get("units") or []
            if units:
                return int(sum(u.get("n_after", 0) for u in units))
        except Exception:
            pass
    return None


def _tokenize_blocks(domain: Domain) -> tuple[int | None, str]:
    """Read block count from the domain's cpt tokenized stats if present."""
    ref = (domain.configs or {}).get("cpt")
    # fallback: common pokemon/edc names
    candidates: list[Path] = []
    if ref:
        # configs may name a yaml whose data.tokenized points at arrow basename
        try:
            import yaml
            cfg_path = Path(ref) if Path(ref).is_absolute() else C.ROOT / ref
            if cfg_path.is_file():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                tok = ((cfg.get("data") or {}).get("tokenized")) or ""
                if tok:
                    base = Path(str(tok)).name.replace(".arrow", "")
                    candidates.append(C.BUILD / "30_tokenized" / f"{base}_stats.json")
        except Exception:
            pass
    # domain-named fallbacks
    for name in (f"{domain.name.replace('-', '_')}_kanto_v1_stats.json",
                 f"{domain.name}_stats.json", "corpus_v1_stats.json"):
        candidates.append(C.BUILD / "30_tokenized" / name)
    for path in candidates:
        if path.is_file():
            try:
                stats = json.loads(path.read_text(encoding="utf-8"))
                blocks = stats.get("num_chunks") or stats.get("n_blocks")
                if blocks is not None:
                    return int(blocks), f"tokenize stats {path.relative_to(C.ROOT)}"
            except Exception:
                continue
    return None, "tokenize stats absent — max_steps deferred until after tokenize"


def _contrast_authorable(domain_name: str) -> bool:
    try:
        groups, preds = load_contrast(domain_name)
        return bool(groups and preds)
    except Exception:
        return False


def _row(knob: str, value: Any, derivation: str, *,
         override: Any = None, source: str = "derived",
         user_note: str | None = None) -> dict[str, Any]:
    row = {
        "knob": knob,
        "value": value,
        "derivation": derivation,
        "source": "override" if override is not None else source,
    }
    if override is not None:
        row["derived_default"] = value
        row["override"] = override
        row["value"] = override
        row["derivation"] = f"operator-override={override} (derived default was {value}: {derivation})"
    if user_note:
        row["user_note"] = user_note
    return row


def derive_knobs(
    domain_name: str,
    editions: str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    device_only: bool = False,
    domain_kind: str = "facts",
) -> dict[str, Any]:
    """Emit the master derivation table for one domain scope."""
    overrides = overrides or {}
    try:
        domain = Domain(domain_name)
    except Exception as exc:
        return {
            "domain": domain_name,
            "ok": False,
            "error": str(exc),
            "knobs": [],
            "ladder": [],
            "failure_sentences": FAILURE_SENTENCES,
        }

    # Prefer manifest kind (Domain.kind / spec.kind); caller override only if facts default.
    manifest_kind = str(getattr(domain, "kind", None) or (domain.spec or {}).get("kind") or domain_kind).lower()
    if manifest_kind in {"facts", "style", "mixed"}:
        domain_kind = manifest_kind

    n_docs = _count_clean_docs(domain_name, editions)
    n_triples = _count_triples(domain_name, editions)
    n_surfaces = _count_hydrated_surfaces(domain_name, editions)
    forms = _effective_forms_estimate(domain_name, editions)
    contrast_ok = _contrast_authorable(domain_name)
    blocks, blocks_src = _tokenize_blocks(domain)

    # Always compute the pure derivation first; apply override only in _row so
    # derived_default stays auditable beside the operator choice.
    drop_frac, drop_der = derive_drop_frac(n_docs, None)
    # recall sample: max(20, 2% kept), cap 100
    recall_n = max(20, int(0.02 * max(n_docs, 1)))
    recall_n = min(100, recall_n)
    if n_docs < 50 and domain_kind == "facts":
        recall_note = "corpus<50 docs → recall audit skipped"
        recall_value: Any = 0
        recall_der = recall_note
    else:
        recall_value = recall_n
        recall_der = f"max(20, 2% of {n_docs} docs)= {recall_n}, cap 100"

    # contrast fraction
    if contrast_ok:
        contrast_frac, contrast_der = 0.10, "contrast block authorable → 10% soft-cap (Phase C)"
    else:
        contrast_frac, contrast_der = 0.0, "no contrast block → 0 + DOMAIN_HARD_NEGATIVES_UNAVAILABLE"

    # SFT hard-neg defaults
    if contrast_ok:
        wrong_n, wrong_der = 20, "contrast authorable → 20 wrong-value negs/unit"
        absent_n, absent_der = 10, "contrast authorable → 10 absent-entity negs/unit"
    else:
        wrong_n, wrong_der = 0, "no contrast → 0 + DOMAIN_HARD_NEGATIVES_UNAVAILABLE"
        absent_n, absent_der = 0, "no contrast → 0 + DOMAIN_HARD_NEGATIVES_UNAVAILABLE"

    # exposure / surface multiple
    exposure_cap = 60
    try:
        from qgre import spec as S
        exposure_cap = int(S.relations_cfg(S.load_spec(domain_name)).get("exposure_cap_per_fact", 60))
    except Exception:
        pass
    if forms is not None and forms > 0:
        exposure_der = f"post-dedup effective surfaces≈{forms}; cap/fact={exposure_cap}"
        exposure_val: Any = forms
    elif n_surfaces > 0:
        exposure_der = f"raw hydrated surfaces={n_surfaces} (dedup not run — forms unmeasured)"
        exposure_val = n_surfaces
    else:
        exposure_der = "no hydrated surfaces yet — exposure deferred until after hydrate/dedup"
        exposure_val = None

    # max_steps from blocks
    micro, accum, epochs = 1, 16, 3
    if blocks is not None and blocks > 0:
        max_steps = max(1, (blocks * epochs) // (micro * accum))
        steps_der = f"blocks({blocks})×epochs({epochs})÷(micro({micro})×accum({accum})) = {max_steps} [{blocks_src}]"
    else:
        max_steps = None
        steps_der = blocks_src

    # anneal
    if max_steps is None:
        anneal_on, anneal_der = None, "deferred until max_steps known"
    elif max_steps >= 500:
        anneal_on, anneal_der = True, f"max_steps({max_steps})≥500 → anneal ON (final 10%)"
    else:
        anneal_on, anneal_der = False, f"max_steps({max_steps})<500 → anneal OFF"

    # generative probe count — needs probe gold size; estimate from probes.jsonl when present
    gen_n = 30
    gen_der = "default min 30 (20% of gold, max 100) — refined when probes.jsonl exists"
    try:
        for ed in iter_editions(domain_name, only=editions, with_raw=False):
            probes = ed.dir / "probes.jsonl"
            if probes.is_file():
                n_p = sum(1 for line in probes.read_text(encoding="utf-8").splitlines() if line.strip())
                gen_n = max(30, min(100, int(0.20 * n_p)))
                gen_der = f"20% of {n_p} probes → {gen_n} (min 30, max 100)"
                break
    except Exception:
        pass

    knobs = [
        _row("domain_kind", domain_kind,
             "agent classifies at ingest; user confirms in the same plan approval",
             override=overrides.get("domain_kind"),
             user_note="judgment call — not a measured knob"),
        _row("drop_frac", drop_frac, drop_der, override=overrides.get("drop_frac")),
        _row("educational_scorer", "teacher",
             "default teacher path; distilled when cached-grade coverage suffices (01 v2)",
             override=overrides.get("educational_scorer")),
        _row("recall_audit_sample", recall_value, recall_der),
        _row("contrast_fraction", contrast_frac, contrast_der,
             override=overrides.get("contrast_fraction")),
        _row("sft_wrong_value_neg", wrong_n, wrong_der,
             override=overrides.get("sft_wrong_value_neg")),
        _row("sft_absent_entity_neg", absent_n, absent_der,
             override=overrides.get("sft_absent_entity_neg")),
        _row("exposure_budget_forms", exposure_val, exposure_der),
        _row("exposure_cap_per_fact", exposure_cap,
             "manifest relations.exposure_cap_per_fact (compute footprint, not quality FAIL)"),
        _row("train.max_steps", max_steps, steps_der,
             override=overrides.get("max_steps")),
        _row("train.epochs_sizing", epochs,
             "default 3 epochs over blocks; size against #760 effective-form floor when forms known"),
        _row("replay_fraction", 0.10,
             "fixed 0.10; raised mechanically if general-ability probe drops",
             override=overrides.get("replay_fraction")),
        _row("anneal.enabled", anneal_on, anneal_der,
             override=overrides.get("anneal_enabled")),
        _row("generative_probe_count", gen_n, gen_der),
        _row("pii_threshold", "default",
             "fixed default; only the operator tightens",
             override=overrides.get("pii_threshold")),
    ]

    # degradation ladder rungs (what the plan will DO given measured state)
    ladder = _degradation_ladder(
        domain_kind=domain_kind,
        n_docs=n_docs,
        n_triples=n_triples,
        contrast_ok=contrast_ok,
        device_only=device_only,
        forms=forms,
        exposure_cap=exposure_cap,
    )

    deviations = [r for r in ladder if r.get("status") in ("skip", "refuse", "degrade")]
    plain = _plain_language_summary(
        domain_name=domain_name,
        n_docs=n_docs,
        n_triples=n_triples,
        n_surfaces=n_surfaces,
        knobs=knobs,
        ladder=ladder,
        device_only=device_only,
    )

    return {
        "domain": domain_name,
        "editions": editions,
        "ok": True,
        "measured": {
            "n_clean_docs": n_docs,
            "n_triples": n_triples,
            "n_hydrated_surfaces": n_surfaces,
            "effective_forms_estimate": forms,
            "tokenize_blocks": blocks,
            "contrast_authorable": contrast_ok,
            "device_only": device_only,
            "domain_kind": domain_kind,
        },
        "knobs": knobs,
        "ladder": ladder,
        "deviations": deviations,
        "failure_sentences": FAILURE_SENTENCES,
        "plain_language": plain,
    }


def _degradation_ladder(
    *,
    domain_kind: str,
    n_docs: int,
    n_triples: int,
    contrast_ok: bool,
    device_only: bool,
    forms: int | None,
    exposure_cap: int,
) -> list[dict[str, Any]]:
    rungs: list[dict[str, Any]] = []

    # thin facts corpus
    if domain_kind == "facts" and n_docs < 50:
        rungs.append({
            "rung": "thin_facts_corpus",
            "status": "skip",
            "when": f"n_docs={n_docs}<50",
            "behavior": "curate skipped (nothing to prune); recall audit skipped",
            "user_sentence": "Not much source material yet — I'll skip pruning and the coverage audit and say so in the report.",
        })
    else:
        rungs.append({
            "rung": "thin_facts_corpus",
            "status": "ok",
            "when": f"n_docs={n_docs}",
            "behavior": "curate + recall audit run normally",
        })

    # style minimum
    if domain_kind == "style" and n_docs < 50:  # proxy; real style uses msg/token counts
        rungs.append({
            "rung": "style_minimum",
            "status": "refuse",
            "when": "style corpus below minimum",
            "behavior": "refuse train",
            "user_sentence": FAILURE_SENTENCES["style_minimum"]["sentence"],
        })
    else:
        rungs.append({
            "rung": "style_minimum",
            "status": "ok" if domain_kind != "style" else "ok",
            "when": f"domain_kind={domain_kind}",
            "behavior": "not a style refuse path",
        })

    # triple exposure feasibility
    if n_triples > 0 and n_triples < 10:
        rungs.append({
            "rung": "exposure_floor_feasibility",
            "status": "degrade",
            "when": f"n_triples={n_triples}<10",
            "behavior": "plan states a low install ceiling; never fake volume",
            "user_sentence": (
                f"This much material supports only a handful of well-installed facts "
                f"(~{n_triples}); I'll target that ceiling."
            ),
        })
    else:
        rungs.append({
            "rung": "exposure_floor_feasibility",
            "status": "ok",
            "when": f"n_triples={n_triples}",
            "behavior": "normal exposure sizing",
        })

    # contrast / hard-negs
    if not contrast_ok:
        rungs.append({
            "rung": "no_contrast_block",
            "status": "degrade",
            "when": "contrast block missing or empty",
            "behavior": "DOMAIN_HARD_NEGATIVES_UNAVAILABLE; weaker anti-confusable recipe",
            "user_sentence": FAILURE_SENTENCES["hard_negatives"]["sentence"],
        })
    else:
        rungs.append({
            "rung": "no_contrast_block",
            "status": "ok",
            "when": "contrast block authorable",
            "behavior": "CPT contrast ~10% + SFT hard-negs 20/10",
        })

    # device-only
    if device_only:
        rungs.append({
            "rung": "device_only",
            "status": "degrade",
            "when": "provider stages refused / device-only",
            "behavior": "score/audit_recall/synth omit with named deviations",
            "user_sentence": (
                "No cloud LLM passes on this run — install will be weaker (template floor only) "
                "and I'll say so in the verdict."
            ),
        })
    else:
        rungs.append({
            "rung": "device_only",
            "status": "ok",
            "when": "provider available",
            "behavior": "full recipe",
        })

    return rungs


def _plain_language_summary(
    *,
    domain_name: str,
    n_docs: int,
    n_triples: int,
    n_surfaces: int,
    knobs: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    device_only: bool,
) -> str:
    knob_lines = []
    for k in knobs:
        if k["value"] is None:
            continue
        knob_lines.append(f"  • {k['knob']} = {k['value']}  ({k['derivation']})")
    devs = [r for r in ladder if r.get("status") != "ok"]
    dev_txt = ""
    if devs:
        dev_txt = "\nRisks / degradations:\n" + "\n".join(
            f"  • {r.get('user_sentence') or r['behavior']}" for r in devs
        )
    return (
        f"Here's what you gave me for `{domain_name}`: {n_docs} clean docs, "
        f"{n_triples} graph facts, {n_surfaces} training surfaces so far.\n"
        f"Here's what I'll do (every number is derived; say the word to override):\n"
        + "\n".join(knob_lines)
        + dev_txt
        + ("\n\nApprove this plan to proceed; nothing GPU runs until you do." if not device_only
           else "\n\nDevice-only mode: provider stages stay off.")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions", default=None)
    ap.add_argument("--device-only", action="store_true")
    ap.add_argument("--domain-kind", default="facts", choices=["facts", "style", "mixed"])
    ap.add_argument("--overrides", default=None, help="JSON object of knob→value overrides")
    args = ap.parse_args()
    overrides = json.loads(args.overrides) if args.overrides else {}
    plan = derive_knobs(
        args.domain,
        args.editions,
        overrides=overrides,
        device_only=args.device_only,
        domain_kind=args.domain_kind,
    )
    json.dump(plan, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
