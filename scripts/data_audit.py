#!/usr/bin/env python3
"""Data-distribution validator — does the corpus match what we now KNOW installs knowledge?

We've learned (from the dissection + the literature) what a corpus needs to install facts and keep
confusable topics apart in latent space:
  • DENSITY      — facts need hundreds→~1000 exposures; the long tail below that is what fabricates.
  • BALANCE      — relation types shouldn't be wildly skewed (9k artist→stage vs 0.8k set_time starves the rest).
  • COVERAGE     — thin editions (few triples) are the ones the model invents venues for.
  • REPLAY       — 5–10% generic text prevents catastrophic forgetting; 0% collapses NLU, >10% wastes budget.
  • DISJOINTNESS — confusable groups need distinctive prose AND zero co-occurrence so they separate in latent
                   space; we verify this statically (vocab + co-occurrence) AND with a GPU pass that embeds the
                   prose with the BASE model — is it distinctive enough to separate *before* any training?

Each dimension yields a deterministic PASS / WARN / FAIL against versioned best-practice thresholds (constants
below, overridable, recorded in the output). Reuses the geometry-probe embed/cosine machinery and the
disjoint-context auditor rather than recomputing. Emits data_audit_data.js (window.DATA_AUDIT) + a ledger;
the standalone data_audit.html renders it and the war-room links to it.

    PYTHONPATH=scripts python scripts/data_audit.py [--editions las_vegas/2026] \
        [--corpus-stats data/_build/30_tokenized/corpus_all97_stats.json] \
        [--exposure checkpoints/cpt/edc_lv_2026_L1w3_int8] [--no-gpu] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import config as C
from pipeline.editions import iter_editions
from pipeline import disjoint_context as DC

# ---- best-practice thresholds (versioned; --override; recorded in output) -----------------------------
DENSITY_MIN = 200        # exposures/fact below this → fabrication-risk long tail (V6 hit stage@1 0.744 at ~170)
DENSITY_TARGET = 1000    # saturation target (EntiGraph 1000×)
REPLAY_BAND = (0.05, 0.10)
BALANCE_COV_MAX = 1.0    # relation-type coefficient-of-variation ceiling (skew)
COVERAGE_MIN_TRIPLES = 10  # an edition with fewer structured triples is "thin"
SEP_MIN = 0.04           # confusable-group latent separation floor (mean intra − inter cosine, base model)
# SFT refusal-balance band (the v4 lesson): too few refusal examples → fabricates (v2: 14% → fab 0.32);
# too many → over-refuses (v4: 34% → coverage 0). Below the band WARNs (fab risk); above FAILs (the cliff).
REFUSAL_BAND = (0.10, 0.25)


def verdict(ok_pass: bool, ok_warn: bool) -> str:
    return "pass" if ok_pass else ("warn" if ok_warn else "fail")


# ------------------------------------------------------------------ density
def dim_density(eds, exposure_ckpt):
    """Per-fact exposure distribution. Real exposures from a Lever-4 exposure.json when available; else the
    hydration line-count per fact (a proxy) so the all-editions audit still has a distribution."""
    counts, source = [], None
    if exposure_ckpt:
        ep = Path(exposure_ckpt) / "exposure.json"
        if ep.exists():
            exp = json.loads(ep.read_text()).get("exposure", {})
            counts = sorted(exp.values()); source = f"exposure.json ({Path(exposure_ckpt).name})"
    if not counts:                                  # proxy: hydration lines per fact (source_triple_ids)
        perfact = Counter()
        for ed in eds:
            rp = ed.dir / "25_hydrated/relations.jsonl"
            if not rp.exists():
                continue
            for l in rp.read_text(encoding="utf-8", errors="replace").splitlines():
                if not l.strip():
                    continue
                try:
                    o = json.loads(l)
                except Exception:
                    continue
                key = (o.get("source_triple_ids") or o.get("entity_ids") or ["?"])[0]
                perfact[str(key)] += 1
        counts = sorted(perfact.values()); source = "hydration line-counts (proxy — no exposure.json)"
    if not counts:
        return {"verdict": "fail", "source": "none", "note": "no density signal found", "n_facts": 0}
    below = sum(1 for c in counts if c < DENSITY_MIN)
    med = statistics.median(counts)
    frac_below = below / len(counts)
    # histogram (log-ish buckets toward the 1000× target)
    edges = [0, 50, 100, 200, 400, 700, 1000, 2000, 10**9]
    hist = [sum(1 for c in counts if edges[i] <= c < edges[i + 1]) for i in range(len(edges) - 1)]
    is_proxy = "proxy" in (source or "")
    # the line-count proxy is NOT on the exposure scale the thresholds were calibrated against (a fact with 2
    # hydration lines may be exposed 800× after tokenization+epochs). Never let it render as a gate-grade FAIL —
    # that would push the density lever off a non-measurement. Gate ONLY on a real exposure.json.
    vd = "skipped" if is_proxy else verdict(frac_below < 0.25 and med >= DENSITY_MIN, frac_below < 0.5)
    return {"verdict": vd, "gated": not is_proxy,
            "source": source, "n_facts": len(counts), "median": med, "max": max(counts),
            "below_min": below, "frac_below_min": frac_below, "hist": hist, "hist_edges": edges,
            "target": DENSITY_TARGET, "min": DENSITY_MIN}


# ------------------------------------------------------------------ relation balance
def dim_relation_balance(eds):
    rel = Counter()
    for ed in eds:
        rp = ed.dir / "25_hydrated/relations.jsonl"
        if not rp.exists():
            continue
        for l in rp.read_text(encoding="utf-8", errors="replace").splitlines():
            if l.strip():
                try:
                    rel[json.loads(l).get("relation_type", "?")] += 1
                except Exception:
                    pass
    if not rel:
        return {"verdict": "fail", "note": "no hydrated relations found", "by_type": {}}
    vals = list(rel.values())
    cov = (statistics.pstdev(vals) / statistics.mean(vals)) if statistics.mean(vals) else 0.0
    return {"verdict": verdict(cov <= BALANCE_COV_MAX, cov <= 2 * BALANCE_COV_MAX),
            "by_type": dict(rel.most_common()), "cov": cov, "cov_max": BALANCE_COV_MAX,
            "n_types": len(rel), "total": sum(vals)}


# ------------------------------------------------------------------ cross-edition coverage
def dim_coverage(eds):
    per_ed, all_preds = {}, Counter()
    for ed in eds:
        tp = ed.dir / "15_structured/triples.jsonl"
        preds = Counter()
        if tp.exists():
            for l in tp.read_text(encoding="utf-8", errors="replace").splitlines():
                if l.strip():
                    try:
                        preds[json.loads(l).get("predicate", "?")] += 1
                    except Exception:
                        pass
        per_ed[ed.slug] = {"n_triples": sum(preds.values()), "predicates": dict(preds)}
        all_preds.update(preds)
    thin = sorted((s for s, d in per_ed.items() if d["n_triples"] < COVERAGE_MIN_TRIPLES),
                  key=lambda s: per_ed[s]["n_triples"])
    n = len(per_ed) or 1
    return {"verdict": verdict(len(thin) / n < 0.25, len(thin) / n < 0.5),
            "n_editions": len(per_ed), "thin_editions": thin, "n_thin": len(thin),
            "top_predicates": dict(all_preds.most_common(14)), "min_triples": COVERAGE_MIN_TRIPLES,
            # compact matrix for the heatmap: edition → counts for the top predicates
            "matrix_preds": [p for p, _ in all_preds.most_common(12)],
            "matrix": {s: [per_ed[s]["predicates"].get(p, 0) for p, _ in all_preds.most_common(12)]
                       for s in sorted(per_ed, key=lambda s: -per_ed[s]["n_triples"])[:30]}}


# ------------------------------------------------------------------ token mix & replay
def dim_token_mix(stats):
    if not stats:
        return {"verdict": "fail", "note": "no corpus stats", "tokens_by_tag": {}}
    total = stats.get("total_tokens", 0) or 1
    replay = stats.get("replay_tokens", 0)
    rfrac = replay / total
    lo, hi = REPLAY_BAND
    return {"verdict": verdict(lo <= rfrac <= hi, rfrac >= lo / 2),   # near-zero replay = real forgetting risk → FAIL, not WARN
            "tokens_by_tag": stats.get("tokens_by_tag", {}), "total_tokens": total,
            "replay_frac": rfrac, "replay_band": list(REPLAY_BAND),
            "entity_mask_frac": stats.get("entity_mask_frac"),
            "src_token_dist": stats.get("src_token_dist", {})}


# ------------------------------------------------------------------ disjoint-context (static)
def dim_disjoint_static(eds):
    groups = DC.GROUPS
    have_vocab = {g["key"]: (g["key"] in DC.SKELETONS) for g in groups}
    # co-occurrence audit over all hydrated lines (two confusable members in one line pulls them together)
    lines = []
    for ed in eds:
        rp = ed.dir / "25_hydrated/relations.jsonl"
        if not rp.exists():
            continue
        for l in rp.read_text(encoding="utf-8", errors="replace").splitlines():
            if l.strip():
                try:
                    lines.append(json.loads(l))
                except Exception:
                    pass
    if not lines:    # audited nothing → honest skip, NOT a vacuous "0 violations" PASS
        return {"verdict": "skipped", "note": "no hydrated relations in scope to audit",
                "scope": "las_vegas/2026 confusable groups", "n_groups": len(groups),
                "groups_with_vocab": sum(have_vocab.values()), "n_cooccurrence_violations": 0, "violation_samples": []}
    violations, per_edition, unchecked = _cooccurrence_by_edition(eds)
    n_missing = sum(1 for v in have_vocab.values() if not v)
    if not per_edition:
        # Scoping the check correctly must not turn a false FAIL into a vacuous PASS. If NO
        # edition in scope declares confusable_groups, nothing was audited and there is no clean
        # bill to give — same rule this dimension already applies when there are no lines at all.
        return {"verdict": "skipped",
                "note": f"no edition in scope declares confusable_groups — 0 of {len(unchecked)} "
                        "audited; the disjoint-context law is UNENFORCED here, not satisfied",
                "scope": "per-edition confusable pins (index_for_meta)",
                "n_groups": len(groups), "groups_with_vocab": sum(have_vocab.values()),
                "missing_vocab": [k for k, v in have_vocab.items() if not v],
                "n_editions_checked": 0, "n_editions_unchecked": len(unchecked),
                "unchecked_editions_sample": unchecked[:10],
                "n_cooccurrence_violations": 0, "violation_samples": []}
    return {"verdict": verdict(not violations and n_missing == 0, len(violations) < 5),
            "scope": "each edition audited against ITS OWN confusable_groups (index_for_meta); "
                     "an edition declaring none has no disjoint-context constraint and is reported "
                     "as unchecked, never judged by another edition's pins",
            "n_groups": len(groups), "groups_with_vocab": sum(have_vocab.values()),
            "missing_vocab": [k for k, v in have_vocab.items() if not v],
            "n_editions_checked": len(per_edition),
            "n_editions_unchecked": len(unchecked),
            "unchecked_editions_sample": unchecked[:10],
            "per_edition_violations": {k: v for k, v in per_edition.items() if v},
            "n_cooccurrence_violations": len(violations),
            "violation_samples": [v.get("text", str(v))[:160] for v in violations[:5]]}


def _cooccurrence_by_edition(eds) -> tuple[list, dict, list]:
    """Audit each edition against its own confusable index (h7c).

    This used one call with no index, which silently defaulted to the gold las_vegas/2026 globals
    and judged all 97 EDC editions by 2026's pins. The 938 "violations" were las_vegas/2016 lines
    naming circuitGROUNDS — factually correct there, and flagged only because 2026 pins Armin van
    Buuren to kineticFIELD. A cross-edition binding mismatch is not contamination.

    Editions declaring no confusable_groups are counted as UNCHECKED and named in the report rather
    than being quietly folded into a pass: no constraint is an honest gap, not a clean bill.
    """
    all_violations: list = []
    per_edition: dict[str, int] = {}
    unchecked: list[str] = []
    for ed in eds:
        rel_path = ed.dir / "25_hydrated/relations.jsonl"
        meta_path = ed.dir / "festival_meta.json"
        if not rel_path.exists():
            continue
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        index = DC.index_for_meta(meta)
        if not index[0]:                      # no confusable members declared for this edition
            unchecked.append(ed.slug)
            continue
        lines = []
        for raw in rel_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.strip():
                try:
                    lines.append(json.loads(raw))
                except Exception:
                    pass
        hits = DC.audit_cooccurrence(lines, *index)
        per_edition[ed.slug] = len(hits)
        all_violations.extend(hits)
    return all_violations, per_edition, unchecked


# ------------------------------------------------------------------ disjoint-context (GPU latent pass)
def _group_separation(X: np.ndarray, labels: list[str]) -> dict:
    from geometry_probe.metrics import cosine_matrix
    from sklearn.metrics import roc_auc_score
    cos = cosine_matrix(X)
    intra, inter = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(float(cos[i, j]))
    if not intra or not inter:
        return {"sep": float("nan"), "auc": float("nan"), "mean_intra": float("nan"), "mean_inter": float("nan")}
    y = [1] * len(intra) + [0] * len(inter)
    return {"sep": float(np.mean(intra) - np.mean(inter)), "auc": float(roc_auc_score(y, intra + inter)),
            "mean_intra": float(np.mean(intra)), "mean_inter": float(np.mean(inter)),
            "n_intra": len(intra), "n_inter": len(inter)}


def dim_disjoint_latent(eds, device, max_per_group=30):
    """Embed each confusable group's canonical disjoint-context prose with the BASE model; is it distinctive
    enough to separate the groups BEFORE training? Best-layer inter/intra cosine separation. The prose comes
    from disjoint_context.generate_for_group (the SAME deterministic skeletons hydration uses) — so this is a
    data-side check of the vocabulary worlds, not the trained model. Reuses geometry_probe.extract."""
    from geometry_probe import extract as E
    groups = [g for g in DC.GROUPS if g.get("skeletons")]
    texts, labels = [], []
    for g in groups:
        for ln in DC.generate_for_group(g)[:max_per_group]:
            texts.append(ln["text"]); labels.append(g["key"])
    if len(set(labels)) < 2 or len(texts) < 6:
        return {"verdict": "skipped", "note": "too few confusable groups/skeletons to embed"}
    model, _ = E.load_model(None, device)            # None = base model (load_model returns (model, resolved))
    tok = E.load_tokenizer()
    H = E.extract_all_layers(model, tok, texts, carrier="{entity}", pooling="mean", device=device)
    import gc, torch
    del model; gc.collect(); torch.cuda.empty_cache()
    per_layer = {li: _group_separation(X, labels) for li, X in H.items()}
    best = max((li for li in per_layer if not np.isnan(per_layer[li]["sep"])),
               key=lambda li: per_layer[li]["sep"], default=max(per_layer))
    b = per_layer[best]
    if b["sep"] != b["sep"]:    # NaN → every layer degenerate (no intra/inter pairs): couldn't MEASURE, not a data verdict
        return {"verdict": "skipped", "note": "latent separation undefined — all layers degenerate (no intra/inter pairs)",
                "scope": "las_vegas/2026 confusable groups", "n_lines": len(texts), "n_groups": len(groups)}
    return {"verdict": verdict(b["sep"] >= SEP_MIN, b["sep"] > 0),
            "scope": "las_vegas/2026 confusable groups", "best_layer": best, "n_layers": len(H),
            "n_lines": len(texts), "n_groups": len(groups),
            "sep": b["sep"], "auc": b["auc"], "mean_intra": b["mean_intra"], "mean_inter": b["mean_inter"],
            "sep_min": SEP_MIN, "per_layer_sep": {str(li): per_layer[li]["sep"] for li in sorted(per_layer)}}


# ------------------------------------------------------------------ SFT tiers (informational)
def dim_sft_tiers(sft_stats):
    if not sft_stats:
        return {"verdict": "skipped", "note": "no sft stats"}
    k, m, u = sft_stats.get("tier_known", 0), sft_stats.get("tier_maybe", 0), sft_stats.get("tier_unknown", 0)
    tot = (k + m + u) or 1
    return {"verdict": "pass", "known": k, "maybe": m, "unknown": u, "known_frac": k / tot,
            "dropped_unknown": sft_stats.get("dropped_unknown", 0), "abstention": sft_stats.get("abstention", 0),
            "raw": sft_stats.get("raw", 0)}


def dim_sft_composition(sft_stats, stats_path):
    """SFT refusal-balance — the dimension that would have caught v4 BEFORE the GPU-hour. The fraction of the
    SFT set that is refusal (abstention + absent-edition negatives) must sit in a safe band: too low fabricates,
    too high over-refuses (the v4 cliff: 34% → coverage collapsed to 0).

    The denominator is the EXACT post-dedup example count read from the sibling .arrow (num_rows) — NOT a tier
    proxy. A tier proxy overcounts answerable (it can't see dedup) and gave v4 a FALSE pass (0.24 vs the true
    0.34); reading the arrow makes it honest."""
    if not sft_stats:
        return {"verdict": "skipped", "note": "no sft stats"}
    lo, hi = REFUSAL_BAND
    ref = sft_stats.get("abstention", 0) + sft_stats.get("absent_neg", 0)
    arrow = Path(str(stats_path).replace("_stats.json", ".arrow"))
    total, src = None, "stats"
    if arrow.exists():
        try:
            from datasets import load_from_disk
            total = load_from_disk(str(arrow)).num_rows; src = "arrow num_rows (exact)"
        except Exception:
            total = None
    if total is None:   # no arrow → honest fallback, but say it's approximate (don't gate confidently)
        total = sft_stats.get("answerable", 0) + ref or (sft_stats.get("tier_known", 0) + sft_stats.get("tier_maybe", 0) + ref)
        src = "proxy (no .arrow — denominator approximate, treat verdict as indicative)"
    frac = ref / max(1, total)
    exact = src.startswith("arrow")
    # below band → fabrication risk (WARN); above → over-refusal cliff (FAIL — the more dangerous side).
    # If we couldn't get an exact denominator, never emit a confident PASS near the boundary — WARN instead.
    if not exact and lo <= frac <= hi:
        vd = "warn"
    else:
        vd = "pass" if lo <= frac <= hi else ("fail" if frac > hi else "warn")
    return {"verdict": vd, "refusal_frac": frac, "exact": exact, "denom_src": src,
            "answerable": total - ref, "refusal_total": ref, "total": total,
            "band": list(REFUSAL_BAND), "abstention": sft_stats.get("abstention", 0),
            "absent_neg": sft_stats.get("absent_neg", 0)}


import re as _re
# The calibration gate's held-out probe phrasings (must match eval_sft_calibration.build_probes). If ANY of
# these appears in the SFT training set — as a positive (direct leak → inflated gate) OR as a refusal-only
# (the v5/v6 confound → over-refusal bias on the probe) — the gate is no longer measuring generalization.
PROBE_PHRASINGS = ["what is the venue for edc <ed>?"]
_ABSTAIN_RE = _re.compile(r"don'?t have|no record|not aware|no data|don'?t know|can'?t (provide|find)", _re.I)


def _phrasing(q: str) -> str:
    """Canonicalize a question to its phrasing template: strip the chat-template scaffolding + role markers,
    blank the edition, collapse case/space. Stripping the turn markers is essential — otherwise the probe
    phrasing never matches (false-negative leak check) because the SFT copy is wrapped in <start_of_turn>…."""
    q = q.lower()
    q = _re.sub(r"<[^>]+>", " ", q)                  # <start_of_turn> / <end_of_turn> / any tag
    q = _re.sub(r"\b(start_of_turn|end_of_turn|user|model)\b", " ", q)
    q = _re.sub(r"edc\s+[a-z .]+?\s+\d{4}", "edc <ed>", q)
    return _re.sub(r"\s+", " ", q).strip()


def dim_sft_phrasing(sft_stats, stats_path):
    """Train/eval HYGIENE — the failure mode v5/v6 taught us: the eval probe phrasing must be HELD OUT of the
    SFT set, and a phrasing trained only as a refusal biases the model to over-refuse that phrasing on real
    editions too. Decodes the .arrow, canonicalizes every question to a phrasing template, and flags:
      • LEAK   — a probe phrasing appears in training (positive = inflated gate; refuse-only = over-refusal bias)
      • ASYM   — a non-IDK phrasing taught ONLY as refuse (never as answer) → blanket-refusal prior
      • THIN   — positive answer-phrasings per real edition (the signal that must beat the refusal prior)"""
    arrow = Path(str(stats_path).replace("_stats.json", ".arrow"))
    if not arrow.exists():
        return {"verdict": "skipped", "note": "no .arrow to decode for phrasing hygiene"}
    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))
        ds = load_from_disk(str(arrow))
    except Exception as e:
        return {"verdict": "skipped", "note": f"{type(e).__name__}: {e}"}
    ans_templates, ref_templates = Counter(), Counter()
    ans_eds = set()
    for r in ds:
        txt = tok.decode(r["input_ids"], skip_special_tokens=True)
        parts = _re.split(r"\bmodel\b", txt, maxsplit=1)
        q, a = parts[0], (parts[1] if len(parts) > 1 else "")
        if "edc" not in q.lower():
            continue
        tpl = _phrasing(q)
        if _ABSTAIN_RE.search(a):
            ref_templates[tpl] += 1
        else:
            ans_templates[tpl] += 1
            m = _re.search(r"edc\s+([a-z .]+?)\s+(\d{4})", q.lower())
            if m:
                ans_eds.add(m.group(1).strip() + "/" + m.group(2))
    leaked = [p for p in PROBE_PHRASINGS if p in ans_templates or p in ref_templates]
    # refuse-only phrasings that are NOT generic IDK forms (those legitimately only refuse) — i.e. a structured
    # question form the model only ever saw refused → it will over-refuse that form on real editions.
    asym = [t for t in ref_templates if t not in ans_templates and "edc <ed>" in t]
    pos_per_ed = (sum(ans_templates.values()) / len(ans_eds)) if ans_eds else 0.0
    verdict = "fail" if leaked else ("warn" if (asym or pos_per_ed < 2.0) else "pass")
    return {"verdict": verdict, "leaked_probe_phrasings": leaked,
            "asym_refuse_only": asym[:8], "n_asym": len(asym),
            "pos_per_edition": round(pos_per_ed, 2), "n_answer_templates": len(ans_templates),
            "n_refuse_templates": len(ref_templates), "n_answer_editions": len(ans_eds),
            "answer_templates": dict(ans_templates.most_common(6))}


def _load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="restrict to these city/year slugs (default: all)")
    ap.add_argument("--domain", default="edc", help="domain slug — audits data/<domain> (was hardcoded 'edc')")
    ap.add_argument("--corpus-stats", default="data/_build/30_tokenized/corpus_all97_stats.json")
    ap.add_argument("--sft-stats", default="data/_build/30_tokenized/sft_v2_stats.json")
    ap.add_argument("--exposure", help="checkpoint dir holding a Lever-4 exposure.json (real density)")
    ap.add_argument("--no-gpu", action="store_true", help="skip the base-model latent pass")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    resolved_device = "CPU ONLY (--no-gpu; latent pass skipped)" if args.no_gpu else args.device
    print(f"\n!!! DATA AUDIT RESOLVED DEVICE: {resolved_device} !!!", file=sys.stderr)

    eds = list(iter_editions(args.domain, only=args.editions, with_raw=False))   # was hardcoded 'edc' — domain-locked the audit
    if not eds:
        raise SystemExit("no editions matched")
    print(f"data audit over {len(eds)} edition(s){' (' + args.editions + ')' if args.editions else ''}")
    stats = _load_json(C.ROOT / args.corpus_stats)
    sft_stats = _load_json(C.ROOT / args.sft_stats)

    dims = {
        "density": dim_density(eds, args.exposure),
        "balance": dim_relation_balance(eds),
        "coverage": dim_coverage(eds),
        "token_mix": dim_token_mix(stats),
        "disjoint_static": dim_disjoint_static(eds),
        "sft_composition": dim_sft_composition(sft_stats, C.ROOT / args.sft_stats),
        "sft_phrasing": dim_sft_phrasing(sft_stats, C.ROOT / args.sft_stats),
        "sft_tiers": dim_sft_tiers(sft_stats),
    }
    if not args.no_gpu:
        try:
            dims["disjoint_latent"] = dim_disjoint_latent(eds, args.device)
        except Exception as e:
            dims["disjoint_latent"] = {"verdict": "skipped", "note": f"{type(e).__name__}: {e}"}
    else:
        dims["disjoint_latent"] = {"verdict": "skipped", "note": "--no-gpu"}

    thresholds = {"density_min": DENSITY_MIN, "density_target": DENSITY_TARGET, "replay_band": list(REPLAY_BAND),
                  "balance_cov_max": BALANCE_COV_MAX, "coverage_min_triples": COVERAGE_MIN_TRIPLES, "sep_min": SEP_MIN,
                  "refusal_band": list(REFUSAL_BAND)}
    order = ["pass", "warn", "skipped", "fail"]   # fail is the worst; a fail+skip corpus must roll up to fail
    worst = max((dims[d]["verdict"] for d in dims), key=lambda v: order.index(v) if v in order else 0)
    payload = {"editions": args.editions or "all", "n_editions": len(eds), "thresholds": thresholds,
               "overall": worst, "dimensions": dims}

    for d, v in dims.items():
        print(f"  {d:>16} [{v['verdict'].upper()}]")
    out = C.ROOT / "data/_build/dissect"; out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit_data.js").write_text("window.DATA_AUDIT = " + json.dumps(payload, default=float) + ";")
    with open(out / "data_audit_results.jsonl", "a") as f:
        f.write(json.dumps({"editions": args.editions or "all",
                            "verdicts": {d: dims[d]["verdict"] for d in dims}}, default=float) + "\n")
    tpl = Path(__file__).parent / "data_audit" / "data_audit.html"
    if tpl.exists():
        (out / "data_audit.html").write_text(tpl.read_text())
        print(f"\n  → data_audit_data.js + data_audit.html (open data/_build/dissect/data_audit.html)")
    else:
        print(f"\n  → data_audit_data.js (data_audit.html template not found — data only)")
    if worst == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
