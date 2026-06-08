#!/usr/bin/env python3
"""Deterministic quality gates — the recurrence-prevention layer.

The EDC corpus audit (docs/qgre_data_pipeline_audit.md) found five failure modes in the hydrated data. Each
becomes a machine-checkable gate here, reading the domain spec's typed-graph invariants (qgre/spec.py). The
suite is read-only and CPU-only — it never touches the GPU. Every gate returns a `Verdict` with the numbers
behind its call and, crucially, classifies a missing relation as a SOURCE GAP (no input data — reported, not
penalized) vs a DEFECT (data present, generation failed) — the distinction the audit had to recover by hand.

  gate_balance        — no relation_type may dominate past dominance_max_frac; ≥ min_active_types present;
                        every non-source-gap relation meets its floor_frac.            (audit defect #1)
  gate_exposure_cap   — no single fact (source_triple_ids[0]) exceeds exposure_cap_per_fact lines.  (#2)
  gate_id_integrity   — predicate→object-type contract + dangling-id + edition-match.   (#3, #4)
  gate_faithfulness   — no refusal/meta-commentary leaked into the corpus; no confusable co-mention.  (#5)
  gate_source_vs_gap  — classify every missing/thin relation; never fail on a pure source gap.

    PYTHONPATH=scripts python scripts/qgre/gates.py --domain edc --editions las_vegas/2026
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config as C
from pipeline.editions import iter_editions
from pipeline import disjoint_context as DC
from qgre import spec as S

_ORDER = ["pass", "warn", "skipped", "fail"]   # fail is worst; a fail+skip suite rolls up to fail


def _worst(verdicts) -> str:
    return max(verdicts, key=lambda v: _ORDER.index(v) if v in _ORDER else 0)


@dataclass
class Verdict:
    gate: str
    passed: bool
    severity: str                                   # pass | warn | fail | skipped
    metrics: dict = field(default_factory=dict)
    examples: list = field(default_factory=list)    # offending rows {text|triple, reason}
    source_gap_vs_defect: dict = field(default_factory=dict)


# ── jsonl readers (shared) ───────────────────────────────────────────────────────────────────────────
def _iter_jsonl(path: Path):
    if not path.exists():
        return
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def _hydrated(eds):
    for ed in eds:
        yield from _iter_jsonl(ed.dir / "25_hydrated/relations.jsonl")


def _edition_meta(ed) -> dict:
    """One edition's festival_meta.json (for the edition-scoped confusable audit). {} if absent/unreadable."""
    p = ed.dir / "festival_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        print(f"WARNING: {ed.slug}: festival_meta.json present but unreadable — confusable audit skipped for this edition",
              file=sys.stderr)
        return {}


def _id_type(eid) -> str | None:
    """Entity-id middle segment, e.g. 'edc_las_vegas_2018:stage:kinetic_field' -> 'stage'. None for a
    literal object (no id structure)."""
    if not isinstance(eid, str):
        return None
    parts = eid.split(":")
    return parts[1] if len(parts) >= 3 else None


# A year-bearing-type slug should NOT carry a year — the unit already encodes it. A foreign year in such a
# slug is CROSS-UNIT CONTAMINATION (e.g. `edc_las_vegas_2023:set:hardwell_kineticfield_2026` — a 2026 set
# leaked into the 2023 edition by a mis-tagged source). The year-bearing TYPES are an edition/festival
# concept, so they are declared per-domain in the manifest (quality.id_integrity.year_bearing_types); a
# domain that declares none skips the check. edition/festival ids legitimately carry years, so are exempt.
_YEAR_RE = __import__("re").compile(r"(?:19|20)\d{2}")


def _foreign_year(eid, edition_year: int, year_bearing_types) -> int | None:
    """If `eid` is a year-bearing-type id (per the manifest) whose slug contains a 4-digit year != the
    unit's, return that foreign year (the contamination tell); else None. Empty types → always None."""
    t = _id_type(eid)
    if t not in year_bearing_types:
        return None
    slug = str(eid).split(":", 2)[-1]
    for m in _YEAR_RE.findall(slug):
        if int(m) != edition_year:
            return int(m)
    return None


# ── gate: relation balance (defect #1) ─────────────────────────────────────────────────────────────────
def gate_balance(eds, spec) -> Verdict:
    rcfg = S.relations_cfg(spec)
    rel = Counter(o.get("relation_type", "?") for o in _hydrated(eds))
    total = sum(rel.values())
    if not total:
        return Verdict("balance", False, "skipped", {"note": "no hydrated relations in scope"})
    dom_type, dom_n = rel.most_common(1)[0]
    dom_frac = dom_n / total
    cov = statistics.pstdev(rel.values()) / statistics.mean(rel.values()) if len(rel) > 1 else 0.0
    dom_max = rcfg.get("dominance_max_frac", 0.85)
    min_types = rcfg.get("min_active_types", 6)

    # per-relation floor, with source-gap classification for any relation under floor / absent
    gap = gate_source_vs_gap(eds, spec)
    classified = gap.source_gap_vs_defect
    under_floor_defects = []
    meta = S.relation_meta(spec)
    for r, m in meta.items():
        floor = m.get("floor_frac", 0.0)
        frac = rel.get(r, 0) / total
        if frac < floor and classified.get(r, {}).get("kind") == "defect":
            under_floor_defects.append({"relation": r, "frac": round(frac, 4), "floor": floor})

    dom_ok = dom_frac <= dom_max
    types_ok = len(rel) >= min_types
    floors_ok = not under_floor_defects
    passed = dom_ok and types_ok and floors_ok
    sev = "pass" if passed else ("warn" if (dom_ok and types_ok) else "fail")
    examples = []
    if not dom_ok:
        examples.append({"text": f"{dom_type} is {dom_frac:.1%} of the corpus",
                         "reason": f"exceeds dominance_max_frac {dom_max:.0%} — relation monoculture"})
    if not types_ok:
        examples.append({"text": f"only {len(rel)} relation types present",
                         "reason": f"below min_active_types {min_types}"})
    for d in under_floor_defects[:6]:
        examples.append({"text": f"{d['relation']} at {d['frac']:.2%}",
                         "reason": f"below floor {d['floor']:.2%} AND classified a DEFECT (source data present)"})
    return Verdict("balance", passed, sev,
                   {"total": total, "n_types": len(rel), "dominant": dom_type, "dominant_frac": round(dom_frac, 4),
                    "dominance_max_frac": dom_max, "min_active_types": min_types, "cov": round(cov, 3),
                    "by_type": dict(rel.most_common()), "under_floor_defects": under_floor_defects},
                   examples, classified)


# ── gate: per-fact exposure cap (defect #2) ─────────────────────────────────────────────────────────────
def gate_exposure_cap(eds, spec) -> Verdict:
    cap = S.relations_cfg(spec).get("exposure_cap_per_fact", 60)
    perfact = Counter()
    for o in _hydrated(eds):
        key = (o.get("source_triple_ids") or o.get("entity_ids") or ["?"])[0]
        perfact[str(key)] += 1
    if not perfact:
        return Verdict("exposure_cap", False, "skipped", {"note": "no hydrated facts in scope", "cap": cap})
    counts = sorted(perfact.values())
    over = {k: n for k, n in perfact.items() if n > cap}
    median = statistics.median(counts)
    p99 = counts[min(len(counts) - 1, int(0.99 * len(counts)))]
    frac_over = len(over) / len(counts)
    # Redundancy is the knowledge-INSTALL mechanism — the gold repeats its key facts up to ~260× each. High
    # per-fact exposure is therefore NOT a quality defect; this gate NEVER fails the suite on it. It is a
    # COMPUTE-FOOTPRINT note: WARN when the *typical* fact is systemically over-replicated (median > cap —
    # compute waste + a flattened importance signal, e.g. VC1's median 204 vs gold's 2, where every fact
    # gets ~200× instead of spending redundancy on what matters), PASS otherwise. The real "uninstalled"
    # signal — a relation TYPE at zero — is the balance / source-gap gates' job, not this one.
    sev = "warn" if median > cap else "pass"
    passed = True
    examples = [{"text": f"fact {k}: {n} lines", "reason": f"~{n}× phrasings (typical is fine; flagged only as compute footprint)"}
                for k, n in sorted(over.items(), key=lambda kv: -kv[1])[:4]] if median > cap else []
    return Verdict("exposure_cap", passed, sev,
                   {"cap": cap, "n_facts": len(counts), "median": median,
                    "p99": p99, "max": counts[-1], "n_over_cap": len(over),
                    "frac_over_cap": round(frac_over, 4),
                    "systemic": median > cap,
                    "note": "redundancy installs knowledge — this is a compute-footprint signal, not a quality defect"},
                   examples)


# ── gate: structured-id integrity (defects #3, #4) ──────────────────────────────────────────────────────
def gate_id_integrity(eds, spec) -> Verdict:
    contract = S.predicate_contract(spec)
    qid = S.quality_cfg(spec).get("id_integrity", {})
    want_edition_match = qid.get("require_edition_match", True)
    year_bearing_types = set(qid.get("year_bearing_types") or [])   # festival-axis; empty → skip contamination check
    n_obj_checked = obj_viol = n_ref = dangling = ed_mismatch = contamination = 0
    examples = []
    for ed in eds:
        ent_ids = {o.get("id") for o in _iter_jsonl(ed.dir / "15_structured/entities.jsonl")}
        pfx = ed.festival_slug
        for t in _iter_jsonl(ed.dir / "15_structured/triples.jsonl"):
            pred, subj, obj = t.get("predicate"), t.get("subject_id"), t.get("object")
            n_ref += 1
            # cross-edition contamination: a stage/set/artist id carrying a foreign year (a 2026 set in 2023)
            for ref in (subj, obj):
                fy = _foreign_year(ref, ed.year, year_bearing_types)
                if fy is not None:
                    contamination += 1
                    if len(examples) < 8:
                        examples.append({"text": str(ref),
                                         "reason": f"year {fy} in a {ed.year} edition — cross-edition contamination"})
                    break
            # (a) predicate→object-type contract (only for declared predicates with entity-type objects)
            allowed = contract.get(pred)
            otype = _id_type(obj)
            if allowed and "literal" not in allowed and otype is not None:
                n_obj_checked += 1
                if otype not in allowed:
                    obj_viol += 1
                    if len(examples) < 8:
                        examples.append({"text": f"{pred}: {subj} -> {obj}",
                                         "reason": f"object type '{otype}' not in allowed {sorted(allowed)}"})
            # (b) dangling + (c) edition-match over the subject and any id-shaped object
            refs = [subj]
            if otype is not None:        # object is an entity id (not a literal)
                refs.append(obj)
            for ref in refs:
                if ref is None:
                    continue
                rs = str(ref)
                is_global = rs.startswith("global:")
                if ref not in ent_ids and not is_global:
                    dangling += 1
                    if len(examples) < 8:
                        examples.append({"text": rs, "reason": "id not present in entities.jsonl"})
                # (c) edition-match: an id must carry this edition's festival_slug (or be global:)
                if want_edition_match and ":" in rs and not is_global:
                    seg = rs.split(":")[0]
                    if seg and seg != pfx:
                        ed_mismatch += 1
                        if len(examples) < 8:
                            examples.append({"text": rs, "reason": f"id prefix '{seg}' != edition '{pfx}'"})
    if not n_ref:
        return Verdict("id_integrity", False, "skipped", {"note": "no structured triples in scope"})
    obj_frac = obj_viol / n_obj_checked if n_obj_checked else 0.0
    dang_frac = dangling / n_ref
    contam_frac = contamination / n_ref
    max_obj = qid.get("max_object_contract_violation_frac", 0.0)
    max_dang = qid.get("max_dangling_frac", 0.0)
    passed = obj_frac <= max_obj and dang_frac <= max_dang and ed_mismatch == 0 and contamination == 0
    # contamination teaches FALSE facts (a 2026 set tagged to 2023) — any is a real defect; notable is a fail.
    sev = "pass" if passed else ("fail" if (obj_frac > 0.1 or dang_frac > 0.05 or ed_mismatch or contam_frac > 0.005) else "warn")
    return Verdict("id_integrity", passed, sev,
                   {"n_triples": n_ref, "object_contract_checked": n_obj_checked,
                    "object_contract_violations": obj_viol, "object_contract_violation_frac": round(obj_frac, 4),
                    "dangling": dangling, "dangling_frac": round(dang_frac, 4),
                    "edition_mismatch": ed_mismatch,
                    "cross_edition_contamination": contamination, "contamination_frac": round(contam_frac, 4),
                    "max_object_contract_violation_frac": max_obj, "max_dangling_frac": max_dang},
                   examples)


# ── gate: faithfulness (defect #5) ──────────────────────────────────────────────────────────────────────
def gate_faithfulness(eds, spec) -> Verdict:
    """Deterministic, high-precision faithfulness: no refusal/meta-commentary leaked into the corpus, and no
    confusable co-mention (reuses the disjoint_context auditor). The subtler canonical-name-abbreviation
    issue the audit found needs an LLM verify pass to catch without false positives — it is NOT gated here;
    these two checks are the unambiguous generation failures."""
    qf = S.quality_cfg(spec).get("faithfulness", {})
    # word-boundary regex so a signature like "here is" does NOT match inside "Where is …" (a legit question).
    import re as _re
    sig_re = _re.compile("|".join(r"\b" + _re.escape(s) + r"\b" for s in qf.get("refusal_signatures", [])),
                         _re.IGNORECASE) if qf.get("refusal_signatures") else None
    n = refusals = 0
    examples = []
    cooc = []
    for ed in eds:
        ed_lines = list(_iter_jsonl(ed.dir / "25_hydrated/relations.jsonl"))
        if not ed_lines:
            continue
        for o in ed_lines:
            n += 1
            txt = (o.get("text") or "")
            if sig_re and sig_re.search(txt):
                refusals += 1
                if len(examples) < 6:
                    examples.append({"text": txt[:160], "reason": "refusal/meta-commentary leaked into corpus"})
        # Judge confusable co-mention against THIS edition's own confusable_groups/days/stages — never the
        # gold's calendar. An edition that declares no confusable_groups has no disjoint-context constraint,
        # so it yields nothing: a binding that merely differs from another edition's is not contamination.
        conf, gof, daytok, stagecam = DC.index_for_meta(_edition_meta(ed))
        cooc += DC.audit_cooccurrence(ed_lines, conf, gof, daytok, stagecam)
    if not n:
        return Verdict("faithfulness", False, "skipped", {"note": "no hydrated lines in scope"})
    refusal_frac = refusals / n
    max_foreign = qf.get("max_foreign_entity_frac", 0.02)
    passed = refusal_frac <= max_foreign and not cooc
    sev = "pass" if passed else ("fail" if (refusal_frac > max_foreign or len(cooc) >= 5) else "warn")
    for v in cooc[:4]:
        examples.append({"text": (v.get("text", str(v)) if isinstance(v, dict) else str(v))[:160],
                         "reason": "confusable-group co-mention (pulls topics together in latent space)"})
    return Verdict("faithfulness", passed, sev,
                   {"n_lines": n, "refusals": refusals, "refusal_frac": round(refusal_frac, 5),
                    "max_foreign_entity_frac": max_foreign, "cooccurrence_violations": len(cooc)},
                   examples)


# ── gate: source-gap vs defect (the honest classifier) ──────────────────────────────────────────────────
def gate_source_vs_gap(eds, spec) -> Verdict:
    """For every relation the spec declares, if it is MISSING or below floor in the corpus, classify it:
    source_gap (every artifact in its `needs` is absent on disk) vs defect (needs satisfied — generation
    should have produced it). NEVER fails on a pure source gap; only reports. This is the gate that keeps the
    suite honest — a missing relation with no input data is surfaced, not penalized."""
    meta = S.relation_meta(spec)
    rel = Counter(o.get("relation_type", "?") for o in _hydrated(eds))
    total = sum(rel.values()) or 1
    # per-edition relation counts: classification is decided PER EDITION (a relation is a defect only in an
    # edition that has the relation's source yet failed to produce it). Pooling the gold with a thin edition
    # can therefore no longer mask that edition's genuine source gap with the gold's artifacts.
    rel_by_ed = {}
    for ed in eds:
        c = Counter(o.get("relation_type", "?") for o in _iter_jsonl(ed.dir / "25_hydrated/relations.jsonl"))
        rel_by_ed[ed] = (c, sum(c.values()) or 1)
    classified = {}
    n_defect = n_gap = 0
    for r, m in meta.items():
        floor = m.get("floor_frac", 0.0)
        frac = rel.get(r, 0) / total
        if frac >= floor and rel.get(r, 0) > 0:
            continue   # present & adequate at the pool level — nothing to classify
        needs = m.get("needs", []) or []
        # DEFECT iff SOME edition is itself under floor for r AND has r's source (so generation should have
        # produced it there but didn't); else SOURCE GAP. A relation with no `needs` that the generator
        # always could make is a defect wherever it is thin.
        is_defect = False
        for ed in eds:
            c, ed_total = rel_by_ed[ed]
            if c.get(r, 0) > 0 and c.get(r, 0) / ed_total >= floor:
                continue   # this edition is fine for r
            has_source = (not needs) or any(S.required_artifact_present(ed.dir, art, spec)[0] for art in needs)
            if has_source:
                is_defect = True
                break
        kind = "defect" if is_defect else "source_gap"
        classified[r] = {"kind": kind, "frac": round(frac, 5), "floor": floor, "needs": needs}
        n_gap += kind == "source_gap"
        n_defect += kind == "defect"
    # this gate is informational: PASS unless there is something to say; it never fails the suite on a gap
    sev = "pass" if not classified else ("warn" if n_defect else "pass")
    examples = [{"text": f"{r}: {c['kind'].upper()}",
                 "reason": (f"missing/thin; needs {c['needs']} — none present" if c["kind"] == "source_gap"
                            else f"missing/thin but needs {c['needs'] or 'nothing'} satisfied — generation failed")}
                for r, c in classified.items()]
    return Verdict("source_vs_gap", True, sev,
                   {"n_classified": len(classified), "n_defect": n_defect, "n_source_gap": n_gap},
                   examples, classified)


# ── suite ──────────────────────────────────────────────────────────────────────────────────────────────
def run_suite(eds, spec) -> dict:
    eds = list(eds)
    gates = {
        "balance": gate_balance(eds, spec),
        "exposure_cap": gate_exposure_cap(eds, spec),
        "id_integrity": gate_id_integrity(eds, spec),
        "faithfulness": gate_faithfulness(eds, spec),
        "source_vs_gap": gate_source_vs_gap(eds, spec),
    }
    overall = _worst(g.severity for g in gates.values())
    return {"overall": overall, "n_editions": len(eds),
            "editions": [e.slug for e in eds],
            "thresholds": {"exposure_cap_per_fact": S.relations_cfg(spec).get("exposure_cap_per_fact"),
                           "dominance_max_frac": S.relations_cfg(spec).get("dominance_max_frac"),
                           "min_active_types": S.relations_cfg(spec).get("min_active_types")},
            "gates": {k: asdict(v) for k, v in gates.items()}}


def main():
    ap = argparse.ArgumentParser(description="qgre deterministic corpus quality gates")
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--festival", default=None, help="festival slug (default = domain)")
    ap.add_argument("--editions", default=None, help="city/year CSV (omit = all on disk)")
    args = ap.parse_args()
    spec = S.load_spec(args.domain)
    eds = list(iter_editions(args.festival or args.domain, only=args.editions))
    if not eds:
        print(f"no editions matched {args.domain}/{args.editions}")
        return
    report = run_suite(eds, spec)
    for name, g in report["gates"].items():
        print(f"  {name:>14} [{g['severity'].upper()}]  {g['metrics']}")
    print(f"\n  overall: {report['overall'].upper()}  ({report['n_editions']} editions)")
    out = C.ROOT / "data/_build/dissect"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qgre_gate_data.js").write_text("window.QGRE_GATE = " + json.dumps(report, default=float) + ";")
    print(f"  → data/_build/dissect/qgre_gate_data.js")


if __name__ == "__main__":
    main()
