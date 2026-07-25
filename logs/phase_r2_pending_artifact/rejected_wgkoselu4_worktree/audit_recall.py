#!/usr/bin/env python3
"""Extraction recall audit — coverage gate at the audit stage (docs/cpt-overhaul/02).

Samples N docs per unit. For each, a judge lists assertable facts about manifest entity_types,
then checks each against the unit's triples → coverage ratio per doc / predicate / unit.

The CLEAN estimand's DENOMINATOR RULE (r3 — the ruler is the fixture + the manifest, nothing else):
  in-denominator  ⟺  predicate is a manifest key  AND  subject resolves into the unit entity set.
  1. judge_rejected — predicate is not a manifest key. Not an assertion about this ontology.
  2. out_of_unit    — subject is outside the unit entity set (15_structured + global: slug rule).
  3. matcher        — subject+object mapped through resolve id_map before keying (match_mode:
     exact | soft-object | canonical).
True extraction gaps remain misses. Gate (default 0.6) applies to CLEAN coverage.

The predicate object-type contract is DIAGNOSTIC ONLY (``object_type_flagged``) and never removes
an assertion from the denominator. Phase F2 filtered on it; that made the entity table and the
domain manifest half the ruler, and both are editable without touching a single triple. Measured
against the pinned kanto fixture 0b50daf3… with a BYTE-IDENTICAL triples file:
  · 51 pure-addition shadow entity rows   → clean 0.1923 → 0.7500 (+0.5577)
  · every predicate's object list → ['nonexistent_type'] → clean 0.1923 → 1.0000 (+0.8077)
Under this rule both are +0.0000 (the contract is not read on the clean path at all), while a
genuinely added triple still moves the number (+0.0037) and an added entity never dilutes the
denominator. ``clean_legacy`` re-reports the Phase-F2 number so the h5 trail stays comparable.

Every report carries a ``ruler`` provenance block (fixture / entities / triples / id_map / domain
manifest sha256 + row counts) and, in fixture mode, fails closed when the fixture does not cover
every sampled document — a partly-covered fixture used to publish a 3-document ratio as a
20-document one with judge_unavailable 0.

    PYTHONPATH=scripts python scripts/pipeline/audit_recall.py \
        --domain <d> [--editions …] [--min-coverage 0.6] [--sample-n 0] \
        [--facts-json path]   # fixture: {doc_path: [fact_dict, …]} skips judge (tests)

Outputs:
  data/_build/recall/<domain>/<unit>.json
  data/_build/recall/report_<domain>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import config as C
from pipeline.editions import iter_editions

DEFAULT_MIN_COVERAGE = 0.6

# A judge call that never returned usable output yields NO ground truth for its document.
# Scoring such a doc as zero coverage measures judge availability, not extraction — the
# Phase-H1 campaign chased two mitigations against a number partly made of these. Above this
# fraction of the sample the coverage estimate is not trustworthy and the audit fails closed.
MAX_JUDGE_UNAVAILABLE_RATIO = 0.10
# Free Nemotron returns unparseable junk live (same finding as eval_generative 2026-07-18);
# paid qwen3-32b is the working OpenRouter default across the measurement stack.
DEFAULT_JUDGE_MODEL = "qwen/qwen3-32b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GLOBAL_PREFIX = "global"

JUDGE_SYSTEM = (
    "You list ASSERTABLE factual claims in a document about a knowledge domain. "
    "Return ONLY a JSON array of objects, each with keys: "
    "subject (string), predicate (string), object (string). "
    "predicate MUST be one of the allowed ontology predicates provided in the user message "
    "(snake_case relation names). Do NOT invent free-form verbs or English phrases as predicates. "
    "subject and object MUST be canonical entity display names (e.g. 'Kanto', 'Giovanni', 'Team Rocket'), "
    "not IDs and not full sentences. No prose, no markdown fences. Empty array if none of the "
    "allowed predicates are assertable from the document."
)
_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")


def body_of(md_text: str) -> str:
    return md_text.split("\n---\n", 1)[-1].strip() if "\n---\n" in md_text else md_text.strip()


def derive_sample_n(n_docs: int) -> int:
    """max(20, 2% of docs), cap 100."""
    if n_docs <= 0:
        return 0
    return max(20, min(100, int(round(0.02 * n_docs))))


def _load_domain_yaml(domain: str) -> dict:
    path = domain_manifest_path(domain)
    if path is None:
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_entity_types(domain: str) -> list[str]:
    raw = _load_domain_yaml(domain)
    types = raw.get("entity_types") or []
    return [str(t) for t in types] if not isinstance(types, str) else [types]


def load_predicates(domain: str) -> list[str]:
    """Ontology predicate names the coverage judge must use (snake_case)."""
    raw = _load_domain_yaml(domain)
    preds = raw.get("predicates") or {}
    if isinstance(preds, dict):
        return [str(k) for k in preds.keys()]
    if isinstance(preds, list):
        return [str(p) for p in preds]
    return []


def load_predicate_contract(domain: str) -> dict[str, list[str]]:
    """predicate → allowed object entity-types (or ['literal']), derived from the domain manifest.

    Same source the structured layer's Ontology.contract / schema_validate predicate enum use —
    never a hardcoded type list.
    """
    raw = _load_domain_yaml(domain)
    preds = raw.get("predicates") or {}
    if not isinstance(preds, dict):
        return {}
    out: dict[str, list[str]] = {}
    for name, meta in preds.items():
        if isinstance(meta, dict):
            obj = meta.get("object") or []
            out[str(name)] = [str(t) for t in obj] if isinstance(obj, list) else [str(obj)]
        else:
            out[str(name)] = []
    return out


def sha256_of(path: Path | None) -> str | None:
    """sha256 of a ruler input, or None when the file is absent/unreadable.

    Every number this audit reports is a function of the fixture AND these files; recording the
    digest is what makes "the ratio moved" attributable to the graph rather than to the ruler.
    """
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def triples_path(ed_dir: Path) -> Path | None:
    """The graph file this audit actually reads (resolved preferred) — the loader's own choice."""
    for name in ("triples.resolved.jsonl", "triples.jsonl"):
        p = ed_dir / "15_structured" / name
        if p.is_file():
            return p
    return None


def entities_path(ed_dir: Path) -> Path | None:
    for name in ("entities.resolved.jsonl", "entities.jsonl"):
        p = ed_dir / "15_structured" / name
        if p.is_file():
            return p
    return None


def _read_jsonl(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_triples(ed_dir: Path) -> list[dict]:
    # Prefer resolved graph when present
    return _read_jsonl(triples_path(ed_dir))


def load_entities(ed_dir: Path) -> list[dict]:
    return _read_jsonl(entities_path(ed_dir))


def id_map_path(domain: str, unit_slug: str) -> Path | None:
    """The resolve-stage id_map this audit reads — the path convention resolve_entities WRITES."""
    unit = unit_slug.replace("/", "_")
    candidates = [
        C.BUILD / "resolve" / domain / unit / "id_map.json",
        C.ROOT / "data" / "_build" / "resolve" / domain / unit / "id_map.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return path
    return None


def load_id_map(domain: str, unit_slug: str) -> dict[str, str]:
    """old_id → canonical_id from the resolve stage (empty if absent)."""
    path = id_map_path(domain, unit_slug)
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def domain_manifest_path(domain: str) -> Path | None:
    path = C.ROOT / "configs" / "domains" / f"{domain}.yaml"
    return path if path.is_file() else None


def collect_clean_docs(ed_dir: Path) -> list[dict]:
    root = ed_dir / "10_clean"
    if not root.is_dir():
        return []
    docs = []
    for md in sorted(root.rglob("*.md")):
        if md.name == "README.md":
            continue
        text = body_of(md.read_text(encoding="utf-8", errors="replace"))
        if text.strip():
            docs.append({"path": str(md), "text": text, "name": md.name})
    return docs


def fact_key(subject: str, predicate: str, obj: str) -> str:
    return f"{subject.strip().lower()}|{predicate.strip().lower()}|{obj.strip().lower()}"


def _id_type_segment(value: object) -> str | None:
    """Middle segment of ``prefix:type:slug``; None if not id-shaped (same rule as schema_validate)."""
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    return parts[1] if len(parts) >= 3 else None


def _norm_surface(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def build_entity_index(
    entities: list[dict],
    id_map: dict[str, str] | None = None,
) -> dict:
    """Name/alias/id surfaces → canonical display name + entity record.

    Incorporates resolve id_map (old_id → canonical_id) so alias forms key to the same canon.
    Unit entity set = every entity in 15_structured plus any ``global:``-prefixed id (ontology
    global-slug rule from extract_structured).
    """
    id_map = id_map or {}
    by_id: dict[str, dict] = {}
    for ent in entities:
        eid = str(ent.get("id") or "")
        if eid:
            by_id[eid] = ent

    def canon_id(eid: str) -> str:
        return id_map.get(eid, eid)

    def display_name(eid: str) -> str:
        cid = canon_id(eid)
        ent = by_id.get(cid) or by_id.get(eid)
        if ent:
            return str(ent.get("name_canonical") or cid)
        return cid

    surface_to_name: dict[str, str] = {}
    surface_to_entity: dict[str, dict] = {}
    unit_names: set[str] = set()

    def index_surface(surface: str, eid: str) -> None:
        text = _norm_surface(surface)
        if not text:
            return
        cname = display_name(eid)
        surface_to_name[text] = cname
        ent = by_id.get(canon_id(eid)) or by_id.get(eid)
        if ent is not None:
            surface_to_entity[text] = ent
        unit_names.add(_norm_surface(cname))

    for ent in entities:
        eid = str(ent.get("id") or "")
        if not eid:
            continue
        index_surface(str(ent.get("name_canonical") or ""), eid)
        index_surface(eid, eid)
        cid = canon_id(eid)
        if cid != eid:
            index_surface(cid, eid)
        for alias in ent.get("aliases") or []:
            index_surface(str(alias), eid)
        parts = eid.split(":")
        if len(parts) >= 3:
            index_surface(parts[2].replace("_", " "), eid)
            # global slug rule: global:type:slug is in-unit everywhere
            if parts[0] == GLOBAL_PREFIX:
                index_surface(f"{GLOBAL_PREFIX}:{parts[1]}:{parts[2]}", eid)

    for old_id, new_id in id_map.items():
        index_surface(old_id, new_id)
        index_surface(new_id, new_id)

    return {
        "surface_to_name": surface_to_name,
        "surface_to_entity": surface_to_entity,
        "unit_names": unit_names,
        "by_id": by_id,
        "id_map": id_map,
    }


def resolve_surface(surface: str, index: dict) -> str | None:
    """Canonical display name for a judge/graph surface, or None if unknown."""
    if not index:
        return None
    return index.get("surface_to_name", {}).get(_norm_surface(surface))


def entity_type_of(surface: str, index: dict) -> str | None:
    """Entity type for a display name / id surface, or None if unresolvable."""
    if not surface:
        return None
    seg = _id_type_segment(surface)
    if seg is not None:
        # Prefer live entity record when present (id_map may retarget type).
        ent = (index or {}).get("surface_to_entity", {}).get(_norm_surface(surface))
        if ent and ent.get("entity_type"):
            return str(ent["entity_type"])
        return seg
    ent = (index or {}).get("surface_to_entity", {}).get(_norm_surface(surface))
    if ent and ent.get("entity_type"):
        return str(ent["entity_type"])
    return None


def validate_judge_fact(
    fact: dict,
    *,
    predicates: list[str] | set[str] | None = None,
    contract: dict[str, list[str]] | None = None,
    entity_index: dict | None = None,
) -> tuple[bool, str | None]:
    """Manifest validation for one judge assertion.

    Mirrors the structured layer: predicate must be a manifest key; object must fit the
    predicate's declared object-type shape (Ontology.contract / schema_validate source).
    When the contract lists only entity types (no ``literal``), the object must resolve to
    an entity of an allowed type — unresolvable junk ("Wednesday") is type-invalid.
    Returns (ok, reason_or_none).
    """
    pred = str(fact.get("predicate") or "").strip()
    obj = str(fact.get("object") or "").strip()
    if not pred:
        return False, "missing_predicate"
    pred_set = set(predicates or [])
    contract = contract or {}
    if pred_set and pred not in pred_set and pred not in contract:
        return False, f"predicate_not_in_manifest:{pred}"
    if contract and pred not in contract and pred_set and pred not in pred_set:
        return False, f"predicate_not_in_manifest:{pred}"
    if not contract:
        return True, None
    if pred not in contract:
        # Predicate allowed by name list but no object-shape declared → accept.
        if pred_set and pred in pred_set:
            return True, None
        return False, f"predicate_not_in_manifest:{pred}"
    allowed = list(contract.get(pred) or [])
    if not allowed or "literal" in allowed:
        return True, None
    otype = entity_type_of(obj, entity_index or {})
    if otype is None:
        return False, f"object_type_unresolved: expected one of {sorted(allowed)}, got {obj!r}"
    if otype not in allowed:
        return False, f"object_type_mismatch: {otype!r} not in {sorted(allowed)}"
    return True, None


def subject_in_unit(subject: str, entity_index: dict | None) -> bool:
    """True when subject resolves into the unit entity set (15_structured + global:)."""
    if not entity_index or not entity_index.get("surface_to_name"):
        # No entity table loaded → scoping inactive (fixture / backward-compat path).
        return True
    if resolve_surface(subject, entity_index) is not None:
        return True
    # global:type:slug form counts even if not yet materialised in the unit table
    parts = str(subject).split(":")
    if len(parts) >= 3 and parts[0] == GLOBAL_PREFIX:
        return True
    return False


def triple_keys(triples: list[dict], id_to_name: dict[str, str] | None = None) -> set[str]:
    keys = set()
    id_to_name = id_to_name or {}
    for t in triples:
        s = id_to_name.get(str(t.get("subject_id") or ""), str(t.get("subject_id") or ""))
        # also keep raw id form
        s_raw = str(t.get("subject_id") or "")
        o = str(t.get("object") or "")
        o_name = id_to_name.get(o, o)
        p = str(t.get("predicate") or "")
        for ss in {s, s_raw}:
            for oo in {o, o_name}:
                keys.add(fact_key(ss, p, oo))
    return keys


def match_fact(
    subject: str,
    predicate: str,
    obj: str,
    graph_keys: set[str],
    entity_index: dict | None = None,
) -> str | None:
    """Return match_mode ('exact' | 'soft-object' | 'canonical') or None if absent."""
    if fact_key(subject, predicate, obj) in graph_keys:
        return "exact"
    fk_s = subject.strip().lower()
    fk_p = predicate.strip().lower()
    fk_o = obj.strip().lower()
    for gk in graph_keys:
        parts = gk.split("|", 2)
        if len(parts) != 3:
            continue
        gs, gp, go = parts
        if gs == fk_s and gp == fk_p and fk_o and go and (fk_o in go or go in fk_o):
            return "soft-object"
    # Canonical: map both sides through resolve id_map / alias index, then re-key.
    if entity_index:
        cs = resolve_surface(subject, entity_index) or subject
        co = resolve_surface(obj, entity_index) or obj
        if fact_key(cs, predicate, co) in graph_keys:
            return "canonical"
        csk, cok = cs.strip().lower(), co.strip().lower()
        for gk in graph_keys:
            parts = gk.split("|", 2)
            if len(parts) != 3:
                continue
            gs, gp, go = parts
            if gs == csk and gp == fk_p and cok and go and (cok in go or go in cok):
                return "canonical"
    return None


def loose_surface_index(entities: list[dict], id_map: dict[str, str] | None = None) -> dict[str, set[str]]:
    """entity id → EVERY surface it can wear (canonical name, aliases, the raw id, id_map preimages).

    Deliberately looser than the audit matcher: it exists to answer "could better matching have
    found this fact?", so it must over-accept. A miss that is absent even here is absent from the
    graph, full stop — an extraction gap, not a keying gap.
    """
    id_map = id_map or {}
    surfaces: dict[str, set[str]] = {}
    for entity in entities:
        eid = str(entity.get("id") or "")
        if not eid:
            continue
        names = {_norm_surface(str(entity.get("name_canonical") or "")), _norm_surface(eid)}
        names |= {_norm_surface(str(alias)) for alias in (entity.get("aliases") or [])}
        surfaces[eid] = {name for name in names if name}
    for old_id, new_id in id_map.items():
        surfaces.setdefault(new_id, set()).add(_norm_surface(old_id))
        if new_id in surfaces:
            surfaces[old_id] = surfaces[new_id]
    return surfaces


def classify_misses(
    misses: list[dict],
    triples: list[dict],
    surfaces: dict[str, set[str]],
) -> dict:
    """Split misses into matcher-recoverable vs genuinely absent — the H1 next-hypothesis instrument.

    The H1 coverage trail spent two iterations against a number nobody could attribute, because a
    miss reported only "not found" and never said WHY. This makes the attribution mechanical:

      matcher_recoverable — the fact is in the graph under some surface the matcher did not key.
                            A matching fix would recover it.
      different_object    — subject+predicate exist in the graph with other objects. The extractor
                            saw the relation and disagreed about the target.
      absent              — nothing for that subject+predicate at all. Only re-extraction recovers it.
    """
    loose: set[tuple[str, str, str]] = set()
    subject_predicate: set[tuple[str, str]] = set()
    for triple in triples:
        predicate = _norm_surface(str(triple.get("predicate") or ""))
        subject_id = str(triple.get("subject_id") or "")
        object_id = str(triple.get("object") or "")
        subject_surfaces = surfaces.get(subject_id) or {_norm_surface(subject_id)}
        object_surfaces = surfaces.get(object_id) or {_norm_surface(object_id)}
        for subject in subject_surfaces:
            subject_predicate.add((predicate, subject))
            for obj in object_surfaces:
                loose.add((predicate, subject, obj))

    counts = {"matcher_recoverable": 0, "different_object": 0, "absent": 0}
    samples: dict[str, list[dict]] = {"matcher_recoverable": [], "different_object": [], "absent": []}
    for miss in misses:
        key = (
            _norm_surface(str(miss.get("predicate") or "")),
            _norm_surface(str(miss.get("subject") or "")),
            _norm_surface(str(miss.get("object") or "")),
        )
        if key in loose:
            bucket = "matcher_recoverable"
        elif key[:2] in subject_predicate:
            bucket = "different_object"
        else:
            bucket = "absent"
        counts[bucket] += 1
        if len(samples[bucket]) < 10:
            samples[bucket].append(miss)
    total = sum(counts.values())
    return {
        **counts,
        "n": total,
        "matcher_recoverable_ratio": round(counts["matcher_recoverable"] / total, 4) if total else 0.0,
        "samples": samples,
        "note": (
            "matcher_recoverable > 0 means a matching fix has headroom. At 0 the coverage gap is "
            "extraction-side and no keying change can move the number — re-extract, do not re-match."
        ),
    }


def coverage_for_doc(
    assertable: list[dict],
    graph_keys: set[str],
    *,
    predicates: list[str] | set[str] | None = None,
    contract: dict[str, list[str]] | None = None,
    entity_index: dict | None = None,
) -> dict:
    """Compare assertable facts to graph with CLEAN class split.

    RAW path (Phase F estimand): every ontology-predicate assertion in the denominator,
    soft-object match only on the object side.
    CLEAN path (r3): denominator = manifest-predicate AND subject_in_unit. Matcher uses
    exact / soft-object / canonical modes. The object-type contract only ANNOTATES
    (``object_type_flagged``); it never removes an assertion from the denominator.
    ``clean_legacy`` reproduces the Phase-F2 contract-filtered number for trail continuity.
    """
    empty = {
        "hits": 0,
        "n": 0,
        "ratio": 1.0,
        "misses": [],
        "per_predicate": {},
        "judge_rejected": [],
        "out_of_unit": [],
        "object_type_flagged": [],
        "hits_detail": [],
        "match_mode_counts": {"exact": 0, "soft-object": 0, "canonical": 0},
        "clean_legacy": {"hits": 0, "n": 0, "ratio": 1.0},
        "raw": {"hits": 0, "n": 0, "ratio": 1.0, "misses": []},
    }
    if not assertable:
        return empty

    # Optional pre-filter identical to Phase F (drop free-form predicates before RAW too).
    pred_set = set(predicates or [])
    if pred_set:
        assertable = [f for f in assertable if str(f.get("predicate") or "") in pred_set]

    raw_hits = 0
    raw_misses: list[dict] = []
    clean_hits = 0
    clean_misses: list[dict] = []
    judge_rejected: list[dict] = []
    out_of_unit: list[dict] = []
    object_type_flagged: list[dict] = []
    legacy_hits = legacy_n = 0
    hits_detail: list[dict] = []
    match_mode_counts = {"exact": 0, "soft-object": 0, "canonical": 0}
    per_pred_raw: dict[str, list[bool]] = defaultdict(list)
    per_pred_clean: dict[str, list[bool]] = defaultdict(list)

    for fact in assertable:
        s = str(fact.get("subject") or "")
        p = str(fact.get("predicate") or "")
        o = str(fact.get("object") or "")

        # ── RAW (Phase F): exact + soft-object only, no validation / scoping ──────────
        raw_mode = None
        if fact_key(s, p, o) in graph_keys:
            raw_mode = "exact"
        else:
            fk_s, fk_p, fk_o = s.strip().lower(), p.strip().lower(), o.strip().lower()
            for gk in graph_keys:
                parts = gk.split("|", 2)
                if len(parts) != 3:
                    continue
                gs, gp, go = parts
                if gs == fk_s and gp == fk_p and fk_o and go and (fk_o in go or go in fk_o):
                    raw_mode = "soft-object"
                    break
        raw_ok = raw_mode is not None
        per_pred_raw[p].append(raw_ok)
        if raw_ok:
            raw_hits += 1
        else:
            raw_misses.append(fact)

        # ── CLEAN: predicate check → unit scope → object-type → canonical match ──────
        # Subject scoping BEFORE object-type so a Hoenn-scoped subject on a kanto doc is
        # out_of_unit (corpus composition), not judge_rejected (even if its object also
        # fails to resolve into the unit table).
        ok_pred, pred_reason = validate_judge_fact(
            fact,
            predicates=pred_set or None,
            contract=None,  # predicate membership only in this pass
            entity_index=None,
        )
        if not ok_pred:
            row = dict(fact)
            row["reject_reason"] = pred_reason
            judge_rejected.append(row)
            continue
        if not subject_in_unit(s, entity_index):
            row = dict(fact)
            row["class"] = "out_of_unit"
            out_of_unit.append(row)
            continue
        # Object-type contract: ANNOTATION, never a denominator filter (r3).
        # Phase F2 dropped these facts. 154 of the pinned fixture's 189 drops were
        # `object_type_unresolved` — "our entity table does not contain this object", which is an
        # extraction/resolve GAP, not judge noise; excluding it made the metric improve when
        # extraction got worse, and handed the entity table and the domain manifest a lever over
        # the gated ratio with a byte-identical graph (measured +0.5577 / +0.8077). A fact that
        # matches the graph is type-valid by construction — the graph passed schema_validate — so
        # the check adds nothing to the numerator either. Flag it and score it.
        ok_obj, obj_reason = validate_judge_fact(
            fact,
            predicates=pred_set or None,
            contract=contract,
            entity_index=entity_index,
        )
        if not ok_obj:
            row = dict(fact)
            row["type_flag"] = obj_reason
            object_type_flagged.append(row)
        mode = match_fact(s, p, o, graph_keys, entity_index)
        if ok_obj:
            legacy_n += 1
        if mode is None:
            clean_misses.append(fact)
            per_pred_clean[p].append(False)
        else:
            clean_hits += 1
            if ok_obj:
                legacy_hits += 1
            match_mode_counts[mode] = match_mode_counts.get(mode, 0) + 1
            hits_detail.append({**fact, "match_mode": mode})
            per_pred_clean[p].append(True)

    raw_n = len(assertable)
    clean_n = clean_hits + len(clean_misses)
    per_predicate = {
        p: {"hits": sum(v), "n": len(v), "ratio": sum(v) / len(v) if v else 1.0}
        for p, v in sorted(per_pred_clean.items())
    }
    per_predicate_raw = {
        p: {"hits": sum(v), "n": len(v), "ratio": sum(v) / len(v) if v else 1.0}
        for p, v in sorted(per_pred_raw.items())
    }
    return {
        "hits": clean_hits,
        "n": clean_n,
        "ratio": clean_hits / clean_n if clean_n else 1.0,
        "misses": clean_misses,
        "per_predicate": per_predicate,
        "judge_rejected": judge_rejected,
        "out_of_unit": out_of_unit,
        "object_type_flagged": object_type_flagged,
        "hits_detail": hits_detail,
        "match_mode_counts": match_mode_counts,
        "clean_legacy": {
            "hits": legacy_hits,
            "n": legacy_n,
            "ratio": legacy_hits / legacy_n if legacy_n else 1.0,
        },
        "raw": {
            "hits": raw_hits,
            "n": raw_n,
            "ratio": raw_hits / raw_n if raw_n else 1.0,
            "misses": raw_misses,
            "per_predicate": per_predicate_raw,
        },
    }


def parse_facts(raw: str) -> list[dict]:
    match = _JSON_ARRAY.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("subject") and item.get("predicate") and item.get("object"):
            out.append({
                "subject": str(item["subject"]),
                "predicate": str(item["predicate"]),
                "object": str(item["object"]),
            })
    return out


def run_audit(
    domain: str,
    editions: str | None,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    sample_n: int = 0,
    facts_json: Path | None = None,
    seed: int = 0,
    dry_run: bool = False,
) -> dict:
    fixture: dict[str, list[dict]] = {}
    fixture_path = Path(facts_json) if facts_json is not None else None
    if fixture_path is not None:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    entity_types = load_entity_types(domain)
    predicates = load_predicates(domain)
    contract = load_predicate_contract(domain)
    manifest_path = domain_manifest_path(domain)
    ruler_units: list[dict] = []
    fixture_matched_keys: set[str] = set()
    all_fixture_missing: list[dict] = []
    unit_reports = []
    overall_hits = overall_n = 0
    raw_hits = raw_n = 0
    pred_agg: dict[str, list[float]] = defaultdict(list)
    pred_agg_raw: dict[str, list[float]] = defaultdict(list)
    class_totals = {
        "judge_rejected": 0,
        "out_of_unit": 0,
        "misses": 0,
        "hits": 0,
        "match_mode": {"exact": 0, "soft-object": 0, "canonical": 0},
    }
    class_totals["object_type_flagged"] = 0
    legacy_hits = legacy_n = 0
    status = "PASS"
    failed_predicates: list[str] = []
    all_judge_rejected: list[dict] = []
    all_out_of_unit: list[dict] = []
    all_judge_unavailable: list[dict] = []
    sampled_docs = 0
    miss_totals = {"matcher_recoverable": 0, "different_object": 0, "absent": 0}

    for ed in iter_editions(domain, only=editions, with_raw=False):
        docs = collect_clean_docs(ed.dir)
        triples = load_triples(ed.dir)
        entities = load_entities(ed.dir)
        id_map = load_id_map(domain, ed.slug)
        entity_index = build_entity_index(entities, id_map)
        ruler_units.append({
            "unit": ed.slug,
            "entities": {
                "path": str(entities_path(ed.dir) or ""),
                "sha256": sha256_of(entities_path(ed.dir)),
                "rows": len(entities),
                "index_surfaces": len(entity_index.get("surface_to_name") or {}),
            },
            "triples": {
                "path": str(triples_path(ed.dir) or ""),
                "sha256": sha256_of(triples_path(ed.dir)),
                "rows": len(triples),
            },
            "id_map": {
                "path": str(id_map_path(domain, ed.slug) or ""),
                "sha256": sha256_of(id_map_path(domain, ed.slug)),
                "entries": len(id_map),
            },
        })

        # id→name from entities (display forms for graph keys)
        id_to_name: dict[str, str] = {}
        for e in entities:
            canon = str(e.get("name_canonical") or e.get("id") or "")
            eid = str(e.get("id") or "")
            if eid:
                id_to_name[eid] = canon
            for alias in e.get("aliases") or []:
                id_to_name[str(alias)] = canon
        # id_map old ids also resolve to the canonical entity's display name
        for old_id, new_id in id_map.items():
            if new_id in id_to_name:
                id_to_name[old_id] = id_to_name[new_id]
            elif new_id:
                id_to_name.setdefault(old_id, id_to_name.get(new_id, new_id))

        gkeys = triple_keys(triples, id_to_name)
        # Also index graph under canonical display names via entity_index
        for t in triples:
            s_id = str(t.get("subject_id") or "")
            o_raw = str(t.get("object") or "")
            p = str(t.get("predicate") or "")
            s_name = resolve_surface(s_id, entity_index) or id_to_name.get(s_id, s_id)
            o_name = resolve_surface(o_raw, entity_index) or id_to_name.get(o_raw, o_raw)
            gkeys.add(fact_key(s_name, p, o_name))

        n_take = sample_n if sample_n > 0 else derive_sample_n(len(docs))
        n_take = min(n_take, len(docs))
        rng = random.Random(seed)
        sample = rng.sample(docs, n_take) if n_take else []

        doc_rows = []
        u_hits = u_n = 0
        u_raw_hits = u_raw_n = 0
        u_judge_rej: list[dict] = []
        u_out: list[dict] = []
        u_unavailable: list[dict] = []
        u_flagged: list[dict] = []
        u_fixture_missing: list[dict] = []
        u_legacy_hits = u_legacy_n = 0
        u_match_mode = {"exact": 0, "soft-object": 0, "canonical": 0}
        # Live path: fan out judge calls (sequential OpenRouter was ~16s/doc → multi-minute silent runs).
        live_need = []
        for doc in sample:
            key = doc["path"]
            assertable = fixture.get(key) or fixture.get(doc["name"]) or fixture.get(str(Path(key).name))
            if assertable is None and not fixture:
                live_need.append(doc)
        live_facts: dict[str, list[dict] | None] = {}
        if live_need:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print(
                f"  {ed.slug}: judging {len(live_need)} docs "
                f"(model={DEFAULT_JUDGE_MODEL}, predicates={len(predicates)})",
                flush=True,
            )
            def _one(d: dict) -> tuple[str, list[dict] | None]:
                return d["path"], _judge_facts(d["text"], entity_types, predicates)
            with ThreadPoolExecutor(max_workers=min(6, len(live_need))) as pool:
                futs = {pool.submit(_one, d): d for d in live_need}
                done = 0
                for fut in as_completed(futs):
                    path, facts = fut.result()
                    live_facts[path] = facts
                    done += 1
                    shown = "unavailable" if facts is None else len(facts)
                    print(f"    judge {done}/{len(live_need)} facts={shown}", flush=True)
            # One sequential retry for unusable judge responses (rate-limit / parse collapse).
            # Only None retries: a genuine empty fact list is an answer, not a failure.
            empties = [d for d in live_need if live_facts.get(d["path"]) is None]
            if empties:
                print(f"  {ed.slug}: retrying {len(empties)} unusable judge responses", flush=True)
                for d in empties:
                    facts = _judge_facts(d["text"], entity_types, predicates)
                    live_facts[d["path"]] = facts
                    shown = "unavailable" if facts is None else len(facts)
                    print(f"    retry {d['name']} facts={shown}", flush=True)
            # Persist the accepted assertions as a REPLAY FIXTURE. The H1 trail was unreadable
            # because every iteration re-rolled a nondeterministic judge; with this, a later
            # iteration re-runs the identical assertion set through --facts-json at zero provider
            # cost and any movement is attributable to the graph alone.
            if not dry_run:
                fixture_out = C.BUILD / "recall" / domain / f"{ed.slug.replace('/', '_')}.assertions.json"
                fixture_out.parent.mkdir(parents=True, exist_ok=True)
                replayable = {
                    Path(path).name: facts
                    for path, facts in sorted(live_facts.items())
                    if facts is not None
                }
                fixture_out.write_text(json.dumps(replayable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"  {ed.slug}: wrote replay fixture ({len(replayable)} docs) -> {fixture_out}", flush=True)
        for doc in sample:
            key = doc["path"]
            # fixture keys may be basename
            hit_key = next(
                (k for k in (key, doc["name"], str(Path(key).name)) if fixture.get(k) is not None),
                None,
            )
            assertable = fixture.get(hit_key) if hit_key is not None else None
            if hit_key is not None:
                fixture_matched_keys.add(hit_key)
            if assertable is None:
                if fixture:
                    # A sampled document the fixture does not cover is NOT "this document asserts
                    # nothing" — it is a ruler/sample disagreement. Scored as [] it contributes 0/0
                    # and vanishes, so the report used to quote a 3-document ratio as a
                    # 20-document one with judge_unavailable 0. Record it; the run fails closed.
                    u_fixture_missing.append({"path": doc["name"]})
                    assertable = []
                else:
                    assertable = live_facts.get(key)
                    if assertable is None:
                        # No ground truth for this doc — unmeasured, not missed. Excluding it
                        # keeps the estimand about extraction instead of judge availability.
                        u_unavailable.append({"path": doc["name"]})
                        continue
            cov = coverage_for_doc(
                assertable,
                gkeys,
                predicates=predicates,
                contract=contract,
                entity_index=entity_index,
            )
            doc_rows.append({
                "path": doc["name"],
                "hits": cov["hits"],
                "n": cov["n"],
                "ratio": round(cov["ratio"], 4),
                "per_predicate": cov["per_predicate"],
                "misses": cov["misses"],
                "judge_rejected": cov["judge_rejected"],
                "out_of_unit": cov["out_of_unit"],
                "object_type_flagged": cov["object_type_flagged"],
                "clean_legacy": cov["clean_legacy"],
                "match_mode_counts": cov["match_mode_counts"],
                "raw": {
                    "hits": cov["raw"]["hits"],
                    "n": cov["raw"]["n"],
                    "ratio": round(cov["raw"]["ratio"], 4),
                },
            })
            u_hits += cov["hits"]
            u_n += cov["n"]
            u_raw_hits += cov["raw"]["hits"]
            u_raw_n += cov["raw"]["n"]
            u_judge_rej.extend(cov["judge_rejected"])
            u_out.extend(cov["out_of_unit"])
            u_flagged.extend(cov["object_type_flagged"])
            u_legacy_hits += cov["clean_legacy"]["hits"]
            u_legacy_n += cov["clean_legacy"]["n"]
            for mode, count in cov["match_mode_counts"].items():
                u_match_mode[mode] = u_match_mode.get(mode, 0) + count
            for p, row in cov["per_predicate"].items():
                pred_agg[p].append(row["ratio"])
            for p, row in cov["raw"].get("per_predicate", {}).items():
                pred_agg_raw[p].append(row["ratio"])

        u_ratio = u_hits / u_n if u_n else 1.0
        u_raw_ratio = u_raw_hits / u_raw_n if u_raw_n else 1.0
        corpus_note = None
        if u_raw_n > 0 and len(u_out) / max(u_raw_n, 1) >= 0.15:
            corpus_note = (
                "out_of_unit is large relative to raw assertions — cross-region content on "
                "cross-region pages is a corpus-composition finding, not an extraction gap."
            )
        u_misses = [miss for row in doc_rows for miss in row.get("misses", [])]
        miss_attribution = classify_misses(
            u_misses, triples, loose_surface_index(entities, id_map),
        )
        unit_reports.append({
            "unit": ed.slug,
            "miss_attribution": miss_attribution,
            "n_docs_sampled": len(sample),
            "n_docs_scored": len(doc_rows),
            "n_docs_judge_unavailable": len(u_unavailable),
            "n_docs_fixture_matched": (len(sample) - len(u_fixture_missing)) if fixture else None,
            "n_docs_fixture_missing": len(u_fixture_missing) if fixture else None,
            "n_docs_total": len(docs),
            "ruler": ruler_units[-1],
            "hits": u_hits,
            "n": u_n,
            "ratio": round(u_ratio, 4),
            "raw": {
                "hits": u_raw_hits,
                "n": u_raw_n,
                "ratio": round(u_raw_ratio, 4),
            },
            "clean_legacy": {
                "hits": u_legacy_hits,
                "n": u_legacy_n,
                "ratio": round(u_legacy_hits / u_legacy_n, 4) if u_legacy_n else 1.0,
            },
            "classes": {
                "judge_rejected": len(u_judge_rej),
                "out_of_unit": len(u_out),
                "object_type_flagged": len(u_flagged),
                "misses": u_n - u_hits,
                "hits": u_hits,
                "match_mode": u_match_mode,
            },
            "corpus_composition_note": corpus_note,
            "docs": doc_rows,
            "judge_rejected_sample": u_judge_rej[:25],
            "out_of_unit_sample": u_out[:25],
            "object_type_flagged_sample": u_flagged[:25],
            "fixture_missing_sample": u_fixture_missing[:25],
            "judge_unavailable_sample": u_unavailable[:25],
        })
        overall_hits += u_hits
        overall_n += u_n
        raw_hits += u_raw_hits
        raw_n += u_raw_n
        class_totals["judge_rejected"] += len(u_judge_rej)
        class_totals["out_of_unit"] += len(u_out)
        class_totals["object_type_flagged"] += len(u_flagged)
        class_totals["misses"] += (u_n - u_hits)
        class_totals["hits"] += u_hits
        legacy_hits += u_legacy_hits
        legacy_n += u_legacy_n
        for mode, count in u_match_mode.items():
            class_totals["match_mode"][mode] = class_totals["match_mode"].get(mode, 0) + count
        all_judge_rejected.extend(u_judge_rej)
        all_out_of_unit.extend(u_out)
        all_judge_unavailable.extend(u_unavailable)
        all_fixture_missing.extend(u_fixture_missing)
        sampled_docs += len(sample)
        for bucket in ("matcher_recoverable", "different_object", "absent"):
            miss_totals[bucket] += miss_attribution[bucket]

        if not dry_run:
            out = C.BUILD / "recall" / domain
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{ed.slug.replace('/', '_')}.json").write_text(
                json.dumps(unit_reports[-1], indent=2) + "\n", encoding="utf-8",
            )

    per_predicate = {
        p: {
            "mean_ratio": round(sum(vs) / len(vs), 4) if vs else 1.0,
            "n_docs": len(vs),
        }
        for p, vs in sorted(pred_agg.items())
    }
    per_predicate_raw = {
        p: {
            "mean_ratio": round(sum(vs) / len(vs), 4) if vs else 1.0,
            "n_docs": len(vs),
        }
        for p, vs in sorted(pred_agg_raw.items())
    }
    for p, row in per_predicate.items():
        if row["n_docs"] > 0 and row["mean_ratio"] < min_coverage:
            status = "FAIL"
            failed_predicates.append(p)

    overall_ratio = overall_hits / overall_n if overall_n else 1.0
    raw_ratio = raw_hits / raw_n if raw_n else 1.0
    if overall_n > 0 and overall_ratio < min_coverage:
        status = "FAIL"

    corpus_note = None
    if raw_n > 0 and class_totals["out_of_unit"] / raw_n >= 0.15:
        corpus_note = (
            "out_of_unit class is large — cross-region content on cross-region pages is a "
            "corpus-composition finding, not an extraction gap."
        )

    report = {
        "domain": domain,
        "status": status,
        "min_coverage": min_coverage,
        "estimand": "clean",
        "estimand_rule": (
            "clean/r3: denominator = manifest-predicate AND subject_in_unit (entity- and "
            "contract-inert against pure additions); numerator = canonical-matched. Run-gate "
            "verdict, never install-strength."
        ),
        "overall": {
            "hits": overall_hits,
            "n": overall_n,
            "ratio": round(overall_ratio, 4),
        },
        "raw": {
            "hits": raw_hits,
            "n": raw_n,
            "ratio": round(raw_ratio, 4),
            "per_predicate": per_predicate_raw,
            "note": "Phase F estimand: ontology-predicate assertions, soft-object match, no validation/scoping",
        },
        "clean": {
            "hits": overall_hits,
            "n": overall_n,
            "ratio": round(overall_ratio, 4),
            "per_predicate": per_predicate,
            "note": (
                "GATED estimand. Denominator = manifest-predicate AND subject_in_unit; numerator = "
                "canonical-matched. The object-type contract only annotates (object_type_flagged) "
                "so neither the entity table nor the domain manifest can move the denominator."
            ),
        },
        "clean_legacy": {
            "hits": legacy_hits,
            "n": legacy_n,
            "ratio": round(legacy_hits / legacy_n, 4) if legacy_n else 1.0,
            "note": (
                "SUPERSEDED Phase-F2 estimand (object-type contract filtered the denominator). "
                "Reported only so the h5 trail stays comparable — NEVER gate on it: measured on the "
                "pinned kanto fixture it moves +0.5577 under 51 pure-addition entity rows and "
                "+0.8077 under a domain-manifest edit, both with a byte-identical triples file."
            ),
        },
        "ruler": {
            "fixture": {
                "path": str(fixture_path) if fixture_path else None,
                "sha256": sha256_of(fixture_path),
                "n_docs": len(fixture),
                "n_assertions": sum(len(v or []) for v in fixture.values()),
            } if fixture_path else None,
            "domain_manifest": {
                "path": str(manifest_path or ""),
                "sha256": sha256_of(manifest_path),
                "n_predicates": len(predicates),
                "n_contract_entries": len(contract),
            },
            "units": ruler_units,
            "note": (
                "Every input the reported ratios are a function of. A ratio that moved while these "
                "digests held is a graph change; a ratio that moved while triples.sha256 held is a "
                "RULER change and is not an extraction result. Quote clean_n, raw_n, entity rows, "
                "triple rows, id_map entries, judge_unavailable and fixture_docs_matched with "
                "every coverage number."
            ),
        },
        "classes": class_totals,
        "corpus_composition_note": corpus_note,
        "per_predicate": per_predicate,
        "failed_predicates": failed_predicates,
        "units": unit_reports,
        "entity_types": entity_types,
        "predicates": predicates,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "RECALL_AUDIT_UNAVAILABLE": False,
        "judge_rejected_sample": all_judge_rejected[:40],
        "out_of_unit_sample": all_out_of_unit[:40],
        "miss_attribution": {
            **miss_totals,
            "n": sum(miss_totals.values()),
            "note": (
                "Where the coverage gap actually lives. matcher_recoverable > 0 means a keying fix "
                "has headroom; at 0 the gap is extraction-side and no matcher change can move the "
                "number. different_object = the extractor saw the relation and chose other targets."
            ),
        },
        "fixture_sample": {
            "active": bool(fixture),
            "n_docs_sampled": sampled_docs,
            "n_docs_matched": sampled_docs - len(all_fixture_missing) if fixture else None,
            "n_docs_missing": len(all_fixture_missing) if fixture else None,
            "n_fixture_entries": len(fixture),
            "n_fixture_entries_unused": len(set(fixture) - fixture_matched_keys) if fixture else None,
            "missing_sample": all_fixture_missing[:40],
            "note": (
                "A sampled document the fixture does not cover scores as 0/0 and disappears from "
                "the estimand while n_docs_sampled and judge_unavailable still read clean. The run "
                "fails closed on any miss: re-pin the fixture, or select the unit with --editions. "
                "Unused fixture entries are reported but not fatal (a deliberately smaller "
                "--sample-n is legitimate)."
            ),
        },
        "judge_unavailable": {
            "n_docs": len(all_judge_unavailable),
            "n_docs_sampled": sampled_docs,
            "ratio": round(len(all_judge_unavailable) / sampled_docs, 4) if sampled_docs else 0.0,
            "max_ratio": MAX_JUDGE_UNAVAILABLE_RATIO,
            "sample": all_judge_unavailable[:40],
            "note": (
                "Documents the judge never answered for. Excluded from the coverage estimand — "
                "unmeasured, not missed. A coverage ratio is only comparable across runs at the "
                "same judge availability."
            ),
        },
    }
    unavailable_ratio = report["judge_unavailable"]["ratio"]
    if fixture and all_fixture_missing:
        # The ruler and the sample disagree. Everything below is a ratio over a SUBSET, so no
        # number is quoted as a verdict — this outranks every other status.
        names = ", ".join(row["path"] for row in all_fixture_missing[:5])
        more = "…" if len(all_fixture_missing) > 5 else ""
        report["status"] = "FAIL"
        report["reason"] = (
            f"fixture covers {sampled_docs - len(all_fixture_missing)}/{sampled_docs} sampled "
            f"documents (missing: {names}{more}); the coverage ratio would be quoted over a "
            "subset of the sample it claims — re-pin the fixture or select the unit with --editions"
        )
    elif overall_n == 0 and raw_n == 0 and not fixture:
        # No judge output at all (e.g. no provider key) — unmeasured, as before. Not a FAIL:
        # an offline/device-only run legitimately cannot score this estimand.
        report["status"] = "SKIP"
        report["RECALL_AUDIT_UNAVAILABLE"] = True
        report["reason"] = "no assertable facts scored (no fixture and no judge output)"
    elif unavailable_ratio > MAX_JUDGE_UNAVAILABLE_RATIO:
        # Partial availability: some docs scored, others never answered. Fail closed rather
        # than publish a coverage number built on a thin, biased sample — and never let
        # dropping the hard documents look like an extraction improvement.
        report["status"] = "FAIL"
        report["reason"] = (
            f"judge unavailable on {len(all_judge_unavailable)}/{sampled_docs} sampled docs "
            f"({unavailable_ratio:.2%} > {MAX_JUDGE_UNAVAILABLE_RATIO:.0%}); the coverage "
            "estimate is not trustworthy at this judge availability"
        )
    elif overall_n == 0 and raw_n > 0:
        # Everything was filtered as noise — clean coverage is vacuously 1.0 but call it out.
        report["status"] = "PASS" if min_coverage <= 1.0 else "FAIL"
        report["reason"] = (
            "all raw assertions were judge_rejected or out_of_unit; clean denominator is 0"
        )

    if not dry_run:
        out = C.BUILD / "recall"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_{domain}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "clean_ratio": report["overall"]["ratio"],
        "raw_ratio": report["raw"]["ratio"],
        "clean_legacy_ratio": report["clean_legacy"]["ratio"],
        "clean_n": report["overall"]["n"],
        "raw_n": report["raw"]["n"],
        "clean_legacy_n": report["clean_legacy"]["n"],
        "classes": class_totals,
        "failed_predicates": failed_predicates,
        "judge_unavailable": report["judge_unavailable"]["n_docs"],
        "judge_unavailable_ratio": report["judge_unavailable"]["ratio"],
        # Ruler guards — a coverage number without these is not comparable to any other run.
        "fixture_docs_matched": report["fixture_sample"]["n_docs_matched"],
        "fixture_docs_sampled": sampled_docs,
        "fixture_sha256": (report["ruler"]["fixture"] or {}).get("sha256"),
        "entity_rows": sum(u["entities"]["rows"] for u in ruler_units),
        "triple_rows": sum(u["triples"]["rows"] for u in ruler_units),
        "id_map_entries": sum(u["id_map"]["entries"] for u in ruler_units),
        "manifest_sha256": report["ruler"]["domain_manifest"]["sha256"],
        "reason": report.get("reason"),
    }))
    return report


def _judge_facts(
    text: str,
    entity_types: list[str],
    predicates: list[str] | None = None,
) -> list[dict] | None:
    """Live judge — requires OpenRouter.

    Returns the assertable facts, ``[]`` when the judge genuinely found none, and ``None``
    when no usable answer was obtained (no key, transport failure, or persistently empty
    completions). ``None`` is NOT zero coverage: the document has no ground truth and must be
    excluded from the estimand rather than scored as an extraction miss.
    """
    import urllib.request
    import subprocess

    def key() -> str | None:
        v = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if v:
            return v
        try:
            p = subprocess.run(
                ["secret-tool", "lookup", "service", "openrouter"],
                capture_output=True, text=True, timeout=5,
            )
            return p.stdout.strip() or None if p.returncode == 0 else None
        except Exception:
            return None

    k = key()
    if not k:
        return None
    types = ", ".join(entity_types) if entity_types else "domain entities"
    preds = ", ".join(predicates) if predicates else "(any snake_case domain relation)"
    # /no_think: qwen3 burns the whole max_tokens budget on reasoning otherwise and
    # returns empty content (measured 2026-07-19). Keep the established model id.
    user = (
        f"Entity types of interest: {types}\n"
        f"ALLOWED predicates (use ONLY these exact strings): {preds}\n\n"
        f"DOCUMENT:\n{text[:6000]}\n\n"
        "List assertable (subject, predicate, object) facts as a JSON array. "
        "predicate must be one of the ALLOWED predicates above.\n"
        "/no_think"
    )
    body = json.dumps({
        "model": DEFAULT_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }).encode()
    req = urllib.request.Request(
        OPENROUTER_BASE_URL + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {k}"},
        method="POST",
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode())
            msg = payload["choices"][0]["message"] or {}
            raw = msg.get("content") or ""
            # fallback: some providers put the JSON only in reasoning when content is empty
            if not raw.strip():
                raw = str(msg.get("reasoning") or msg.get("reasoning_content") or "")
            if not raw.strip():
                # No usable answer — retry rather than reporting "this document asserts nothing".
                last_err = ValueError("empty judge completion")
                import time as _time
                _time.sleep(1.5 * (2 ** attempt))
                continue
            facts = parse_facts(raw)
            if predicates:
                allowed = set(predicates)
                facts = [f for f in facts if str(f.get("predicate") or "") in allowed]
            return facts
        except Exception as exc:
            last_err = exc
            import time as _time
            _time.sleep(1.5 * (2 ** attempt))
    if last_err:
        print(f"    judge error: {type(last_err).__name__}: {last_err}", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="extraction recall audit")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    ap.add_argument("--sample-n", type=int, default=0, help="0 = derive from corpus size")
    ap.add_argument("--facts-json", help="fixture path for offline red-green")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        report = run_audit(
            args.domain, args.editions,
            min_coverage=args.min_coverage,
            sample_n=args.sample_n,
            facts_json=Path(args.facts_json) if args.facts_json else None,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"audit_recall failed closed: {error}")
    sys.exit(0 if report["status"] in ("PASS", "SKIP") else 1)


if __name__ == "__main__":
    main()
