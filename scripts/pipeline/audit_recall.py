#!/usr/bin/env python3
"""Extraction recall audit — coverage gate at the audit stage (docs/cpt-overhaul/02).

Samples N docs per unit. For each, a judge lists assertable facts about manifest entity_types,
then checks each against the unit's triples → coverage ratio per doc / predicate / unit.

The CLEAN estimand (Phase F2) drops metric noise before the gate:
  1. judge_rejected — assertion fails manifest validation (predicate not a key, or object does
     not fit the predicate's declared object-type shape; same contract the structured layer uses).
  2. out_of_unit   — subject is outside the unit entity set (15_structured + global: slug rule).
  3. matcher       — subject+object mapped through resolve id_map before keying (match_mode:
     exact | soft-object | canonical).
True extraction gaps remain misses. Gate (default 0.6) applies to CLEAN coverage.

THE RULER PIN (r3) — the assertion fixture is only ONE QUARTER of the ruler.
``--facts-json`` freezes the 376 assertions, but the CLEAN denominator is also a function of the
entity table (``subject_in_unit``) and of the domain manifest (the object-type contract), and the
document sample is a function of the on-disk .md list. All three were unpinned and unreported.
Measured on kanto against fixture 0b50daf3… with a BYTE-IDENTICAL triples file:
  · +55 pure-addition entity rows (one per distinct clean miss object, retyped)
        → clean 0.1923 → 1.0000 (+0.8077), clean_n 78 → 11, status PASS
  · +13 pure-addition entity rows (one per distinct out_of_unit subject)
        → clean 0.1923 → 0.1579 (-0.0344), clean_n 78 → 95
  · manifest only, every predicate's object list → ['nonexistent_type']
        → clean 0.1923 → 1.0000 (+0.8077), clean_n 78 → 0, status PASS
  · manifest only, every predicate's object list → ['literal']
        → clean 0.1923 → 0.0749 (-0.1174), clean_n 78 → 267
  · one .md inserted at sorted position 0 → the seed-0 sample overlaps the fixture 3/20 and the
        17 unmatched docs score as ``assertable = []``, so the report quoted a 3-document ratio as
        a 20-document one with judge_unavailable 0.
A denominator selectable anywhere in [0, 376] by editing files the fixture does not cover makes
every extraction change unfalsifiable, so:

  * ``ruler_fingerprint`` records the sha256 + row count of every ruler input on every report.
  * ``<unit>.ruler.json`` (written by ``--pin-ruler``) PINS four of them — fixture, entities,
    id_map, domain manifest. A run whose live digests differ FAILS CLOSED and quotes no ratio.
  * The triples file is recorded and deliberately NOT pinned: it is the thing under measurement.
    A graph-only change scores normally at full sensitivity (+1 genuine triple = +0.0128 clean /
    +0.0027 raw, denominator unchanged at 78/376) — a denominator pinned so hard that real new
    triples stop registering would be equally broken.
  * In fixture mode a sampled document the fixture does not cover is a ruler/sample disagreement,
    not an empty document: the run FAILS CLOSED instead of silently scoring it 0/0.
  * A clean denominator of 0 is FAIL, never PASS. "Everything was filtered as noise" is the
    zero-from-an-uncalibrated-detector shape, not a cleared gate.
Re-pinning after a deliberate entity/manifest change is one command and a visible diff — that is
the point: the number cannot move unless the recorded ruler moves with it.

    PYTHONPATH=scripts python scripts/pipeline/audit_recall.py \
        --domain <d> [--editions …] [--min-coverage 0.6] [--sample-n 0] \
        [--facts-json path]   # fixture: {doc_path: [fact_dict, …]} skips judge (tests)
        [--pin-ruler]         # (re)write <unit>.ruler.json from the CURRENT ruler inputs
        [--no-ruler-pin]      # ignore an existing pin (diagnostic; never for a quoted number)

Outputs:
  data/_build/recall/<domain>/<unit>.json
  data/_build/recall/report_<domain>.json
  data/_build/recall/<domain>/<unit>.ruler.json   (only with --pin-ruler)
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

# The ruler inputs a pin freezes. The triples file is deliberately ABSENT: pinning the graph
# would make a genuine extraction gain fail closed, which is the equal-and-opposite defect.
PINNED_RULER_FIELDS = ("fixture", "entities", "id_map", "domain_manifest", "documents",
                       "editions_registry", "sample")

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
    path = C.ROOT / "configs" / "domains" / f"{domain}.yaml"
    if not path.is_file():
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
    """sha256 of one ruler input, or None when absent/unreadable.

    Every ratio this audit prints is a function of the fixture AND these files. Recording the
    digest is what makes "the number moved" attributable to the graph instead of to the ruler.
    """
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def triples_path(ed_dir: Path) -> Path | None:
    """The graph file the audit actually reads. The PATH CHOICE lives here, not in each caller:
    a provenance block that re-derives the convention can name a file the loader never opened."""
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
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_triples(ed_dir: Path) -> list[dict]:
    # Prefer resolved graph when present
    return _read_jsonl(triples_path(ed_dir))


def load_entities(ed_dir: Path) -> list[dict]:
    return _read_jsonl(entities_path(ed_dir))


def id_map_path(domain: str, unit_slug: str) -> Path | None:
    """The resolve-stage id_map file this audit reads, or None.

    Thin wrapper over `pipeline.split.id_map_path` — ONE loader, ONE path rule. The digest and the
    data must come from the same file; both now resolve through split so a nested unit
    (`las_vegas/2011`) cannot silently disagree with the split-time map.
    """
    from pipeline.split import id_map_path as split_id_map_path

    ed_dir = C.ROOT / "data" / domain / Path(unit_slug)
    return split_id_map_path(ed_dir, domain=domain)


def load_id_map(domain: str, unit_slug: str) -> dict[str, str]:
    """old_id → canonical_id from the resolve stage (empty if absent).

    Delegates to `pipeline.split.load_id_map` — the single loader. Callers that already have an
    edition directory should call split's form directly; this wrapper exists for the domain+slug
    call shape used by audit/probes/quarantine/reasoning.
    """
    from pipeline.split import load_id_map as split_load_id_map

    ed_dir = C.ROOT / "data" / domain / Path(unit_slug)
    return split_load_id_map(ed_dir, domain=domain)


def domain_manifest_path(domain: str) -> Path | None:
    path = C.ROOT / "configs" / "domains" / f"{domain}.yaml"
    return path if path.is_file() else None


def documents_digest(ed_dir: Path) -> dict:
    """Digest of the DOCUMENT CORPUS the sample is drawn from — the fifth ruler input.

    `derive_sample_n` and the seed-0 sample are functions of what `collect_clean_docs` returns, so
    the corpus is a denominator input exactly like the entity table. Pinning the other four while
    leaving this open was a real hole: parking 104 of 105 kanto .md files reproduces the full
    ratio exploit with every other pinned digest byte-identical, because the sample then lands on
    documents the fixture has no entry for.

    Digest is over the sorted (relpath, sha256) pairs rather than the directory mtime, so it is
    stable across checkouts and moves with content, additions AND deletions.
    """
    root = ed_dir / "10_clean"
    if not root.is_dir():
        return {"path": None, "sha256": None, "n_docs": 0}
    rows = []
    for md in sorted(root.rglob("*.md")):
        if md.name == "README.md":
            continue
        rows.append(f"{md.relative_to(root).as_posix()}\0{sha256_of(md)}")
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest() if rows else None
    return {"path": str(root), "sha256": digest, "n_docs": len(rows)}


def select_sample(docs: list[dict], sample_n: int, seed: int) -> tuple[int, list[dict]]:
    """The scored document sample. One definition, used by both the scorer and the ruler.

    Factored out so the ruler fingerprint digests the SAME selection the scorer scores; computing
    it twice would let the two drift and the pin would verify a sample nobody used.
    """
    n_take = sample_n if sample_n > 0 else derive_sample_n(len(docs))
    n_take = min(n_take, len(docs))
    rng = random.Random(seed)
    return n_take, (rng.sample(docs, n_take) if n_take else [])


def sample_digest(ed_dir: Path, docs: list[dict], sample_n: int, seed: int) -> dict:
    """Digest of the SAMPLE SELECTOR — the seventh ruler input, and the first that is not a file.

    All three round-4 reviewers found this independently. The argument that licensed pinning the
    document corpus names three operands — the corpus, the count and the seed — and only the corpus
    was pinned. `--sample-n 5` moves kanto's clean_ratio from 0.1923 to 0.3529 with every one of the
    six pinned digests byte-identical, `ruler_verified: true` and `quotable: true`, without touching
    a byte on disk; on the production path seed 0 and seed 7 disagree the same way. Read-set
    enumeration can never reach these, which is exactly why three earlier reviewers converged on the
    registry and none reached the sampler.

    The digest is over the RELPATHS ACTUALLY SAMPLED, because that — not the seed integer — is what
    the denominator is a function of.
    """
    n_take, sample = select_sample(docs, sample_n, seed)
    rel = sorted(str(Path(d.get("path", "")).name) for d in sample)
    digest = hashlib.sha256("\n".join(rel).encode("utf-8")).hexdigest() if rel else None
    return {"seed": seed, "sample_n_requested": sample_n, "n_take": n_take,
            "n_docs_available": len(docs), "sha256": digest}


def editions_registry_digest(domain: str, ed_dir: Path) -> dict:
    """Digest of the UNIT REGISTRY — the sixth ruler input, and the one that decides how many
    units exist at all.

    The other five pin the inputs of ONE unit. `overall_ratio` is
    `sum(unit hits) / sum(unit n)` re-aggregated over whatever `iter_editions` yields, and
    `iter_editions` yields whatever `configs/editions_<domain>.json` lists — so the registry moves
    the aggregate denominator without touching a single pinned file. All three round-3 reviewers
    found this independently. The sharpest reproduction: point the registry at a `kanto_alias`
    unit whose `15_structured` is symlinked to kanto's and whose `10_clean` holds 20
    fixture-covered copies of the one document that scores 1/1, and the audit reports
    `clean_ratio 1.0000, status PASS, quotable true` while every pinned digest and row count is
    identical to the pinned 0.1923 baseline — the pinned kanto.ruler.json is never even opened.

    `resolved_dir` is recorded alongside the digest because a registry entry can silently
    RE-ADDRESS a unit (`{"unit": ["kanto", ""]}` yields the same directory under a different slug,
    which changes which pin file is looked up). The digest alone would not see that.
    """
    # Resolved from C.ROOT at CALL time, not from the module-level C.CONFIGS, which is computed at
    # import and would ignore a relocated root — the same convention the rest of this file uses.
    path = C.ROOT / "configs" / f"editions_{domain}.json"
    return {
        "path": str(path),
        "sha256": sha256_of(path),
        "resolved_dir": str(ed_dir),
    }


def ruler_pin_path(domain: str, unit_slug: str) -> Path:
    """Where a unit's ruler pin lives — beside its assertion fixture, same naming rule."""
    return C.BUILD / "recall" / domain / f"{unit_slug.replace('/', '_')}.ruler.json"


def ruler_fingerprint(
    domain: str,
    ed_dir: Path,
    unit_slug: str,
    *,
    fixture_path: Path | None = None,
    fixture: dict | None = None,
    docs: list[dict] | None = None,
    sample_n: int = 0,
    seed: int = 0,
) -> dict:
    """Digest + row count of every input the reported ratios depend on.

    ``pinned`` are the four the pin freezes; ``triples`` is recorded and deliberately unpinned.
    """
    ent_p = entities_path(ed_dir)
    trip_p = triples_path(ed_dir)
    idm_p = id_map_path(domain, unit_slug)
    man_p = domain_manifest_path(domain)
    fixture = fixture if fixture is not None else {}
    return {
        "unit": unit_slug,
        "fixture": {
            "path": str(fixture_path) if fixture_path else None,
            "sha256": sha256_of(fixture_path),
            "n_docs": len(fixture),
            "n_assertions": sum(len(v or []) for v in fixture.values()),
        },
        "entities": {
            "path": str(ent_p) if ent_p else None,
            "sha256": sha256_of(ent_p),
            "rows": len(_read_jsonl(ent_p)),
        },
        "id_map": {
            "path": str(idm_p) if idm_p else None,
            "sha256": sha256_of(idm_p),
            "entries": len(load_id_map(domain, unit_slug)),
        },
        "domain_manifest": {
            "path": str(man_p) if man_p else None,
            "sha256": sha256_of(man_p),
        },
        "documents": documents_digest(ed_dir),
        "editions_registry": editions_registry_digest(domain, ed_dir),
        "sample": sample_digest(ed_dir, docs or [], sample_n, seed),
        "triples": {
            "path": str(trip_p) if trip_p else None,
            "sha256": sha256_of(trip_p),
            "rows": len(_read_jsonl(trip_p)),
            "pinned": False,
            "note": "the graph under measurement — recorded, never pinned",
        },
    }


def load_ruler_pin(domain: str, unit_slug: str) -> dict | None:
    path = ruler_pin_path(domain, unit_slug)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def compare_ruler_pin(pin: dict, fingerprint: dict, *, fixture_supplied: bool = True) -> list[str]:
    """Human-readable mismatches between a pin and the live ruler. Empty list = verified.

    Only PINNED_RULER_FIELDS are compared. A field the pin does not carry is reported as a
    mismatch rather than skipped: a pin written by an older/edited writer must not silently
    verify a ruler it never covered.

    `fixture_supplied=False` skips ONLY the fixture digest. The fixture is a replay input, not a
    property of the corpus: the production invocation scores against the live judge and passes no
    `--facts-json`, so enforcing that digest unconditionally failed every real run closed with
    "fixture: file absent on disk" and quoted no ratio at all. The other four still bind, and the
    caller records `fixture: {supplied: false}` so the skip is visible rather than silent.
    """
    problems: list[str] = []
    pinned = pin.get("pinned") or {}
    for field in PINNED_RULER_FIELDS:
        if field == "fixture" and not fixture_supplied:
            continue
        # Key absence (an old/edited pin that never covered this field) is different from an
        # explicit null (the input was absent at pin time — e.g. resolve never wrote id_map.json).
        # Agreed absence (null==null) is a stable ruler state and must verify; a missing key is not.
        key_present = field in pinned
        want = pinned.get(field)
        got = (fingerprint.get(field) or {}).get("sha256")
        if not key_present:
            problems.append(f"{field}: pin carries no digest (pin is incomplete)")
        elif want is None and got is None:
            continue
        elif want is None and got is not None:
            problems.append(f"{field}: pinned absent but live {got[:12]}…")
        elif want is not None and got is None:
            problems.append(f"{field}: file absent on disk (pinned {want[:12]}…)")
        elif want != got:
            problems.append(f"{field}: pinned {want[:12]}… but live {got[:12]}…")
    # A registry can RE-ADDRESS a unit without changing its own bytes — `{"unit": ["kanto", ""]}`
    # resolves to the same directory under the slug "kanto/", which changes which pin file is
    # looked up. The digest cannot see that, so the resolved directory is compared separately.
    want_dir = (pin.get("recorded") or {}).get("resolved_dir")
    got_dir = (fingerprint.get("editions_registry") or {}).get("resolved_dir")
    if want_dir and got_dir and want_dir != got_dir:
        problems.append(f"editions_registry: unit resolved to {got_dir} but pinned {want_dir}")
    return problems


def write_ruler_pin(domain: str, unit_slug: str, fingerprint: dict) -> Path:
    """Freeze the four ruler digests. Re-pinning is legitimate and deliberately cheap — it is a
    committed file, so the diff is the record of why a coverage number is allowed to move."""
    path = ruler_pin_path(domain, unit_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = load_ruler_pin(domain, unit_slug)
    payload = {
        "domain": domain,
        "unit": unit_slug,
        "pinned": {f: (fingerprint.get(f) or {}).get("sha256") for f in PINNED_RULER_FIELDS},
        "recorded": {
            "fixture_n_docs": fingerprint["fixture"]["n_docs"],
            "fixture_n_assertions": fingerprint["fixture"]["n_assertions"],
            "entity_rows": fingerprint["entities"]["rows"],
            "id_map_entries": fingerprint["id_map"]["entries"],
            "triples_sha256_at_pin_time": fingerprint["triples"]["sha256"],
            "triples_rows_at_pin_time": fingerprint["triples"]["rows"],
            "document_count": (fingerprint.get("documents") or {}).get("n_docs"),
            "resolved_dir": (fingerprint.get("editions_registry") or {}).get("resolved_dir"),
            "sample": {k: v for k, v in (fingerprint.get("sample") or {}).items() if k != "sha256"},
        },
        "note": (
            "Coverage ruler pin. The four digests under `pinned` must match at score time or the "
            "audit fails closed and quotes no ratio. `triples_*_at_pin_time` is RECORDED ONLY — "
            "the graph is the thing under measurement and is never pinned, so a genuine "
            "extraction change still scores at full sensitivity. RE-PINNING IS THE ESCAPE HATCH "
            "AND IT IS NOT SEALED: whoever re-pins can move the ruler and then quote a number. "
            "`supersedes` is why that cannot be silent — it carries the digests and row counts "
            "this pin replaced, so the pin's own diff shows the ruler change that licensed the "
            "new series. Same trust model as the assertion fixture: committed, diffed, reviewed."
        ),
    }
    if previous:
        payload["supersedes"] = {
            "pinned": previous.get("pinned"),
            "recorded": previous.get("recorded"),
            "supersedes": previous.get("supersedes"),
        }
        changed = [
            f for f in PINNED_RULER_FIELDS
            if (previous.get("pinned") or {}).get(f) != payload["pinned"][f]
        ]
        if changed:
            before = previous.get("recorded") or {}
            print(
                f"  RE-PIN {domain}/{unit_slug}: ruler changed on {', '.join(changed)} — "
                f"entities {before.get('entity_rows')}→{payload['recorded']['entity_rows']} rows, "
                f"id_map {before.get('id_map_entries')}→{payload['recorded']['id_map_entries']} "
                f"entries, fixture {before.get('fixture_n_assertions')}→"
                f"{payload['recorded']['fixture_n_assertions']} assertions. Every coverage number "
                f"after this belongs to a NEW series and is not comparable to the previous one.",
                flush=True,
            )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
    CLEAN path (GATED, unchanged from Phase F2): judge_rejected + out_of_unit removed from
    denominator; matcher uses exact / soft-object / canonical modes.
    CLEAN_UNFILTERED (r3 diagnostic, NOT gated): denominator = manifest-predicate AND
    subject_in_unit, with the object-type contract ignored. It exists so the estimand choice is
    an argued number instead of a silent default — on the pinned kanto fixture the three arms read
    clean 0.1923 (78) / clean_unfiltered 0.0749 (267) / raw 0.0532 (376). The 154 assertions
    dropped as ``object_type_unresolved`` are "the object is not in our entity table", which is
    arguably an extraction gap rather than judge noise; that argument would move the gate to a
    third estimand and start a new eval series, so it is REPORTED here and decided elsewhere.
    """
    empty = {
        "hits": 0,
        "n": 0,
        "ratio": 1.0,
        "misses": [],
        "per_predicate": {},
        "judge_rejected": [],
        "out_of_unit": [],
        "reject_reasons": {},
        "hits_detail": [],
        "match_mode_counts": {"exact": 0, "soft-object": 0, "canonical": 0},
        "clean_unfiltered": {"hits": 0, "n": 0, "ratio": 1.0},
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
    reject_reasons: dict[str, int] = defaultdict(int)
    unfiltered_hits = unfiltered_n = 0
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
            reject_reasons[str(pred_reason or "").split(":")[0]] += 1
            continue
        if not subject_in_unit(s, entity_index):
            row = dict(fact)
            row["class"] = "out_of_unit"
            out_of_unit.append(row)
            continue
        ok_obj, obj_reason = validate_judge_fact(
            fact,
            predicates=pred_set or None,
            contract=contract,
            entity_index=entity_index,
        )
        # clean_unfiltered: same denominator MINUS the object-type contract. Scored for every
        # fact that got this far, including the ones CLEAN is about to drop — that is the whole
        # point of the arm, so the match has to run before the `continue`.
        unfiltered_n += 1
        unfiltered_mode = match_fact(s, p, o, graph_keys, entity_index)
        if unfiltered_mode is not None:
            unfiltered_hits += 1
        if not ok_obj:
            row = dict(fact)
            row["reject_reason"] = obj_reason
            judge_rejected.append(row)
            reject_reasons[str(obj_reason or "").split(":")[0]] += 1
            continue
        mode = unfiltered_mode
        if mode is None:
            clean_misses.append(fact)
            per_pred_clean[p].append(False)
        else:
            clean_hits += 1
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
        "reject_reasons": dict(reject_reasons),
        "hits_detail": hits_detail,
        "match_mode_counts": match_mode_counts,
        "clean_unfiltered": {
            "hits": unfiltered_hits,
            "n": unfiltered_n,
            "ratio": unfiltered_hits / unfiltered_n if unfiltered_n else 1.0,
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
    pin_ruler: bool = False,
    use_ruler_pin: bool = True,
) -> dict:
    fixture: dict[str, list[dict]] = {}
    fixture_path = Path(facts_json) if facts_json is not None else None
    if fixture_path is not None:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    entity_types = load_entity_types(domain)
    predicates = load_predicates(domain)
    contract = load_predicate_contract(domain)
    ruler_units: list[dict] = []
    ruler_problems: list[str] = []
    ruler_pins_written: list[str] = []
    ruler_pins_ignored: list[str] = []
    ruler_units_pinned = 0
    all_fixture_missing: list[dict] = []
    fixture_keys_used: set[str] = set()
    unfiltered_hits = unfiltered_n = 0
    reject_reason_totals: dict[str, int] = defaultdict(int)
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

        # ── Ruler provenance + pin verification ──────────────────────────────────────
        # Done BEFORE any scoring so a mismatch is attributable to a named file, not inferred
        # from a moved number. The fingerprint reads the same paths the loaders above used.
        fingerprint = ruler_fingerprint(
            domain, ed.dir, ed.slug, fixture_path=fixture_path, fixture=fixture,
            docs=docs, sample_n=sample_n, seed=seed,
        )
        if pin_ruler and not dry_run:
            ruler_pins_written.append(str(write_ruler_pin(domain, ed.slug, fingerprint)))
        # The pin is read whether or not it is enforced. --no-ruler-pin must not be able to
        # make an ignored pin invisible: skipping the check is itself a reportable condition,
        # or the flag is just a bypass with a polite name.
        pin = load_ruler_pin(domain, ed.slug)
        # The fixture is a REPLAY input: the production call scores against the live judge and
        # supplies no --facts-json, so its digest binds only when one was actually given.
        fixture_supplied = fixture_path is not None
        unit_problems = (compare_ruler_pin(pin, fingerprint, fixture_supplied=fixture_supplied)
                         if (pin and use_ruler_pin) else [])
        fingerprint["fixture"]["supplied"] = fixture_supplied
        fingerprint["pin"] = {
            "path": str(ruler_pin_path(domain, ed.slug)),
            "present": pin is not None,
            "enforced": bool(pin) and use_ruler_pin,
            "ignored": bool(pin) and not use_ruler_pin,
            "verified": (not unit_problems) if (pin and use_ruler_pin) else None,
            "problems": unit_problems,
            "fixture_compared": fixture_supplied,
        }
        if pin and use_ruler_pin:
            ruler_units_pinned += 1
        if pin and not use_ruler_pin:
            ruler_pins_ignored.append(ed.slug)
        ruler_problems.extend(f"{ed.slug}: {p}" for p in unit_problems)
        ruler_units.append(fingerprint)

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

        n_take, sample = select_sample(docs, sample_n, seed)

        doc_rows = []
        u_hits = u_n = 0
        u_raw_hits = u_raw_n = 0
        u_judge_rej: list[dict] = []
        u_out: list[dict] = []
        u_unavailable: list[dict] = []
        u_fixture_missing: list[dict] = []
        u_unfiltered_hits = u_unfiltered_n = 0
        u_reject_reasons: dict[str, int] = defaultdict(int)
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
            # fixture keys may be basename. `is not None` (not truthiness): a fixture entry of []
            # is a real "this document asserts nothing", and must not be confused with absence.
            hit_key = next(
                (k for k in (key, doc["name"], str(Path(key).name)) if fixture.get(k) is not None),
                None,
            )
            assertable = fixture.get(hit_key) if hit_key is not None else None
            if hit_key is not None:
                fixture_keys_used.add(hit_key)
            if assertable is None:
                if fixture:
                    # A SAMPLED document the fixture does not cover is a ruler/sample disagreement,
                    # not an empty document. Scored as [] it contributes 0/0 and vanishes: insert
                    # one .md at sorted position 0 and the seed-0 sample overlaps the fixture 3/20,
                    # yet the report still printed n_docs_sampled 20 / judge_unavailable 0 over a
                    # 3-document ratio. Record it; the run fails closed below.
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
            u_unfiltered_hits += cov["clean_unfiltered"]["hits"]
            u_unfiltered_n += cov["clean_unfiltered"]["n"]
            for reason, count in cov["reject_reasons"].items():
                u_reject_reasons[reason] += count
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
            "ruler": fingerprint,
            "hits": u_hits,
            "n": u_n,
            "ratio": round(u_ratio, 4),
            "raw": {
                "hits": u_raw_hits,
                "n": u_raw_n,
                "ratio": round(u_raw_ratio, 4),
            },
            "clean_unfiltered": {
                "hits": u_unfiltered_hits,
                "n": u_unfiltered_n,
                "ratio": round(u_unfiltered_hits / u_unfiltered_n, 4) if u_unfiltered_n else 1.0,
            },
            "classes": {
                "judge_rejected": len(u_judge_rej),
                "out_of_unit": len(u_out),
                "misses": u_n - u_hits,
                "hits": u_hits,
                "match_mode": u_match_mode,
            },
            "reject_reasons": dict(u_reject_reasons),
            "corpus_composition_note": corpus_note,
            "docs": doc_rows,
            "judge_rejected_sample": u_judge_rej[:25],
            "out_of_unit_sample": u_out[:25],
            "fixture_missing_sample": u_fixture_missing[:25],
            "judge_unavailable_sample": u_unavailable[:25],
        })
        overall_hits += u_hits
        overall_n += u_n
        raw_hits += u_raw_hits
        raw_n += u_raw_n
        class_totals["judge_rejected"] += len(u_judge_rej)
        class_totals["out_of_unit"] += len(u_out)
        class_totals["misses"] += (u_n - u_hits)
        class_totals["hits"] += u_hits
        for mode, count in u_match_mode.items():
            class_totals["match_mode"][mode] = class_totals["match_mode"].get(mode, 0) + count
        all_judge_rejected.extend(u_judge_rej)
        all_out_of_unit.extend(u_out)
        all_judge_unavailable.extend(u_unavailable)
        all_fixture_missing.extend(u_fixture_missing)
        unfiltered_hits += u_unfiltered_hits
        unfiltered_n += u_unfiltered_n
        for reason, count in u_reject_reasons.items():
            reject_reason_totals[reason] += count
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
            "note": "validated + in-unit + canonical-matched; gate applies here",
        },
        "clean_unfiltered": {
            "hits": unfiltered_hits,
            "n": unfiltered_n,
            "ratio": round(unfiltered_hits / unfiltered_n, 4) if unfiltered_n else 1.0,
            "note": (
                "DIAGNOSTIC, NOT GATED. CLEAN minus the object-type contract filter. The gap "
                "between this and `clean` is the object_type_* reject mass — see reject_reasons. "
                "Reported so the choice of gated estimand is an argued number: moving the gate "
                "here would restate every figure in the h5 trail and is a campaign, not a patch."
            ),
        },
        "ruler": {
            "note": (
                "Every input the ratios above are a function of. `pinned` digests are enforced: a "
                "run whose live ruler differs FAILS CLOSED and quotes no ratio. `triples` is "
                "recorded and never pinned — it is the thing under measurement, so a graph-only "
                "change still scores at full sensitivity. Quote clean_n, raw_n, entity rows, "
                "triple rows, id_map entries, judge_unavailable and fixture_docs_matched with "
                "every coverage number; a ratio without them is not comparable to any other run."
            ),
            "pinned_fields": list(PINNED_RULER_FIELDS),
            "n_units": len(ruler_units),
            "n_units_pinned": ruler_units_pinned,
            "verified": (not ruler_problems) if ruler_units_pinned else None,
            "problems": ruler_problems,
            "pins_written": ruler_pins_written,
            "pins_ignored": ruler_pins_ignored,
            # The registry DIGEST alone cannot see this: `--editions` narrows the aggregated unit
            # set without touching a byte of the registry file, and overall_ratio is a sum over
            # whatever set was actually iterated. Two runs quoting comparable-looking numbers over
            # different unit sets is the same failure as an unpinned ruler, so the resolved set and
            # the filter that produced it are recorded on every report.
            "editions_filter": editions,
            "units_iterated": [u.get("unit") for u in ruler_units],
            "units_unpinned": [u.get("unit") for u in ruler_units
                               if not ((u.get("pin") or {}).get("present"))],
            "units": ruler_units,
        },
        "fixture_sample": {
            "active": bool(fixture),
            "n_docs_sampled": sampled_docs,
            "n_docs_matched": (sampled_docs - len(all_fixture_missing)) if fixture else None,
            "n_docs_missing": len(all_fixture_missing) if fixture else None,
            "n_fixture_entries": len(fixture),
            "n_fixture_entries_unused": len(set(fixture) - fixture_keys_used) if fixture else None,
            "missing_sample": all_fixture_missing[:40],
            "note": (
                "Fixture/sample alignment. A sampled document the fixture does not cover scores "
                "0/0 and disappears while n_docs_sampled and judge_unavailable still read clean, "
                "so any miss is FAIL: re-pin the fixture or narrow with --editions. Unused fixture "
                "entries are reported, not fatal — a deliberately smaller --sample-n is legitimate."
            ),
        },
        "reject_reasons": dict(reject_reason_totals),
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
    if ruler_problems:
        # The ruler moved. Every ratio below was computed against a different ruler than the one
        # the pin recorded, so none of them is comparable to anything — this outranks the gate,
        # the judge-availability check and the sample check, because it invalidates all three.
        report["status"] = "FAIL"
        report["reason"] = (
            "ruler pin mismatch (" + "; ".join(ruler_problems[:4])
            + (" …" if len(ruler_problems) > 4 else "")
            + ") — the coverage ruler changed, so no ratio here is comparable to the pinned "
              "series; re-pin with --pin-ruler if the change was intended and expect a NEW series"
        )
    elif fixture and all_fixture_missing:
        # The fixture and the document sample disagree: every ratio is over a SUBSET of the
        # sample it claims. Quoting one is how a 3-document number got published as a
        # 20-document number with judge_unavailable 0.
        names = ", ".join(row["path"] for row in all_fixture_missing[:5])
        report["status"] = "FAIL"
        report["reason"] = (
            f"fixture covers {sampled_docs - len(all_fixture_missing)}/{sampled_docs} sampled "
            f"documents (missing: {names}{' …' if len(all_fixture_missing) > 5 else ''}); the "
            "coverage ratio would be quoted over a subset of the sample it claims — re-pin the "
            "fixture or select the unit with --editions"
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
        # Everything was filtered as noise. This used to be PASS (clean coverage is vacuously
        # 1.0) — a gate cleared by a denominator of zero. Measured: rewriting every predicate's
        # object list to ['nonexistent_type'] in the domain manifest, with a byte-identical
        # graph, emptied the denominator and turned kanto's 0.1923 FAIL into a 1.0000 PASS.
        # A ratio over nothing is the zero-from-an-uncalibrated-detector shape, not a result.
        report["status"] = "FAIL"
        report["reason"] = (
            f"clean denominator is 0 of {raw_n} raw assertions (all judge_rejected or "
            "out_of_unit) — a coverage ratio over an empty denominator measures nothing and "
            "must never clear the gate; check the predicate contract and the entity table"
        )

    # Quotability is a property of the run, not of the reader. Anything that made the ratios
    # incomparable is recorded ON the number so a receipt cannot copy it out of context.
    # ABSENCE OF A PIN IS NOT VERIFICATION. `unit_problems` is only computed when a pin exists, so
    # a unit with no pin file produced no problems and sailed through as quotable — and the unit
    # slug decides the pin's filename, which means renaming a unit (or adding one) silently unpins
    # it rather than failing closed. That is the mechanism by which the ruler could be swapped
    # WITHOUT touching any pinned file. A run may quote a ratio only if every unit it aggregated
    # was actually verified against a pin.
    units_unpinned = [u.get("unit") for u in (report.get("ruler") or {}).get("units") or []
                      if not ((u.get("pin") or {}).get("present"))]
    quotable = not (
        ruler_problems
        or ruler_pins_ignored          # --no-ruler-pin over a pinned unit is a diagnostic, not a number
        or units_unpinned              # an unpinned unit was never checked against anything
        or (fixture and all_fixture_missing)
        or overall_n == 0
    )
    report["quotable"] = quotable
    report["quotable_note"] = (
        "ratios are comparable to the pinned series"
        if quotable
        else "DO NOT QUOTE: " + str(
            report.get("reason")
            or (f"ruler pin ignored (--no-ruler-pin) for {', '.join(ruler_pins_ignored)}"
                if ruler_pins_ignored else "ruler/sample integrity failure")
        )
    )
    for block in ("overall", "clean", "raw", "clean_unfiltered"):
        report[block]["quotable"] = quotable

    if not dry_run:
        out = C.BUILD / "recall"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_{domain}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # The printed line is the surface receipts copy from, so an unquotable run prints NULL
    # ratios — not a number plus a caveat somewhere else in a 40k-line JSON.
    print(json.dumps({
        "status": report["status"],
        "quotable": quotable,
        "clean_ratio": report["overall"]["ratio"] if quotable else None,
        "raw_ratio": report["raw"]["ratio"] if quotable else None,
        "clean_unfiltered_ratio": report["clean_unfiltered"]["ratio"] if quotable else None,
        "clean_n": report["overall"]["n"],
        "raw_n": report["raw"]["n"],
        "clean_unfiltered_n": report["clean_unfiltered"]["n"],
        "classes": class_totals,
        "reject_reasons": dict(reject_reason_totals),
        "failed_predicates": failed_predicates,
        "judge_unavailable": report["judge_unavailable"]["n_docs"],
        "judge_unavailable_ratio": report["judge_unavailable"]["ratio"],
        # ── ruler guards: every coverage number carries these or it is not comparable ──
        "ruler_verified": report["ruler"]["verified"],
        "ruler_units_pinned": ruler_units_pinned,
        "ruler_problems": ruler_problems,
        "ruler_pins_ignored": ruler_pins_ignored,
        "fixture_sha256": (ruler_units[0]["fixture"]["sha256"] if ruler_units else None),
        "fixture_docs_matched": report["fixture_sample"]["n_docs_matched"],
        "fixture_docs_sampled": sampled_docs,
        "entity_rows": sum(u["entities"]["rows"] for u in ruler_units),
        "triple_rows": sum(u["triples"]["rows"] for u in ruler_units),
        "id_map_entries": sum(u["id_map"]["entries"] for u in ruler_units),
        "manifest_sha256": (ruler_units[0]["domain_manifest"]["sha256"] if ruler_units else None),
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
    ap.add_argument(
        "--pin-ruler", action="store_true",
        help="(re)write <unit>.ruler.json from the CURRENT fixture/entities/id_map/manifest "
             "digests. Deliberate and cheap: the pin is a committed file, so the diff is the "
             "record of why a coverage number was allowed to move.",
    )
    ap.add_argument(
        "--no-ruler-pin", action="store_true",
        help="score without enforcing an existing pin. DIAGNOSTIC ONLY — the resulting ratio is "
             "not comparable to the pinned series and must not be quoted as one.",
    )
    args = ap.parse_args()
    try:
        report = run_audit(
            args.domain, args.editions,
            min_coverage=args.min_coverage,
            sample_n=args.sample_n,
            facts_json=Path(args.facts_json) if args.facts_json else None,
            seed=args.seed,
            dry_run=args.dry_run,
            pin_ruler=args.pin_ruler,
            use_ruler_pin=not args.no_ruler_pin,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"audit_recall failed closed: {error}")
    sys.exit(0 if report["status"] in ("PASS", "SKIP") else 1)


if __name__ == "__main__":
    main()
