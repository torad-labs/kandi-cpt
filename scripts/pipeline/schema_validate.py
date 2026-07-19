"""Draft-2020-12 validators for the 15_structured records.

entity/triple/discourse `$ref` shared `$defs` in `metadata.schema.json` by its `$id`, so they
are not standalone — all four go into one `referencing.Registry` before building a validator.
Phase 3 validates EVERY record before writing JSONL (substrate spec §2.3).

DOMAIN-GENERAL: the domain-specific enums — entity_type + the id-pattern type list (entity), the
predicate enum (triple), the stance enum (discourse), the topic enum (metadata $defs) — are sourced
from the domain MANIFEST (`configs/domains/<d>.yaml`: `entity_types` / `predicates` / `ontology.topics`
/ `ontology.stances`), not hardcoded, so ANY domain validates against its own ontology. `domain=None`
loads the static schema files verbatim (backward-compatible); edc's manifest reproduces them byte-for-byte.
"""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import config as C
from pipeline.domain import Domain

_SCHEMA_DIR = C.ROOT / "schema"
_NAMES = ("metadata", "entity", "triple", "discourse", "relation")


def _load(name: str) -> dict:
    return json.loads((_SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _patch(name: str, s: dict, dom: Domain) -> dict:
    """Overwrite the schema's domain-specific enums with the manifest's ontology (in memory; files untouched)."""
    spec = dom.spec
    ont = spec.get("ontology") or {}
    if name == "entity" and spec.get("entity_types"):
        ets = list(spec["entity_types"])
        s["properties"]["entity_type"]["enum"] = ets
        s["properties"]["id"]["pattern"] = f"^[a-z0-9_]+:({'|'.join(ets)}):[a-z0-9_]+$"
    elif name == "triple" and spec.get("predicates"):
        s["properties"]["predicate"]["enum"] = list(spec["predicates"].keys())
    elif name == "discourse" and ont.get("stances"):
        s["properties"]["stance"]["enum"] = list(ont["stances"])
    elif name == "metadata":
        if ont.get("topics"):
            s["$defs"]["topic"]["enum"] = list(ont["topics"])
        # provenance vocabulary: the manifest's confidence_map keys ARE the domain's domain_class enum.
        # Only override when the SET differs — edc's map matches the static list, and byte-identity with
        # the static schema is a golden contract (test_edc_patched_equals_static).
        cmap = ont.get("confidence_map")
        if cmap and set(cmap) != set(s["$defs"]["domainClass"]["enum"]):
            s["$defs"]["domainClass"]["enum"] = list(cmap)
    elif name == "relation":
        # The 25_hydrated relation_type vocabulary, pinned to the manifest (the 15_structured contract
        # moved to the hydrate seam). KEY: the enum source MUST match the WRITER's relation_type source —
        #   • a domain whose hydrate writer emits SEMANTIC labels (EDC's Rust factory → 'artist_to_stage')
        #     declares relations.taxonomy → enum = the taxonomy keys;
        #   • a domain whose generic writer emits the RAW predicate (generic_expand/synth → 'made_of')
        #     declares NO taxonomy → enum = the predicates keys.
        # So a semantic-taxonomy domain that (mis)uses the generic predicate writer now FAILS at write.
        rels = spec.get("relations") or {}
        tax = rels.get("taxonomy") or {}
        if tax:
            s["properties"]["relation_type"]["enum"] = list(tax.keys())
        elif spec.get("predicates"):
            s["properties"]["relation_type"]["enum"] = list(spec["predicates"].keys())
    return s


def _schema(name: str, dom: Domain | None) -> dict:
    s = _load(name)
    return _patch(name, s, dom) if dom is not None else s


def _build_registry(dom: Domain | None) -> Registry:
    reg = Registry()
    for n in _NAMES:
        s = _schema(n, dom)
        reg = reg.with_resource(uri=s["$id"], resource=Resource.from_contents(s))
    return reg


_DOMAINS: dict[str, Domain] = {}
_VALIDATORS: dict[tuple[str, str], Draft202012Validator] = {}


def _domain(name: str | None) -> Domain | None:
    if name is None:
        return None
    if name not in _DOMAINS:
        _DOMAINS[name] = Domain(name)
    return _DOMAINS[name]


def validator_for(kind: str, domain: str | None = None) -> Draft202012Validator:
    """kind ∈ {'entity','triple','discourse'} — cached validator for the domain's ontology (None = static)."""
    key = (kind, domain or "_static")
    if key not in _VALIDATORS:
        dom = _domain(domain)
        _VALIDATORS[key] = Draft202012Validator(_schema(kind, dom), registry=_build_registry(dom))
    return _VALIDATORS[key]


def _id_type_segment(eid: str) -> str | None:
    """Middle segment of ``prefix:type:slug``; None if unstructured."""
    if not isinstance(eid, str):
        return None
    parts = eid.split(":")
    return parts[1] if len(parts) >= 3 else None


def id_type_mismatch(record: dict) -> dict | None:
    """Return a mismatch descriptor when id middle-segment ≠ entity_type; else None.

    Loud-warning primitive (not a validate hard-fail): EDC golden data has intentional
    cross-label cases (e.g. ``…:service:medical_tents`` with entity_type ``medical``), so
    a hard error would false-fail the reference domain. Extract/resolve call this to surface
    pokemon-style minting bugs (``…:location:kanto`` + entity_type ``region``).
    """
    eid = record.get("id")
    et = record.get("entity_type")
    seg = _id_type_segment(eid) if isinstance(eid, str) else None
    if seg is None or not isinstance(et, str) or not et or seg == et:
        return None
    return {
        "id": eid,
        "entity_type": et,
        "id_type_segment": seg,
        "name_canonical": record.get("name_canonical"),
    }


def warn_entity_id_type_mismatches(records: list[dict], *, where: str = "structured") -> list[dict]:
    """Print a loud warning for id/type-segment contradictions; return the mismatch list."""
    import sys
    bad = [m for m in (id_type_mismatch(r) for r in records) if m is not None]
    if bad:
        print(
            f"[{where}] WARNING: {len(bad)} entit(y/ies) with id middle-segment ≠ entity_type "
            f"(extract should mint prefix:entity_type:slug). sample={bad[:5]}",
            file=sys.stderr,
        )
    return bad


def validate(kind: str, record: dict, domain: str | None = None) -> list[str]:
    """Return a list of human-readable error strings ([] = valid). `domain` selects the ontology (None = static)."""
    v = validator_for(kind, domain)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(record)]
