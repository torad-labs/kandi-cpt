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
_NAMES = ("metadata", "entity", "triple", "discourse")


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
    elif name == "metadata" and ont.get("topics"):
        s["$defs"]["topic"]["enum"] = list(ont["topics"])
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


def validate(kind: str, record: dict, domain: str | None = None) -> list[str]:
    """Return a list of human-readable error strings ([] = valid). `domain` selects the ontology (None = static)."""
    v = validator_for(kind, domain)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(record)]
