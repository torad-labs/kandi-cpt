"""Draft-2020-12 validators for the 15_structured records.

entity/triple/discourse `$ref` shared `$defs` in `metadata.schema.json` by its `$id`, so they
are not standalone — all four go into one `referencing.Registry` before building a validator.
Phase 3 validates EVERY record before writing JSONL (substrate spec §2.3).
"""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import config as C

_SCHEMA_DIR = C.ROOT / "schema"
_NAMES = ("metadata", "entity", "triple", "discourse")


def _load(name: str) -> dict:
    return json.loads((_SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _build_registry() -> Registry:
    reg = Registry()
    for n in _NAMES:
        s = _load(n)
        reg = reg.with_resource(uri=s["$id"], resource=Resource.from_contents(s))
    return reg


_REGISTRY = _build_registry()
_VALIDATORS: dict[str, Draft202012Validator] = {}


def validator_for(kind: str) -> Draft202012Validator:
    """kind ∈ {'entity','triple','discourse'} — cached, registry-resolved validator."""
    if kind not in _VALIDATORS:
        _VALIDATORS[kind] = Draft202012Validator(_load(kind), registry=_REGISTRY)
    return _VALIDATORS[kind]


def validate(kind: str, record: dict) -> list[str]:
    """Return a list of human-readable error strings ([] = valid)."""
    v = validator_for(kind)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(record)]
