"""Emit JSON Schema for the config contract → schema/cpt_config.schema.json (Phase 5).

This file is the machine-readable contract a future control-plane app consumes (and editors validate against).
It is GENERATED from scripts/cpt_config.py — never hand-edit it. Regenerate + drift-gate:

  PYTHONPATH=scripts python scripts/emit_schema.py            # write schema/cpt_config.schema.json
  PYTHONPATH=scripts python scripts/emit_schema.py --check    # nonzero-exit if on-disk drifts from the dataclasses

The --check mode is the drift gate: if someone adds a field to a sub-config but forgets to regenerate, CI
(or a pre-commit) catches it. Stdlib only (no pydantic — matches the repo's dataclass approach).
"""
from __future__ import annotations

import argparse
import json
import sys
import typing
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpt_config as CC  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "cpt_config.schema.json"

_PRIM = {bool: "boolean", int: "integer", float: "number", str: "string"}


def _schema_for(tp) -> dict:
    """Map a (resolved) Python type annotation to a JSON-Schema fragment."""
    origin = typing.get_origin(tp)
    if origin is typing.Literal:                       # Literal["a","b"] → enum
        return {"enum": list(typing.get_args(tp))}
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):  # X | None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        has_none = len(args) != len(typing.get_args(tp))
        base = _schema_for(args[0]) if len(args) == 1 else {"anyOf": [_schema_for(a) for a in args]}
        if has_none and "type" in base:
            base["type"] = [base["type"], "null"]
        elif has_none:
            base = {"anyOf": [base, {"type": "null"}]}
        return base
    if origin in (list, typing.List):                  # list[T]
        (item,) = typing.get_args(tp) or (str,)
        return {"type": "array", "items": _schema_for(item)}
    if is_dataclass(tp):
        return _object_schema(tp)
    if tp in _PRIM:
        return {"type": _PRIM[tp]}
    return {}                                          # unknown → permissive


def _object_schema(cls) -> dict:
    hints = typing.get_type_hints(cls)
    props, required = {}, []
    for f in fields(cls):
        props[f.name] = _schema_for(hints[f.name])
        # a field with no default and no default_factory is required
        if f.default is MISSING and f.default_factory is MISSING:  # type: ignore[misc]
            required.append(f.name)
    out = {"type": "object", "additionalProperties": False, "properties": props}
    if required:
        out["required"] = required
    return out


def build_schema() -> dict:
    root = _object_schema(CC.CptConfig)
    root["$schema"] = "http://json-schema.org/draft-07/schema#"
    root["title"] = "kandi-cpt run config (CptConfig)"
    root["description"] = ("THE config contract — generated from scripts/cpt_config.py by scripts/emit_schema.py. "
                           "One training entrypoint + one eval entrypoint consume this; do not hand-edit.")
    # reorder so $schema/title/description lead
    return {k: root[k] for k in ("$schema", "title", "description", "type", "additionalProperties",
                                 "properties", "required") if k in root}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit nonzero if the on-disk schema drifts from the dataclasses")
    args = ap.parse_args()
    schema = build_schema()
    text = json.dumps(schema, indent=2) + "\n"
    if args.check:
        if not SCHEMA_PATH.exists():
            sys.exit(f"✗ {SCHEMA_PATH} missing — run `python scripts/emit_schema.py` to generate it.")
        on_disk = SCHEMA_PATH.read_text()
        if on_disk != text:
            sys.exit(f"✗ {SCHEMA_PATH} is STALE vs scripts/cpt_config.py — regenerate with "
                     f"`python scripts/emit_schema.py` and commit.")
        print(f"✓ schema up to date ({len(schema['properties'])} top-level sections)")
    else:
        SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA_PATH.write_text(text)
        print(f"wrote {SCHEMA_PATH}  ({len(schema['properties'])} sections)")


if __name__ == "__main__":
    main()
