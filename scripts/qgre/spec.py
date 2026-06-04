"""Domain-spec reader for the quality gates.

Reads the four quality sections a domain manifest declares (configs/domains/<name>.yaml: entity_types,
predicates, relations, quality) via the existing `pipeline.domain.Domain` loader, and exposes the
typed-graph invariants the gates enforce. Nothing here is festival-specific — a non-festival domain
populates the same sections and the gates read them identically.

The one non-trivial helper is `required_artifact_present`: it resolves a logical `needs` key (e.g.
"set_times.jsonl" or "festival_meta.days") to an on-disk probe relative to an edition dir, which is how
`gate_source_vs_gap` distinguishes a SOURCE GAP (no data) from a DEFECT (data present, generation failed).
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.domain import Domain


def load_spec(domain: str = "edc") -> dict:
    """The parsed domain manifest (configs/domains/<domain>.yaml)."""
    return Domain(domain).spec


def entity_types(spec: dict) -> set[str]:
    return set(spec.get("entity_types", []))


def predicate_contract(spec: dict) -> dict[str, set[str]]:
    """predicate -> allowed object entity_types (or {'literal'} for free values). Predicates absent from
    the spec are unconstrained — the gate only enforces the contracts we've actually declared."""
    return {p: set(meta.get("object", [])) for p, meta in (spec.get("predicates") or {}).items()}


def relations_cfg(spec: dict) -> dict:
    """The `relations` block: exposure_cap_per_fact, dominance_max_frac, min_active_types, taxonomy{}."""
    return spec.get("relations", {})


def relation_meta(spec: dict) -> dict[str, dict]:
    """relation_type -> {needs:[...], floor_frac, target_frac?}."""
    return (spec.get("relations") or {}).get("taxonomy", {})


def quality_cfg(spec: dict) -> dict:
    return spec.get("quality", {})


def required_artifacts(spec: dict) -> dict[str, str]:
    """logical needs-key -> on-disk probe string ('file' or 'file#jsonkey')."""
    return ((spec.get("quality") or {}).get("source_gap") or {}).get("required_artifacts", {})


def required_artifact_present(edition_dir: Path, logical: str, spec: dict) -> tuple[bool, str]:
    """Resolve a `needs` artifact for one edition. Returns (present, probe_str). Probe forms:

      'file'              present iff the file exists and is non-empty.
      'file#key'          present iff the JSON file has a non-empty top-level value at `key`.
      'file#key.subkey'   present iff `key` is a dict/list of objects and SOME element carries a
                          non-empty `subkey`. This is what tells a real DEFECT from a SOURCE GAP for
                          relations whose source is a *field inside* a scaffolded skeleton — e.g.
                          'festival_meta.json#days.date': every edition is scaffolded with a days
                          skeleton {Friday:'',…}, so the bare '#days' probe wrongly reports present on a
                          dateless edition; the hydrator only emits day_of_week_map when a day has a real
                          date, so requiring '.date' makes the dateless majority a correct SOURCE GAP.
      'file~substring'    present iff the file exists and contains `substring` (a content probe — used to
                          detect e.g. a B2B billing in the lineup, the genuine source for b2b_pairing).
    """
    probe = required_artifacts(spec).get(logical, logical)
    # content probe: "file~substring"
    if "~" in probe:
        fname, _, substr = probe.partition("~")
        fp = edition_dir / fname
        try:
            return (fp.exists() and substr in fp.read_text(encoding="utf-8", errors="replace")), probe
        except OSError:
            return False, probe
    fname, _, jpath = probe.partition("#")
    fp = edition_dir / fname
    if not fp.exists():
        return False, probe
    if not jpath:
        try:
            return fp.stat().st_size > 0, probe
        except OSError:
            return False, probe
    try:
        doc = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False, probe
    jkey, _, subkey = jpath.partition(".")
    val = doc.get(jkey)
    if subkey:
        members = val.values() if isinstance(val, dict) else (val if isinstance(val, list) else [])
        present = any(isinstance(m, dict) and str(m.get(subkey, "")).strip() for m in members)
        return present, probe
    # non-empty: a populated list/dict/str/number counts as present; [], {}, "", None do not.
    present = bool(val) if not isinstance(val, (int, float)) else True
    return present, probe
