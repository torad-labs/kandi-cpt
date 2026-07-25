#!/usr/bin/env python3
"""Generic multi-relation hydrate generator — the domain-agnostic analog of the festival-specialized
`fact_expand`.

`fact_expand` handles ONE relation in a festival-specific shape (artist→stage→day, the `plays_at`
lineup) with bespoke balance/contrast/disjoint machinery. But a structured corpus is generally
multi-relation (even EDC's gold has `located_at`/`serves`/`requires`/… besides `plays_at`), and a
non-festival domain (pokemon-lore: `originates_from`/`member_of`/`leads`/… across 18 lore predicates)
has NO single primary relation. This generator covers that case: it loads EVERY triple generically and
renders the domain's per-predicate hydrate templates over them — the deterministic, faithful-by-
construction hydrate FLOOR (every line is one verified triple; no invention, no LLM).

A NEW domain points its `hydrate` stage at this and authors `hydrate_templates.predicate_templates` in
configs/templates/<domain>.yaml (predicate → list of templates, placeholders {subject} {object} {unit}
{attr_<k>}). EDC keeps its specialized fact_expand + the Rust factory; this is the reusable skeleton for
everything else. The LLM diversity layer (synth_corpus) sits on top of this floor.

    PYTHONPATH=scripts python scripts/pipeline/generic_expand.py --domain <d> [--editions city/year] [--dry-run]
Output: <edition>/25_hydrated/relations.jsonl ({text, relation_type, entity_ids, source_triple_ids,
source_doc_ids?, split}).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.schema_validate import validate
from pipeline.split import load_split, triple_id


def load_all_triples(ed) -> list[dict]:
    """Load every triple with display names resolved while retaining provenance/confidence when present.

    Domain-agnostic — any predicate, any object (an entity id OR a free literal). The display fields are
    additive; source lineage must not disappear merely because hydration needs human-readable names.
    """
    struct = ed.dir / "15_structured"
    ef, tf = struct / "entities.jsonl", struct / "triples.jsonl"
    if not tf.exists():
        return []
    names: dict[str, str] = {}
    if ef.exists():
        for line in ef.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    e = json.loads(line)
                    names[e["id"]] = e.get("name_canonical", "")
                except Exception:
                    pass

    def disp(idstr) -> str:
        s = str(idstr)
        return names.get(s) or (s.split(":")[-1].replace("_", " ") if ":" in s else s)

    out: list[dict] = []
    bad = 0
    for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            bad += 1                                       # a corrupt triple = a verified fact silently lost — count it
            continue
        loaded = {"subject_id": t.get("subject_id", ""), "subject": disp(t.get("subject_id", "")),
                  "predicate": t.get("predicate", ""), "object_id": t.get("object", ""),
                  "object": disp(t.get("object", "")), "attrs": t.get("attrs") or {}}
        for key in ("provenance", "confidence"):
            if key in t:
                loaded[key] = t[key]
        out.append(loaded)
    if bad:
        print(f"  WARNING: {tf}: skipped {bad} malformed JSONL line(s)", file=sys.stderr)
    return out


def source_doc_ids(triple: dict) -> list[str]:
    """Known clean-document ids for a loaded structured triple (empty for legacy records)."""
    provenance = triple.get("provenance")
    doc_id = provenance.get("doc_id") if isinstance(provenance, dict) else None
    return [doc_id] if isinstance(doc_id, str) and doc_id else []


def predicate_templates(domain: str) -> dict:
    """The domain's per-predicate hydrate template lists (configs/templates/<domain>.yaml)."""
    return ((Domain(domain).templates or {}).get("hydrate_templates") or {}).get("predicate_templates") or {}


def balance_by_utilization(records: list[dict], *, quota: int | None = None) -> tuple[list[dict], dict]:
    """Allocate surfaces by LOWEST utilization first instead of emitting every render uniformly.

    Reference: Source2Synth / SoG's secondary-sampling + controlled-allocation mechanism — generation
    budget goes to under-covered facts and entities rather than being spread evenly, so a popular
    subject stops soaking up surfaces a long-tail one never gets.

    Uniform rendering makes per-fact counts a function of how many templates happen to fit a
    predicate, which is a property of the template file rather than of the domain. Measured on
    pokemon-lore before this change: template surfaces ran 2 per fact at the median and 21 at the
    max (10.5x), llm_synth 10 and 70 (7.0x).

    `quota` is DERIVED from the corpus (median renders per fact) unless supplied. Facts at or below
    quota keep everything — nothing is culled that was not over-emitting. Facts above it keep the
    quota, choosing the renders whose entities are currently least-used, which is where the entity
    balancing happens. Deterministic: ties break on the record's own text.
    """
    by_fact: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        by_fact[tuple(rec.get("source_triple_ids") or [rec["text"]])].append(rec)
    if not by_fact:
        return records, {"quota": 0, "facts": 0, "kept": 0, "dropped": 0}

    sizes = sorted(len(v) for v in by_fact.values())
    derived_quota = quota if quota is not None else max(1, int(statistics.median(sizes)))

    entity_use: Counter = Counter()
    kept: list[dict] = []
    dropped = 0
    # Facts with the FEWEST candidates are allocated first: they are the long tail, and letting
    # them claim their entities' budget before the popular facts is the whole point.
    for fact in sorted(by_fact, key=lambda k: (len(by_fact[k]), k)):
        members = sorted(by_fact[fact], key=lambda r: r["text"])
        if len(members) <= derived_quota:
            chosen = members
        else:
            chosen = sorted(
                members,
                key=lambda r: (sum(entity_use[e] for e in (r.get("entity_ids") or [])), r["text"]),
            )[:derived_quota]
            dropped += len(members) - len(chosen)
        for rec in chosen:
            for entity in rec.get("entity_ids") or []:
                entity_use[entity] += 1
        kept.extend(chosen)

    report = {
        "quota": derived_quota,
        "facts": len(by_fact),
        "kept": len(kept),
        "dropped": dropped,
        "per_fact_before": {"min": sizes[0], "median": statistics.median(sizes), "max": sizes[-1]},
        "entity_utilization": {
            "n_entities": len(entity_use),
            "min": min(entity_use.values()) if entity_use else 0,
            "max": max(entity_use.values()) if entity_use else 0,
        },
    }
    return kept, report


def expand_records(triples: list[dict], ptmpls: dict, unit_name: str, domain: str | None = None) -> list[dict]:
    """Render the per-predicate templates over the triples → hydrate records (the 25_hydrated shape).
    A triple whose predicate has no template is skipped; a template referencing an attr the triple lacks
    is skipped (KeyError) rather than emitting a malformed line. Deterministic, faithful by construction."""
    out: list[dict] = []
    counts: dict = defaultdict(int)              # predicate -> n triples seen
    rendered: dict = defaultdict(int)            # (predicate, template_index) -> n successful renders
    for t in triples:
        p = t["predicate"]
        tmpls = ptmpls.get(p)
        if not tmpls:
            continue
        counts[p] += 1
        kw = {"subject": t["subject"], "object": t["object"], "unit": unit_name}
        kw.update({f"attr_{k}": v for k, v in t["attrs"].items()})
        for ti, tm in enumerate(tmpls):
            try:
                text = tm.format(**kw)
            except (KeyError, IndexError):
                continue
            rendered[(p, ti)] += 1
            rec = {"text": text, "relation_type": p, "direction": "forward",
                   "surface_form": "template", "source_triple_ids": [triple_id(t)],
                   "entity_ids": [t["subject"], t["object"]], "split": "cpt"}
            if doc_ids := source_doc_ids(t):
                rec["source_doc_ids"] = doc_ids
            _errs = validate("relation", rec, domain=domain)
            if _errs:
                raise SystemExit(
                    f"[generic_expand] 25_hydrated relation record fails schema for domain "
                    f"'{domain}': {_errs[:3]} | rec={rec}. relation_type '{p}' is not in the manifest's "
                    f"relations.taxonomy / predicates enum (configs/domains/{domain}.yaml) — the hydrated "
                    f"vocabulary must match the manifest the gates classify against.")
            out.append(rec)
    # A template that rendered for ZERO of a predicate's triples is a placeholder typo, NOT an optional-attr
    # skip (those render for SOME triples). Surface it — a silently-zero template is a missing training fact.
    for p, n in counts.items():
        for ti in range(len(ptmpls.get(p) or [])):
            if n and not rendered[(p, ti)]:
                print(f"  WARNING: predicate '{p}' template #{ti} rendered 0/{n} triples — likely a "
                      f"placeholder typo in hydrate_templates.predicate_templates", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions", help="restrict to 'unit,…' (default: all on disk)")
    ap.add_argument("--dry-run", action="store_true", help="print per-unit counts + uncovered predicates, write nothing")
    args = ap.parse_args()
    ptmpls = predicate_templates(args.domain)
    if not ptmpls:
        raise SystemExit(f"domain '{args.domain}' has no hydrate_templates.predicate_templates — author them in "
                         f"configs/templates/{args.domain}.yaml (or, if that file exists, check the "
                         f"hydrate_templates.predicate_templates key spelling/nesting).")
    wrote = 0
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):   # consumes 15_structured, not 00_raw
        triples = load_all_triples(ed)
        if not triples:
            continue
        sp = load_split(ed.dir)
        if sp:
            before = len(triples)
            triples = [t for t in triples if triple_id(t) in sp["train_ids"]]
            print(f"  {ed.slug}: split {sp['version']} — hydrating {len(triples)}/{before} triples "
                  f"({before - len(triples)} holdout EXCLUDED)")
        else:
            print(f"  {ed.slug}: no split manifest — hydrating ALL triples (probe deltas are recall, "
                  f"never held-out; carve with scripts/pipeline/split.py)")
        # Three-tier routing (trust-loop/04): hot-tier predicates NEVER reach the train shard.
        try:
            from pipeline.tier_routing import filter_triples_by_tier, load_predicate_tiers
            _tiers = load_predicate_tiers(args.domain)
            if _tiers:
                before_tier = len(triples)
                triples, _drops = filter_triples_by_tier(
                    triples, _tiers, allow=frozenset({"stable", "proven"}),
                )
                hot_n = int(_drops.get("hot") or 0)
                if hot_n:
                    print(f"  {ed.slug}: tier filter — dropped {hot_n} hot-tier triples "
                          f"({before_tier}→{len(triples)}; stable+proven only)")
        except Exception as _tier_err:  # noqa: BLE001 — never block expand on optional tier module issues
            print(f"  {ed.slug}: tier filter skipped ({type(_tier_err).__name__}: {_tier_err})", flush=True)
        mp = ed.dir / "festival_meta.json"
        unit_name = (json.loads(mp.read_text()).get("name") if mp.exists() else None) or ed.slug
        recs = expand_records(triples, ptmpls, unit_name, args.domain)
        recs, util = balance_by_utilization(recs)
        print(f"  {unit_name}: utilization-balanced {util['kept']} kept / {util['dropped']} dropped "
              f"(quota={util['quota']}/fact, {util['facts']} facts; per-fact before "
              f"min={util['per_fact_before']['min']} median={util['per_fact_before']['median']} "
              f"max={util['per_fact_before']['max']})", flush=True)
        uncovered = sorted({t["predicate"] for t in triples} - set(ptmpls))
        if args.dry_run:
            print(f"  {ed.slug}: {len(recs)} lines over {len({r['relation_type'] for r in recs})} relations; "
                  f"uncovered predicates: {uncovered or 'none'}")
            continue
        if not recs:
            continue
        out_dir = ed.dir / "25_hydrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "relations.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        os.replace(tmp, out_path)
        wrote += 1
        if uncovered:
            print(f"  WARNING: predicates with NO template (uncovered, silently skipped): {uncovered}",
                  file=sys.stderr)
        print(f"  {ed.slug}: wrote {len(recs)} hydrated lines "
              f"({len({r['relation_type'] for r in recs})} relations); uncovered: {uncovered or 'none'}")
    if not args.dry_run:
        print(f"\nwrote {wrote} relations.jsonl")


if __name__ == "__main__":
    main()
