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
Output: <edition>/25_hydrated/relations.jsonl ({text, relation_type, entity_ids, source_triple_ids, split}).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from pipeline.domain import Domain
from pipeline.editions import iter_editions


def load_all_triples(ed) -> list[dict]:
    """Every triple as {subject_id, subject, predicate, object_id, object, attrs}, display names resolved
    from entities.jsonl. Domain-agnostic — any predicate, any object (an entity id OR a free literal)."""
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
        out.append({"subject_id": t.get("subject_id", ""), "subject": disp(t.get("subject_id", "")),
                    "predicate": t.get("predicate", ""), "object_id": t.get("object", ""),
                    "object": disp(t.get("object", "")), "attrs": t.get("attrs") or {}})
    if bad:
        print(f"  WARNING: {tf}: skipped {bad} malformed JSONL line(s)", file=sys.stderr)
    return out


def predicate_templates(domain: str) -> dict:
    """The domain's per-predicate hydrate template lists (configs/templates/<domain>.yaml)."""
    return ((Domain(domain).templates or {}).get("hydrate_templates") or {}).get("predicate_templates") or {}


def expand_records(triples: list[dict], ptmpls: dict, unit_name: str) -> list[dict]:
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
        i = counts[p]
        counts[p] += 1
        kw = {"subject": t["subject"], "object": t["object"], "unit": unit_name}
        kw.update({f"attr_{k}": v for k, v in t["attrs"].items()})
        for ti, tm in enumerate(tmpls):
            try:
                text = tm.format(**kw)
            except (KeyError, IndexError):
                continue
            rendered[(p, ti)] += 1
            out.append({"text": text, "relation_type": p, "direction": "forward",
                        "surface_form": "template", "source_triple_ids": [f"{p}:{i}"],
                        "entity_ids": [t["subject"], t["object"]], "split": "cpt"})
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
        mp = ed.dir / "festival_meta.json"
        unit_name = (json.loads(mp.read_text()).get("name") if mp.exists() else None) or ed.slug
        recs = expand_records(triples, ptmpls, unit_name)
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
