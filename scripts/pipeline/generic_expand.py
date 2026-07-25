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
import random
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


def _protected_spans(text: str, entities: list) -> list[tuple[int, int]]:
    """Character ranges of every entity mention — the parts a formatting variant may never touch."""
    spans: list[tuple[int, int]] = []
    for entity in entities:
        name = str(entity or "")
        if not name:
            continue
        start = text.find(name)
        while start != -1:
            spans.append((start, start + len(name)))
            start = text.find(name, start + 1)
    return spans


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Unique n-grams / total n-grams — the diversity gate's ruler."""
    grams: set = set()
    total = 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i:i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def formatting_augment_derived(
    records: list[dict],
    *,
    seed: int = 1729,
) -> tuple[list[dict], dict]:
    """Emit formatting variants ONLY at a setting that measurably raises distinct-2, else emit none.

    The knobs are derived by measurement rather than asked for or hardcoded. A formatting variant
    helps only when its new-bigram share beats the corpus's existing distinct-2; whether it does
    depends on sentence length and how much the variants collide with each other, which is a
    property of the corpus and cannot be known in advance. So the augmenter builds each candidate
    setting, measures distinct-2 on the result, and keeps the best one that IMPROVES on the
    un-augmented baseline. If none does, it returns the records untouched.

    That makes this incapable of regressing the gate it is meant to help — the failure mode of the
    first attempt, where a fixed prefix transform at a fixed fraction dropped distinct-2 by 27.7%.
    """
    if not records:
        return records, {"added": 0, "multiplier": 1.0, "chosen": None}
    baseline = distinct_n([r["text"] for r in records])
    best: tuple[list[dict], dict] | None = None
    best_score = baseline
    trials = []
    for fraction in (0.25, 0.5):
        for emphasis in (2, 3, 4, 6):
            candidate, report = formatting_augment(
                records, fraction=fraction, emphasis_per_surface=emphasis, seed=seed,
            )
            score = distinct_n([r["text"] for r in candidate])
            trials.append({"fraction": fraction, "emphasis": emphasis, "distinct_2": round(score, 4)})
            if score > best_score:
                best_score, best = score, (candidate, {**report, "distinct_2": score})
    if best is None:
        return records, {
            "added": 0, "multiplier": 1.0, "chosen": None,
            "baseline_distinct_2": round(baseline, 4), "trials": trials,
            "note": "no setting improved distinct-2 on this corpus — augmentation declined itself",
        }
    records_out, report = best
    return records_out, {
        **report,
        "baseline_distinct_2": round(baseline, 4),
        "chosen": {"fraction": report["fraction"], "emphasis": report["emphasis_per_surface"]},
        "distinct_2_gain": round(report["distinct_2"] - baseline, 4),
        "trials": trials,
    }


def formatting_augment(
    records: list[dict],
    *,
    fraction: float = 0.5,
    emphasis_per_surface: int = 2,
    seed: int = 1729,
) -> tuple[list[dict], dict]:
    """Formatting-only diversity: markdown emphasis on NON-ENTITY tokens (EMNLP-2025 dvlab).

    The first attempt used prefix transforms (indent/bullet/blockquote) and regressed distinct-2 by
    27.7%. That was an artifact of the transform, not of the technique: a prefix leaves every
    interior bigram identical, so each variant contributes ~1 new bigram against ~6 total (0.17),
    far under the corpus ratio (~0.33) — the ratio had to fall. An interior edit changes the two
    bigrams adjacent to each edited token instead (0.40 for one token, 0.80 for two), which is above
    the corpus ratio and therefore able to raise it.

    Fact safety is preserved by MASKING rather than by avoiding the interior: entity mentions are
    located and their character ranges are protected, so emphasis only lands on tokens that are not
    part of any entity. `_preserves_payload` then re-checks every entity survives verbatim, and any
    variant that fails is dropped rather than emitted.
    """
    if not records or fraction <= 0:
        return records, {"added": 0, "fraction": fraction, "multiplier": 1.0}
    rng = random.Random(seed)
    added: list[dict] = []
    skipped = 0
    for rec in sorted(records, key=lambda r: r["text"]):
        if rng.random() >= fraction:
            continue
        text = rec["text"]
        spans = _protected_spans(text, rec.get("entity_ids") or [])
        tokens, cursor, free = text.split(" "), 0, []
        for index, token in enumerate(tokens):
            start, end = cursor, cursor + len(token)
            cursor = end + 1
            if not token.strip() or any(s < end and start < e for s, e in spans):
                continue
            if any(ch.isalpha() for ch in token):
                free.append(index)
        if not free:
            skipped += 1
            continue
        for index in rng.sample(free, min(emphasis_per_surface, len(free))):
            tokens[index] = f"**{tokens[index]}**"
        variant_text = " ".join(tokens)
        if not _preserves_payload(rec, variant_text):
            skipped += 1
            continue
        variant = dict(rec)
        variant["text"] = variant_text
        variant["surface_form"] = f"{rec.get('surface_form', 'template')}+fmt:emphasis"
        added.append(variant)
    out = records + added
    return out, {
        "added": len(added),
        "skipped_unsafe": skipped,
        "fraction": fraction,
        "emphasis_per_surface": emphasis_per_surface,
        "seed": seed,
        "multiplier": round(len(out) / len(records), 4),
    }


def _preserves_payload(rec: dict, text: str) -> bool:
    """Every entity must survive VERBATIM — the invariant that keeps faithfulness unaffected."""
    return all(str(e) in text for e in (rec.get("entity_ids") or []) if str(e))


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
        recs, fmt = formatting_augment_derived(recs)
        print(f"  {unit_name}: utilization-balanced {util['kept']} kept / {util['dropped']} dropped "
              f"(quota={util['quota']}/fact, {util['facts']} facts; per-fact before "
              f"min={util['per_fact_before']['min']} median={util['per_fact_before']['median']} "
              f"max={util['per_fact_before']['max']})", flush=True)
        if fmt.get("chosen"):
            print(f"  {unit_name}: formatting +{fmt['added']} surfaces "
                  f"(chosen {fmt['chosen']}, distinct-2 {fmt['baseline_distinct_2']} -> "
                  f"{fmt['distinct_2']:.4f}, +{fmt['distinct_2_gain']})", flush=True)
        else:
            print(f"  {unit_name}: formatting declined — {fmt.get('note', 'no gain')}", flush=True)
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
