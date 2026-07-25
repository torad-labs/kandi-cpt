#!/usr/bin/env python3
"""Generic forced-choice probe generator — the domain-agnostic eval-items skeleton.

The festival `eval_cloze.py` builds forced-choice items over stages/days ("Which stage does X play?",
choices = the 7 stages). This is the generic analog: for each verified triple it builds a forced-choice
item — the question from `sft_templates.rel_q[predicate][0]`, the correct object, and N distractors drawn
from the SAME predicate's object pool (plausible wrong answers of the right type). Domain-agnostic: a new
domain authors NO probe code — the questions are its `rel_q`, the distractors are its own entities. The
probe *design* (held-out strategy, era tags, base-delta) stays the human-reviewed `domains/<d>.probes.md`;
this generates the *items* deterministically.

    PYTHONPATH=scripts python scripts/pipeline/generic_probes.py --domain <d> [--editions ...] [--distractors 3] [--dry-run]
Output: <edition>/probes.jsonl ({question, choices, answer, predicate, subject, source_triple_id, split}).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

from pipeline.audit_recall import load_id_map
from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples
from pipeline.split import load_split, triple_id


def rel_q_for(domain: str) -> dict:
    """The domain's sft_templates.rel_q (predicate → [question, answer]) — the probe questions."""
    return ((Domain(domain).templates or {}).get("sft_templates") or {}).get("rel_q") or {}


def question_surfaces(question: str, count: int) -> list[str]:
    """Deterministic prompt surfaces over one gold question; never changes its answer or source id."""
    if count < 1 or count > 4:
        raise ValueError(f"surfaces_per_gold must be in [1,4], got {count}")
    stem = question[0].lower() + question[1:] if question else question
    surfaces = [
        question,
        f"Answer briefly: {question}",
        f"Using the stated lore, {stem}",
        f"Give only the named fact: {question}",
    ]
    return surfaces[:count]


def surface_id(source_id: str, variant: int, question: str) -> str:
    payload = f"{source_id}\0{variant}\0{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def subject_key(t: dict) -> str:
    """The raw id a triple records its subject under (display name only when there is no id)."""
    return str(t.get("subject_id", "") or t.get("subject", ""))


def entity_classes(pool_triples: list[dict], id_map: dict[str, str] | None = None) -> dict[str, str]:
    """Map every subject_id to a stable representative of the ENTITY it denotes.

    Two ids denote one entity — and therefore share one truth set — when either independent evidence
    says so: they render the same display name (a question built from that name cannot distinguish
    them, so an object true under either id answers it correctly), or resolve merged them (surface
    variants of one entity, e.g. 'Above & Beyond' / 'Above and Beyond'). Neither test subsumes the
    other: merged ids can render different names, and same-name ids are not always merged (a
    character and a species_lore entry legitimately stay distinct in resolve yet render alike).

    Keying the truth subtraction on the RAW id splits one entity's truths across keys, so an object
    missing from the probed id's key survives into the distractor pool — and ships as a "wrong"
    answer that is in fact right. Representative is the class minimum, so probes stay deterministic.
    """
    id_map = id_map or {}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            low, high = sorted((ra, rb))
            parent[high] = low

    first_id_for_name: dict[str, str] = {}
    for t in pool_triples:
        sid = subject_key(t)
        find(sid)
        canonical = id_map.get(sid)
        if canonical:
            union(sid, canonical)
        name = str(t.get("subject", "") or "")
        if name:
            if name in first_id_for_name:
                union(sid, first_id_for_name[name])
            else:
                first_id_for_name[name] = sid
    return {sid: find(sid) for sid in list(parent)}


def build_probes(gold_triples: list[dict], rel_q: dict, unit_name: str = "", n_distractors: int = 3,
                 seed: int = 0, pool_triples: list[dict] | None = None,
                 surfaces_per_gold: int = 1, id_map: dict[str, str] | None = None) -> list[dict]:
    """Forced-choice items over `gold_triples`, with distractors drawn from `pool_triples` (default:
    the same records) after excluding every object true for that entity+predicate. A gold triple is
    skipped if its predicate has no question template or no safe distractor pool (can't force a
    choice). Deterministic given `seed`."""
    rng = random.Random(seed)
    pool_triples = gold_triples if pool_triples is None else pool_triples
    classes = entity_classes(pool_triples, id_map)
    by_pred: dict[str, set] = defaultdict(set)
    true_by_entity_pred: dict[tuple[str, str], set] = defaultdict(set)
    for t in pool_triples:
        if t["object"]:
            by_pred[t["predicate"]].add(t["object"])
            entity = classes[subject_key(t)]
            true_by_entity_pred[(entity, t["predicate"])].add(t["object"])
    items: list[dict] = []
    seen_source_ids: set[str] = set()
    duplicate_sources = 0
    seen_q: dict = defaultdict(int)                        # predicate WITH a question template -> n triples seen
    made: dict = defaultdict(int)                          # predicate -> n probes produced
    drops: dict = defaultdict(lambda: defaultdict(int))    # predicate -> reason -> n
    for t in gold_triples:
        source_id = triple_id(t)
        if source_id in seen_source_ids:
            duplicate_sources += 1
            continue
        seen_source_ids.add(source_id)
        p, obj = t["predicate"], t["object"]
        q = rel_q.get(p)
        if not q:
            continue                                       # no question template → not a probe axis (no warn)
        seen_q[p] += 1
        if not obj:
            drops[p]["no_object"] += 1
            continue
        sid = subject_key(t)
        pool = sorted(by_pred[p] - true_by_entity_pred[(classes.get(sid, sid), p)])
        if not pool:                                       # need ≥1 distractor of the right type
            drops[p]["no_safe_distractor"] += 1
            continue
        try:
            question = q[0].format(s=t["subject"], n=unit_name).strip()
        except (KeyError, IndexError):
            drops[p]["template_error"] += 1
            continue
        for variant, rendered in enumerate(question_surfaces(question, surfaces_per_gold)):
            choices = rng.sample(pool, min(n_distractors, len(pool))) + [obj]
            rng.shuffle(choices)
            items.append({"question": rendered, "choices": choices, "answer": obj,
                          "predicate": p, "subject": t["subject"], "source_triple_id": source_id,
                          "probe_surface_id": surface_id(source_id, variant, rendered),
                          "surface_variant": variant, "split": "held_out"})
            made[p] += 1
    empty_pool_drops = sum(d.get("no_safe_distractor", 0) for d in drops.values())
    print(f"  source-id guard: dropped {duplicate_sources} duplicate semantic triple(s)")
    print(f"  multi-correct guard: dropped {empty_pool_drops} probe(s) with an empty distractor pool "
          f"after excluding all objects true for the same (entity, predicate) — entity folds together "
          f"ids that render one name or that resolve merged")
    # A predicate the operator authored a question for that yields ZERO probes is an UNTESTED fact axis — in an
    # eval set the absence of a test must not silently read as a passing test.
    for p, seen in seen_q.items():
        d = dict(drops[p])
        if not made[p]:
            print(f"  WARNING: predicate '{p}' has a question template but produced 0 probes "
                  f"({d} of {seen} triples) — that fact axis is UNTESTED", file=sys.stderr)
        elif d:                                            # partial drop — some instances silently left untested
            print(f"  WARNING: predicate '{p}': {made[p]}/{seen} triples → probes; {sum(d.values())} dropped "
                  f"({d}) — those fact instances are not in the eval set", file=sys.stderr)
    return items


def agreement_key(t: dict) -> tuple[str, str, str]:
    return (str(t.get("subject_id") or ""), str(t.get("predicate") or ""), str(t.get("object") or ""))


def two_model_gold_filter(
    gold_triples: list[dict],
    second_pass: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Keep intersection of gold with a second-extractor pass; disagreements → adjudication queue.

    Gold that ships is only the intersection. Disagreements never enter scored items silently.
    """
    second_keys = {agreement_key(t) for t in second_pass}
    kept, disagreed = [], []
    for t in gold_triples:
        if agreement_key(t) in second_keys:
            kept.append(t)
        else:
            disagreed.append(t)
    return kept, disagreed


def load_second_pass(path) -> list[dict]:
    """Load a second-model extraction jsonl (same triple shape) for --gold-agreement."""
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"gold-agreement second pass missing: {p}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions")
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--surfaces-per-gold", type=int, default=1, choices=range(1, 5),
                    help="deterministic question surfaces per held-out gold triple (1-4)")
    ap.add_argument("--dry-run", action="store_true", help="print per-unit counts, write nothing")
    ap.add_argument(
        "--gold-agreement",
        metavar="SECOND_PASS_JSONL",
        help="two-model gold: keep intersection with SECOND_PASS_JSONL; disagreements → "
             "data/_build/gold_agreement/<domain>/<unit>.review.jsonl (never scored silently)",
    )
    args = ap.parse_args()
    rel_q = rel_q_for(args.domain)
    if not rel_q:
        raise SystemExit(f"domain '{args.domain}' has no sft_templates.rel_q — author it in "
                         f"configs/templates/{args.domain}.yaml (or check the key spelling/nesting if the file exists).")
    second_pass_all = load_second_pass(args.gold_agreement) if args.gold_agreement else None
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):   # consumes 15_structured, not 00_raw
        all_triples = load_all_triples(ed)
        if not all_triples:
            continue
        sp = load_split(ed.dir)
        if sp:
            gold_triples = [t for t in all_triples if triple_id(t) in sp["holdout_ids"]]
            split_label = "held_out"
            print(f"  {ed.slug}: split {sp['version']} held-out mode — {len(gold_triples)} gold / "
                  f"{len(all_triples)} distractor-pool triples")
        else:
            gold_triples = all_triples
            split_label = "trained"
            print(f"  {ed.slug}: trained mode — {len(gold_triples)} gold / {len(all_triples)} "
                  f"distractor-pool triples (no split manifest)")
        gold_agreement_stamp = None
        if second_pass_all is not None:
            kept, disagreed = two_model_gold_filter(gold_triples, second_pass_all)
            print(f"  {ed.slug}: gold-agreement two_model — keep {len(kept)} / drop {len(disagreed)} to adjudication")
            if not args.dry_run:
                import config as C
                from pathlib import Path as _P
                review_dir = C.BUILD / "gold_agreement" / args.domain
                review_dir.mkdir(parents=True, exist_ok=True)
                review_path = review_dir / f"{ed.slug.replace('/', '_')}.review.jsonl"
                review_path.write_text(
                    "\n".join(json.dumps(t, ensure_ascii=False) for t in disagreed) + ("\n" if disagreed else ""),
                    encoding="utf-8",
                )
                print(f"    adjudication queue → {review_path}")
            gold_triples = kept
            gold_agreement_stamp = "two_model"
        mp = ed.dir / "festival_meta.json"
        unit_name = (json.loads(mp.read_text()).get("name") if mp.exists() else None) or ed.slug
        # resolve's id_map is one of the two entity equivalences the multi-correct guard needs; read it
        # via the same (domain, slug) convention resolve_entities writes under.
        id_map = load_id_map(args.domain, ed.slug)
        items = build_probes(gold_triples, rel_q, unit_name, args.distractors,
                             pool_triples=all_triples, surfaces_per_gold=args.surfaces_per_gold,
                             id_map=id_map)
        for item in items:
            item["split"] = split_label
            if sp:
                item["split_version"] = sp["version"]
            if gold_agreement_stamp:
                item["gold_agreement"] = gold_agreement_stamp
            # Always stamp source id when available (eval_probes bind requires it on held_out).
            # build_probes already sets source_triple_id.
        if args.dry_run:
            print(f"  {ed.slug}: {len(items)} forced-choice probes over {len({i['predicate'] for i in items})} predicates")
            continue
        out_path = ed.dir / "probes.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
        os.replace(tmp, out_path)
        print(f"  {ed.slug}: wrote {len(items)} {split_label} probes from {len(gold_triples)} gold triples "
              f"({len(all_triples)} distractor-pool triples)"
              + (f" [gold_agreement={gold_agreement_stamp}]" if gold_agreement_stamp else ""))


if __name__ == "__main__":
    main()
