#!/usr/bin/env python3
"""Assemble V6 (distributional-hypothesis approach — see pipeline/disjoint_context.py).

V4/V5 proved the failure is confusable facts sharing a representation basin, not undertraining. The fix
is context engineering: confusable entities must appear in DISJOINT contexts (pushes reps apart);
related facts share context (binds). The hard rule: NEVER co-mention a confusable pair — co-mention is
shared context and pulls them together (the v3 failure).

V6 = V4 base (de-contaminated, confusable CO-MENTIONS stripped) + heavy disjoint-context over-exposure
for each confusable entity in its own vocabulary world. Keep the density floor, DROP the equalizer
(v5 cost ~0.20 stage install). Standard CE (V7 unlikelihood held in reserve).

    PYTHONPATH=scripts python scripts/build_v6_corpus.py [--confusable-target 150] [--dry-run]

V4 base is read from relations.jsonl.pre-equalize.bak (preserved); V6 → relations.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

import config as C
from pipeline import disjoint_context as D
from pipeline.set_times_lines import set_time_lines

ED = C.ROOT / "data/edc/las_vegas/2026"
BASE = ED / "25_hydrated/relations.jsonl.pre-equalize.bak"     # V4, un-equalized
OUT = ED / "25_hydrated/relations.jsonl"                       # V6 target

REFUSAL = re.compile("|".join(map(re.escape, [
    "cannot fulfill", "i cannot", "without inventing", "the provided facts", "given facts",
    "as an ai", "i am unable", "i'm unable", "does not specify", "do not specify", "not specified in",
    "strict faithfulness", "i can only", "cannot generate", "no information", "unable to comply"])), re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusable-target", type=int, default=150,
                    help="disjoint-context lines per confusable group (over the floor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) load V4 base; strip refusal/meta contamination
    base, dropped_refusal = [], 0
    for l in open(BASE):
        if not l.strip():
            continue
        r = json.loads(l)
        if REFUSAL.search(r.get("text", "")):
            dropped_refusal += 1
            continue
        r.setdefault("context_profile", "shared_v4")          # tag the existing pool
        base.append(r)

    # 2) STRIP confusable CO-MENTION violations from the V4 base (the v3 guardrail). These are the light
    #    contrastives ("Armin plays Sunday — not Saturday", "…— not cosmicMEADOW") that co-locate the
    #    confusable tokens. Their job is replaced by disjoint context below.
    kept, stripped = [], []
    for r in base:
        why = D.violates(r.get("text", ""))
        (stripped if why else kept).append((r, why))
    base = [r for r, _ in kept]

    # 3) disjoint-context over-exposure for each confusable group, in its own vocabulary world
    disjoint = []
    per_group = {}
    for g in D.GROUPS:
        variants = D.generate_for_group(g)                    # distinct lines, this group's world only
        target = args.confusable_target
        lines = [dict(variants[i % len(variants)]) for i in range(max(len(variants), target))]
        disjoint.extend(lines)
        per_group[g["key"]] = len(lines)

    # 4) report + GATES -----------------------------------------------------------------------------
    pos_per_entity = collections.Counter()
    for r in base:
        if r.get("relation_type") == "artist_to_stage":
            e = r.get("entity_ids") or []
            if e:
                pos_per_entity[e[0]] += 1

    print(f"v4 base         : dropped {dropped_refusal} refusal/meta")
    print(f"co-mention strip: removed {len(stripped)} confusable co-mention lines (v3 guardrail)")
    if stripped:
        for r, why in stripped[:5]:
            print(f"    - [{why}] {r['text'][:90]}")
    print(f"disjoint context: +{len(disjoint)} lines across {len(per_group)} groups {per_group}")

    # 3b) authoritative set-time lines (#47) — adds the time dimension + reinforces day/stage/sunrise
    settimes = set_time_lines(str(ED / "set_times.jsonl"), str(ED / "15_structured/entities.jsonl")) \
        if (ED / "set_times.jsonl").exists() else []
    print(f"set-time lines  : +{len(settimes)} (from set_times.jsonl)")

    full = base + disjoint + settimes

    # GATE 1 — CO-OCCURRENCE AUDIT (the critical one): must be ZERO after assembly
    viol = D.audit_cooccurrence(full)
    print(f"\nGATE 1 co-occurrence audit: {len(viol)} violation(s) " + ("✓ CLEAN" if not viol else "✗ FAIL"))
    for v in viol[:8]:
        print(f"    ✗ [{v['_why']}] {v['text'][:90]}")

    # GATE 2 — floor: every confusable entity meets the floor (min positive density among all facts)
    floor = min(pos_per_entity.values()) if pos_per_entity else 0
    conf_counts = {}
    for e in D.CONFUSABLE:
        n = pos_per_entity.get(e, 0) + sum(1 for r in disjoint if e in (r.get("entity_ids") or []))
        conf_counts[e] = n
    below = {e: n for e, n in conf_counts.items() if n < floor}
    print(f"GATE 2 floor (≥{floor}/entity): " + ("✓ all confusable entities meet floor" if not below else f"✗ below: {below}"))

    # GATE 3 — hard-negative boost: confusable entities EXCEED the floor
    boosted = all(n > floor for n in conf_counts.values())
    print(f"GATE 3 boost (> floor): " + ("✓ confusable clusters over-exposed" if boosted else f"✗ not all boosted: {conf_counts}"))

    # GATE 4 — context-disjointness: neighbor-token overlap between groups (lower = better)
    overlap = D.neighbor_overlap()
    worst = max(overlap.values()) if overlap else 0
    print(f"GATE 4 context-disjointness (group vocab Jaccard, want low): worst={worst:.3f}")
    for k, v in sorted(overlap.items(), key=lambda x: -x[1])[:4]:
        print(f"    {k}: {v:.3f}")

    # GATE 5 — dedup / distinct ratio per relation
    by_rel = collections.defaultdict(list)
    for r in full:
        by_rel[r.get("relation_type", "?")].append(r["text"])
    print("GATE 5 distinct ratio per relation:")
    for rel, texts in sorted(by_rel.items(), key=lambda x: -len(x[1]))[:8]:
        print(f"    {rel:20} n={len(texts):5} distinct={len(set(texts)):5} ratio={len(set(texts))/len(texts):.2f}")

    print(f"\nV6 total: {len(full)} lines  (base {len(base)} + disjoint {len(disjoint)} + set-time {len(settimes)})  | "
          f"confusable per-entity: {conf_counts}")

    if args.dry_run:
        print("\n-- sample disjoint-context lines (each entity in its own world) --")
        seen = set()
        for r in disjoint:
            if r["context_profile"] not in seen:
                seen.add(r["context_profile"])
                print(f"    [{r['context_profile']}] {r['text']}")
        print("\n[dry-run] nothing written.")
        return

    if viol:
        print("\n✗ REFUSING to write — co-occurrence audit must be clean first.")
        return
    rng.shuffle(full)
    with open(OUT, "w") as f:
        for r in full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT}  ({len(full)} lines)")


if __name__ == "__main__":
    main()
