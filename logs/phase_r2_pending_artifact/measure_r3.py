#!/usr/bin/env python3
"""r3 measurement harness — reproduce the audit's baseline and synthetics in-process.

No files written into the repo.
"""
from __future__ import annotations
import json, sys, copy
from pathlib import Path

import config as C
from pipeline.audit_recall import (
    load_entities, load_triples, load_id_map, load_predicates,
    load_predicate_contract, build_entity_index, coverage_for_doc,
    triple_keys, resolve_surface, fact_key, collect_clean_docs, derive_sample_n,
)

DOMAIN = "pokemon-lore"
SLUG = "kanto"
ED_DIR = C.ROOT / "data" / DOMAIN / SLUG
FIXTURE = C.BUILD / "recall" / DOMAIN / "kanto.assertions.json"


def build_gkeys(entities, triples, entity_index, id_map):
    id_to_name = {}
    for e in entities:
        canon = str(e.get("name_canonical") or e.get("id") or "")
        eid = str(e.get("id") or "")
        if eid:
            id_to_name[eid] = canon
        for alias in e.get("aliases") or []:
            id_to_name[str(alias)] = canon
    for old_id, new_id in id_map.items():
        if new_id in id_to_name:
            id_to_name[old_id] = id_to_name[new_id]
        elif new_id:
            id_to_name.setdefault(old_id, id_to_name.get(new_id, new_id))
    gkeys = triple_keys(triples, id_to_name)
    for t in triples:
        s_id = str(t.get("subject_id") or "")
        o_raw = str(t.get("object") or "")
        p = str(t.get("predicate") or "")
        s_name = resolve_surface(s_id, entity_index) or id_to_name.get(s_id, s_id)
        o_name = resolve_surface(o_raw, entity_index) or id_to_name.get(o_raw, o_raw)
        gkeys.add(fact_key(s_name, p, o_name))
    return gkeys


def score(fixture, entities, triples, id_map, contract, predicates):
    entity_index = build_entity_index(entities, id_map)
    gkeys = build_gkeys(entities, triples, entity_index, id_map)
    hits = n = rh = rn = 0
    rej = {}
    misses = []
    for doc, assertable in sorted(fixture.items()):
        cov = coverage_for_doc(assertable, gkeys, predicates=predicates,
                               contract=contract, entity_index=entity_index)
        hits += cov["hits"]; n += cov["n"]
        rh += cov["raw"]["hits"]; rn += cov["raw"]["n"]
        misses.extend(cov["misses"])
        for r in cov["judge_rejected"]:
            reason = str(r.get("reject_reason") or "").split(":")[0]
            rej[reason] = rej.get(reason, 0) + 1
    return {
        "clean_hits": hits, "clean_n": n,
        "clean_ratio": round(hits / n, 4) if n else 1.0,
        "raw_hits": rh, "raw_n": rn,
        "raw_ratio": round(rh / rn, 4) if rn else 1.0,
        "reject_reasons": rej,
        "misses": misses,
    }


def main():
    fixture = json.loads(FIXTURE.read_text())
    entities = load_entities(ED_DIR)
    triples = load_triples(ED_DIR)
    id_map = load_id_map(DOMAIN, SLUG)
    contract = load_predicate_contract(DOMAIN)
    predicates = load_predicates(DOMAIN)

    print(f"fixture docs={len(fixture)} assertions={sum(len(v) for v in fixture.values())}")
    print(f"entities={len(entities)} triples={len(triples)} id_map={len(id_map)} preds={len(predicates)}")
    base = score(fixture, entities, triples, id_map, contract, predicates)
    print("BASELINE:", json.dumps({k: v for k, v in base.items() if k != "misses"}))

    # --- synthetic A: addition-only shadow entities, 0 new triples ---
    miss_objects = []
    seen = set()
    for m in base["misses"]:
        o = str(m.get("object") or "")
        if o and o.lower() not in seen:
            seen.add(o.lower()); miss_objects.append(o)
    print(f"distinct miss objects={len(miss_objects)}")
    # choose a shadow type that is outside every predicate's allowed set
    all_types = set()
    for v in contract.values():
        all_types |= set(v)
    shadow_type = "gadget_widget_zzz"
    print("shadow type not in any contract:", shadow_type not in all_types)
    shadow = [
        {"id": f"kanto:{shadow_type}:shadow_{i}", "name_canonical": o,
         "entity_type": shadow_type, "aliases": []}
        for i, o in enumerate(miss_objects)
    ]
    synth_a = score(fixture, entities + shadow, triples, id_map, contract, predicates)
    print("SYNTH_A (+%d shadow entities, 0 triples):" % len(shadow),
          json.dumps({k: v for k, v in synth_a.items() if k != "misses"}))
    print("  delta clean = %+.4f   delta raw = %+.4f" % (
        synth_a["clean_ratio"] - base["clean_ratio"],
        synth_a["raw_ratio"] - base["raw_ratio"]))

    # --- synthetic B: contract widened to ['literal'] ---
    wide = {k: ["literal"] for k in contract}
    synth_b = score(fixture, entities, triples, id_map, wide, predicates)
    print("SYNTH_B (contract -> literal):",
          json.dumps({k: v for k, v in synth_b.items() if k != "misses"}))
    print("  delta clean = %+.4f   delta raw = %+.4f" % (
        synth_b["clean_ratio"] - base["clean_ratio"],
        synth_b["raw_ratio"] - base["raw_ratio"]))

    # --- synthetic C: +1 genuine triple that IS in the fixture as a clean miss ---
    # pick a clean miss whose subject/object already resolve (so it is a real recovery)
    ei = build_entity_index(entities, id_map)
    cand = None
    for m in base["misses"]:
        s, p, o = m.get("subject"), m.get("predicate"), m.get("object")
        if resolve_surface(s, ei) and resolve_surface(o, ei):
            cand = m; break
    print("clean-miss candidate (both sides resolve):", cand)
    if cand:
        # find ids
        def id_for(surface):
            ent = ei["surface_to_entity"].get(surface.strip().lower())
            return str(ent.get("id")) if ent else None
        t = {"subject_id": id_for(cand["subject"]), "predicate": cand["predicate"],
             "object": id_for(cand["object"]) or cand["object"]}
        synth_c = score(fixture, entities, triples + [t], id_map, contract, predicates)
        print("SYNTH_C (+1 genuine triple):",
              json.dumps({k: v for k, v in synth_c.items() if k != "misses"}))
        print("  delta clean = %+.4f   delta raw = %+.4f" % (
            synth_c["clean_ratio"] - base["clean_ratio"],
            synth_c["raw_ratio"] - base["raw_ratio"]))

    # --- synthetic D: +1 entity +1 triple (real re-extraction shape) ---
    # pick a judge_rejected assertion with object_type_unresolved
    ei2 = build_entity_index(entities, id_map)
    blind = None
    for doc, facts in sorted(fixture.items()):
        for f in facts:
            p = str(f.get("predicate") or "")
            if p not in contract:
                continue
            allowed = contract[p]
            if not allowed or "literal" in allowed:
                continue
            from pipeline.audit_recall import entity_type_of
            if entity_type_of(str(f.get("object") or ""), ei2) is None and resolve_surface(str(f.get("subject") or ""), ei2):
                blind = f; break
        if blind: break
    print("blind-spot candidate (object unresolved):", blind)
    if blind:
        allowed = contract[str(blind["predicate"])]
        new_ent = {"id": f"kanto:{allowed[0]}:new_obj", "name_canonical": blind["object"],
                   "entity_type": allowed[0], "aliases": []}
        def id_for2(surface, idx):
            ent = idx["surface_to_entity"].get(surface.strip().lower())
            return str(ent.get("id")) if ent else None
        t = {"subject_id": id_for2(blind["subject"], ei2), "predicate": blind["predicate"],
             "object": new_ent["id"]}
        synth_d = score(fixture, entities + [new_ent], triples + [t], id_map, contract, predicates)
        print("SYNTH_D (+1 entity +1 triple):",
              json.dumps({k: v for k, v in synth_d.items() if k != "misses"}))
        print("  delta clean = %+.4f   delta raw = %+.4f" % (
            synth_d["clean_ratio"] - base["clean_ratio"],
            synth_d["raw_ratio"] - base["raw_ratio"]))

    # --- sample sensitivity ---
    import random
    docs = collect_clean_docs(ED_DIR)
    print(f"clean docs={len(docs)} derive_sample_n={derive_sample_n(len(docs))}")
    n_take = min(derive_sample_n(len(docs)), len(docs))
    sample = random.Random(0).sample(docs, n_take)
    names = {Path(d["path"]).name for d in sample}
    print("sample∩fixture =", len(names & set(fixture.keys())), "/", n_take)
    # insert a synthetic doc at sorted position 0
    fake = {"path": str(ED_DIR / "10_clean" / "0000_aaa.md"), "text": "x", "name": "0000_aaa.md"}
    docs2 = sorted(docs + [fake], key=lambda d: d["path"])
    s2 = random.Random(0).sample(docs2, min(derive_sample_n(len(docs2)), len(docs2)))
    n2 = {Path(d["path"]).name for d in s2}
    print("with 1 doc inserted at pos 0: sample∩fixture =", len(n2 & set(fixture.keys())), "/", len(s2))


if __name__ == "__main__":
    main()
