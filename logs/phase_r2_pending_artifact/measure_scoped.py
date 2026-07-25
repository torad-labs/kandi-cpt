#!/usr/bin/env python3
"""Prototype the `scoped` estimand (pred-valid AND subject-in-unit; object-type demoted
from denominator filter to reported class) and measure it against every synthetic."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import config as C
from pipeline.audit_recall import (
    load_entities, load_triples, load_id_map, load_predicates,
    load_predicate_contract, build_entity_index, coverage_for_doc,
    triple_keys, resolve_surface, fact_key, validate_judge_fact,
    subject_in_unit, match_fact, entity_type_of,
)

DOMAIN, SLUG = "pokemon-lore", "kanto"
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


def scoped_for_doc(assertable, gkeys, pred_set, contract, entity_index):
    """Prototype: denominator = predicate-valid AND subject-in-unit. object-type is a CLASS."""
    hits = misses = 0
    type_invalid_kept = 0
    rejected = out = 0
    for fact in assertable:
        s = str(fact.get("subject") or "")
        p = str(fact.get("predicate") or "")
        o = str(fact.get("object") or "")
        if pred_set and p not in pred_set:
            rejected += 1
            continue
        ok_pred, _ = validate_judge_fact(fact, predicates=pred_set or None,
                                         contract=None, entity_index=None)
        if not ok_pred:
            rejected += 1
            continue
        if not subject_in_unit(s, entity_index):
            out += 1
            continue
        ok_obj, _ = validate_judge_fact(fact, predicates=pred_set or None,
                                        contract=contract, entity_index=entity_index)
        if not ok_obj:
            type_invalid_kept += 1
        if match_fact(s, p, o, gkeys, entity_index) is not None:
            hits += 1
        else:
            misses += 1
    return {"hits": hits, "n": hits + misses, "type_invalid": type_invalid_kept,
            "judge_rejected": rejected, "out_of_unit": out}


def score_all(fixture, entities, triples, id_map, contract, predicates):
    ei = build_entity_index(entities, id_map)
    gkeys = build_gkeys(entities, triples, ei, id_map)
    pred_set = set(predicates)
    tot = {"clean_h": 0, "clean_n": 0, "raw_h": 0, "raw_n": 0,
           "sc_h": 0, "sc_n": 0, "sc_ti": 0, "sc_rej": 0, "sc_out": 0}
    misses = []
    for doc, ass in sorted(fixture.items()):
        cov = coverage_for_doc(ass, gkeys, predicates=predicates,
                               contract=contract, entity_index=ei)
        tot["clean_h"] += cov["hits"]; tot["clean_n"] += cov["n"]
        tot["raw_h"] += cov["raw"]["hits"]; tot["raw_n"] += cov["raw"]["n"]
        misses.extend(cov["misses"])
        sc = scoped_for_doc(ass, gkeys, pred_set, contract, ei)
        tot["sc_h"] += sc["hits"]; tot["sc_n"] += sc["n"]
        tot["sc_ti"] += sc["type_invalid"]; tot["sc_rej"] += sc["judge_rejected"]
        tot["sc_out"] += sc["out_of_unit"]
    tot["clean_r"] = round(tot["clean_h"] / tot["clean_n"], 4) if tot["clean_n"] else 1.0
    tot["raw_r"] = round(tot["raw_h"] / tot["raw_n"], 4) if tot["raw_n"] else 1.0
    tot["sc_r"] = round(tot["sc_h"] / tot["sc_n"], 4) if tot["sc_n"] else 1.0
    tot["_misses"] = misses
    return tot


def show(label, base, cur):
    print(f"{label:<46} clean {cur['clean_h']:>3}/{cur['clean_n']:<3}={cur['clean_r']:.4f} "
          f"({cur['clean_r']-base['clean_r']:+.4f})   "
          f"raw {cur['raw_h']:>3}/{cur['raw_n']:<3}={cur['raw_r']:.4f} "
          f"({cur['raw_r']-base['raw_r']:+.4f})   "
          f"SCOPED {cur['sc_h']:>3}/{cur['sc_n']:<3}={cur['sc_r']:.4f} "
          f"({cur['sc_r']-base['sc_r']:+.4f})")


def main():
    fixture = json.loads(FIXTURE.read_text())
    E = load_entities(ED_DIR); T = load_triples(ED_DIR)
    M = load_id_map(DOMAIN, SLUG)
    Ct = load_predicate_contract(DOMAIN); P = load_predicates(DOMAIN)

    base = score_all(fixture, E, T, M, Ct, P)
    print(f"classes: judge_rejected={base['sc_rej']} out_of_unit={base['sc_out']} "
          f"type_invalid(kept in scoped)={base['sc_ti']}")
    show("BASELINE", base, base)

    ei = build_entity_index(E, M)
    miss_objs, seen = [], set()
    for m in base["_misses"]:
        o = str(m.get("object") or "")
        if o and o.lower() not in seen:
            seen.add(o.lower()); miss_objs.append(o)

    # (a) 55 shadow entities, type outside every contract, 0 triples
    shadow = [{"id": f"kanto:zzz_shadow:s{i}", "name_canonical": o,
               "entity_type": "zzz_shadow", "aliases": []} for i, o in enumerate(miss_objs)]
    show("(a) +55 shadow ents / 0 triples", base, score_all(fixture, E + shadow, T, M, Ct, P))

    # (a2) shadow entities named after the out_of_unit SUBJECTS (worst case for scoping)
    out_subj, seen2 = [], set()
    for doc, ass in sorted(fixture.items()):
        for f in ass:
            s = str(f.get("subject") or "")
            p = str(f.get("predicate") or "")
            if p in set(P) and s and not subject_in_unit(s, ei) and s.lower() not in seen2:
                seen2.add(s.lower()); out_subj.append(s)
    print(f"    distinct out_of_unit subjects = {len(out_subj)}")
    shadow2 = [{"id": f"kanto:zzz_shadow:o{i}", "name_canonical": s,
                "entity_type": "zzz_shadow", "aliases": []} for i, s in enumerate(out_subj)]
    show("(a2) +%d out-of-unit SUBJECT ents / 0 triples" % len(shadow2), base,
         score_all(fixture, E + shadow2, T, M, Ct, P))

    # (a3) both
    show("(a3) a + a2 combined / 0 triples", base,
         score_all(fixture, E + shadow + shadow2, T, M, Ct, P))

    # (a4) entity DELETION: drop every entity whose canonical name is a miss object
    keep = [e for e in E if str(e.get("name_canonical") or "").strip().lower() not in seen]
    print(f"    deletion variant drops {len(E)-len(keep)} entities")
    show("(a4) DELETE %d entities / 0 triples" % (len(E) - len(keep)), base,
         score_all(fixture, keep, T, M, Ct, P))

    # (b) contract widened to literal
    show("(b) contract -> ['literal']", base,
         score_all(fixture, E, T, M, {k: ["literal"] for k in Ct}, P))
    # (b2) contract narrowed to a bogus type
    show("(b2) contract -> ['nonesuch']", base,
         score_all(fixture, E, T, M, {k: ["nonesuch"] for k in Ct}, P))

    # (c) +1 genuine triple, both sides already resolve
    cand = None
    for m in base["_misses"]:
        if resolve_surface(m.get("subject"), ei) and resolve_surface(m.get("object"), ei):
            cand = m; break
    def id_for(sfc):
        e = ei["surface_to_entity"].get(str(sfc).strip().lower())
        return str(e.get("id")) if e else None
    tc = {"subject_id": id_for(cand["subject"]), "predicate": cand["predicate"],
          "object": id_for(cand["object"]) or cand["object"]}
    print(f"    (c) triple = {tc}")
    show("(c) +1 genuine triple (both sides known)", base,
         score_all(fixture, E, T + [tc], M, Ct, P))

    # (d) +1 entity +1 triple — the object_type_unresolved rescue
    blind = None
    for doc, ass in sorted(fixture.items()):
        for f in ass:
            p = str(f.get("predicate") or "")
            if p not in Ct: continue
            al = Ct[p]
            if not al or "literal" in al: continue
            if entity_type_of(str(f.get("object") or ""), ei) is None and subject_in_unit(str(f.get("subject") or ""), ei):
                blind = f; break
        if blind: break
    print(f"    (d) blind assertion = {blind}")
    al = Ct[str(blind["predicate"])]
    ne = {"id": f"kanto:{al[0]}:rescued", "name_canonical": blind["object"],
          "entity_type": al[0], "aliases": []}
    td = {"subject_id": id_for(blind["subject"]), "predicate": blind["predicate"],
          "object": ne["id"]}
    show("(d) +1 entity +1 triple (rescue unresolved)", base,
         score_all(fixture, E + [ne], T + [td], M, Ct, P))

    # (d2) rescue an out_of_unit assertion (subject scope grows) — the honest limit case
    oos = None
    for doc, ass in sorted(fixture.items()):
        for f in ass:
            if str(f.get("predicate") or "") in set(P) and not subject_in_unit(str(f.get("subject") or ""), ei):
                oos = f; break
        if oos: break
    print(f"    (d2) out_of_unit assertion = {oos}")
    ns = {"id": "kanto:character:oos_subj", "name_canonical": oos["subject"],
          "entity_type": "character", "aliases": []}
    no = {"id": "kanto:character:oos_obj", "name_canonical": oos["object"],
          "entity_type": (Ct.get(oos["predicate"]) or ["character"])[0], "aliases": []}
    t2 = {"subject_id": ns["id"], "predicate": oos["predicate"], "object": no["id"]}
    show("(d2) rescue an out_of_unit assertion", base,
         score_all(fixture, E + [ns, no], T + [t2], M, Ct, P))


if __name__ == "__main__":
    main()
