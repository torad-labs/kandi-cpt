"""Synthetic red-green: does the entity table move the pinned coverage ratio?

MECHANICS 2 — before trusting any number from this estimand, feed it a synthetic
perturbation that touches ONLY the entity table (zero new triples, byte-identical
fixture) and prove the ratio moves. If it moves, the fixture is not the ruler.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

import pipeline.audit_recall as A

FIXTURE = Path("data/_build/recall/pokemon-lore/kanto.assertions.json")
_orig_load_entities = A.load_entities


def run(label, extra_entities):
    def patched(ed_dir):
        return _orig_load_entities(ed_dir) + list(extra_entities)

    A.load_entities = patched
    try:
        rep = A.run_audit(
            "pokemon-lore", "kanto",
            facts_json=FIXTURE, seed=0, dry_run=True,
        )
    finally:
        A.load_entities = _orig_load_entities
    u = rep["units"][0]
    print(f"{label:38s} ratio={u['ratio']:.4f} hits={u['hits']} clean_n={u['n']} "
          f"raw_n={u['raw']['n']} out_of_unit={u['classes']['out_of_unit']} "
          f"judge_rejected={u['classes']['judge_rejected']} "
          f"absent={u['miss_attribution']['absent']}")
    return u


base = run("BASELINE (0 synthetic entities)", [])

ten = [
    {"id": f"pokemon_lore_kanto:region:synthzzz{i}", "entity_type": "region",
     "name_canonical": f"SynthZZZ{i}", "aliases": [], "attributes": {}}
    for i in range(10)
]
run("PLUS 10 unrelated entities", ten)

# Escalation: type every distinct miss OBJECT as an entity of an allowed type.
report = json.loads(Path("data/_build/recall/pokemon-lore/kanto.json").read_text())
miss_objects = sorted({m["object"] for d in report["docs"] for m in d["misses"]})
print(f"# distinct miss objects: {len(miss_objects)}")
typed = [
    {"id": f"pokemon_lore_kanto:item:synthobj{i}", "entity_type": "item",
     "name_canonical": obj, "aliases": [], "attributes": {}}
    for i, obj in enumerate(miss_objects)
]
run(f"PLUS {len(typed)} miss-objects typed 'item'", typed)

ten_objs = [
    {"id": f"pokemon_lore_kanto:item:synthten{i}", "entity_type": "item",
     "name_canonical": obj, "aliases": [], "attributes": {}}
    for i, obj in enumerate(miss_objects[:10])
]
run("PLUS 10 miss-objects typed 'item'", ten_objs)
