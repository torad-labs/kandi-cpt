#!/usr/bin/env python3
"""Independent confirmation of the audit's measured pools for kanto TRAIN."""
from __future__ import annotations
import itertools
from collections import defaultdict

from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples
from pipeline.split import load_split, triple_id

eds = list(iter_editions("pokemon-lore", with_raw=False))
print("editions:", [e.slug for e in eds])
ed = eds[0]
rows = load_all_triples(ed)
print("rows:", len(rows))
uniq = {}
for t in rows:
    uniq.setdefault(triple_id(t), t)
print("unique triple_ids:", len(uniq))
sp = load_split(ed.dir)
print("split version:", sp["version"], "n_triples in manifest:", sp["n_triples"],
      "train_ids:", len(sp["train_ids"]), "holdout_ids:", len(sp["holdout_ids"]))

train = {tid: t for tid, t in uniq.items() if tid in sp["train_ids"]}
hold = {tid: t for tid, t in uniq.items() if tid in sp["holdout_ids"]}
neither = {tid: t for tid, t in uniq.items()
           if tid not in sp["train_ids"] and tid not in sp["holdout_ids"]}
print(f"train {len(train)} / holdout {len(hold)} / NEITHER {len(neither)}")

# entity nodes touched by train triples: subject_id + object_id
nodes = set()
for t in train.values():
    nodes.add(str(t["subject_id"]))
    nodes.add(str(t["object_id"]))
print("entity nodes over train triples:", len(nodes))

items = sorted(train.items())
tids = [k for k, _ in items]
ts = [v for _, v in items]
n = len(ts)

import yaml
tm = yaml.safe_load(open("configs/templates/pokemon-lore.yaml"))
relq = (tm.get("sft_templates") or {}).get("rel_q") or {}
print("rel_q predicates:", len(relq))

def sid(t): return str(t["subject_id"])
def oid(t): return str(t["object_id"])

chain = 0
chain_relq = 0
shared_subj = 0
shared_obj = 0
connected2 = 0
pairs_connected = []
for i, j in itertools.combinations(range(n), 2):
    a, b = ts[i], ts[j]
    ea = {sid(a), oid(a)}
    eb = {sid(b), oid(b)}
    if not (ea & eb):
        continue
    connected2 += 1
    pairs_connected.append((i, j))
    is_chain = (oid(a) == sid(b)) or (oid(b) == sid(a))
    if is_chain:
        chain += 1
        if a["predicate"] in relq and b["predicate"] in relq:
            chain_relq += 1
    if sid(a) == sid(b):
        shared_subj += 1
    if oid(a) == oid(b):
        shared_obj += 1

print(f"POOL A (chain + both rel_q): {chain_relq}")
print(f"POOL B (directed 2-hop chain): {chain}")
print(f"POOL C (B + shared-subject): approx {chain + shared_subj}")
print(f"POOL D (all connected 2-triple): {connected2}")

# 3-triple connected sets: connected as a graph over triples (edge = share an entity)
adj = defaultdict(set)
for i, j in pairs_connected:
    adj[i].add(j)
    adj[j].add(i)
tri = 0
for i, j in pairs_connected:
    # third triple k connected to i or j, k != i,j, dedupe by requiring k > max? no —
    # count sets {i,j,k} once: enumerate sorted triples
    pass
seen3 = set()
for i in range(n):
    for j in adj[i]:
        for k in (adj[i] | adj[j]):
            if k in (i, j):
                continue
            key = tuple(sorted((i, j, k)))
            if key in seen3:
                continue
            seen3.add(key)
tri = len(seen3)
print(f"POOL E (connected 3-triple sets): {tri}")
print(f"POOL F (2+3): {connected2 + tri}")
if chain:
    print(f"RATIO narrow:widest = {chain} : {connected2+tri} = {(connected2+tri)/chain:.1f}x")

cov_chain = set()
for i, j in pairs_connected:
    a, b = ts[i], ts[j]
    if (oid(a) == sid(b)) or (oid(b) == sid(a)):
        cov_chain.add(tids[i]); cov_chain.add(tids[j])
cov_all = set()
for i, j in pairs_connected:
    cov_all.add(tids[i]); cov_all.add(tids[j])
print(f"graph coverage — chain pool: {len(cov_chain)}/{len(train)} = {len(cov_chain)/len(train):.3f}")
print(f"graph coverage — connected pool: {len(cov_all)}/{len(train)} = {len(cov_all)/len(train):.3f}")

# budget base
from pathlib import Path
hyd = ed.dir / "25_hydrated"
tot = 0
for p in sorted(hyd.glob("*.jsonl")):
    c = sum(1 for L in p.read_text(encoding="utf-8", errors="replace").splitlines() if L.strip())
    print(f"  {p.name}: {c}")
    tot += c
print("TOTAL hydrated lines:", tot)
