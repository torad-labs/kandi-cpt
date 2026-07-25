#!/usr/bin/env python3
"""Reasoning-linked surface class — connected TRAIN triple sets, reformulated by the LLM CHANNEL.

The hydrate FLOOR (`generic_expand`) renders one triple per line and `generic_synth` paraphrases one
triple per line, so nothing in the corpus states two facts in a way that makes their COMPOSITION
available. This stage adds that channel: enumerate connected sets of 2–3 TRAIN triples
deterministically, then ask the generator model for reformulations that pose and answer how the two
endpoints are related — the "multi-hop path sentence" lever `docs/cpt_installation_best_practices.md`
names, and the model-generated reformulation EntiGraph/GraphGen actually describe.

WHY THE LLM CHANNEL AND NOT TEMPLATES (binding prior art: commit f3cc97ad, the P3 revert, and
logs/phase_r6_llm_channel_audit.json). A first version of this class rendered three fixed f-string
question skeletons. Measured on kanto with `diversity_lib.effective_forms` (jaccard 0.55): 906
records collapsed to 19 forms, 869 of them ONE cluster, against 571±22 for an equal-mass sample of
`relations_llm.jsonl` — the same corpus, the same budget, rendered through this same generator seam.
The deterministic per-triple FLOOR scores 356 on the same ruler, so the template renderer was an
order of magnitude below even the floor it was meant to extend. f3cc97ad's own conclusion: "a prefix
or a bold tag is not model-generated reformulation. I spent three attempts improving the channel that
matters least." Enumeration is deterministic; RENDERING is the model's job.

    NAME THE RULER. This repo's two diversity measures disagree — `diversity_lib.distinct_n` (the
    audit gate's ruler) and `diversity_lib.effective_forms` (dedup's floor accounting) score the same
    pair differently, and effective_forms is lexical jaccard, so it over-credits a mere reversal
    (r6 measured "Giovanni leads Team Rocket." vs "Team Rocket is led by Giovanni." at 0.43, i.e.
    two forms). Every diversity claim here is against effective_forms(0.55) and is a FLOOR against
    collapse, never a proof of semantic variety.

GROUNDING IS ENFORCED HERE. `scripts/qgre/gates.py` reads only 25_hydrated/relations.jsonl
(faithfulness/balance/exposure/source-vs-gap) and relations_llm + relations_contrast (diversity), so
this file is invisible to EVERY corpus gate. The model may only reformulate the STATED edges: a
generated line that names an entity outside the walk, introduces a numeral the walk does not license,
drops an endpoint, negates, or repeats an earlier line is DROPPED with a counted reason. Zero-drop is
never trusted on faith — every reason is red-green calibrated in test_generic_reasoning.py.

ENTITY IDENTITY IS RESOLVED, NOT RAW. `load_all_triples` reads the UNRESOLVED 15_structured, where
kanto carries 42 display names under two node ids and `resolve` recorded 88 alias→canonical entries.
Keying the walk graph on the raw id fabricates a walk whose two endpoints are the SAME entity under
two ids ("What links Pikachu to Pikachu?") and fragments the graph. `entity_classes` union-finds over
(same display name) OR (resolve-merged via id_map) and keys on the class minimum — the same two
equivalences and the same representative rule as `generic_probes.entity_classes`, generalised from
subjects to BOTH endpoints because a walk graph is keyed on both. Measured on kanto: pool
10,904 → 15,440 (+42%), coverage 499 → 501 of 521, alias self-connections 2 → 0.

ESTIMAND — TRAIN-ONLY, TWO LEVELS. (1) id level: every constituent triple must satisfy
`triple_id(t) in split["train_ids"]`, positive membership on EVERY triple, never `not in
holdout_ids` (140 of kanto's 736 unique triples are in NEITHER side because 15_structured was
rewritten after the immutable split was cut). (2) FACT level: a walk whose two endpoints are exactly
a HOLDOUT triple's endpoint pair teaches that held-out fact as a connection, and the id check cannot
see it — 76 such walks exist in kanto's pool. They are dropped, and the drop costs zero coverage
(501/521 with and without). Every number produced by this stage is a RUN-GATE corpus-construction
number on the TRAIN side; nothing here estimates install strength.

    PYTHONPATH=scripts python scripts/pipeline/generic_reasoning.py --domain <d>
        [--editions unit,…] [--max-fraction 0.10] [--variations 2] [--max-walk 3] [--dry-run]
Output: <edition>/25_hydrated/relations_reasoning.jsonl
`--dry-run` plans and reports the LLM call count a real run WOULD make, making ZERO provider calls.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import config as C
from pipeline import diversity_lib as Div
from pipeline.audit_recall import load_id_map
from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples, source_doc_ids
from pipeline.generic_synth import (
    DEFAULT_PERSONAS,
    DEFAULT_TEMPERATURES,
    _persona_for,
    _temperature_for,
    openrouter_generate,
)
from pipeline.schema_validate import validate
from pipeline.split import load_split, triple_id

OUT_NAME = "relations_reasoning.jsonl"
SURFACE_FORM = "reasoning_walk"

# The reformulation instruction. It states the walk's own edges and NOTHING else; the domain framing
# is the `synth_templates.system_preamble` prepended to it, exactly as generic_synth does.
WALK_PROMPT = (
    "Here are {k} verified facts. Read together they link {x} to {y}{via}:\n\n{facts}\n\n"
    "Write {n} DIVERSE ways to pose AND answer the question of how {x} and {y} are related, using "
    "ONLY these facts. Vary the form — a question then its answer, a short explanation, a reader's "
    "query answered in prose, a quiz item, a step-by-step trace. Every variation must name both {x} "
    "and {y} and must follow the chain exactly as stated above. Assert NOTHING beyond these facts: "
    "introduce no other people, places, groups, organisations, works, dates or numbers, and deny "
    "nothing. Output ONE self-contained variation per line (question and answer on the same line), "
    "no numbering, no preamble."
)

MIN_LINE_CHARS = 20
_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_DIGITS_RE = re.compile(r"\d+")
# A reformulation of positive stated edges has no reason to deny anything; a denial is either
# redundant or a fabricated negative, and this corpus has a dedicated contrast class for negatives.
_NEGATION_RE = re.compile(
    r"(?<!\w)(?:not|never|no longer|nor|neither|unlike|false|incorrect|untrue|isn't|aren't|wasn't|"
    r"weren't|doesn't|don't|didn't|cannot|can't)(?!\w)", re.I)

DROP_REASONS = (
    "unauthorable",        # a constituent edge has no fillable declarative template
    "not_a_path",          # the triple set is not a simple path with two nameable endpoints
    "degenerate",          # two path nodes render the SAME display name (alias self-connection)
    "unlabelable",         # the head predicate is not admissible as this domain's relation_type
    "holdout_bridge",      # the endpoint pair IS a holdout triple's endpoint pair (fact-level leak)
    "empty_completion",    # the generator returned nothing for this walk
    "too_short",           # a returned line is not a surface
    "ungrounded",          # a returned line names an entity the walk does not license
    "foreign_numeral",     # a returned line carries a number the walk does not license
    "missing_endpoint",    # a returned line does not name both endpoints
    "negated",             # a returned line denies something
    "duplicate",           # a returned line repeats one already emitted for this unit
)


# ---------------------------------------------------------------------------- domain surface


def rel_q_for(domain: str) -> dict:
    """The domain's sft_templates.rel_q (predicate → [question, declarative]) — the block
    `generic_probes` reads. Index 1 is the declarative; only declaratives are stated to the model."""
    return ((Domain(domain).templates or {}).get("sft_templates") or {}).get("rel_q") or {}


def preamble_for(domain: str) -> str:
    """The domain's synth framing — reused verbatim from the proven LLM stage."""
    return ((Domain(domain).templates or {}).get("synth_templates") or {}).get("system_preamble", "") or ""


def admitted_relation_types(domain: str | None, predicates) -> set[str]:
    """Which raw predicates this domain's relation_type enum accepts.

    `relation_type: "reasoning_walk"` is REJECTED — `schema_validate._patch` pins the enum to the
    domain manifest, so this class borrows one constituent predicate (contrast's precedent,
    generic_contrast.py:204). A domain declaring a `relations.taxonomy` pins the enum to SEMANTIC
    labels instead and EDC's raw predicates are then inadmissible: that is a class-UNAVAILABLE state,
    reported and skipped BEFORE enumeration — never a crash, and never a record under a wrong label.
    """
    if not domain:
        return {str(p) for p in predicates}
    ok: set[str] = set()
    for predicate in sorted({str(p) for p in predicates}):
        probe = {"text": "probe", "relation_type": predicate, "entity_ids": ["probe"],
                 "source_triple_ids": ["probe"], "split": "cpt"}
        if not validate("relation", probe, domain=domain):
            ok.add(predicate)
    return ok


def declarative(t: dict, rel_q: dict, unit_name: str) -> str | None:
    """One edge as its domain-authored declarative, or None when unauthorable.

    Fail-closed on a template this driver cannot fill (EDC's rel_q carries `{n}`; a domain may add
    more). A missing render is a counted drop, never a partially-formatted sentence.
    """
    forms = rel_q.get(str(t.get("predicate", "")))
    if not forms or len(forms) < 2 or not isinstance(forms[1], str):
        return None
    kw = {"s": str(t.get("subject", "")), "o": str(t.get("object", "")), "n": unit_name}
    kw.update({f"attr_{k}": v for k, v in (t.get("attrs") or {}).items()})
    try:
        text = forms[1].format(**kw).strip()
    except (KeyError, IndexError, ValueError):
        return None
    return text or None


# ---------------------------------------------------------------------------- entity identity


def endpoint_ids(t: dict) -> tuple[str, str]:
    """The raw ids a triple records its two endpoints under (display name only when there is no id)."""
    return (str(t.get("subject_id", "") or t.get("subject", "")),
            str(t.get("object_id", "") or t.get("object", "")))


def entity_classes(triples: list[dict], id_map: dict[str, str] | None = None) -> dict[str, str]:
    """Map every endpoint id to a stable representative of the ENTITY it denotes.

    Two ids denote one entity when either independent evidence says so: they render the same display
    name (a walk keyed on the raw ids would then "connect" an entity to itself through its own
    alias), or `resolve` merged them (surface variants of one entity). Neither test subsumes the
    other: merged ids can render different names, and same-name ids are not always merged.

    Same union-find, same two equivalences and the same class-minimum representative as
    `generic_probes.entity_classes` — generalised from `subject_key` alone to BOTH endpoints, because
    a walk graph is keyed on both. Representative is the class minimum so walks stay deterministic.
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
    for t in triples:
        subject_id, object_id = endpoint_ids(t)
        for idk, name in ((subject_id, str(t.get("subject", "") or "")),
                          (object_id, str(t.get("object", "") or ""))):
            if not idk:
                continue
            find(idk)
            canonical = id_map.get(idk)
            if canonical:
                union(idk, canonical)
            if name:
                if name in first_id_for_name:
                    union(idk, first_id_for_name[name])
                else:
                    first_id_for_name[name] = idk
    return {idk: find(idk) for idk in list(parent)}


def class_names(triples: list[dict], classes: dict[str, str]) -> dict[str, str]:
    """One canonical display name per entity class — the most frequent, ties broken lexicographically.

    A class merged by `id_map` can carry two spellings; rendering must pick one deterministically or
    the question and the stated edge disagree about what the entity is called.
    """
    seen: dict[str, Counter] = defaultdict(Counter)
    for t in triples:
        subject_id, object_id = endpoint_ids(t)
        for idk, name in ((subject_id, str(t.get("subject", "") or "")),
                          (object_id, str(t.get("object", "") or ""))):
            if idk and name:
                seen[classes.get(idk, idk)][name] += 1
    return {cls: min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0] for cls, counter in seen.items()}


def class_aliases(triples: list[dict], classes: dict[str, str]) -> dict[str, set[str]]:
    """EVERY display name a class is written under — the licensed vocabulary for its walks."""
    out: dict[str, set[str]] = defaultdict(set)
    for t in triples:
        subject_id, object_id = endpoint_ids(t)
        for idk, name in ((subject_id, str(t.get("subject", "") or "")),
                          (object_id, str(t.get("object", "") or ""))):
            if idk and name:
                out[classes.get(idk, idk)].add(name)
    return out


def canonical_endpoints(t: dict, classes: dict[str, str]) -> tuple[str, str]:
    subject_id, object_id = endpoint_ids(t)
    return classes.get(subject_id, subject_id), classes.get(object_id, object_id)


# ---------------------------------------------------------------------------- walk enumeration


def enumerate_walks(train: dict[str, dict], classes: dict[str, str], max_walk: int = 3,
                    max_hub_degree: int = 64) -> tuple[list[tuple[str, ...]], dict]:
    """Every CONNECTED set of 2..max_walk train triples that forms a SIMPLE PATH over ENTITY CLASSES.

    A walk is kept when its own entity subgraph has exactly two degree-1 nodes and one more node than
    edges — a path with two nameable endpoints and no repeats. That drops (a) pairs sharing BOTH
    endpoints (an inverse pair: no (X, Y) to ask about) and (b) 3-edge stars, whose three endpoints do
    not reduce to one question. Measured on kanto TRAIN the filter costs ZERO graph coverage.

    HUB GUARD. An entity touching more than `max_hub_degree` train triples contributes no pairs and
    no adjacency. Two reasons, one of each kind. Substantive: a walk through a 268-degree hub is
    "A is at the festival, B is at the festival, so A connects to B" — 42,779 near-identical walks on
    edc/las_vegas/2026 alone. Structural: without it the 3-set expansion over that hub is ~10^7
    tuples held in memory, and an 18.7G pipeline python is what put 115 earlyoom kills on this box.
    kanto's maximum entity degree is 31, so the guard is inert there — a measured no-op, reported as
    `hub_skipped: []`, not an assumed one.

    Deterministic: sorted throughout, each walk a tuple of triple ids in PATH order.
    """
    stats = {"hub_skipped": [], "connected_sets": 0}
    if max_walk < 2:
        return [], stats
    touch_all: dict[str, set[str]] = defaultdict(set)
    for tid, t in train.items():
        for e in canonical_endpoints(t, classes):
            touch_all[e].add(tid)
    touch: dict[str, set[str]] = {}
    for e in sorted(touch_all):
        if len(touch_all[e]) > max_hub_degree:
            stats["hub_skipped"].append({"entity_class": e, "degree": len(touch_all[e])})
        else:
            touch[e] = touch_all[e]

    def path_order(ids: tuple[str, ...]) -> tuple[str, ...] | None:
        """Order the edges head→tail, or None when the set is not a simple path."""
        deg: Counter = Counter()
        for tid in ids:
            for e in set(canonical_endpoints(train[tid], classes)):
                deg[e] += 1
        ends = sorted(e for e, n in deg.items() if n == 1)
        if len(ends) != 2 or len(deg) != len(ids) + 1:
            return None
        remaining = set(ids)
        node = ends[0]
        ordered: list[str] = []
        while remaining:
            nxt = [tid for tid in sorted(remaining) if node in canonical_endpoints(train[tid], classes)]
            if len(nxt) != 1:
                return None
            tid = nxt[0]
            ordered.append(tid)
            remaining.discard(tid)
            a, b = canonical_endpoints(train[tid], classes)
            node = b if a == node else a
        return tuple(ordered)

    pairs: set[tuple[str, ...]] = set()
    for ids in touch.values():
        for a, b in combinations(sorted(ids), 2):
            pairs.add((a, b))

    adjacency: dict[str, set[str]] = defaultdict(set)
    for ids in touch.values():
        for tid in ids:
            adjacency[tid] |= ids
    for tid in adjacency:
        adjacency[tid].discard(tid)

    candidates: set[tuple[str, ...]] = set(pairs)
    frontier = pairs
    for _ in range(max_walk - 2):
        grown: set[tuple[str, ...]] = set()
        for walk in frontier:
            neighbours: set[str] = set()
            for tid in walk:
                neighbours |= adjacency[tid]
            for tid in neighbours - set(walk):
                grown.add(tuple(sorted(set(walk) | {tid})))
        candidates |= grown
        frontier = grown
    stats["connected_sets"] = len(candidates)

    out: list[tuple[str, ...]] = []
    for ids in sorted(candidates):
        ordered = path_order(ids)
        if ordered is not None:
            out.append(ordered)
    return sorted(out), stats


def walk_shape(walk: tuple[str, ...], train: dict[str, dict], classes: dict[str, str]) -> str:
    """Diagnostic label for the receipt: how the two edges of a pair meet."""
    if len(walk) != 2:
        return f"path_{len(walk)}"
    (s1, o1) = canonical_endpoints(train[walk[0]], classes)
    (s2, o2) = canonical_endpoints(train[walk[1]], classes)
    if o1 == s2 or o2 == s1:
        return "chain"
    if s1 == s2:
        return "shared_subject"
    if o1 == o2:
        return "shared_object"
    return "connected"


def holdout_bridge_pairs(uniq: dict[str, dict], holdout_ids: set[str],
                         classes: dict[str, str]) -> set[frozenset]:
    """Endpoint CLASS pairs of the held-out triples — the fact-level estimand boundary.

    `assert_train_only` checks triple ids and cannot see this: a walk from Agatha to Red through
    Lorelei states the connection the holdout triple (Red|opposes|Agatha) is probed on. Measured on
    kanto: 76 of 15,440 pool walks; dropping them costs 0 graph coverage.
    """
    out: set[frozenset] = set()
    for tid, t in uniq.items():
        if tid in holdout_ids:
            a, b = canonical_endpoints(t, classes)
            if a and b and a != b:
                out.add(frozenset((a, b)))
    return out


# ---------------------------------------------------------------------------- grounding


def _boundary(name: str) -> str:
    return r"(?<!\w)" + re.escape(name) + r"(?!\w)"


def universe_matcher(names) -> re.Pattern | None:
    """One compiled alternation over every entity display name in the unit (BOTH split sides).

    Longest-first so `Kanto Region` is preferred over `Kanto`. Case-SENSITIVE on purpose: display
    names are capitalised proper nouns and a case-insensitive scan turns the entity `May` into a
    false hit on the modal verb. Stated limit: a lowercase foreign entity name is not caught here.
    """
    names = sorted({str(n) for n in names if str(n).strip()}, key=lambda n: (-len(n), n))
    if not names:
        return None
    return re.compile("|".join(_boundary(n) for n in names))


def grounding_violations(text: str, allowed: set[str], matcher: re.Pattern | None) -> list[str]:
    """Entity names in `text` the walk does not license — the class's grounding gate.

    `allowed` is EVERY display name of the walk's own entity classes (a class merged by resolve can
    carry two spellings, and the stated edge may use either). They are masked longest-first BEFORE
    the scan, so an allowed `Kanto Region` cannot be reported as a foreign `Kanto`.
    """
    if matcher is None:
        return []
    masked = text
    for name in sorted(allowed, key=lambda n: (-len(n), n)):
        if name:
            masked = re.sub(_boundary(name), " \x00 ", masked)
    return sorted({m.group(0) for m in matcher.finditer(masked)})


def numeral_violations(text: str, licensed: set[str]) -> list[str]:
    """Digit runs the walk does not license — a model inventing a year, a count or an ordinal.

    The entity scan cannot see these: "1996" is not an entity display name, so a fabricated date
    passes a purely entity-level check.
    """
    return sorted({m.group(0) for m in _DIGITS_RE.finditer(text or "")} - set(licensed))


def negation_markers(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _NEGATION_RE.finditer(text or "")})


def missing_endpoints(text: str, x_names: set[str], y_names: set[str]) -> list[str]:
    """Which endpoint the line fails to name. A surface that drops an endpoint is not about the walk."""
    out = []
    for label, names in (("x", x_names), ("y", y_names)):
        if not any(re.search(_boundary(n), text) for n in names if n):
            out.append(label)
    return out


# ---------------------------------------------------------------------------- walk context


def walk_context(walk: tuple[str, ...], train: dict[str, dict], rel_q: dict, unit_name: str,
                 classes: dict[str, str], names_of: dict[str, str],
                 aliases_of: dict[str, set[str]]) -> tuple[dict | None, str]:
    """Everything the prompt and the grounding gate need for one walk, or (None, reason)."""
    decls = [declarative(train[tid], rel_q, unit_name) for tid in walk]
    if any(d is None for d in decls):
        return None, "unauthorable"
    deg: Counter = Counter()
    for tid in walk:
        for e in set(canonical_endpoints(train[tid], classes)):
            deg[e] += 1
    if sorted(deg.values()).count(1) != 2 or len(deg) != len(walk) + 1:
        return None, "not_a_path"
    start = next((e for e in canonical_endpoints(train[walk[0]], classes) if deg[e] == 1), None)
    if start is None:
        return None, "not_a_path"
    nodes = [start]
    node = start
    for tid in walk:
        a, b = canonical_endpoints(train[tid], classes)
        node = b if a == node else a
        nodes.append(node)
    if len(set(nodes)) != len(nodes) or deg[nodes[-1]] != 1:
        return None, "not_a_path"
    rendered = [names_of.get(n, n) for n in nodes]
    if len(set(rendered)) != len(rendered):
        # Belt-and-braces after the class merge: alias tables are never complete, and a walk whose
        # endpoints render one name is a fabricated self-connection ("What links Pikachu to Pikachu?").
        return None, "degenerate"
    licensed_names: set[str] = set()
    for n in nodes:
        licensed_names |= {names_of.get(n, n)} | set(aliases_of.get(n, ()))
    licensed_numbers = {m.group(0) for d in decls for m in _DIGITS_RE.finditer(d)}
    licensed_numbers |= {m.group(0) for n in licensed_names for m in _DIGITS_RE.finditer(n)}
    return {
        "walk": tuple(walk),
        "decls": [str(d) for d in decls],
        "nodes": nodes,
        "rendered": rendered,
        "x": rendered[0],
        "y": rendered[-1],
        "bridges": rendered[1:-1],
        "x_names": {names_of.get(nodes[0], nodes[0])} | set(aliases_of.get(nodes[0], ())),
        "y_names": {names_of.get(nodes[-1], nodes[-1])} | set(aliases_of.get(nodes[-1], ())),
        "licensed_names": licensed_names,
        "licensed_numbers": licensed_numbers,
        "relation_type": str(train[walk[0]].get("predicate", "")),
        "doc_ids": _doc_ids(walk, train),
    }, ""


def _doc_ids(walk: tuple[str, ...], train: dict[str, dict]) -> list[str]:
    out: list[str] = []
    for tid in walk:
        for d in source_doc_ids(train[tid]):
            if d not in out:
                out.append(d)
    return out


# ---------------------------------------------------------------------------- budget + selection


def budget_for_fraction(base_lines: int, max_fraction: float) -> tuple[int, dict]:
    """How many records make this class `max_fraction` of the corpus it JOINS.

    The base is the unit's OTHER hydrated surface files, so the fraction is a share of the FINAL
    corpus: N / (base + N) == max_fraction, i.e. N = base * f/(1-f). Named explicitly because the two
    readings of "10% of the corpus" differ 6x on kanto — 10% of the 8,159-line hydrated layer is 906
    records, while `clamp_contrast_to_fraction`'s base (relations.jsonl alone, 1,518) gives 152, and
    152 records cannot cover more than 304 of the 521 train triples at 2 ids each, i.e. the narrower
    base structurally caps coverage below the directed-chain pool this class exists to beat.
    """
    if base_lines <= 0 or not (0.0 < max_fraction < 1.0):
        return 0, {"limit": 0, "base_lines": base_lines, "max_fraction": max_fraction,
                   "reason": "no base surfaces or max_fraction outside (0,1) — class unavailable"}
    limit = int(base_lines * max_fraction / (1.0 - max_fraction))
    return limit, {
        "limit": limit,
        "base_lines": base_lines,
        "max_fraction": max_fraction,
        "formula": "limit = int(base_lines * f / (1 - f))  # share of base+limit, not of base",
        "share_of_final": round(limit / (base_lines + limit), 5) if (base_lines + limit) else None,
        "reason": (f"{limit} of {base_lines + limit} final hydrated lines = "
                   f"{limit / (base_lines + limit):.2%} (target {max_fraction:.0%})"),
    }


def select_walks(walks: list[tuple[str, ...]], budget: int) -> list[tuple[str, ...]]:
    """Coverage first, then lowest-utilization fill — the r1 allocation policy, applied to walks.

    Pass 1 gives every train triple appearing in ANY candidate walk at least one surface, which is
    the number separating this class from the directed-chain pin (304/521 vs 501/521). Pass 2 spends
    the remainder on the triples used least, so the class does not reintroduce the per-fact
    amplification skew logs/phase_r1_utilization_balance.json collapsed to 1.00x. Deterministic.
    """
    if budget <= 0 or not walks:
        return []
    by_tid: dict[str, list[int]] = defaultdict(list)
    for i, walk in enumerate(walks):
        for tid in walk:
            by_tid[tid].append(i)
    used: Counter = Counter()
    chosen: list[int] = []
    chosen_set: set[int] = set()

    for tid in sorted(by_tid):
        if len(chosen) >= budget:
            break
        if used[tid]:
            continue
        pick = min(by_tid[tid], key=lambda i: (sum(used[x] for x in walks[i]), -len(walks[i]), walks[i]))
        chosen.append(pick)
        chosen_set.add(pick)
        for x in walks[pick]:
            used[x] += 1

    while len(chosen) < budget:
        progressed = False
        for tid in sorted(by_tid, key=lambda t: (used[t], t)):
            if len(chosen) >= budget:
                break
            free = [i for i in by_tid[tid] if i not in chosen_set]
            if not free:
                continue
            # Same rule as pass 1: the walk whose OTHER triples are least used. Taking the next
            # unchosen walk instead drags hub triples along as passengers — measured on kanto, the
            # cursor version produced a max of 83 surfaces for one triple against a mean of 4.8.
            pick = min(free, key=lambda i: (sum(used[x] for x in walks[i]), -len(walks[i]), walks[i]))
            chosen.append(pick)
            chosen_set.add(pick)
            for x in walks[pick]:
                used[x] += 1
            progressed = True
        if not progressed:
            break
    return [walks[i] for i in chosen]


# ---------------------------------------------------------------------------- estimand + health


def assert_train_only(records: list[dict], train_ids: set[str], holdout_ids: set[str]) -> None:
    """Every constituent triple of every emitted walk is on the TRAIN side. Positive membership only:
    `not in holdout_ids` admits the triples in NEITHER set (kanto: 140 of 736, because 15_structured
    was rewritten after the immutable split was cut)."""
    leaks = []
    for rec in records:
        for tid in rec.get("source_triple_ids") or []:
            if tid in holdout_ids:
                leaks.append((tid, "HOLDOUT", rec.get("text", "")[:80]))
            elif tid not in train_ids:
                leaks.append((tid, "UNSPLIT", rec.get("text", "")[:80]))
    if leaks:
        raise SystemExit(
            f"[generic_reasoning] estimand violation: {len(leaks)} emitted walk triple(s) are not in "
            f"split TRAIN — {leaks[:5]}"
        )


def form_metrics(texts: list[str], threshold: float = 0.55) -> dict:
    """The NAMED diversity ruler for this class: diversity_lib.effective_forms (jaccard `threshold`),
    NOT distinct_n. Both are reported because this repo's two measures disagree."""
    forms = Div.effective_forms(list(texts), threshold) if texts else 0
    return {
        "ruler": f"diversity_lib.effective_forms(jaccard={threshold})",
        "records": len(texts),
        "effective_forms": forms,
        "form_ratio": round(forms / len(texts), 4) if texts else 0.0,
        "distinct_2": round(Div.distinct_n(texts, 2), 4) if texts else 0.0,
        "also_reported": "diversity_lib.distinct_n(n=2) — the audit gate's ruler; the two disagree",
    }


def assert_not_form_collapsed(texts: list[str], min_form_ratio: float, where: str) -> dict:
    """Fail-closed on a form-collapsed class. THIS is the gate for this class's diversity.

    No corpus gate reads this file, so an in-file check is the only one there is. The bar is a ratio
    so it is scale-free. Calibration on kanto, all at 906 records, ruler effective_forms(0.55):
        3 fixed f-string skeletons (the renderer this repair replaces) ....  19 forms, ratio 0.021
        relations.jsonl, deterministic per-triple floor ................... 356 forms, ratio 0.393
        relations_llm.jsonl, this same generator seam, 5 seeds ....... 552-596 forms, ratio ~0.63
    The default 0.20 sits an order of magnitude above the collapsed renderer and half the
    deterministic floor, so it fails the artifact this repair exists to remove and passes the channel
    that replaces it. Stated limit: effective_forms is lexical jaccard, so it counts a mere reversal
    as two forms (r6 measured 0.43 for one). This is a floor against collapse, not proof of variety.
    """
    metrics = form_metrics(texts)
    metrics["min_form_ratio"] = min_form_ratio
    if not texts or min_form_ratio <= 0:
        metrics["verdict"] = "not_applicable"
        return metrics
    if metrics["form_ratio"] < min_form_ratio:
        raise SystemExit(
            f"[generic_reasoning] {where}: form-collapsed class — {metrics['effective_forms']} "
            f"effective forms over {len(texts)} records (ratio {metrics['form_ratio']:.3f} < "
            f"{min_form_ratio}) under {metrics['ruler']}. Refusing to write. A deterministic template "
            f"renderer scores ~0.021 here and that is exactly what this gate exists to stop; check "
            f"the generator response (an outage returns the same fallback line every call)."
        )
    metrics["verdict"] = "pass"
    return metrics


def assert_channel_healthy(stats: dict, max_empty_fraction: float = 0.5) -> None:
    """A mostly-empty generator run must FAIL the stage, not ship a thin corpus green.

    Mirrors generic_synth's abort. Empty completions are COUNTED at the call site rather than
    smoothed into a successful-looking empty string — qwen3-family models burn max_tokens on
    reasoning and return empty content, and `content or ""` surfaces that far from its cause.
    """
    calls = int(stats.get("llm_calls", 0))
    empty = int(stats.get("drops", {}).get("empty_completion", 0))
    if calls and empty / calls > max_empty_fraction:
        raise SystemExit(
            f"[generic_reasoning] {empty}/{calls} generator calls returned nothing "
            f"(>{max_empty_fraction:.0%}) — the reasoning layer is mostly empty; aborting the stage "
            f"(check the API key / quota / outage / a model that answers only in `reasoning`) rather "
            f"than writing a thin corpus green."
        )
    if calls and not stats.get("emitted"):
        raise SystemExit(
            f"[generic_reasoning] {calls} generator calls produced 0 admissible surfaces "
            f"(drops: {stats.get('drops')}). Refusing to write an empty corpus or overwrite a prior "
            f"good one."
        )


# ---------------------------------------------------------------------------- plan + render


def plan_walks(train: dict[str, dict], uniq: dict[str, dict], rel_q: dict, unit_name: str,
               domain: str | None, *, classes: dict[str, str], names_of: dict[str, str],
               aliases_of: dict[str, set[str]], holdout_ids: set[str] | frozenset = frozenset(),
               max_walk: int = 3, max_hub_degree: int = 64,
               n_walks: int | None = None) -> tuple[list[dict], dict]:
    """Deterministic half: enumerate, guard, select. ZERO generator calls happen here."""
    drops = {reason: 0 for reason in DROP_REASONS}
    stats: dict = {"drops": drops, "examples": {}, "class_unavailable": None,
                   "walks_enumerated": 0, "walks_renderable": 0, "hub_skipped": []}
    admitted = admitted_relation_types(domain, [t.get("predicate", "") for t in train.values()])
    if not admitted:
        # Checked BEFORE enumeration: a domain whose taxonomy admits no raw predicate must cost
        # nothing, not enumerate a 10^7-tuple pool and then discard it.
        stats["class_unavailable"] = (
            f"domain '{domain}' pins relation_type to a taxonomy admitting none of this unit's raw "
            f"predicates; the class cannot label a walk without inventing a relation label")
        print(f"  [generic_reasoning] {stats['class_unavailable']} — 0 records", file=sys.stderr)
        return [], stats

    walks, enum_stats = enumerate_walks(train, classes, max_walk=max_walk, max_hub_degree=max_hub_degree)
    stats["walks_enumerated"] = len(walks)
    stats["connected_sets"] = enum_stats["connected_sets"]
    stats["hub_skipped"] = enum_stats["hub_skipped"]

    bridges = holdout_bridge_pairs(uniq, set(holdout_ids), classes)
    contexts: list[dict] = []
    for walk in walks:
        ctx, reason = walk_context(walk, train, rel_q, unit_name, classes, names_of, aliases_of)
        if ctx is None:
            drops[reason] += 1
            if reason in ("degenerate",) and len(stats["examples"].setdefault(reason, [])) < 3:
                stats["examples"][reason].append(list(walk))
            continue
        if ctx["relation_type"] not in admitted:
            drops["unlabelable"] += 1
            continue
        # EVERY non-adjacent node pair, not just the endpoints. A walk asserts a connection between
        # each pair of nodes it names; adjacent pairs are its own declared TRAIN edges and are
        # exempt, but any pair at distance >= 2 is a claim the walk makes without stating the edge —
        # exactly what a held-out triple is probed on. Checking endpoints alone let a longer walk
        # narrate the held-out fact between two interior nodes: measured on kanto at shipped
        # defaults, the selected walk [Professor Oak, Agatha, Lorelei, Red] states the held-out
        # (Agatha, Red) pair while its outer pair (Professor Oak, Red) is not held out at all.
        nodes = ctx["nodes"]
        stated = next(((i, j) for i in range(len(nodes)) for j in range(i + 2, len(nodes))
                       if frozenset((nodes[i], nodes[j])) in bridges), None)
        if stated is not None:
            drops["holdout_bridge"] += 1
            if len(stats["examples"].setdefault("holdout_bridge", [])) < 3:
                i, j = stated
                stats["examples"]["holdout_bridge"].append(
                    {"walk": list(walk), "endpoints": [ctx["x"], ctx["y"]],
                     "stated_pair": [ctx["rendered"][i], ctx["rendered"][j]]})
            continue
        contexts.append(ctx)
    stats["walks_renderable"] = len(contexts)

    pool = [ctx["walk"] for ctx in contexts]
    by_walk = {ctx["walk"]: ctx for ctx in contexts}
    chosen = select_walks(pool, len(pool) if n_walks is None else n_walks)
    selected = [by_walk[w] for w in chosen]
    stats["walks_selected"] = len(selected)
    stats["llm_calls_planned"] = len(selected)
    stats["shape_counts"] = dict(sorted(Counter(
        walk_shape(ctx["walk"], train, classes) for ctx in selected).items()))
    stats["walk_train_triples_covered"] = len({tid for ctx in selected for tid in ctx["walk"]})
    return selected, stats


def build_prompt(ctx: dict, preamble: str, persona: str, variations: int) -> str:
    via = ""
    if ctx["bridges"]:
        via = " through " + " and ".join(ctx["bridges"])
    head = preamble + "\n\n" if preamble else ""
    if persona:
        head = head + persona + "\n\n"
    return head + WALK_PROMPT.format(
        k=len(ctx["decls"]), x=ctx["x"], y=ctx["y"], via=via,
        facts="\n".join(f"    {d}" for d in ctx["decls"]), n=variations)


def parse_variations(raw: str) -> list[str]:
    """Generator response → candidate surfaces. Marker stripping mirrors generic_synth."""
    out: list[str] = []
    for line in (raw or "").splitlines():
        line = _MARKER_RE.sub("", line).strip().strip('"').strip()
        if line:
            out.append(line)
    return out


def render_records(selected: list[dict], domain: str | None, universe_names, *, preamble: str = "",
                   generate_fn=openrouter_generate, variations: int = 2, budget: int | None = None,
                   concurrency: int = 16,
                   personas: tuple[str, ...] | list[str] | None = DEFAULT_PERSONAS,
                   temperatures: tuple[float, ...] | list[float] | None = DEFAULT_TEMPERATURES,
                   vary_conditioning: bool = True) -> tuple[list[dict], dict]:
    """LLM half: one call per selected walk, then the fail-closed grounding gate on every line.

    `generate_fn(system, user, **kw) -> str` is the SAME injectable seam generic_synth uses
    (generic_synth.py:114-121) with the same 2-arg TypeError fallback for stubs, so the whole class is
    provable offline at zero provider cost. `executor.map` preserves input order, so concurrency
    changes wall time and never corpus order.
    """
    drops = {reason: 0 for reason in DROP_REASONS}
    stats: dict = {"drops": drops, "examples": {}, "llm_calls": 0, "variations_requested": variations}
    if not selected:
        stats.update({"emitted": 0, "train_triples_covered": 0})
        return [], stats
    matcher = universe_matcher(universe_names)
    personas = tuple(personas or ()) if vary_conditioning else ()
    temperatures = tuple(temperatures or (0.85,)) if vary_conditioning else (0.85,)

    prepared = []
    for index, ctx in enumerate(selected):
        persona = _persona_for(index, personas)
        temperature = _temperature_for(index, temperatures)
        prepared.append((ctx, build_prompt(ctx, preamble, persona, variations), temperature, persona))

    def call(item):
        ctx, system, temperature, persona = item
        try:
            raw = generate_fn(system, "Write the variations now.", temperature=temperature)
        except TypeError:
            raw = generate_fn(system, "Write the variations now.")          # stubs taking (system, user)
        return ctx, persona, raw

    results = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        for completed, item in enumerate(executor.map(call, prepared), start=1):
            results.append(item)
            if completed % 25 == 0 or completed == len(prepared):
                print(f"  …{completed}/{len(prepared)} reasoning calls", flush=True)
    stats["llm_calls"] = len(results)

    out: list[dict] = []
    seen_text: set[str] = set()
    for ctx, persona, raw in results:
        if not (raw or "").strip():
            drops["empty_completion"] += 1
            continue
        kept = 0
        for line in parse_variations(raw):
            if kept >= variations:
                break
            if len(line) < MIN_LINE_CHARS:
                drops["too_short"] += 1
                continue
            foreign = grounding_violations(line, ctx["licensed_names"], matcher)
            if foreign:
                drops["ungrounded"] += 1
                _example(stats, "ungrounded", {"walk": list(ctx["walk"]), "foreign_entities": foreign,
                                               "text": line[:160]})
                continue
            numerals = numeral_violations(line, ctx["licensed_numbers"])
            if numerals:
                drops["foreign_numeral"] += 1
                _example(stats, "foreign_numeral", {"walk": list(ctx["walk"]), "numerals": numerals,
                                                    "text": line[:160]})
                continue
            missing = missing_endpoints(line, ctx["x_names"], ctx["y_names"])
            if missing:
                drops["missing_endpoint"] += 1
                _example(stats, "missing_endpoint", {"walk": list(ctx["walk"]), "missing": missing,
                                                     "text": line[:160]})
                continue
            denials = negation_markers(line)
            if denials:
                drops["negated"] += 1
                _example(stats, "negated", {"walk": list(ctx["walk"]), "markers": denials,
                                            "text": line[:160]})
                continue
            if line in seen_text:
                drops["duplicate"] += 1
                continue
            seen_text.add(line)
            rec = {
                "text": line,
                "relation_type": ctx["relation_type"],
                "surface_form": SURFACE_FORM,
                "source_triple_ids": list(ctx["walk"]),
                "entity_ids": list(ctx["rendered"]),
                "split": "cpt",
            }
            if ctx["doc_ids"]:
                rec["source_doc_ids"] = ctx["doc_ids"]
            if persona:
                rec["synth_persona"] = persona[:80]
            errors = validate("relation", rec, domain=domain)
            if errors:
                raise SystemExit(
                    f"[generic_reasoning] 25_hydrated relation record fails schema for domain "
                    f"'{domain}': {errors[:3]} | rec={rec}")
            out.append(rec)
            kept += 1

    if budget is not None and len(out) > budget:
        out = out[:budget]
    stats["emitted"] = len(out)
    stats["budget_shortfall"] = max(0, (budget or len(out)) - len(out))
    stats["train_triples_covered"] = len({tid for rec in out for tid in rec["source_triple_ids"]})
    stats["surfaces_per_walk"] = _spread(Counter(tuple(r["source_triple_ids"]) for r in out))
    stats["surfaces_per_triple"] = _spread(Counter(t for r in out for t in r["source_triple_ids"]))
    stats["form_metrics"] = form_metrics([r["text"] for r in out])
    return out, stats


def _example(stats: dict, reason: str, payload: dict, limit: int = 5) -> None:
    bucket = stats["examples"].setdefault(reason, [])
    if len(bucket) < limit:
        bucket.append(payload)


def _spread(counter: Counter) -> dict:
    values = sorted(counter.values())
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": values[0], "p50": values[len(values) // 2], "max": values[-1],
            "mean": round(sum(values) / len(values), 3)}


def reasoning_records(train: dict[str, dict], uniq: dict[str, dict], rel_q: dict, unit_name: str,
                      domain: str | None, universe_names, *, classes: dict[str, str],
                      names_of: dict[str, str], aliases_of: dict[str, set[str]],
                      holdout_ids: set[str] | frozenset = frozenset(), preamble: str = "",
                      generate_fn=openrouter_generate, variations: int = 2,
                      budget: int | None = None, max_walk: int = 3, max_hub_degree: int = 64,
                      concurrency: int = 16, **render_kw) -> tuple[list[dict], dict]:
    """Plan then render for one unit. Convenience wrapper; `main` uses the two halves directly so
    `--dry-run` cannot reach the generator at all."""
    n_walks = None if budget is None else max(1, math.ceil(budget / max(1, variations)))
    selected, plan_stats = plan_walks(
        train, uniq, rel_q, unit_name, domain, classes=classes, names_of=names_of,
        aliases_of=aliases_of, holdout_ids=holdout_ids, max_walk=max_walk,
        max_hub_degree=max_hub_degree, n_walks=n_walks)
    records, render_stats = render_records(
        selected, domain, universe_names, preamble=preamble, generate_fn=generate_fn,
        variations=variations, budget=budget, concurrency=concurrency, **render_kw)
    return records, merge_stats(plan_stats, render_stats)


def merge_stats(plan_stats: dict, render_stats: dict) -> dict:
    merged = dict(plan_stats)
    drops = dict(plan_stats.get("drops") or {})
    for reason, n in (render_stats.get("drops") or {}).items():
        drops[reason] = drops.get(reason, 0) + n
    examples = dict(plan_stats.get("examples") or {})
    examples.update(render_stats.get("examples") or {})
    merged.update({k: v for k, v in render_stats.items() if k not in ("drops", "examples")})
    merged["drops"] = drops
    merged["examples"] = examples
    return merged


# ---------------------------------------------------------------------------- unit plumbing


def _line_count(path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _unit_name(ed) -> str:
    meta = ed.dir / "festival_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("name") or ed.slug
        except (json.JSONDecodeError, OSError):
            pass
    return ed.slug


def base_surface_lines(ed) -> tuple[int, dict[str, int]]:
    """Lines in the unit's OTHER hydrated surface files — this class's budget base.

    Reads `dedup_surfaces.HYDRATED_NAMES` (the registry this class is listed in) MINUS this class's
    own output, so re-running is idempotent rather than compounding.
    """
    from pipeline.dedup_surfaces import HYDRATED_NAMES

    hyd = ed.dir / "25_hydrated"
    per = {name: _line_count(hyd / name) for name in HYDRATED_NAMES if name != OUT_NAME}
    return sum(per.values()), per


def dedup_delta(ed, records: list[dict], cap: int) -> dict:
    """What registering this file in HYDRATED_NAMES actually costs — a NAMED instrument, not a claim.

    Registering does NOT buy a meaningful exposure cap: `diversity_lib.fact_key` is
    source_triple_ids[0], so the class scatters across one bucket per head triple, far under the cap.
    What it DOES do is put the class into the per-fact cull, where it can displace pre-existing
    surfaces. Measured on kanto, both with a deterministic renderer standing in for the generator:
    excluded 49 → 56 at 906 records (7 pre-existing displaced) and 49 → 57 at 904 (8 displaced); in
    both, 0 reasoning records culled and floor_violations 0 before AND after. Reported per run
    because the displacement depends on what the generator actually wrote.
    """
    from pipeline import dedup_surfaces as DS

    base_rows = DS.load_unit_surfaces(ed)
    extra = []
    for i, rec in enumerate(records):
        text = str(rec.get("text") or "").strip()
        if not text:
            continue
        extra.append({"text": text, "relation_type": rec.get("relation_type") or "?",
                      "source_triple_ids": rec.get("source_triple_ids") or [],
                      "surface_form": rec.get("surface_form") or SURFACE_FORM,
                      "path": OUT_NAME, "line": i,
                      "key": DS.surface_key(OUT_NAME, i, text), "fact": Div.fact_key(rec)})
    base_rows = [row for row in base_rows if not str(row.get("path", "")).endswith(OUT_NAME)]
    before = DS.dedup_unit(base_rows, budget=cap, cap=cap)
    after = DS.dedup_unit(base_rows + extra, budget=cap, cap=cap)
    new_keys = {row["key"] for row in extra}
    newly_excluded = set(after["exclude_keys"]) - set(before["exclude_keys"]) - new_keys
    return {
        "cap": cap,
        "n_excluded_before": before["n_excluded"],
        "n_excluded_after": after["n_excluded"],
        "reasoning_records_culled": len(new_keys & set(after["exclude_keys"])),
        "pre_existing_newly_excluded": len(newly_excluded),
        "pre_existing_examples": sorted(newly_excluded)[:5],
        "floor_violations_before": len(before["floor_violations"]),
        "floor_violations_after": len(after["floor_violations"]),
        "head_fact_buckets": len({row["fact"] for row in extra}),
        "note": ("fact_key is source_triple_ids[0], so the exposure cap does not bind on this class; "
                 "the honest control is floor_violations 0 before AND after, not relations.jsonl's "
                 "line count (dedup is non-destructive and can never move it)"),
    }


def unit_frames(ed, domain: str) -> dict:
    """Everything a unit contributes: triples, split sides, entity classes, display vocabulary."""
    all_triples = load_all_triples(ed)
    uniq: dict[str, dict] = {}
    for t in all_triples:
        uniq.setdefault(triple_id(t), t)
    split = load_split(ed.dir)
    if split:
        train = {tid: t for tid, t in uniq.items() if tid in split["train_ids"]}
        mode = f"split {split['version']} TRAIN"
    else:
        train = dict(uniq)
        mode = "no split manifest (ALL triples)"
    id_map = load_id_map(domain, ed.slug)
    triples = list(uniq.values())
    classes = entity_classes(triples, id_map)
    names_of = class_names(triples, classes)
    aliases_of = class_aliases(triples, classes)
    universe: set[str] = set()
    for names in aliases_of.values():
        universe |= names
    return {"uniq": uniq, "train": train, "split": split, "mode": mode, "id_map": id_map,
            "classes": classes, "names_of": names_of, "aliases_of": aliases_of, "universe": universe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="pokemon-lore")
    ap.add_argument("--editions", help="restrict to 'unit,…' (default: all on disk)")
    ap.add_argument("--max-walk", type=int, default=3, help="largest connected triple set (2 or 3)")
    ap.add_argument("--max-hub-degree", type=int, default=64,
                    help="an entity touching more train triples than this bridges no walk")
    ap.add_argument("--variations", type=int, default=2,
                    help="surfaces requested per walk; ONE LLM call per walk, so this is the spend "
                         "knob. Measured on kanto at budget 906, all three at coverage 501/521 "
                         "(the graph ceiling): v=1 906 calls, per-triple max/p50 6/4 = 1.5x; "
                         "v=2 453 calls, 8/4 = 2.0x; v=4 227 calls, 12/4 = 3.0x. Default 2 halves "
                         "the spend while keeping the per-fact skew nearer r1's 1.00x than v=4 does")
    ap.add_argument("--max-fraction", type=float, default=0.10,
                    help="this class's share of the FINAL hydrated corpus (default 0.10); the base is "
                         "the unit's other hydrated surface files, printed with the arithmetic")
    ap.add_argument("--min-form-ratio", type=float, default=0.20,
                    help="fail-closed floor on effective_forms/records for the emitted class")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: counts, coverage and the LLM call count a real run would make. "
                         "Makes ZERO provider calls and writes nothing.")
    args = ap.parse_args()

    rel_q = rel_q_for(args.domain)
    if not rel_q:
        print(f"[generic_reasoning] domain '{args.domain}': reasoning surfaces unavailable — no "
              f"sft_templates.rel_q authored")
        return
    preamble = preamble_for(args.domain)

    cap = None
    if not args.dry_run:
        try:
            from qgre import spec as S
            cap = int(S.relations_cfg(S.load_spec(args.domain)).get("exposure_cap_per_fact", 60))
        except (FileNotFoundError, KeyError, ValueError) as error:
            print(f"  [generic_reasoning] dedup delta unavailable: {type(error).__name__}: {error}",
                  file=sys.stderr)

    wrote = 0
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        frames = unit_frames(ed, args.domain)
        train, uniq, split = frames["train"], frames["uniq"], frames["split"]
        if len(train) < 2:
            print(f"  {ed.slug}: {frames['mode']} — {len(train)} train triple(s); no walk possible")
            continue
        base_lines, base_per_file = base_surface_lines(ed)
        budget, budget_info = budget_for_fraction(base_lines, args.max_fraction)
        n_walks = max(1, math.ceil(budget / max(1, args.variations))) if budget else 0
        selected, stats = plan_walks(
            train, uniq, rel_q, _unit_name(ed), args.domain, classes=frames["classes"],
            names_of=frames["names_of"], aliases_of=frames["aliases_of"],
            holdout_ids=(split or {}).get("holdout_ids") or frozenset(),
            max_walk=args.max_walk, max_hub_degree=args.max_hub_degree, n_walks=n_walks or None)
        walk_cov = stats.get("walk_train_triples_covered", 0)
        live_drops = {k: v for k, v in stats["drops"].items() if v}
        print(f"  {ed.slug}: {frames['mode']} — {len(train)}/{len(uniq)} unique triples; "
              f"entity ids {len(frames['classes'])} → {len(set(frames['classes'].values()))} classes; "
              f"walks {stats['walks_enumerated']} enumerated / {stats['walks_renderable']} renderable "
              f"(drops {live_drops}); "
              f"budget {budget} records = {args.max_fraction:.0%} of base {base_lines} + budget; "
              f"selected {stats.get('walks_selected', 0)} walks × {args.variations} variations = "
              f"{stats.get('llm_calls_planned', 0)} LLM calls; "
              f"walk coverage {walk_cov}/{len(train)}; shapes {stats.get('shape_counts')}")
        if stats.get("hub_skipped"):
            print(f"    hub guard skipped {len(stats['hub_skipped'])} entity class(es): "
                  f"{stats['hub_skipped'][:3]}")
        if args.dry_run:
            print("    --dry-run: 0 provider calls made, nothing written")
            continue
        if not selected:
            print("    nothing renderable — no file written")
            continue

        records, render_stats = render_records(
            selected, args.domain, frames["universe"], preamble=preamble, variations=args.variations,
            budget=budget or None, concurrency=args.concurrency)
        stats = merge_stats(stats, render_stats)
        assert_channel_healthy(stats)
        if split:
            assert_train_only(records, split["train_ids"], split["holdout_ids"])
        form = assert_not_form_collapsed([r["text"] for r in records], args.min_form_ratio, ed.slug)
        coverage = stats["train_triples_covered"] / len(train) if train else 0.0
        achieved = len(records) / (base_lines + len(records)) if (base_lines + len(records)) else 0.0

        out_dir = ed.dir / "25_hydrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / OUT_NAME
        tmp = out_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) +
                       ("\n" if records else ""), encoding="utf-8")
        os.replace(tmp, out_path)

        receipt_dir = C.BUILD / "reasoning" / args.domain
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{ed.slug.replace('/', '_')}_budget.json"
        payload = {
            "domain": args.domain, "unit": ed.slug, "split": (split or {}).get("version"),
            "train_triples": len(train), "unique_triples": len(uniq),
            "entity_ids": len(frames["classes"]), "entity_classes": len(set(frames["classes"].values())),
            "id_map_entries": len(frames["id_map"]),
            "base_lines_per_file": base_per_file, "budget": budget_info,
            "variations": args.variations, "form_gate": form,
            "train_triples_covered_frac": round(coverage, 4),
            "achieved_share_of_final": round(achieved, 5),
            "estimand": ("RUN-GATE, TRAIN side of the unit's split. Not an install-strength number: "
                         "no checkpoint, no eval, no CPT run."),
            **{k: v for k, v in stats.items()},
        }
        if cap is not None:
            payload["dedup_delta"] = dedup_delta(ed, records, cap)
        receipt_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        wrote += 1
        print(f"    EMITTED {len(records)} = {achieved:.2%} of the final hydrated corpus; "
              f"covered {stats['train_triples_covered']}/{len(train)} = {coverage:.1%}; "
              f"{form['effective_forms']} effective forms (ratio {form['form_ratio']:.3f}); "
              f"wrote {out_path} (receipt {receipt_path})")
    if not args.dry_run:
        print(f"\nwrote {wrote} {OUT_NAME} file(s)")


if __name__ == "__main__":
    main()
