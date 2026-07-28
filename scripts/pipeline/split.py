#!/usr/bin/env python3
"""Triple-id split — the seam that makes "held-out" TRUE instead of a label.

Before this module, `generic_probes.py` stamped every probe `split: held_out` while every triple
hydrated: deltas measured trained-fact recall + format transfer, never fact-holdout generalization
(the measured 2026-07-12 finding; the agent handbook's `eval/split-and-decontamination.md` carries
the design law this implements).

The contract:
  1. Run AFTER `structured`, BEFORE `hydrate`: writes a versioned, immutable split manifest per unit
     (`<unit>/splits/split_v1.json`) of content-addressed triple ids — sha1(subject|predicate|object).
  2. Holdout is carved by SUBJECT GROUP (all triples sharing a subject travel to one side — the
     cheap component grouping that keeps a held-out subject's other facts from training next to it),
     with a guard that no predicate loses its entire train side.
  3. `generic_expand` / `generic_synth` hydrate TRAIN ids only; `generic_probes` draws gold from
     HOLDOUT ids and stamps the split version. With no manifest present, everything behaves as
     before EXCEPT the probe stamp, which becomes the truthful `trained`.
  4. The manifest is immutable: re-running refuses to overwrite (a new split is a new version file,
     never a rewrite — re-cutting after a weak eval is goalpost-moving).

    PYTHONPATH=scripts python scripts/pipeline/split.py --domain <d> [--editions u1,u2]
        [--holdout-frac 0.10] [--seed 42] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SPLIT_VERSION = "v1"
SPLIT_VERSION_V2 = "v2"
SPLIT_VERSION_V3 = "v3"
# Preference order for load_split when no version is requested. v3 is the alias+inverse series
# cut with the unified id_map loader (nested unit slugs resolve); v2 is the same algorithm cut
# when load_id_map keyed on ed_dir.name and returned {} for every EDC city/year unit.
SPLIT_VERSION_PREFERENCE = (SPLIT_VERSION_V3, SPLIT_VERSION_V2, SPLIT_VERSION)

# Common inverse pairs — both directions listed so lookup is symmetric.
_INVERSE_PAIRS = (
    ("leads", "led_by"),
    ("member_of", "has_member"),
    ("located_in", "contains"),
    ("evolves_into", "evolves_from"),
    ("parent_of", "child_of"),
    ("rival_of", "rival_of"),
    ("allied_with", "allied_with"),
)
INVERSE_OF: dict[str, str] = {}
for _a, _b in _INVERSE_PAIRS:
    INVERSE_OF[_a] = _b
    INVERSE_OF[_b] = _a


def triple_id(t: dict) -> str:
    """Content-addressed id: stable across runs, derivable by every consumer from the record alone.
    Keyed on the semantic triple (subject id | predicate | object), not on file position."""
    key = f"{t.get('subject_id', '')}|{t.get('predicate', '')}|{t.get('object', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def split_path(ed_dir, version: str = SPLIT_VERSION):
    return ed_dir / "splits" / f"split_{version}.json"


def load_split(ed_dir, version: str | None = None) -> dict | None:
    """The unit's split manifest, or None (no split carved — consumers fail open, probes stamp
    `trained`). Prefer the newest series present (v3 > v2 > v1) when version is None."""
    if version is None:
        for ver in SPLIT_VERSION_PREFERENCE:
            p = split_path(ed_dir, ver)
            if p.exists():
                version = ver
                break
        else:
            return None
    p = split_path(ed_dir, version)
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    m["train_ids"] = set(m["train_ids"])
    m["holdout_ids"] = set(m["holdout_ids"])
    return m


def _domain_and_slug(ed_dir: Path, domain: str | None = None) -> tuple[str | None, str]:
    """Resolve (domain, unit_slug) for an edition directory.

    Nested units (EDC `city/year`) live at `data/<domain>/<city>/<year>`. The slug is the path
    *relative to* `data/<domain>` (e.g. `las_vegas/2011`), never `ed_dir.name` alone — that is only
    the year, and looking up `resolve/edc/2011/id_map.json` silently returns {} for every EDC unit.
    Measured 2026-07-25: 50 EDC units carry a non-empty resolve map under the full slug and zero
    under the year-only key; every on-disk split_v2 recorded `id_map_size: 0` because of this.
    """
    ed_dir = Path(ed_dir)
    # Prefer the live config root; fall back to a `data/` segment anywhere in the path so off-tree
    # test fixtures and relocated checkouts still yield city/year rather than year-only.
    parts_full = ed_dir.resolve().parts if ed_dir.exists() else ed_dir.parts
    try:
        import config as C
        data_root = Path(C.ROOT).resolve() / "data"
        rel = ed_dir.resolve().relative_to(data_root) if ed_dir.exists() else None
    except Exception:
        rel = None
    if rel is not None:
        parts = rel.parts
    elif "data" in parts_full:
        i = parts_full.index("data")
        parts = parts_full[i + 1 :]
    else:
        return domain, ed_dir.name
    if not parts:
        return domain, ed_dir.name
    inferred_domain = parts[0]
    slug = "/".join(parts[1:]) if len(parts) > 1 else ed_dir.name
    return (domain or inferred_domain), slug


def id_map_path(ed_dir, domain: str | None = None) -> Path | None:
    """The resolve-stage id_map file for this unit, or None. Single source of path truth —
    loaders AND digests must read the same file (the coverage ruler pins this path's sha256)."""
    ed_dir = Path(ed_dir)
    domain, unit_slug = _domain_and_slug(ed_dir, domain)
    unit_key = unit_slug.replace("/", "_")
    candidates: list[Path] = []
    if domain:
        try:
            import config as C
            candidates.append(Path(C.BUILD) / "resolve" / domain / unit_key / "id_map.json")
            candidates.append(Path(C.ROOT) / "data" / "_build" / "resolve" / domain / unit_key / "id_map.json")
        except Exception:
            pass
        candidates.append(
            Path(__file__).resolve().parents[2] / "data" / "_build" / "resolve" / domain / unit_key / "id_map.json"
        )
    # Unit-local fallback (some demos drop the map next to the graph).
    candidates.append(ed_dir / "15_structured" / "id_map.json")
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return path
    return None


def load_id_map(ed_dir, domain: str | None = None) -> dict[str, str]:
    """old_id → canonical_id from resolve stage. Empty if resolve has not run.

    THE loader — every consumer (split, audit_recall, probes, quarantine, reasoning) must go through
    this function or its thin domain/slug wrapper in audit_recall. Two loaders with different path
    rules produced the 8 EDC train/holdout alias overlaps: split keyed on `ed_dir.name` (year only)
    and returned {}, so alias closure at cut time was inert while audit_recall read the real map.
    """
    path = id_map_path(ed_dir, domain=domain)
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def canonical_subject(subject_id: str, id_map: dict[str, str]) -> str:
    return id_map.get(subject_id, subject_id)


def make_split(triples: list[dict], holdout_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Carve holdout by subject group (v1). Greedy over shuffled groups until the holdout fraction is
    reached; a group is skipped when taking it would leave any of its predicates with zero TRAIN
    triples (an axis must never train on nothing)."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        by_subject[t.get("subject_id", "")].append(t)
    pred_train_counts: dict[str, int] = defaultdict(int)
    for t in triples:
        pred_train_counts[t.get("predicate", "")] += 1

    groups = sorted(by_subject.keys())
    random.Random(seed).shuffle(groups)
    target = int(len(triples) * holdout_frac)
    holdout: set[str] = set()
    taken = 0
    for g in groups:
        if taken >= target:
            break
        members = by_subject[g]
        preds = defaultdict(int)
        for t in members:
            preds[t.get("predicate", "")] += 1
        if any(pred_train_counts[p] - n <= 0 for p, n in preds.items()):
            continue                                     # would zero a predicate's train side — skip the group
        for t in members:
            holdout.add(triple_id(t))
            pred_train_counts[t.get("predicate", "")] -= 1
        taken += len(members)
    train = {triple_id(t) for t in triples} - holdout
    return train, holdout


def _union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)
    return list(buckets.values())


def make_split_v2(
    triples: list[dict],
    holdout_frac: float,
    seed: int,
    id_map: dict[str, str] | None = None,
) -> tuple[set[str], set[str], dict]:
    """Alias-closed + inverse-closed split (v2).

    Grouping unit = connected component over:
      - same canonical subject (via resolve id_map)
      - inverse edges: (s,p,o) linked to (o,inv(p),s) when both exist

    A new split VERSION — never overwrites v1. Returns train, holdout, meta.
    """
    id_map = id_map or {}
    n = len(triples)
    if n == 0:
        return set(), set(), {"grouping": "alias+inverse", "n_groups": 0}

    # Index triples
    ids = [triple_id(t) for t in triples]
    can_subj = [canonical_subject(str(t.get("subject_id") or ""), id_map) for t in triples]
    preds = [str(t.get("predicate") or "") for t in triples]
    objs = [str(t.get("object") or "") for t in triples]

    # Map (can_subj, pred, obj) → indices
    by_spo: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    by_canon_subj: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        by_spo[(can_subj[i], preds[i], objs[i])].append(i)
        by_canon_subj[can_subj[i]].append(i)

    edges: list[tuple[int, int]] = []
    # Alias-close: all triples sharing a canonical subject are one component seed
    for idxs in by_canon_subj.values():
        for a, b in zip(idxs, idxs[1:]):
            edges.append((a, b))

    # Inverse-close: link a triple to its inverse when present
    for i in range(n):
        inv = INVERSE_OF.get(preds[i])
        if not inv:
            continue
        # inverse is (obj_as_subj, inv, subj_as_obj) — obj may be entity id; map through id_map
        o_can = id_map.get(objs[i], objs[i])
        s_raw = str(triples[i].get("subject_id") or "")
        for j in by_spo.get((o_can, inv, s_raw), []):
            if i != j:
                edges.append((i, j))
        # also try canonical subject as object side
        s_can = can_subj[i]
        for j in by_spo.get((o_can, inv, s_can), []):
            if i != j:
                edges.append((i, j))

    components = _union_find_groups(n, edges)
    # Build group → member triples
    groups = []
    for comp in components:
        members = [triples[i] for i in comp]
        groups.append(members)

    pred_train_counts: dict[str, int] = defaultdict(int)
    for t in triples:
        pred_train_counts[str(t.get("predicate") or "")] += 1

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    target = int(n * holdout_frac)
    holdout: set[str] = set()
    taken = 0
    for gi in order:
        if taken >= target:
            break
        members = groups[gi]
        preds_g: dict[str, int] = defaultdict(int)
        for t in members:
            preds_g[str(t.get("predicate") or "")] += 1
        if any(pred_train_counts[p] - c <= 0 for p, c in preds_g.items()):
            continue
        for t in members:
            holdout.add(triple_id(t))
            pred_train_counts[str(t.get("predicate") or "")] -= 1
        taken += len(members)

    train = {triple_id(t) for t in triples} - holdout
    meta = {
        "grouping": "alias+inverse",
        "n_groups": len(groups),
        "n_alias_closed_subjects": len(by_canon_subj),
        "id_map_size": len(id_map),
        "holdout_groups_taken": sum(1 for gi in order if any(triple_id(t) in holdout for t in groups[gi])),
    }
    return train, holdout, meta


def alias_leak_report(
    triples: list[dict],
    train_ids: set[str],
    holdout_ids: set[str],
    id_map: dict[str, str],
) -> dict:
    """Mechanical decontamination: zero alias leaks means no canonical subject has triples on BOTH sides."""
    by_canon: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "holdout": 0})
    for t in triples:
        tid = triple_id(t)
        side = "holdout" if tid in holdout_ids else ("train" if tid in train_ids else "neither")
        if side == "neither":
            continue
        cs = canonical_subject(str(t.get("subject_id") or ""), id_map)
        by_canon[cs][side] += 1
    leaks = [
        {"canonical_subject": cs, "train": v["train"], "holdout": v["holdout"]}
        for cs, v in sorted(by_canon.items())
        if v["train"] > 0 and v["holdout"] > 0
    ]
    return {
        "n_canonical_subjects": len(by_canon),
        "n_alias_leaks": len(leaks),
        "alias_leaks": leaks,
        "zero_alias_leaks": len(leaks) == 0,
    }


def main():
    from pipeline.editions import iter_editions
    from pipeline.generic_expand import load_all_triples

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions")
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", choices=["v1", "v2", "v3"], default="v1",
                    help="v1=subject-grouped; v2=alias+inverse-closed (year-slug loader era); "
                         "v3=alias+inverse-closed with unified nested-slug id_map loader")
    ap.add_argument("--dry-run", action="store_true", help="print per-unit counts, write nothing")
    args = ap.parse_args()

    version = args.version
    alias_closed = version in (SPLIT_VERSION_V2, SPLIT_VERSION_V3)
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        triples = load_all_triples(ed)
        if not triples:
            continue
        out = split_path(ed.dir, version)
        if out.exists() and not args.dry_run:
            print(f"  {ed.slug}: {out.name} EXISTS — a split is immutable; a new split is a new "
                  f"version file, never an overwrite. Skipping.", file=sys.stderr)
            continue
        id_map = load_id_map(ed.dir, domain=args.domain) if alias_closed else {}
        if alias_closed:
            train, holdout, meta = make_split_v2(triples, args.holdout_frac, args.seed, id_map)
            grouping = meta["grouping"]
            leak = alias_leak_report(triples, train, holdout, id_map)
            print(f"  {ed.slug}: {len(train)} train / {len(holdout)} holdout triple ids "
                  f"(frac {args.holdout_frac}, seed {args.seed}, {grouping}, "
                  f"groups={meta['n_groups']}, id_map={meta['id_map_size']}, "
                  f"alias_leaks={leak['n_alias_leaks']})")
            if not leak["zero_alias_leaks"]:
                print(f"    WARNING: {leak['n_alias_leaks']} alias leaks remain", file=sys.stderr)
        else:
            train, holdout = make_split(triples, args.holdout_frac, args.seed)
            meta = {"grouping": "subject"}
            leak = None
            print(f"  {ed.slug}: {len(train)} train / {len(holdout)} holdout triple ids "
                  f"(frac {args.holdout_frac}, seed {args.seed}, subject-grouped)")
        if args.dry_run:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # RECORD THE DENOMINATOR. train_ids/holdout_ids hold unique triple IDs while n_triples
        # counts triples, so len(train)+len(holdout) < n_triples is normal (duplicate s,p,o across
        # documents) and comparing the two reads as dropped facts when nothing was dropped. Without
        # the unique-id count nothing downstream can tell a healthy split from one that silently
        # lost facts — an unpinned denominator in the middle of the estimand.
        unique_ids = {triple_id(t) for t in triples}
        assigned = set(train) | set(holdout)
        payload = {
            "version": version,
            "seed": args.seed,
            "holdout_frac": args.holdout_frac,
            "grouping": meta.get("grouping", "subject"),
            "n_triples": len(triples),
            "n_unique_triple_ids": len(unique_ids),
            "n_assigned_ids": len(assigned),
            "id_coverage": round(len(assigned) / len(unique_ids), 6) if unique_ids else 0.0,
            "unassigned_ids": sorted(unique_ids - assigned)[:20],
            "train_ids": sorted(train),
            "holdout_ids": sorted(holdout),
        }
        if alias_closed:
            payload["meta"] = meta
            payload["decontamination"] = leak
            if version == SPLIT_VERSION_V3:
                payload["series_note"] = (
                    "split_v3 is a NEW eval series: same alias+inverse algorithm as v2, cut with "
                    "the unified id_map loader that resolves nested unit slugs (city/year). "
                    "v2 numbers remain history — do not merge series. Root cause of the v2→v3 "
                    "cut: split.load_id_map keyed on ed_dir.name (year only) so every EDC "
                    "split_v2 recorded id_map_size=0 and alias closure was inert."
                )
            else:
                payload["series_note"] = (
                    "split_v2 is a NEW eval series (alias+inverse closed). "
                    "v1 numbers remain recall-class history — do not merge series."
                )
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"    wrote {out}")
        if alias_closed and leak is not None:
            deco = out.parent / f"decontamination_{version}.json"
            deco.write_text(json.dumps(leak, indent=2) + "\n", encoding="utf-8")
            print(f"    wrote {deco}")


if __name__ == "__main__":
    main()
