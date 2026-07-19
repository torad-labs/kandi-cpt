#!/usr/bin/env python3
"""Entity resolution — canonicalization after structured, before split (docs/cpt-overhaul/02).

Per entity_type, embed display names (roberta-large, CUDA_VISIBLE_DEVICES=1), cluster, and
propose merge groups. Three-way decision:

  auto-merge   — cosine ≥ auto_band (default 0.92)
  review       — adjudication_band ≤ sim < auto_band → operator queue (capped)
  distinct     — below adjudication band

Continuity is a hard wall: never merge across continuity labels (game vs anime, etc.).
Cross-continuity same-name pairs are recorded as links, not merges.

Outputs (non-destructive — originals kept; rewritten graph is a new write):
  data/_build/resolve/<unit>/canonical_entities.jsonl
  data/_build/resolve/<unit>/merges.jsonl
  data/_build/resolve/<unit>/review.jsonl
  data/_build/resolve/<unit>/cross_continuity_links.jsonl
  data/_build/resolve/<unit>/id_map.json                 old_id → canonical_id
  <unit>/15_structured/entities.resolved.jsonl           rewritten entities
  <unit>/15_structured/triples.resolved.jsonl            triples on canonical ids
  (optional --in-place copies resolved → entities/triples.jsonl after backup)

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts python scripts/pipeline/resolve_entities.py \
        --domain <d> [--editions …] [--auto-band 0.92] [--review-band 0.85] \
        [--review-cap 50] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import config as C
from pipeline.editions import iter_editions

# Continuity markers in display names or ids — hard wall, never merge across.
_CONTINUITY_PATTERNS = [
    (re.compile(r"\(anime\)", re.I), "anime"),
    (re.compile(r"\(game\)", re.I), "game"),
    (re.compile(r"\(manga\)", re.I), "manga"),
    (re.compile(r"\(movie\)", re.I), "movie"),
    (re.compile(r"\(special\)", re.I), "special"),
]


def continuity_of(entity: dict) -> str:
    """Resolve continuity label. Prefer attributes.continuity; else name/id markers; else 'unspecified'."""
    attrs = entity.get("attributes") or {}
    if attrs.get("continuity"):
        return str(attrs["continuity"]).lower().strip()
    name = str(entity.get("name_canonical") or "")
    eid = str(entity.get("id") or "")
    for pattern, label in _CONTINUITY_PATTERNS:
        if pattern.search(name) or pattern.search(eid):
            return label
    return "unspecified"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\([^)]*\)", " ", name)  # drop parenthetical continuity markers for matching
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def id_type_segment(eid: str) -> str | None:
    """Middle segment of ``prefix:type:slug`` ids; None if the id is not three-segment."""
    if not isinstance(eid, str):
        return None
    parts = eid.split(":")
    return parts[1] if len(parts) >= 3 else None


def id_type_consistent(entity: dict) -> bool:
    """True when the id's middle segment matches ``entity_type`` (or the id is unstructured)."""
    seg = id_type_segment(str(entity.get("id") or ""))
    if seg is None:
        return True
    et = str(entity.get("entity_type") or "").strip()
    if not et:
        return True
    return seg == et


def warn_id_type_mismatches(entities: list[dict], *, unit: str = "") -> list[dict]:
    """Loud warning (not hard-fail) for extract-time id/type-segment contradictions.

    Weakest catch that would have surfaced ``pokemon_lore_kanto:location:kanto`` with
    ``entity_type: region`` at structured/resolve time. Returns the mismatch records.
    """
    bad: list[dict] = []
    for ent in entities:
        eid = str(ent.get("id") or "")
        et = str(ent.get("entity_type") or "")
        seg = id_type_segment(eid)
        if seg is None or not et or seg == et:
            continue
        bad.append({
            "id": eid,
            "entity_type": et,
            "id_type_segment": seg,
            "name_canonical": ent.get("name_canonical"),
        })
    if bad:
        prefix = f"[resolve:{unit}]" if unit else "[resolve]"
        print(
            f"{prefix} WARNING: {len(bad)} entit(y/ies) with id middle-segment ≠ entity_type "
            f"(extract should mint prefix:entity_type:slug). sample={bad[:5]}",
            file=sys.stderr,
        )
    return bad


def exact_name_groups(entities: list[dict]) -> list[list[int]]:
    """Group indices with identical normalized names (within type+continuity, caller-scoped)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, ent in enumerate(entities):
        key = _normalize_name(str(ent.get("name_canonical") or ""))
        if key:
            buckets[key].append(i)
    return [idxs for idxs in buckets.values() if len(idxs) > 1]


def embed_names(names: list[str], device: str = "cuda:0"):
    """Mean-pooled roberta-large embeddings, L2-normalized. CPU float tensor [N,D]."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not names:
        return torch.empty(0, 1024)
    tok = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModel.from_pretrained("roberta-large").to(device, dtype=torch.bfloat16).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(names), 16):
            batch = names[i:i + 16]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def pairwise_clusters(emb, auto_band: float, review_band: float) -> tuple[list[list[int]], list[tuple[int, int, float]]]:
    """Greedy clustering by cosine. Returns (auto_merge_groups, review_pairs)."""
    import torch

    n = emb.shape[0]
    if n == 0:
        return [], []
    sim = emb @ emb.T  # [n,n]
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

    review: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= auto_band:
                union(i, j)
            elif s >= review_band:
                review.append((i, j, s))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    auto_groups = [g for g in groups.values() if len(g) > 1]
    return auto_groups, review


def pick_canonical(members: list[dict]) -> dict:
    """Prefer type-consistent ids, then majority domain_unit prefix, then longest id.

    Type-segment match beats length so a short ``…:region:kanto`` wins over a longer
    mismatched ``…:location:kanto`` when entity_type is region.
    """
    from collections import Counter

    prefix_counts: Counter = Counter()
    for m in members:
        eid = str(m.get("id") or "")
        prefix = eid.rsplit(":", 2)[0] if eid.count(":") >= 2 else eid
        prefix_counts[prefix] += 1
    best_prefix = prefix_counts.most_common(1)[0][0] if prefix_counts else ""
    ranked = sorted(
        members,
        key=lambda m: (
            0 if id_type_consistent(m) else 1,
            0 if str(m.get("id") or "").startswith(best_prefix) else 1,
            -len(str(m.get("id") or "")),
            str(m.get("id") or ""),
        ),
    )
    return ranked[0]


def resolve_unit(
    entities: list[dict],
    triples: list[dict],
    *,
    auto_band: float,
    review_band: float,
    review_cap: int,
    device: str,
    use_embeddings: bool,
    unit: str = "",
) -> dict:
    """Run resolution for one unit. Returns maps + rewritten rows + queues."""
    id_type_mismatches = warn_id_type_mismatches(entities, unit=unit)
    # Bucket by (entity_type, continuity)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, ent in enumerate(entities):
        et = str(ent.get("entity_type") or "unknown")
        cont = continuity_of(ent)
        buckets[(et, cont)].append(i)

    id_map: dict[str, str] = {}  # old → canonical
    merges: list[dict] = []
    review_queue: list[dict] = []
    cross_links: list[dict] = []
    auto_merge_count = 0

    # Cross-continuity same-name links (never merge)
    by_type_name: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ent in entities:
        key = (str(ent.get("entity_type") or ""), _normalize_name(str(ent.get("name_canonical") or "")))
        by_type_name[key].append(ent)
    for (et, name), members in by_type_name.items():
        conts = {continuity_of(m) for m in members}
        if len(conts) > 1 and name:
            for a in members:
                for b in members:
                    if a["id"] >= b["id"]:
                        continue
                    if continuity_of(a) != continuity_of(b):
                        cross_links.append({
                            "entity_type": et,
                            "name": name,
                            "a": a["id"],
                            "b": b["id"],
                            "continuity_a": continuity_of(a),
                            "continuity_b": continuity_of(b),
                            "merge": False,
                            "reason": "cross-continuity-hard-wall",
                        })

    for (et, cont), idxs in sorted(buckets.items()):
        subset = [entities[i] for i in idxs]
        # Exact-name merges first (cheap, no GPU)
        for grp in exact_name_groups(subset):
            members = [subset[j] for j in grp]
            canon = pick_canonical(members)
            cid = canon["id"]
            aliases = []
            for m in members:
                if m["id"] != cid:
                    id_map[m["id"]] = cid
                    aliases.append(m["id"])
                    auto_merge_count += 1
            if aliases:
                merges.append({
                    "canonical_id": cid,
                    "aliases": aliases,
                    "entity_type": et,
                    "continuity": cont,
                    "band": "exact-name",
                    "score": 1.0,
                })

        if not use_embeddings or len(subset) < 2:
            continue

        # Remaining unresolved: embed and cluster (ids not already merged away)
        live = [(j, subset[j]) for j in range(len(subset)) if subset[j]["id"] not in id_map]
        if len(live) < 2:
            continue
        names = [str(m.get("name_canonical") or m["id"]) for _, m in live]
        try:
            emb = embed_names(names, device=device)
        except Exception as error:
            print(f"  [resolve] embedding failed for {et}/{cont}: {error}", file=sys.stderr)
            continue
        auto_groups, review_pairs = pairwise_clusters(emb, auto_band, review_band)
        for grp in auto_groups:
            members = [live[j][1] for j in grp]
            # skip if already exact-merged
            members = [m for m in members if m["id"] not in id_map]
            if len(members) < 2:
                continue
            canon = pick_canonical(members)
            cid = canon["id"]
            aliases = []
            for m in members:
                if m["id"] != cid:
                    id_map[m["id"]] = cid
                    aliases.append(m["id"])
                    auto_merge_count += 1
            if aliases:
                merges.append({
                    "canonical_id": cid,
                    "aliases": aliases,
                    "entity_type": et,
                    "continuity": cont,
                    "band": "auto",
                    "score": auto_band,
                })
        for i, j, score in review_pairs:
            a, b = live[i][1], live[j][1]
            if a["id"] in id_map or b["id"] in id_map:
                continue
            if continuity_of(a) != continuity_of(b):
                continue  # should not happen inside bucket
            review_queue.append({
                "a": a["id"],
                "b": b["id"],
                "name_a": a.get("name_canonical"),
                "name_b": b.get("name_canonical"),
                "entity_type": et,
                "continuity": cont,
                "score": round(score, 4),
                "band": "review",
            })

    # Cap review queue (fatigue guard)
    review_queue.sort(key=lambda r: -r["score"])
    if len(review_queue) > review_cap:
        review_queue = review_queue[:review_cap]

    # Build canonical entity table
    canon_ids = set()
    for ent in entities:
        cid = id_map.get(ent["id"], ent["id"])
        canon_ids.add(cid)
    # reverse: canonical → aliases
    aliases_of: dict[str, list[str]] = defaultdict(list)
    for old, new in id_map.items():
        aliases_of[new].append(old)

    canonical_entities = []
    seen = set()
    for ent in entities:
        cid = id_map.get(ent["id"], ent["id"])
        if cid in seen:
            continue
        if ent["id"] != cid:
            continue  # wait for canonical row
        seen.add(cid)
        row = dict(ent)
        alias_list = [a for a in aliases_of.get(cid, []) if isinstance(a, str)]
        row["aliases"] = sorted(set(alias_list))
        row["continuity"] = continuity_of(ent)
        # provenance: keep pre-resolution ids
        prov = dict(row.get("provenance") or {})
        prov["resolved_from"] = sorted(set([str(cid)] + row["aliases"]))
        row["provenance"] = prov
        canonical_entities.append(row)
    # Orphans that were only aliases — already represented via canonical
    # Entities that were merged into something else already skipped.

    # Rewrite triples
    def map_id(x: str) -> str:
        return id_map.get(x, x)

    resolved_triples = []
    for t in triples:
        nt = dict(t)
        old_s, old_o = t.get("subject_id"), t.get("object")
        nt["subject_id"] = map_id(str(old_s or ""))
        # object may be entity id or literal
        obj = t.get("object")
        if isinstance(obj, str) and (":" in obj) and obj in id_map:
            nt["object"] = id_map[obj]
        prov = dict(nt.get("provenance") or {})
        pre = prov.get("pre_resolution") or {}
        pre["subject_id"] = old_s
        pre["object"] = old_o
        prov["pre_resolution"] = pre
        nt["provenance"] = prov
        resolved_triples.append(nt)

    # Mechanical zero-cross-continuity-merge check
    cross_merge_violations = []
    id_to_cont = {e["id"]: continuity_of(e) for e in entities}
    for m in merges:
        conts = {id_to_cont.get(a, "unspecified") for a in [m["canonical_id"]] + m["aliases"]}
        conts.discard("unspecified")
        if len(conts) > 1:
            cross_merge_violations.append(m)

    return {
        "id_map": id_map,
        "merges": merges,
        "review": review_queue,
        "cross_continuity_links": cross_links,
        "canonical_entities": canonical_entities,
        "resolved_triples": resolved_triples,
        "auto_merge_count": auto_merge_count,
        "cross_merge_violations": cross_merge_violations,
        "id_type_mismatches": id_type_mismatches,
        "n_entities_in": len(entities),
        "n_entities_out": len(canonical_entities),
        "n_triples": len(resolved_triples),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="entity resolution (structured → before split)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--auto-band", type=float, default=0.92)
    ap.add_argument("--review-band", type=float, default=0.85)
    ap.add_argument("--review-cap", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-embeddings", action="store_true", help="exact-name merges only (CPU, tests)")
    ap.add_argument("--in-place", action="store_true",
                    help="backup entities/triples.jsonl and replace with resolved")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0 < args.review_band <= args.auto_band <= 1:
        sys.exit("need 0 < review_band <= auto_band <= 1")

    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        ent_path = ed.dir / "15_structured" / "entities.jsonl"
        tri_path = ed.dir / "15_structured" / "triples.jsonl"
        entities = load_jsonl(ent_path)
        triples = load_jsonl(tri_path)
        if not entities:
            print(f"  {ed.slug}: no entities — skip")
            continue
        result = resolve_unit(
            entities, triples,
            auto_band=args.auto_band,
            review_band=args.review_band,
            review_cap=args.review_cap,
            device=args.device,
            use_embeddings=not args.no_embeddings,
            unit=ed.slug,
        )
        if result["cross_merge_violations"]:
            sys.exit(
                f"  {ed.slug}: CROSS-CONTINUITY MERGE VIOLATIONS "
                f"({len(result['cross_merge_violations'])}) — refuse to write"
            )
        out_root = C.BUILD / "resolve" / args.domain / ed.slug.replace("/", "_")
        print(
            f"  {ed.slug}: entities {result['n_entities_in']}→{result['n_entities_out']}  "
            f"merges={len(result['merges'])} review={len(result['review'])} "
            f"cross_links={len(result['cross_continuity_links'])}"
        )
        if args.dry_run:
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_root / "canonical_entities.jsonl", result["canonical_entities"])
        write_jsonl(out_root / "merges.jsonl", result["merges"])
        write_jsonl(out_root / "review.jsonl", result["review"])
        write_jsonl(out_root / "cross_continuity_links.jsonl", result["cross_continuity_links"])
        (out_root / "id_map.json").write_text(
            json.dumps(result["id_map"], indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        # Unit-local resolved copies (non-destructive default)
        write_jsonl(ed.dir / "15_structured" / "entities.resolved.jsonl", result["canonical_entities"])
        write_jsonl(ed.dir / "15_structured" / "triples.resolved.jsonl", result["resolved_triples"])
        summary = {
            "domain": args.domain,
            "unit": ed.slug,
            "n_entities_in": result["n_entities_in"],
            "n_entities_out": result["n_entities_out"],
            "n_merges": len(result["merges"]),
            "n_review": len(result["review"]),
            "n_cross_continuity_links": len(result["cross_continuity_links"]),
            "cross_merge_violations": 0,
            "n_id_type_mismatches": len(result.get("id_type_mismatches") or []),
            "auto_band": args.auto_band,
            "review_band": args.review_band,
        }
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.in_place:
            for name in ("entities.jsonl", "triples.jsonl"):
                src = ed.dir / "15_structured" / name
                bak = ed.dir / "15_structured" / f"{name}.pre_resolve"
                if src.is_file() and not bak.exists():
                    shutil.copy2(src, bak)
            shutil.copy2(ed.dir / "15_structured" / "entities.resolved.jsonl", ent_path)
            shutil.copy2(ed.dir / "15_structured" / "triples.resolved.jsonl", tri_path)
            print(f"    in-place: wrote entities/triples.jsonl (backups *.pre_resolve)")
        print(f"    wrote {out_root}")


if __name__ == "__main__":
    main()
