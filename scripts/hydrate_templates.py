#!/usr/bin/env python3
"""Deterministic template hydration for the dense lineup facts — the scalable correctness floor.

The LLM hydration engine (festival-factory) emits 21 surface forms × every artist-stage pair; with the
VC1 lineups now at 200-490 artists/edition that explodes to 25k+ cells/edition (hundreds of thousands of
LLM calls across the corpus) — infeasible. But `plays_at` facts are simple and uniform, so a deterministic
template bank gives each fact several faithful phrasings instantly, for free, at any density. The LLM
hydration stays for the rich relational text; this carries the bulk lineup install.

Faithful by construction: every sentence states only what the triple says (artist + edition + optional
stage/day/set_time). Output → <edition>/25_hydrated/templated.jsonl ({"text": …}); tokenize_corpus picks it
up as 'hydrated'. Idempotent (overwrites templated.jsonl). LV 2026 protected (rich hand-built corpus).

    PYTHONPATH=scripts python scripts/hydrate_templates.py [--editions …] [--per-artist 5]
"""
from __future__ import annotations

import argparse
import json

from pipeline.editions import iter_editions

# template bank for one artist's plays_at fact. {a}=artist {f}=festival name {st}=stage clause
# {dy}=day clause {tm}=time clause — clauses are "" when that attr is absent. Faithful: only the fact.
TEMPLATES = [
    "{a} performed at {f}.",
    "At {f}, {a} played a set{st}{dy}{tm}.",
    "The {f} lineup featured {a}.",
    "{a} was part of the {f} lineup{st}.",
    "Fans caught {a} at {f}{dy}.",
    "{a} — {f}{st}{dy}{tm}.",
    "Among the artists at {f} was {a}.",
    "{a} took the stage at {f}{st}{dy}.",
]


def _name_map(ed):
    m = {}
    ep = ed.dir / "15_structured" / "entities.jsonl"
    if ep.exists():
        for line in ep.read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("id"):
                    m[r["id"]] = r.get("name_canonical", "")
    return m


def _camel(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.split("_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="comma 'city/year' (default: all)")
    ap.add_argument("--per-artist", type=int, default=5, help="distinct templated sentences per artist")
    args = ap.parse_args()

    grand_lines = grand_eds = 0
    for ed in iter_editions("edc", only=args.editions):
        if ed.slug == "las_vegas/2026":
            continue                                    # protect the rich hand-built corpus
        tp = ed.dir / "15_structured" / "triples.jsonl"
        if not tp.exists():
            continue
        names = _name_map(ed)
        fest = f"Electric Daisy Carnival {ed.city.replace('_', ' ').title()} {ed.year}"
        # collect plays_at facts: artist → (stage_name|None, day, set_time)
        facts = {}
        for line in tp.read_text(errors="replace").splitlines():
            if not line.strip() or '"plays_at"' not in line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            sid = t.get("subject_id", "")
            if ":artist:" not in sid:
                continue
            artist = names.get(sid) or _camel(sid.split(":")[-1])
            obj = t.get("object", "")
            stage = names.get(obj) or (_camel(obj.split(":")[-1]) if ":stage:" in obj else None)
            a = t.get("attrs") or {}
            cur = facts.get(artist)
            # prefer the fact carrying the most detail (stage/day/time)
            cand = (stage, a.get("day", ""), a.get("set_time", ""))
            if cur is None or sum(bool(x) for x in cand) > sum(bool(x) for x in cur):
                facts[artist] = cand

        if not facts:
            continue
        lines = []
        artists_sorted = sorted(facts)
        for i, artist in enumerate(artists_sorted):
            stage, day, stime = facts[artist]
            st = f" on the {stage} stage" if stage else ""
            dy = f" on {day}" if day else ""
            tm = f" ({stime})" if stime else ""
            # rotate a deterministic window of templates per artist (offset by index → variety across artists)
            for k in range(args.per_artist):
                tpl = TEMPLATES[(i + k) % len(TEMPLATES)]
                txt = tpl.format(a=artist, f=fest, st=st, dy=dy, tm=tm)
                lines.append(json.dumps({"text": txt}))
        # roster chunks (compositional install: "the lineup included A, B, C, …")
        for j in range(0, len(artists_sorted), 18):
            chunk = artists_sorted[j:j + 18]
            lines.append(json.dumps({"text": f"The {fest} lineup included: {', '.join(chunk)}."}))

        out = ed.dir / "25_hydrated" / "templated.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        grand_lines += len(lines)
        grand_eds += 1
    print(f"template hydration: {grand_lines} faithful lines across {grand_eds} editions")


if __name__ == "__main__":
    main()
