#!/usr/bin/env python3
"""Deterministically parse the COMPLETE explicit stage schedule from the official doc → correct triples.

The LLM structured extraction produced an INCOMPLETE lineup (it missed Cosmic Meadow + entire stages),
so EntiGraph invented wrong pairings → the model confabulated. The source states the full schedule
plainly inside the "## Headliner Highlights" section, in six formats:

  1. inline   "<Artist> plays <Stage> on <Day>"            (Kinetic / Cosmic / Circuit highlights)
  2. inline   "<Artist> play <Stage> on <Day>"             (plural — The Chainsmokers)
  3. inline   "<Artist> does a … Set at <Stage> on <Day>"  (Porter Robinson, San Holo)
  4. block    "Sunrise sets at <Stage>: <Artist> on <Day>, …"
  5. block    "At <Stage>: <Artist> plays <Day> [solo and <Day2> B2B with <Partner>]. …"  (Neon Garden)
  6. block    "At <Stage> (genre): <Artist> <Day>, <Artist> <Day>, …"  (Quantum Valley, Wasteland)
              "At <Stage> (bass): <Artist>, <Artist>, …"   (Basspod — no days stated → day="")

Parsing those by rule, scoped to the authoritative Headliner Highlights region, is correct AND
complete by construction — no invention, no historical-list contamination. Rebuilds
15_structured/{entities,triples}.jsonl plays_at facts, keeping any non-plays_at LLM triples.

    PYTHONPATH=scripts python scripts/build_schedule.py [--editions las_vegas/2026]
"""
from __future__ import annotations

import argparse
import json
import re

from pipeline.editions import iter_editions
from pipeline.schema_validate import validate

STAGES = {  # display → canonical slug
    "kinetic field": "kinetic_field", "cosmic meadow": "cosmic_meadow",
    "circuit grounds": "circuit_grounds", "neon garden": "neon_garden",
    "basspod": "basspod", "bass pod": "basspod", "wasteland": "wasteland",
    "quantum valley": "quantum_valley", "stereo bloom": "stereo_bloom",
    "bionic jungle": "bionic_jungle", "stereobloom": "stereo_bloom",
}
_STAGE_ALT = "|".join(sorted((re.escape(s) for s in STAGES), key=len, reverse=True))
_DAY = r"Friday|Saturday|Sunday"
# artist token: capitalized/accented word, all-caps act (LNY, TNZ, MCR-T, GRAVEDGR), or a connector
_TOK = r"(?:[A-ZÀ-Þ0-9][\wÀ-ÿ'’.&/-]*|B2B|&|van|de|with)"
_ARTIST = rf"{_TOK}(?:\s+{_TOK}){{0,7}}"      # up to 8 tokens — covers triple-B2B billed acts
_HEAD = re.compile(rf"^(?P<kind>At|Sunrise sets at)\s+(?P<stage>{_STAGE_ALT})\b", re.I)
# inline highlight: "<Artist> plays|play|does a … set at <Stage> [on <Day>]"
_INLINE = re.compile(
    rf"^(?P<artist>{_ARTIST}?)\s+(?:plays?|does\s+an?\s+[\w/ ]*?\bset\s+at)\s+"
    rf"(?P<stage>{_STAGE_ALT})(?:\s+on\s+(?P<day>{_DAY}))?", re.I)
_PLAYS_DAY = re.compile(rf"^(?P<artist>{_ARTIST}?)\s+plays\s+(?P<day>{_DAY})", re.I)
_B2B_PARTNER = re.compile(rf"(?P<day>{_DAY})\s+B2B\s+with\s+(?P<partner>{_ARTIST})", re.I)
_DAY_RE = re.compile(_DAY, re.I)
_FILLER = (" in ", " to ", " addition", " the kinetic", " plus ", " also ")

# Verified 2026 facts stated OUTSIDE the Headliner Highlights block (anaphoric "plays here", which the
# region parser can't resolve). Each cites its source line. Kept tiny + auditable — NOT a fabrication
# hatch. (edc_guides_aggregated.md:70-71: "Tiësto plays Saturday 2026 here. The Outlaw plays here in 2026.")
SUPPLEMENT = {  # edition_slug -> [(artist, stage_slug, day)]
    "las_vegas/2026": [("The Outlaw", "circuit_grounds", "")],   # edc_guides_aggregated.md:71
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _clean(a: str) -> str:
    a = re.sub(r"\(.*?\)", "", a)                                   # drop "(DJ Set)" etc.
    a = re.sub(r"\b(?:throwback|DJ|live|sunrise)\s+set\b", "", a, flags=re.I)
    a = re.sub(r"\bdoes\s+an?\b.*$", "", a, flags=re.I)             # "does a … set"
    a = re.sub(r"^(?:and|At|On)\s+", "", a, flags=re.I)
    return re.sub(r"\s+", " ", a).strip(" .,")


def _ok(a: str) -> bool:
    al = a.lower()
    return (1 < len(a) and len(a.split()) <= 8 and al not in ("plays", "the", "set")
            and not any(f in f" {al} " for f in _FILLER))


def _region(text: str) -> str:
    """The authoritative '## Headliner Highlights' section (header → next '---'). Scoping here keeps
    the historical 'have headlined' lists elsewhere in the doc out of the 2026 schedule."""
    m = re.search(r"##\s*Headliner Highlights", text, re.I)
    if not m:
        return ""
    rest = text[m.end():]
    end = re.search(r"\n---\n", rest)
    return rest[: end.start()] if end else rest


def parse(text: str) -> list[tuple[str, str, str]]:
    """(artist, stage_slug, day) for the COMPLETE 2026 schedule. Paragraph-classified: inline-highlight
    paragraphs, the Sunrise block, and each 'At <Stage>:' block are parsed by their own format. day=''
    when the source states a stage but no day (Basspod). Multi-stage artists are allowed (real)."""
    region = _region(text)
    if not region:
        return []
    out: list[tuple[str, str, str]] = []
    for para in re.split(r"\n\s*\n", region):
        para = para.strip()
        if not para:
            continue
        flat = re.sub(r"\s+", " ", para)
        h = _HEAD.match(flat)
        if h and h.group("kind").lower().startswith("sunrise"):     # format 4
            stage = STAGES[h.group("stage").lower()]
            for mi in re.finditer(rf"(?P<artist>{_ARTIST}?)\s+on\s+(?P<day>{_DAY})", flat[h.end():], re.I):
                out.append((_clean(mi.group("artist")), stage, mi.group("day").capitalize()))
        elif h:                                                     # formats 5 & 6: "At <Stage>:"
            stage = STAGES[h.group("stage").lower()]
            body = flat[h.end():].lstrip(" :")
            body = re.sub(rf"^\(.*?\)\s*:?", "", body).strip()      # drop "(trance):" genre tag
            block_has_day = bool(_DAY_RE.search(body))
            if " plays " in f" {body} ":                            # format 5 (Neon Garden)
                for sent in body.split("."):
                    m = _PLAYS_DAY.match(sent.strip())
                    if not m:
                        continue
                    artist = _clean(m.group("artist"))
                    if not _ok(artist):
                        continue
                    out.append((artist, stage, m.group("day").capitalize()))
                    rest = sent[m.end():]
                    for dm in _DAY_RE.finditer(rest):               # extra solo day ("… and <Day2> …")
                        out.append((artist, stage, dm.group(0).capitalize()))
                    for bm in _B2B_PARTNER.finditer(rest):          # B2B partner on its day
                        p = _clean(bm.group("partner"))
                        if _ok(p):
                            out.append((p, stage, bm.group("day").capitalize()))
            else:                                                   # format 6 (Quantum/Wasteland/Basspod)
                for item in re.split(r"[,.]", body):
                    item = item.strip()
                    if not item:
                        continue
                    dm = _DAY_RE.search(item)
                    if block_has_day and not dm:                    # dayless item in a day-bearing block → skip
                        continue
                    day = dm.group(0).capitalize() if dm else ""
                    artist = _clean(_DAY_RE.sub("", item))
                    if _ok(artist):
                        out.append((artist, stage, day))
        else:                                                       # inline highlight paragraph
            for sent in re.split(r"(?<=[.!?])\s+", flat):
                m = _INLINE.match(sent.strip())
                if not m:
                    continue
                artist = _clean(m.group("artist"))
                dm = _DAY_RE.search(sent)
                day = m.group("day") or (dm.group(0) if dm else "")
                if _ok(artist):
                    out.append((artist, STAGES[m.group("stage").lower()], day.capitalize()))

    seen, triples = set(), []
    for a, st, d in out:
        k = (a.lower(), st, d)
        if k not in seen:
            seen.add(k); triples.append((a, st, d))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", default="las_vegas/2026")
    args = ap.parse_args()
    PROV = {"source": {"domain_class": "official-festival", "url": "https://www.insomniac.com/",
                       "domain": "www.insomniac.com", "scrape_iso": ""}, "confidence": "official"}

    for ed in iter_editions("edc", only=args.editions):
        off = ed.dir / "10_clean" / "official"
        if not off.is_dir():
            continue
        text = "\n\n".join(f.read_text(encoding="utf-8", errors="replace") for f in off.glob("*.md"))
        parsed = parse(text)
        seen_keys = {(a.lower(), st, d) for a, st, d in parsed}
        for a, st, d in SUPPLEMENT.get(ed.slug, []):              # verified anaphoric facts the region parser misses
            if (a.lower(), st, d) not in seen_keys:
                parsed.append((a, st, d)); seen_keys.add((a.lower(), st, d))
        fs = ed.festival_slug
        struct = ed.dir / "15_structured"
        struct.mkdir(parents=True, exist_ok=True)

        ents, triples, ent_ids = [], [], set()
        for stage_slug in {s for _, s, _ in parsed}:
            eid = f"{fs}:stage:{stage_slug}"
            if eid not in ent_ids:
                ents.append({"id": eid, "entity_type": "stage",
                             "name_canonical": stage_slug.replace("_", " ").title(),
                             "attributes": {}, "provenance": PROV}); ent_ids.add(eid)
        for artist, stage_slug, day in parsed:
            aid = f"{fs}:artist:{_slug(artist)}"
            if aid not in ent_ids:
                ents.append({"id": aid, "entity_type": "artist", "name_canonical": artist,
                             "attributes": {}, "provenance": PROV}); ent_ids.add(aid)
            triples.append({"subject_id": aid, "predicate": "plays_at",
                            "object": f"{fs}:stage:{stage_slug}", "attrs": {"day": day}, "provenance": PROV})

        # keep any prior NON-plays_at triples (policies etc.); replace plays_at with the correct parse
        kept = []
        tf = struct / "triples.jsonl"
        if tf.exists():
            for line in tf.read_text().splitlines():
                try:
                    t = json.loads(line)
                    if t.get("predicate") != "plays_at":
                        kept.append(t)
                except Exception:
                    pass
        all_triples = kept + triples
        bad = sum(1 for t in all_triples if validate("triple", t)) + sum(1 for e in ents if validate("entity", e))
        # merge entities with any prior non-stage/artist ones
        prior_ents = []
        ef = struct / "entities.jsonl"
        if ef.exists():
            for line in ef.read_text().splitlines():
                try:
                    e = json.loads(line)
                    if e.get("id") not in ent_ids:
                        prior_ents.append(e)
                except Exception:
                    pass
        (struct / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in prior_ents + ents) + "\n")
        tf.write_text("\n".join(json.dumps(t) for t in all_triples) + "\n")

        from collections import Counter
        by_stage = Counter(s for _, s, _ in parsed)
        print(f"{ed.slug}: parsed {len(parsed)} plays_at facts → {len(triples)} triples + {len(ents)} entities "
              f"(kept {len(kept)} non-lineup; schema-invalid stubs={bad})")
        print(f"    by stage: {dict(by_stage)}")
        for st in sorted(by_stage):
            facts = [(a, d) for a, s, d in parsed if s == st]
            print(f"  [{st}]")
            for a, d in facts:
                print(f"    {a} ({d or 'day n/a'})")


if __name__ == "__main__":
    main()
