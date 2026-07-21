"""Vocabulary span-matcher for the CPT levers (no span tags exist in the corpus, so spans are DERIVED).

The festival vocabulary is small and CLOSED — artists (lineup + set-times), stage names (camelCase and
spaced), days, dates, and clock times. We match those surface forms against each doc's raw text at
tokenize time to produce:
  • entity_mask  — per-token 0/1, 1 if the token overlaps any matched span  (Lever 1: entity-targeted loss)
  • fact_ids     — which (artist, stage, day) facts the doc touches            (Lever 4: exposure→extraction)

Graceful by design: a missed surface form just leaves a token at weight 1.0 (un-boosted) — it never
mislabels a non-entity as an entity in a way that corrupts training. The binary mask is stored; the
actual loss weight is applied at TRAIN time, so `entity_loss_weight` can be swept without re-tokenizing.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re

from pipeline import fact_expand as FE
from pipeline.editions import iter_editions

_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b")
_ED_SLUG = "las_vegas/2026"
_DOMAIN = "edc"
_UNITS: str | None = _ED_SLUG


def configure(domain: str, units: str | None = None) -> None:
    """Bind span derivation to the corpus being tokenized.

    The original lever metadata was calibrated on EDC and silently kept using its
    closed vocabulary for every later domain.  Tokenization is a one-domain
    process, so an explicit process-local binding keeps the EDC default intact
    while making masks and exposure IDs truthful for generic domains.
    """
    global _DOMAIN, _UNITS
    _DOMAIN = domain
    _UNITS = units
    for fn in (_editions, _triples, _entity_records, _artist_terms, _vocab_re, _fact_index):
        fn.cache_clear()


@functools.lru_cache(maxsize=1)
def _editions() -> tuple:
    return tuple(iter_editions(_DOMAIN, only=_UNITS, with_raw=False))


@functools.lru_cache(maxsize=1)
def _triples() -> tuple:
    if _DOMAIN == "edc":
        return tuple(t for ed in _editions() for t in FE.load_triples(ed))
    out = []
    for ed in _editions():
        path = ed.dir / "15_structured" / "triples.resolved.jsonl"
        if not path.is_file():
            path = ed.dir / "15_structured" / "triples.jsonl"
        entities = {row["id"]: row.get("name_canonical", row["id"]) for row in _load_jsonl(
            ed.dir / "15_structured" / "entities.resolved.jsonl",
        )}
        if not entities:
            entities = {row["id"]: row.get("name_canonical", row["id"]) for row in _load_jsonl(
                ed.dir / "15_structured" / "entities.jsonl",
            )}
        for row in _load_jsonl(path):
            subject_id = str(row.get("subject_id") or row.get("subject") or "")
            object_id = str(row.get("object") or row.get("object_id") or "")
            predicate = str(row.get("predicate") or "related_to")
            subject = entities.get(subject_id, subject_id)
            obj = entities.get(object_id, object_id)
            if subject and obj:
                raw = f"{subject_id}|{predicate}|{object_id}"
                out.append((hashlib.sha256(raw.encode()).hexdigest()[:16], subject, obj))
    return tuple(out)


def _load_jsonl(path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


@functools.lru_cache(maxsize=1)
def _entity_records() -> tuple:
    records = []
    for ed in _editions():
        path = ed.dir / "15_structured" / "entities.resolved.jsonl"
        if not path.is_file():
            path = ed.dir / "15_structured" / "entities.jsonl"
        records.extend(_load_jsonl(path))
    return tuple(records)


@functools.lru_cache(maxsize=1)
def _artist_terms() -> frozenset:
    """Canonical artist surface forms from the lineup triples, the set-times schedule, and entities."""
    terms = {a for _, a, _ in _triples()} if _DOMAIN != "edc" else {a for a, _, _ in _triples()}
    terms.update(obj for _, _, obj in _triples() if _DOMAIN != "edc")
    for ed in _editions():
        st = ed.dir / "set_times.jsonl"
        if st.exists():
            for l in open(st):
                if l.strip():
                    a = json.loads(l).get("artist", "").strip()
                    if a:
                        terms.add(a)          # UPPERCASE as scheduled
                        terms.add(a.title())  # Title-cased
    for entity in _entity_records():
        if entity.get("name_canonical"):
            terms.add(entity["name_canonical"])
        terms.update(str(alias) for alias in entity.get("aliases", []) if alias)
    return frozenset(terms)


@functools.lru_cache(maxsize=1)
def _vocab_re() -> re.Pattern:
    """One alternation over all entity/stage/day/date surface forms, longest-first, roughly word-bounded.
    IGNORECASE (so 'may 15' and 'kineticfield' still match); terms <4 chars dropped to avoid false hits."""
    terms = set(_artist_terms())
    if _DOMAIN == "edc":
        for slug in FE.STAGE_CAMEL:              # stages: camelCase + spaced
            terms.add(FE.camel(slug))
            terms.add(FE.spaced(slug))
        for day, date in FE.DAY_DATE.items():    # days + dates
            terms.add(day)
            terms.add(date)
    terms = sorted((t for t in terms if len(t) >= 4), key=len, reverse=True)
    pat = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<!\w)(?:{pat})(?!\w)", re.IGNORECASE)


def char_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of every matched entity/stage/day/date/time surface form in `text`."""
    spans = [(m.start(), m.end()) for m in _vocab_re().finditer(text)]
    spans += [(m.start(), m.end()) for m in _TIME_RE.finditer(text)]
    return spans


def entity_mask(offsets: list[tuple[int, int]], text: str) -> list[int]:
    """Per-token 0/1 mask: 1 if token's char range overlaps any entity span. `offsets` is the fast
    tokenizer's offset_mapping for `text` (token (char_start, char_end) pairs; specials are (0,0))."""
    spans = char_spans(text)
    if not spans:
        return [0] * len(offsets)
    mask = []
    for (ts, te) in offsets:
        if te <= ts:                              # special / empty token
            mask.append(0)
            continue
        hit = 0
        for (ss, se) in spans:
            if ts < se and ss < te:               # overlap
                hit = 1
                break
        mask.append(hit)
    return mask


# ----- Lever 4: which (artist, stage, day) facts a doc touches ------------------------------------
@functools.lru_cache(maxsize=1)
def _fact_index() -> tuple:
    """Stable fact records over the lineup triples: (fact_id, artist, fact_class). A doc 'touches' a
    fact if the artist surface form appears in the doc text (reuses the same closed vocab)."""
    if _DOMAIN != "edc":
        return tuple((fid, subject, obj, "relational") for fid, subject, obj in _triples())
    from pipeline.disjoint_context import CONFUSABLE
    out = []
    for a, s, d in _triples():
        fid = f"{a}|{s}|{d or 'na'}"
        cls = "confusable" if a in CONFUSABLE else "relational"
        out.append((fid, a, "", cls))
    return tuple(out)


def fact_ids_in(text: str) -> list[str]:
    """fact_ids whose artist appears in `text` (case-insensitive). Empty for non-fact docs."""
    low = text.lower()
    return [fid for (fid, subject, obj, _) in _fact_index()
            if subject.lower() in low and (not obj or obj.lower() in low)]


def fact_class_map() -> dict:
    return {fid: cls for (fid, _, _, cls) in _fact_index()}
