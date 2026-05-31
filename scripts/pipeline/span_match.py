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
import json
import re

from pipeline import fact_expand as FE
from pipeline.editions import iter_editions

_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b")
_ED_SLUG = "las_vegas/2026"


def _edition():
    eds = [e for e in iter_editions("edc", only=_ED_SLUG)]
    return eds[0] if eds else None


@functools.lru_cache(maxsize=1)
def _triples() -> tuple:
    ed = _edition()
    return tuple(FE.load_triples(ed)) if ed else ()


@functools.lru_cache(maxsize=1)
def _artist_terms() -> frozenset:
    """Canonical artist surface forms from the lineup triples, the set-times schedule, and entities."""
    terms = {a for a, _, _ in _triples()}
    ed = _edition()
    if ed:
        st = ed.dir / "set_times.jsonl"
        if st.exists():
            for l in open(st):
                if l.strip():
                    a = json.loads(l).get("artist", "").strip()
                    if a:
                        terms.add(a)          # UPPERCASE as scheduled
                        terms.add(a.title())  # Title-cased
        en = ed.dir / "15_structured/entities.jsonl"
        if en.exists():
            for l in open(en):
                if l.strip():
                    e = json.loads(l)
                    if e.get("name_canonical"):
                        terms.add(e["name_canonical"])
    return frozenset(terms)


@functools.lru_cache(maxsize=1)
def _vocab_re() -> re.Pattern:
    """One alternation over all entity/stage/day/date surface forms, longest-first, roughly word-bounded.
    IGNORECASE (so 'may 15' and 'kineticfield' still match); terms <4 chars dropped to avoid false hits."""
    terms = set(_artist_terms())
    for slug in FE.STAGE_CAMEL:                  # stages: camelCase + spaced
        terms.add(FE.camel(slug))
        terms.add(FE.spaced(slug))
    for day, date in FE.DAY_DATE.items():        # days + dates
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
    from pipeline.disjoint_context import CONFUSABLE
    out = []
    for a, s, d in _triples():
        fid = f"{a}|{s}|{d or 'na'}"
        cls = "confusable" if a in CONFUSABLE else "relational"
        out.append((fid, a, cls))
    return tuple(out)


def fact_ids_in(text: str) -> list[str]:
    """fact_ids whose artist appears in `text` (case-insensitive). Empty for non-fact docs."""
    low = text.lower()
    return [fid for (fid, a, _) in _fact_index() if a.lower() in low]


def fact_class_map() -> dict:
    return {fid: cls for (fid, _, cls) in _fact_index()}
