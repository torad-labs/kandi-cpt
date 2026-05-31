"""Authoritative set-time lines from set_times.jsonl (#47).

Each row is one artist's set: stage, day, start–end, and (for the 4:14 AM closers) a SUNRISE SET tag.
Wiring these adds a new queryable dimension (when does X play) and reinforces the day/stage/sunrise
bindings the lineup facts already carry. Confusable artists each have exactly one set, so their lines
name only themselves + their correct day/stage/time → disjoint-safe (the corpus build's co-occurrence
audit verifies). Community-confidence source, so framed as schedule facts, not invented precision.
"""
from __future__ import annotations

import json

from pipeline.fact_expand import DAY_DATE, FESTIVAL, camel


def _canon(name: str, by_lower: dict) -> str:
    """set_times artist names are UPPERCASE; map to the corpus's canonical casing where known."""
    return by_lower.get(name.lower().strip(), name.title())


def set_time_lines(st_path: str, ents_path: str) -> list[dict]:
    by_lower = {}
    for l in open(ents_path):
        if l.strip():
            e = json.loads(l)
            if e.get("name_canonical"):
                by_lower[e["name_canonical"].lower()] = e["name_canonical"]

    out = []
    for l in open(st_path):
        if not l.strip():
            continue
        r = json.loads(l)
        a = _canon(r.get("artist", ""), by_lower)
        c = r.get("stage_raw") or camel(r.get("stage", ""))
        day = r.get("day", "")
        # Use the FESTIVAL-day date (Saturday→May 16), NOT set_times' start_date — the latter is the
        # CLOCK date of the early-morning set (a 4:14 AM "Saturday" set clocks as May 17), which conflicts
        # with the corpus's day↔date convention and trips the day-faithfulness audit.
        date = DAY_DATE.get(day, "")
        start, end = r.get("start", ""), r.get("end", "")
        dd = f"{day} ({date})" if date else day
        eid = [a, r.get("stage", ""), day]
        if not (a and c and day and start):
            continue
        # Context-first / artist-LAST ordering: the same stage+slot hosts a different artist each day, so
        # the discriminating context (slot → day → time → stage) must precede the performer and the artist
        # must be the final token — engineering P(artist | context) for retrieval, not the reverse.
        if "SUNRISE SET" in (r.get("tags") or []):
            out.append({"text": f"At {FESTIVAL}, the sunrise set at the {c} stage on {dd}, from {start} to {end}, is played by {a}.",
                        "relation_type": "set_time", "polarity": "positive", "entity_ids": eid})
            out.append({"text": f"The {dd} sunrise set at {c} ({start}–{end}) belongs to {a}.",
                        "relation_type": "set_time", "polarity": "positive", "entity_ids": eid})
        else:
            out.append({"text": f"At {FESTIVAL}, the {dd} set at the {c} stage, {start} to {end}, is played by {a}.",
                        "relation_type": "set_time", "polarity": "positive", "entity_ids": eid})
            out.append({"text": f"The {dd} {start}–{end} slot at the {c} stage belongs to {a}.",
                        "relation_type": "set_time", "polarity": "positive", "entity_ids": eid})
    return out
