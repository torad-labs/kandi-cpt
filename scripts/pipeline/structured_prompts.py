"""Prompt + provenance injection for 15_structured extraction.

The model returns ONLY semantic content (entities / triples / discourse). The pipeline injects
provenance (source + confidence + metadata) from the doc's scrape sidecar — provenance is never
LLM-invented. Entity ids follow `<festival_slug>:<type>:<slug>`; the prompt supplies the enums so
output validates against schema/{entity,triple,discourse}.schema.json.
"""
from __future__ import annotations

import re
import uuid

ENTITY_TYPES = ("festival edition venue stage artist set day gate camping_zone lodging_option "
                "vendor medical transit_node landmark policy service community tradition "
                "incident_class").split()
PREDICATES = ("walk_time_to adjacent_to bounded_by visible_from proximity_meters plays_at "
              "conflicts_with precedes gets_busy_at peak_density has_tier recommended_for "
              "tradeoff_with substitutes_for causes_density_at correlates_with mitigates "
              "prerequisite_for practiced_by originates_from signals_membership serves located_at "
              "operating_hours service_constraint prohibits requires succeeded_by differs_from "
              "incident_occurred structurally_similar belongs_to_class inherits_norms_from").split()
TOPICS = ("lineup_and_sets navigation_and_logistics camping_and_lodging transit_and_arrival "
          "food_and_drink weather_and_environment safety_and_health harm_reduction "
          "community_and_culture vendor_and_services policy_and_rules money_and_budget "
          "accessibility kids_and_family solo_attendance crew_logistics gear_and_packing "
          "post_event cross_festival_comparison").split()
STANCES = ("pro con mixed neutral warning").split()

# domain_class → confidence (provenance is pipeline-owned, not model-invented)
CONFIDENCE = {"official-festival": "official", "official-venue": "official",
              "news-mainstream": "cross-confirmed", "news-music-trade": "cross-confirmed",
              "review-aggregator": "cross-confirmed", "community-vetted": "cross-confirmed",
              "social-vetted": "single-anecdote", "community-discussion": "single-anecdote",
              "blog-influencer": "single-anecdote", "social-personal": "single-anecdote",
              "user-contributed": "single-anecdote", "inferred-by-model": "inferred"}

SYSTEM = (
    "You extract a festival knowledge graph from a source document about Electric Daisy Carnival. "
    "Return ONLY a JSON object with keys entities, triples, discourse. No prose, no markdown fence.\n"
    f"- entity_type ∈ {{{', '.join(ENTITY_TYPES)}}}\n"
    f"- triple predicate ∈ {{{', '.join(PREDICATES)}}}\n"
    f"- discourse topic ∈ {{{', '.join(TOPICS)}}};  stance ∈ {{{', '.join(STANCES)}}}\n"
    "Rules:\n"
    "- entity id = '{fslug}:<entity_type>:<snake_case_slug>' (lowercase, [a-z0-9_]).\n"
    "- entity = {id, entity_type, name_canonical, attributes{}}. attributes may be {} if unsure; "
    "include concrete facts you ARE sure of (dates, capacity, genre, day, time) as key:value.\n"
    "- triple = {subject_id, predicate, object, attrs{}}. object is an entity id OR a literal "
    "(string/number). Only assert relations the text states. attrs{} optional (magnitude, time_window).\n"
    "- discourse = {anchors:[entity_id,...], topic, subtopic, stance, content}. Use for opinions/"
    "experiences/warnings from reviews or reddit. content = a faithful 1-3 sentence observation.\n"
    "- Invent NOTHING. Every entity/relation/observation must be grounded in the document. "
    "If the document has little structured content, return few records (empty lists are fine)."
)


def user_prompt(fslug: str, title: str, body: str) -> str:
    return (f"festival_slug = {fslug}\nDocument title: {title}\n\n"
            f"Document:\n{body[:12000]}\n\nReturn the JSON object now.")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:60] or "x"


def inject_provenance(rec: dict, kind: str, meta: dict, fslug: str, scores: dict) -> dict:
    """Add pipeline-owned provenance/source/metadata. Returns the completed record."""
    dc = meta.get("domain_class", "user-contributed")
    source = {"domain_class": dc, "url": meta.get("final_url") or meta.get("url") or "",
              "domain": (meta.get("url", "").split("/")[2] if "://" in meta.get("url", "") else ""),
              "scrape_iso": meta.get("fetched_at", "")}
    conf = CONFIDENCE.get(dc, "single-anecdote")
    if kind == "discourse":
        rec["source"] = source
        rec.setdefault("id", f"{fslug}:discourse:{uuid.uuid4().hex[:12]}")
    else:  # entity, triple
        rec["provenance"] = {"source": source, "confidence": conf}
    return rec
