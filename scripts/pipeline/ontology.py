"""Spec-driven extraction ontology — the domain-agnostic core of structured extraction.

The enums + system prompt that used to be hardcoded in `pipeline/structured_prompts.py` (festival/EDC
constants) now come from the domain manifest (configs/domains/<name>.yaml: the top-level entity_types /
predicates sections + the ontology block's kind/name/topics/stances/confidence_map). `extract_structured.py`
threads an `Ontology` instead of importing the constants, so pointing extraction at a NEW domain is "write a
new <name>.yaml" — no code change.

Parity guarantee: for EDC the spec re-declares the current lists verbatim, so `Ontology("edc").system()`
reproduces the old hardcoded `structured_prompts.SYSTEM` byte-for-byte (asserted in test_ontology.py). The
refactor is a provable no-op; the predicate→object contract fix lands separately on top.
"""
from __future__ import annotations

import re
import uuid

from pipeline.domain import Domain


class Ontology:
    """The extraction ontology for one domain, read from its manifest."""

    def __init__(self, domain: str = "edc"):
        spec = Domain(domain).spec
        ont = spec.get("ontology", {})
        self.domain = domain
        self.kind = ont.get("kind", "festival")
        self.name = ont.get("name", "")
        self.entity_types = list(spec.get("entity_types", []))
        self.predicates = list((spec.get("predicates") or {}).keys()) or list(ont.get("predicates", []))
        self.topics = list(ont.get("topics", []))
        self.stances = list(ont.get("stances", []))
        self.confidence_map = dict(ont.get("confidence_map", {}))
        # the predicate→object-type contract (for the extraction-time enforcement that fixes plays_at→festival)
        self.contract = {p: list(m.get("object", [])) for p, m in (spec.get("predicates") or {}).items()}

    # ── the system prompt (spec-driven; byte-identical to the old hardcoded SYSTEM for EDC) ──
    def system(self) -> str:
        return (
            f"You extract a {self.kind} knowledge graph from a source document about {self.name}. "
            "Return ONLY a JSON object with keys entities, triples, discourse. No prose, no markdown fence.\n"
            f"- entity_type ∈ {{{', '.join(self.entity_types)}}}\n"
            f"- triple predicate ∈ {{{', '.join(self.predicates)}}}\n"
            f"- discourse topic ∈ {{{', '.join(self.topics)}}};  stance ∈ {{{', '.join(self.stances)}}}\n"
            "Rules:\n"
            "- entity id = '{fslug}:<entity_type>:<snake_case_slug>' (lowercase, [a-z0-9_]).\n"
            "- entity = {id, entity_type, name_canonical, attributes{}}. attributes may be {} if unsure; "
            "include concrete facts you ARE sure of (dates, capacity, genre, day, time) as key:value.\n"
            "- triple = {subject_id, predicate, object, attrs{}, quote?}. object is an entity id OR a literal "
            "(string/number). Only assert relations the text states. attrs{} optional (magnitude, time_window). "
            "quote is optional; when present, copy the minimal exact supporting sentence verbatim (300 characters "
            "max), never a paraphrase.\n"
            "- plays_at REQUIRES a stage/set/venue object. If an artist is in the lineup but the text gives "
            "NO stage, use predicate in_lineup with the festival/edition id as object — NEVER plays_at to "
            "the festival.\n"
            "- discourse = {anchors:[entity_id,...], topic, subtopic, stance, content}. Use for opinions/"
            "experiences/warnings from reviews or reddit. content = a faithful 1-3 sentence observation.\n"
            "- Invent NOTHING. Every entity/relation/observation must be grounded in the document. "
            "If the document has little structured content, return few records (empty lists are fine)."
        )

    def user_prompt(self, fslug: str, title: str, body: str) -> str:
        return (f"festival_slug = {fslug}\nDocument title: {title}\n\n"
                f"Document:\n{body[:12000]}\n\nReturn the JSON object now.")

    def confidence_for(self, domain_class: str) -> str:
        return self.confidence_map.get(domain_class, "single-anecdote")

    def inject_provenance(self, rec: dict, kind: str, meta: dict, fslug: str) -> dict:
        """Inject pipeline-owned source/confidence/document identity and an optional exact model quote.

        The model never supplies identity, hashes, confidence, or offsets. Mirrors the old
        structured_prompts.inject_provenance signature, minus the unused ``scores`` arg.
        """
        dc = meta.get("domain_class", "user-contributed")
        source = {"domain_class": dc, "url": meta.get("final_url") or meta.get("url") or "",
                  "domain": (meta.get("url", "").split("/")[2] if "://" in meta.get("url", "") else ""),
                  "scrape_iso": meta.get("fetched_at", "")}
        if kind == "discourse":
            rec["source"] = source
            rec.setdefault("id", f"{fslug}:discourse:{uuid.uuid4().hex[:12]}")
        else:  # entity, triple
            provenance = {"source": source, "confidence": self.confidence_for(dc)}
            if kind == "triple":
                # Document identity/hash are pipeline-owned. `quote` is the sole model-returned provenance
                # field; it is removed from the semantic triple before schema validation and exact-matched
                # against the same clean-body prefix the model saw. Never synthesize or fuzzy-match offsets.
                doc_id = meta.get("doc_id")
                doc_sha256 = meta.get("doc_sha256")
                if doc_id:
                    provenance["doc_id"] = str(doc_id)
                if doc_sha256:
                    provenance["doc_sha256"] = str(doc_sha256)
                raw_quote = rec.pop("quote", None)
                if isinstance(raw_quote, str) and (quote := raw_quote.strip()[:300]):
                    provenance["quote"] = quote
                    body = meta.get("doc_body")
                    start = body.find(quote) if isinstance(body, str) else -1
                    provenance["span"] = (
                        {"start": start, "end": start + len(quote)} if start >= 0 else None
                    )
            rec["provenance"] = provenance
        return rec


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:60] or "x"
