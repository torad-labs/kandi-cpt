"""Disjoint-context generation for confusable entities (V6 — distributional-hypothesis approach).

Firth (1957) / Harris (1954), still how transformers behave: a token's representation is shaped by the
company it keeps. SHARED context pulls representations together; DISJOINT context pushes them apart. The
Kinetic-sunrise / Cosmic-Friday cluster collapsed (v5 cosine showed the stages getting LESS distinct)
because the confusable facts shared context. v3 made it worse by CO-MENTIONING the wrong pairing
("X is NOT at CosmicMeadow") — co-mention IS shared context, so it pulled them CLOSER.

THE HARD RULE (never reintroduce): never name two confusable entities — or a confusable entity together
with another's day/stage — in the same line. Separation comes ONLY from disjoint context: each
confusable entity's facts live in their own vocabulary world, naming only that entity + its CORRECT
stage/day. Related facts (artist↔correct stage/day) co-occur freely — binding is wanted there. We
engineer the geometry by engineering co-occurrence, then read the cosine to confirm.

Each confusable group below has: its members, their one correct (stage, day), and a set of skeletons
written in a DISTINCT vocabulary world (different framing, scene-setting, neighbor tokens) that the other
groups do not share. The audit (audit_cooccurrence) is the v3 guardrail.
"""
from __future__ import annotations

import json
import re

import config as C
from pipeline.fact_expand import FESTIVAL, _daydate, camel

# All days/stages as surface tokens, for the co-occurrence audit (camel + spaced + date).
_DAY_TOKENS = {"Friday": ["friday", "may 15"], "Saturday": ["saturday", "may 16"], "Sunday": ["sunday", "may 17"]}


# Distinctive vocabulary WORLDS, keyed by confusable-group `key`. This is the GENERATION detail (the
# probe does not need it). Group MEMBERSHIP (entities + correct stage/day) is festival DATA — read from
# festival_meta.json below, the single source of truth shared with the geometry probe. Each skeleton is
# context-first: the distinctive block LEADS, then {a} entity, then {c} stage, with {dd} day LAST (the
# discriminator computed in distinct left-context + a late retroactive-sharpening token).
SKELETONS = {
    "armin_sunday_finale": [
        "The grand finale of the entire weekend, an uplifting trance send-off as the last sunrise breaks — {a} on the {c} stage, {dd}.",
        "Closing night, the festival's very last main-stage moment before it ends: {a} headlines {c}, {dd}.",
        "An uplifting close to three days, the dawn farewell of {FEST}: {a} takes {c}, {dd}.",
        "Last set of the whole weekend, a Sunday-into-dawn trance send-off: {a} on {c}, {dd}.",
        "The weekend's grand finale, uplifting and trance-driven as the final sunrise comes — {a} at {c}, {dd}.",
        "When the very last sunrise of the festival arrives, the closing finale belongs to {a}: {c}, {dd}.",
        "The festival's final breath, an uplifting closing-night farewell: {a} headlines {c}, {dd}.",
        "A grand uplifting send-off to end three days at {FEST}: {a} on the {c} stage, {dd}.",
    ],
    "aboveandbeyond_saturday_heart": [
        # MUST carry the queryable attribute 'sunrise' (eval tests "Saturday sunrise") — discriminated by
        # Saturday, distinct A&B framing (anthemic/group-therapy/midpoint), never co-mentions Armin.
        "Saturday's sunrise set at the {c} stage — anthemic, hands-in-the-air group therapy: {a}, {dd}.",
        "The {c} sunrise on {dd} belongs to {a}: the weekend's emotional, tearful midpoint.",
        "At the {c} stage, the Saturday sunrise is {a} — anthemic singalongs, the heart of the three days, {dd}.",
        "{a} plays the {c} sunrise set on {dd}, the mid-festival emotional centerpiece — group therapy at dawn.",
        "The Saturday (May 16) sunrise: {a} on {c}, anthemic and full of singalongs, the weekend's peak.",
        "On {dd}, {a} delivers the {c} sunrise set — the tearful midpoint of {FEST}.",
        "{a}'s {c} sunrise is Saturday's centerpiece: anthemic, emotional, hands in the air, {dd}.",
        "The mid-festival {c} sunrise on {dd} is {a} — group therapy, singalongs, the emotional core.",
    ],
    "underworld_friday_opening": [
        "Opening night, the launch of the entire weekend by a veteran live act — {a} on the {c} stage, {dd}.",
        "The festival's first night, the set that kicks everything off: {a} headlines {c}, {dd}.",
        "Before anything else happens at {FEST}, the opening live performance: {a} on {c}, {dd}.",
        "The weekend's launch, a veteran live show christening the festival: {a} takes {c}, {dd}.",
        "First night, first headline set, the very start of three days: {a} opens {c}, {dd}.",
        "Kicking off the whole weekend as the first sunrise nears: {a} on the {c} stage, {dd}.",
        "The opening-night launch of {FEST} by a live electronic veteran: {a} on {c}, {dd}.",
        "The very first headline moment of the festival, opening night: {a} at {c}, {dd}.",
    ],
    "cosmic_sunday_melodic": [
        "Closing day, the final-day melodic-bass showcase of the weekend — {a} on the {c} stage, {dd}.",
        "The last day of the festival, a melodic lineup in full: {a} headlines {c}, {dd}.",
        "Final-day melodic bass, the closing session of three days: {a} on {c}, {dd}.",
        "On the festival's last day, the melodic-bass showcase: {a} takes {c}, {dd}.",
        "Closing-day melodic bass at {FEST}, the final session of the weekend: {a} on {c}, {dd}.",
        "The weekend's last day, a melodic showcase to close it out: {a} headlines {c}, {dd}.",
    ],
}

_META = C.ROOT / "data/edc/las_vegas/2026/festival_meta.json"


def _load_groups() -> list[dict]:
    """Confusable-group MEMBERSHIP from festival_meta.json (single source of truth, also read by the
    geometry probe), merged with the in-code SKELETONS vocabulary world for each group key."""
    meta = json.loads(_META.read_text())
    return [{**g, "skeletons": SKELETONS.get(g["key"], [])} for g in meta.get("confusable_groups", [])]


GROUPS = _load_groups()
CONFUSABLE = {m: (g["stage"], g["day"]) for g in GROUPS for m in g["members"]}
GROUP_OF = {m: g["key"] for g in GROUPS for m in g["members"]}


def _ctx(day_token: str) -> str:
    return _daydate(day_token)


def generate_for_group(group: dict) -> list[dict]:
    """Distinct lines for one group, each in the group's own vocabulary world. Single-member groups
    name only that member; the multi-member group (Sunday-Cosmic) may name its OWN members together
    (same group → binding wanted) but NEVER another group's entity/day/stage."""
    c = camel(group["stage"])
    dd = _ctx(group["day"])
    out = []
    for a in group["members"]:
        for sk in group["skeletons"]:
            out.append({"text": sk.format(a=a, c=c, dd=dd, FEST=FESTIVAL),
                        "relation_type": "disjoint_context", "polarity": "positive",
                        "context_profile": group["key"], "entity_ids": [a, group["stage"], group["day"]]})
    return out


def violates(text: str) -> str | None:
    """THE v3 guardrail as a predicate → reason string, or None if clean. A line violates if it
    co-mentions, in one window:
      (1) confusable entities from TWO DIFFERENT groups (same-group co-mention is allowed — binding), or
      (2) a confusable entity + a DAY token that is not that entity's correct day, or
      (3) a confusable entity + a STAGE (camel) that is not that entity's correct stage.
    Separation must come from disjoint context, never from co-mention."""
    from pipeline.fact_expand import STAGE_CAMEL
    low = text.lower()
    present = [e for e in CONFUSABLE if e.lower() in low]
    if not present:
        return None
    if len({GROUP_OF[e] for e in present}) >= 2:
        return f"co-mentions different confusable groups: {present}"
    good_stage, good_day = CONFUSABLE[present[0]]            # same group → shared (stage, day)
    for day, toks in _DAY_TOKENS.items():
        if day != good_day and any(tok in low for tok in toks):
            return f"{present[0]} (day={good_day}) co-mentions wrong day {day}"
    for slug, cam in STAGE_CAMEL.items():
        if slug != good_stage and len(cam) > 3 and cam.lower() in low:
            return f"{present[0]} (stage={good_stage}) co-mentions wrong stage {cam}"
    return None


# Confusable stages (camelCase) — cultural/conversational synth must not bind these to a DAY. The day↔
# stage binding for a confusable stage is the EXACT signal V6 is sharpening (kineticFIELD sunrise = Armin
# = Sunday); a cultural line saying "kineticFIELD sunrise on night two" injects a competing Saturday
# binding. Keep culture day-agnostic about confusable stages — the day belongs to the facts, in disjoint
# context. (Non-confusable stages are unconstrained: culture may freely mention bassPOD, neonGARDEN, …)
_CONF_STAGE_CAMEL = {camel(g["stage"]) for g in GROUPS}
_NIGHT_RE = re.compile(r"night\s*(one|two|three|1|2|3)\b", re.I)


def cultural_conflict(text: str) -> str | None:
    """Guard for cultural/conversational synth: a cultural line must not pin a CONFUSABLE stage to a day
    (named day, date, or 'night N'). That competes with the disjoint fact install. Returns reason/None.
    Confusable stage with NO day is fine (kineticFIELD+sunrise reinforces the correct binding)."""
    low = text.lower()
    stages = [c for c in _CONF_STAGE_CAMEL if len(c) > 3 and c.lower() in low]
    if not stages:
        return None
    days = [d for d, toks in _DAY_TOKENS.items() if any(t in low for t in toks)]
    if days:
        return f"cultural line binds confusable stage {stages} to day {days}"
    if _NIGHT_RE.search(low):
        return f"cultural line binds confusable stage {stages} to a specific night"
    return None


def audit_cooccurrence(lines: list[dict]) -> list[dict]:
    """Run violates() over a corpus; return the violating lines tagged with the reason. Must be EMPTY
    before training (the v3 guardrail)."""
    out = []
    for r in lines:
        why = violates(r.get("text", ""))
        if why:
            out.append({**r, "_why": why})
    return out


def neighbor_overlap() -> dict:
    """Context-disjointness metric: Jaccard overlap of the content-word vocabulary between each pair of
    confusable groups' skeletons. Lower = more disjoint context = the goal."""
    import re
    STOP = set("the a an of on at in to and or is are be for as it its this that with into from by "
               "{a} {c} {dd} {fest} - — : , . play plays playing set stage day weekend night festival".split())

    def words(g):
        toks = re.findall(r"[a-zA-Z]+", " ".join(g["skeletons"]).lower())
        return {w for w in toks if w not in STOP and len(w) > 2}

    vocab = {g["key"]: words(g) for g in GROUPS}
    keys = list(vocab)
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = vocab[keys[i]], vocab[keys[j]]
            jac = len(a & b) / max(1, len(a | b))
            out[f"{keys[i]} ∩ {keys[j]}"] = round(jac, 3)
    return out
