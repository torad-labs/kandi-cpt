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

The distinctive vocabulary WORLDS (the disjoint-context TEMPLATE SHAPES) are domain-authored
(configs/templates/<domain>.yaml `hydrate_templates.disjoint_skeletons`), so a new domain swaps them;
group MEMBERSHIP stays festival DATA in each unit's festival_meta.json.
"""
from __future__ import annotations

import json
import re
import sys

import config as C
from pipeline.domain import Domain
from pipeline.fact_expand import FESTIVAL, _daydate, camel

# All days/stages as surface tokens, for the co-occurrence audit (camel + spaced + date).
_DAY_TOKENS = {"Friday": ["friday", "may 15"], "Saturday": ["saturday", "may 16"], "Sunday": ["sunday", "may 17"]}


def _skeletons(domain: str = "edc") -> dict:
    """The domain's disjoint-context vocabulary worlds, keyed by confusable-group `key`. This is the
    GENERATION detail (the probe does not need it); group MEMBERSHIP is festival DATA in festival_meta.json.
    Each skeleton is context-first: the distinctive block LEADS, then {a} entity, then {c} stage, with {dd}
    day LAST (the discriminator computed in distinct left-context + a late retroactive-sharpening token)."""
    try:
        return (Domain(domain).templates or {}).get("hydrate_templates", {}).get("disjoint_skeletons", {})
    except Exception:
        return {}


# Module-level edc skeletons — config-sourced, kept as a module global for back-compat (data_audit reads
# `disjoint_context.SKELETONS` to report which confusable groups have a vocabulary world).
SKELETONS = _skeletons("edc")

_META = C.ROOT / "data/edc/las_vegas/2026/festival_meta.json"


def _load_groups() -> list[dict]:
    """Confusable-group MEMBERSHIP from festival_meta.json (single source of truth, also read by the
    geometry probe), merged with the domain-authored disjoint_skeletons vocabulary world for each key."""
    try:
        meta = json.loads(_META.read_text())
    except (OSError, ValueError):
        print(f"WARNING: disjoint_context could not read {_META} — confusable groups empty (V6 lever OFF this run)",
              file=sys.stderr)
        return []
    return [{**g, "skeletons": SKELETONS.get(g["key"], [])} for g in meta.get("confusable_groups", [])]


GROUPS = _load_groups()
# Guard (silent-failure): skeletons are now config-sourced, so a missing/misparsed
# hydrate_templates.disjoint_skeletons key silently empties the V6 disjoint-context lever. A meta that
# DECLARES confusable_groups but loads no skeletons is a config error, not a no-op — surface it loudly.
if GROUPS and not any(g.get("skeletons") for g in GROUPS):
    print("  WARNING: confusable_groups are declared but NO disjoint_skeletons loaded — the V6 disjoint-context "
          "corpus will be EMPTY (check hydrate_templates.disjoint_skeletons key/nesting in the templates file)",
          file=sys.stderr)
CONFUSABLE = {m: (g["stage"], g["day"]) for g in GROUPS for m in g["members"]}
GROUP_OF = {m: g["key"] for g in GROUPS for m in g["members"]}


def index_for_meta(meta: dict) -> tuple[dict, dict, dict, dict]:
    """Build one edition's confusable index — `(confusable: member→(stage,day), group_of: member→key,
    day_tokens: day→[surface tokens], stage_camels: slug→camel)` — from ITS OWN festival_meta.json. The
    auditor must judge an edition against its own calendar/lineup, never the gold's: an edition that
    declares no `confusable_groups` has no disjoint-context constraint, so the maps come back empty and
    `audit_cooccurrence` finds nothing (a cross-edition binding mismatch is NOT contamination). Mirrors the
    in-code globals when fed the gold LV-2026 meta, so the V6 callers are unaffected."""
    groups = meta.get("confusable_groups", [])
    confusable = {m: (g["stage"], g["day"]) for g in groups for m in g["members"]}
    group_of = {m: g["key"] for g in groups for m in g["members"]}
    day_tokens: dict[str, list[str]] = {}
    for day, info in meta.get("days", {}).items():
        toks = [day.lower()]
        date = ((info or {}).get("date") or "").strip().lower()
        if date:
            toks.append(date)
        day_tokens[day] = toks
    stage_camels = {slug: (s.get("camel") or "") for slug, s in meta.get("stages", {}).items()}
    return confusable, group_of, day_tokens, stage_camels


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


def violates(text: str, confusable=None, group_of=None, day_tokens=None, stage_camels=None) -> str | None:
    """THE v3 guardrail as a predicate → reason string, or None if clean. A line violates if it
    co-mentions, in one window:
      (1) confusable entities from TWO DIFFERENT groups (same-group co-mention is allowed — binding), or
      (2) a confusable entity + a DAY token that is not that entity's correct day, or
      (3) a confusable entity + a STAGE (camel) that is not that entity's correct stage.
    Separation must come from disjoint context, never from co-mention.

    The four index args (confusable/group_of/day_tokens/stage_camels) default to the gold LV-2026 globals,
    so the V6 callers keep their exact behavior. Pass an edition's own index (see `index_for_meta`) to judge
    THAT edition against its own calendar/lineup — the only correct thing to do off the gold."""
    from pipeline.fact_expand import STAGE_CAMEL
    confusable = CONFUSABLE if confusable is None else confusable
    group_of = GROUP_OF if group_of is None else group_of
    day_tokens = _DAY_TOKENS if day_tokens is None else day_tokens
    stage_camels = STAGE_CAMEL if stage_camels is None else stage_camels
    low = text.lower()
    present = [e for e in confusable if e.lower() in low]
    if not present:
        return None
    if len({group_of[e] for e in present}) >= 2:
        return f"co-mentions different confusable groups: {present}"
    good_stage, good_day = confusable[present[0]]            # same group → shared (stage, day)
    for day, toks in day_tokens.items():
        if day != good_day and any(tok in low for tok in toks):
            return f"{present[0]} (day={good_day}) co-mentions wrong day {day}"
    for slug, cam in stage_camels.items():
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


def audit_cooccurrence(lines: list[dict], confusable=None, group_of=None,
                       day_tokens=None, stage_camels=None) -> list[dict]:
    """Run violates() over a corpus; return the violating lines tagged with the reason. Must be EMPTY
    before training (the v3 guardrail). Pass an edition's own index (`index_for_meta`) to audit a non-gold
    edition against its own bindings; omit them to use the gold LV-2026 globals (the V6 default)."""
    out = []
    for r in lines:
        why = violates(r.get("text", ""), confusable, group_of, day_tokens, stage_camels)
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
