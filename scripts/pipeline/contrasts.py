"""Verbose NEGATIVE→POSITIVE contrast phrases (V6).

Deterministic by design: a wrong negative is the worst poison in the corpus, and v3's LLM-generated
contrast backfired (it co-located the wrong pairing). Every line here is built from verified triples.

Structure rule (the V6 lever): LEAD with the wrong binding behind a negation cue, then RESOLVE on the
correct binding so the sentence ENDS on the right token. Under autoregressive CE the resolving clause
sits in the "answer slot" — the correct token takes the recency/answer gradient while the negated one
is subordinated. ("X is not Saturday; X plays Sunday" trains predict(Sunday|…) at the end, not Saturday.)

The day/stage/joint contrast SHAPES are domain-authored (configs/templates/<domain>.yaml
`hydrate_templates.contrast_*`) and render against the unit's `Ctx`; EDC carries today's exact strings.
The attractor stages (the confusion sinks the contrasts negate against) are `hydrate_templates.attractors`.
"""
from __future__ import annotations

from pipeline.fact_expand import _dd, camel, default_ctx


def _wrong_stages(stage_slug: str, ctx) -> tuple[str, str]:
    """Two attractor stages to negate against (the confusion sinks), excluding the true one."""
    w = [s for s in ctx.t.get("attractors", []) if s != stage_slug]
    if len(w) < 2:
        w = (w + ["circuit_grounds", "quantum_valley"])
    return camel(w[0]), camel(w[1])


def day_contrasts(a: str, stage_slug: str, day: str, ctx=None) -> list[str]:
    if not day:
        return []
    ctx = ctx or default_ctx()
    c, date, dd = camel(stage_slug), ctx.day_date.get(day, ""), _dd(day, ctx)
    w1, w2 = [d for d in ctx.days if d != day]
    kw = dict(a=a, c=c, date=date, dd=dd, day=day, w1=w1, w2=w2, festival=ctx.festival)
    forms = [t.format(**kw) for t in ctx.t["contrast_day"]]
    # both wrong-day orderings double the distinct surface without new facts
    swapped = [f.replace(w1, "\0").replace(w2, w1).replace("\0", w2) for f in forms[:8]]
    return forms + swapped


def stage_contrasts(a: str, stage_slug: str, day: str, ctx=None) -> list[str]:
    ctx = ctx or default_ctx()
    c = camel(stage_slug)
    W1, W2 = _wrong_stages(stage_slug, ctx)
    dc = f", on {_dd(day, ctx)}" if day else ""
    kw = dict(a=a, c=c, W1=W1, W2=W2, dc=dc, festival=ctx.festival)
    return [t.format(**kw) for t in ctx.t["contrast_stage"]]


def joint_contrasts(a: str, stage_slug: str, day: str, ctx=None) -> list[str]:
    """Bind the wrong stage AND wrong day in one negation, then resolve on the correct (stage, day)."""
    if not day:
        return []
    ctx = ctx or default_ctx()
    c, dd = camel(stage_slug), _dd(day, ctx)
    W1, W2 = _wrong_stages(stage_slug, ctx)
    w1, w2 = [d for d in ctx.days if d != day]
    kw = dict(a=a, c=c, dd=dd, W1=W1, W2=W2, w1=w1, w2=w2, festival=ctx.festival)
    return [t.format(**kw) for t in ctx.t["contrast_joint"]]


# Verified hard-negatives for the trio the eval/cloze keeps confusing (Armin Sun / A&B Sat / Underworld
# Fri-Cosmic) and the Sunday-Cosmic compositional miss. All facts checked against the triple store +
# set_times.jsonl (A&B/Armin carry the SUNRISE SET tag at kineticFIELD).
HARD_NEGATIVES = [
    "The Saturday (May 16) sunrise set at the kineticFIELD stage is not Armin van Buuren, and it is not Underworld — the Saturday sunrise belongs to Above & Beyond.",
    "The Sunday (May 17) sunrise set at kineticFIELD is not Above & Beyond, and it is not Underworld — it is Armin van Buuren who plays the Sunday sunrise.",
    "Underworld does not play a sunrise set, and Underworld is not on the kineticFIELD stage; Underworld performs at the cosmicMEADOW stage on Friday (May 15).",
    "Three EDC Las Vegas 2026 sets get confused: Above & Beyond is the Saturday (May 16) kineticFIELD sunrise, Armin van Buuren is the Sunday (May 17) kineticFIELD sunrise, and Underworld is neither — Underworld is Friday (May 15) at cosmicMEADOW.",
    "It is not Armin van Buuren who plays the Saturday kineticFIELD sunrise; that is Above & Beyond. Armin van Buuren's sunrise is the next night, Sunday (May 17).",
    "It is not Above & Beyond who closes Sunday at kineticFIELD; the Sunday (May 17) sunrise is Armin van Buuren. Above & Beyond played the night before, Saturday (May 16).",
    "Do not place Underworld on Sunday: Underworld's cosmicMEADOW set is Friday (May 15). The Sunday cosmicMEADOW artists include Seven Lions, Alison Wonderland, and San Holo.",
    "The Sunday cosmicMEADOW lineup is not Underworld — Underworld is the Friday (May 15) cosmicMEADOW act. On Sunday, cosmicMEADOW features Seven Lions, Alison Wonderland, and San Holo.",
    "Above & Beyond and Armin van Buuren both play the kineticFIELD sunrise, but not on the same day: Above & Beyond is Saturday (May 16), Armin van Buuren is Sunday (May 17).",
    "If you are looking for the Friday cosmicMEADOW headliner, it is Underworld — not Seven Lions, Alison Wonderland, or San Holo, who play cosmicMEADOW on Sunday (May 17).",
    "The kineticFIELD sunrise is not the same act two nights running: Saturday (May 16) is Above & Beyond, Sunday (May 17) is Armin van Buuren.",
    "Armin van Buuren does not play Saturday at EDC Las Vegas 2026 — his kineticFIELD sunrise set is Sunday (May 17). The Saturday (May 16) sunrise is Above & Beyond.",
    "Above & Beyond does not play Sunday at EDC Las Vegas 2026 — their kineticFIELD sunrise set is Saturday (May 16). The Sunday (May 17) sunrise is Armin van Buuren.",
    "Underworld is a Friday act, not a Saturday or Sunday one: Underworld plays cosmicMEADOW on Friday (May 15), not the kineticFIELD sunrise on either weekend night.",
    "On Sunday (May 17), the cosmicMEADOW artists are Seven Lions, Alison Wonderland, and San Holo — not Underworld, whose cosmicMEADOW set is Friday (May 15).",
    "Do not confuse the kineticFIELD sunrise acts with Underworld: the sunrise sets are Above & Beyond (Saturday, May 16) and Armin van Buuren (Sunday, May 17), while Underworld plays cosmicMEADOW on Friday (May 15).",
    "It is Armin van Buuren, not Above & Beyond, who plays the Sunday (May 17) kineticFIELD sunrise that closes EDC Las Vegas 2026.",
    "It is Above & Beyond, not Armin van Buuren, who plays the Saturday (May 16) kineticFIELD sunrise at EDC Las Vegas 2026.",
    "The Friday (May 15) cosmicMEADOW headliner is Underworld — and Underworld appears nowhere on the kineticFIELD sunrise schedule.",
]


def generate_for_fact(a: str, stage_slug: str, day: str, ctx=None) -> list[str]:
    """All distinct contrast lines for one (artist, stage, day) fact."""
    return (day_contrasts(a, stage_slug, day, ctx) + stage_contrasts(a, stage_slug, day, ctx)
            + joint_contrasts(a, stage_slug, day, ctx))
