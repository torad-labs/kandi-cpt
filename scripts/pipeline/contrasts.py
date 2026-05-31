"""Verbose NEGATIVE→POSITIVE contrast phrases (V6).

Deterministic by design: a wrong negative is the worst poison in the corpus, and v3's LLM-generated
contrast backfired (it co-located the wrong pairing). Every line here is built from verified triples.

Structure rule (the V6 lever): LEAD with the wrong binding behind a negation cue, then RESOLVE on the
correct binding so the sentence ENDS on the right token. Under autoregressive CE the resolving clause
sits in the "answer slot" — the correct token takes the recency/answer gradient while the negated one
is subordinated. ("X is not Saturday; X plays Sunday" trains predict(Sunday|…) at the end, not Saturday.)
"""
from __future__ import annotations

from pipeline.fact_expand import DAY_DATE, DAYS, FESTIVAL, _daydate, camel

ATTRACTORS = ["kinetic_field", "neon_garden", "cosmic_meadow"]   # the stages facts wrongly collapse toward


def _wrong_stages(stage_slug: str) -> tuple[str, str]:
    """Two attractor stages to negate against (the confusion sinks), excluding the true one."""
    w = [s for s in ATTRACTORS if s != stage_slug]
    if len(w) < 2:
        w = (w + ["circuit_grounds", "quantum_valley"])
    return camel(w[0]), camel(w[1])


def day_contrasts(a: str, stage_slug: str, day: str) -> list[str]:
    if not day:
        return []
    c, date, dd = camel(stage_slug), DAY_DATE.get(day, ""), _daydate(day)
    w1, w2 = [d for d in DAYS if d != day]
    forms = [
        f"It is a common mix-up, but {a} does not appear on {w1} or {w2} at {FESTIVAL} — {a}'s only set is {dd}, at the {c} stage.",
        f"Not {w1}. Not {w2}. {a} performs at {FESTIVAL} on {dd}, taking the {c} stage.",
        f"Do not plan around a {w1} or {w2} appearance: {a} plays {FESTIVAL} on {day} ({date}), 2026, at {c}.",
        f"{a} is absent from both the {w1} and the {w2} schedules at {FESTIVAL}; the correct day is {dd}, on {c}.",
        f"Whatever the rumor of a {w1} or {w2} slot, {a} actually plays {dd} at {FESTIVAL}, on the {c} stage.",
        f"Skip {w1} and {w2} if you are there for {a} — {a}'s {FESTIVAL} set is {dd}, at {c}.",
        f"There is no {a} set on {w1}, and none on {w2}: at {FESTIVAL}, {a} is a {day} ({date}) act, on the {c} stage.",
        f"People often guess {w1} or {w2}, but both are wrong — {a} plays {dd}, at {c}, at {FESTIVAL}.",
        f"{a} does not play {w1}; {a} does not play {w2}; {a} plays {dd} at {FESTIVAL}, on {c}.",
        f"Contrary to any {w1}-or-{w2} assumption, {a}'s single {FESTIVAL} appearance is {dd}, at the {c} stage.",
        f"To be clear about the day — not {w1}, not {w2}: {a} is on {dd} at {FESTIVAL}, playing {c}.",
        f"If a schedule lists {a} on {w1} or {w2}, it is wrong; {a} plays {dd} at {FESTIVAL}, on {c}.",
        f"Mark it as {day}, not {w1} or {w2}: {a}'s {FESTIVAL} set is {dd}, at the {c} stage.",
        f"Anyone expecting {a} on {w1} or {w2} will miss it — {a} plays {dd} at {FESTIVAL}, on {c}.",
        f"The day to remember for {a} is {dd}, not {w1} and not {w2}, on the {c} stage at {FESTIVAL}.",
        f"{a} has no {w1} slot and no {w2} slot at {FESTIVAL}; {a} is a {dd} act, on {c}.",
        f"Cross off {w1} and {w2} for {a} — the {FESTIVAL} set is {dd}, at the {c} stage.",
        f"It would be a mistake to catch {a} on {w1} or {w2}; {a} performs {dd} at {FESTIVAL}, on {c}.",
    ]
    # both wrong-day orderings double the distinct surface without new facts
    swapped = [f.replace(w1, "\0").replace(w2, w1).replace("\0", w2) for f in forms[:8]]
    return forms + swapped


def stage_contrasts(a: str, stage_slug: str, day: str) -> list[str]:
    c = camel(stage_slug)
    W1, W2 = _wrong_stages(stage_slug)
    dc = f", on {_daydate(day)}" if day else ""
    return [
        f"You will not find {a} at {W1} or {W2} at {FESTIVAL} — {a} plays the {c} stage{dc}.",
        f"It is not {W1}, and it is not {W2}: {a}'s stage at {FESTIVAL} is {c}{dc}.",
        f"Despite their pull, neither {W1} nor {W2} hosts {a}; {a} appears on {c}{dc} at {FESTIVAL}.",
        f"Do not head to {W1} or {W2} for {a} — {a} is on the {c} stage{dc} at {FESTIVAL}.",
        f"{a} is not a {W1} act and not a {W2} act; {a} belongs to the {c} stage{dc}.",
        f"Forget {W1} and {W2}: {a}'s {FESTIVAL} home is the {c} stage{dc}.",
        f"The stage is not {W1}, nor {W2} — {a} plays {c}{dc} at {FESTIVAL}.",
        f"Among the big stages, {a} skips {W1} and {W2}, playing instead on {c}{dc}.",
        f"Checking the stage: not {W1}, not {W2}; {a} is on {c}{dc} at {FESTIVAL}.",
        f"Neither {W1} nor {W2} is right for {a} — the {FESTIVAL} stage is {c}{dc}.",
        f"{a} will not be at {W1} or {W2}; look for {a} at the {c} stage{dc}.",
        f"It is a misconception that {a} plays {W1} or {W2}; {a}'s stage is {c}{dc} at {FESTIVAL}.",
    ]


def joint_contrasts(a: str, stage_slug: str, day: str) -> list[str]:
    """Bind the wrong stage AND wrong day in one negation, then resolve on the correct (stage, day)."""
    if not day:
        return []
    c, dd = camel(stage_slug), _daydate(day)
    W1, W2 = _wrong_stages(stage_slug)
    w1, w2 = [d for d in DAYS if d != day]
    return [
        f"It is not {W1} on {w1}, and not {W2} on {w2} — {a} plays the {c} stage on {dd} at {FESTIVAL}.",
        f"Neither a {w1} set at {W1} nor a {w2} set at {W2}: {a}'s only {FESTIVAL} appearance is {dd}, at {c}.",
        f"Do not look for {a} at {W1} or on {w1}; {a} is {dd}, at the {c} stage, at {FESTIVAL}.",
        f"{a} is not at {W1}, not at {W2}, not on {w1}, not on {w2}; {a} plays {c} on {dd} at {FESTIVAL}.",
        f"Forget a {W1}/{w1} pairing for {a} — the answer is the {c} stage on {dd} at {FESTIVAL}.",
    ]


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


def generate_for_fact(a: str, stage_slug: str, day: str) -> list[str]:
    """All distinct contrast lines for one (artist, stage, day) fact."""
    return day_contrasts(a, stage_slug, day) + stage_contrasts(a, stage_slug, day) + joint_contrasts(a, stage_slug, day)
