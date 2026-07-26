#!/usr/bin/env python3
"""Synthetic corpus expansion for Kandi CPT.

Combines techniques validated in the synthetic-data-for-CPT literature:

1. EntiGraph (Yang et al. 2024, NeurIPS): extract entities from source docs,
   then generate text describing relations between entity pairs/triplets.
   Scales to 455M tokens log-linearly; flat-paraphrase plateaus much earlier.
2. Nemotron-HQ-Synth (Su et al. 2025): 5 rephrasing strategies — Diverse QA,
   Extract-Knowledge (textbook style), Distill (condensed), Wikipedia, List.
3. Phi textbook approach (Microsoft): vary the target audience for natural
   stylistic + lexical diversity (grade school / college / expert / general).
4. Focal Rewriting (Yang 2026): condition generation on a specific question
   to drive topic diversity, not just stylistic diversity.

Plus 3 Kandi-specific formats: friend-chat, wayfinding narrative,
festival-veteran tutorial.

Generator: Gemma 4 31B via OpenRouter (same family as the on-device E2B
target — voice patterns and tokenizer space align naturally).

Output: data/edc/las_vegas/2026/20_synthetic/<source_doc_stem>/<strategy>__<n>.md
        — preserves source attribution and strategy lineage for ablation.
        Single edition for now; the rewrite iterates data/<festival>/<city>/<year>/.

Run:
    python scripts/synth_corpus.py [--max-chunks-per-doc N] [--targets N]
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
from config import CLEAN as CORPUS_DIR, SYNTHETIC as SYNTH_DIR
from pipeline import fact_expand as FE
LOG_FILE = ROOT / "logs" / "synth_progress.md"

MODEL = "google/gemma-4-31b-it"   # exact same family as on-device E2B target — voice + tokenizer aligned
# Deprecated name — resolved at runtime via pipeline.llm_provider (googleai plan A).
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_AI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MAX_CONCURRENT = 6     # OpenRouter rate limits — be polite
CHUNK_TOKENS = 800     # rough chunk size; we count by chars/4
MAX_TOKENS_OUT = 1300  # per-generation cap (fact-variation calls request up to ~36 lines when rebalanced)
TEMPERATURE = 0.85     # high enough for diversity, low enough for coherence

# -------- Generator strategies --------------------------------------------

# Task-framing preamble prepended to EVERY strategy/relation system prompt. Telling the generator
# what the data is FOR (diverse faithful representations to install EDC-2026 knowledge into an
# on-device model's weights) measurably sharpens on-target, factual, specific output.
SYSTEM_PREAMBLE = (
    "You are generating synthetic training data for the continued pre-training (CPT) of Kandi, an "
    "on-device AI festival companion. The objective is to install accurate, specific, durable "
    "knowledge about Electric Daisy Carnival (EDC) Las Vegas 2026 (May 15-17, 2026, Las Vegas Motor "
    "Speedway) into the model's weights, through many DIVERSE representations of the same underlying "
    "facts. Every output MUST be: (1) faithful to the source — invent no artists, stages, days, "
    "times, prices, or policies not present; (2) concrete and specific — name real artists, stages "
    "(in Insomniac camelCase: kineticFIELD, cosmicMEADOW, circuitGROUNDS, neonGARDEN, bassPOD, "
    "wasteLAND, quantumVALLEY, stereoBLOOM, bionicJUNGLE), days, and times; (3) self-contained and "
    "free of meta-commentary about the source. Your specific writing task for this passage:\n\n"
)

# Each strategy gets its own system prompt (prepended with SYSTEM_PREAMBLE). We feed the source
# chunk in the user turn. Strategies are mostly orthogonal — running them all on the same chunk
# produces distinct generations with high lexical + semantic diversity.

STRATEGIES = {
    "qa_pairs": (
        "You are writing high-quality factual question-answer pairs for a "
        "knowledge corpus about Electric Daisy Carnival 2026 (Las Vegas Motor "
        "Speedway, May 15-17 2026). For the source content below, produce "
        "5-10 specific Q&A pairs that capture every concrete fact. Each "
        "question should be specific (not 'tell me about X'), each answer "
        "should commit to a definite claim. Use the exact artist names, "
        "stage names (with Insomniac's camelCase: kineticFIELD, cosmicMEADOW, "
        "etc.), days, times, and venue details from the source. No "
        "hedging, no clarifying questions. Format:\n\n"
        "Q: <specific question>\nA: <definite answer>\n\n"
        "Source content follows."
    ),
    "extract_knowledge": (
        "You are writing a textbook-style passage that distills the facts in "
        "the source text about Electric Daisy Carnival 2026 into a dense, "
        "factual prose. Use the structure of a reference work: define terms, "
        "name entities concretely, cite specific artists/stages/days/times. "
        "Multiple short paragraphs. Avoid filler. Avoid second-person 'you' "
        "framing. Avoid promotional language. Just facts in clean prose."
    ),
    "distill_summary": (
        "Write a tight, fact-dense summary of the source content about EDC "
        "Las Vegas 2026. Preserve every artist name, stage assignment, day, "
        "time, location, and policy. Compress everything else. ~200 words. "
        "Maintain Insomniac's stage camelCase (kineticFIELD, cosmicMEADOW, "
        "circuitGROUNDS, neonGARDEN, bassPOD, wasteLAND, quantumVALLEY, "
        "stereoBLOOM, bionicJUNGLE)."
    ),
    "wikipedia": (
        "Rewrite the source content in the formal encyclopedic style of "
        "Wikipedia — neutral tone, third person, factual claims with "
        "specific entity names. Cover the same facts but reorganize into "
        "thematic sections (Overview, Stages, Lineup, Logistics, Safety) "
        "as appropriate. Use Insomniac's stage camelCase. No 'in this "
        "article we will…' meta-language; just write the entry directly."
    ),
    "knowledge_list": (
        "Convert the source content into a flat list of single-fact bullet "
        "points about EDC 2026. Each bullet: one concrete fact, one short "
        "sentence. Cover artists (with stage and day), stage descriptions, "
        "ops policies, safety knowledge, wayfinding tips. Use Insomniac's "
        "stage camelCase. 25-50 bullets total."
    ),
    "friend_chat": (
        "Rewrite the source content as a casual but informative two-person "
        "chat where one friend is asking specific questions about EDC 2026 "
        "and a festival-veteran friend is answering with concrete details. "
        "Cover the same facts. Use natural conversation flow. The veteran "
        "uses Insomniac's stage camelCase (kineticFIELD, etc.) and commits "
        "to specific artist/stage/day claims. Format:\n\n"
        "Friend: <question>\nVet: <answer with specifics>\n\n"
        "Generate 8-15 turns."
    ),
    "wayfinding_pov": (
        "Rewrite the source content as a first-person wayfinding narrative "
        "from a festival veteran walking through EDC 2026. Describe what's "
        "where: 'Walking from kineticFIELD east toward cosmicMEADOW, you "
        "pass the merchant area on the left and the carnival rides on the "
        "right…' Use concrete spatial relationships. Name stages with "
        "Insomniac camelCase. Name specific artists playing the stages "
        "you pass. Cover the lineup facts the source mentions. Like a "
        "walking-tour transcript."
    ),
}

# Per-fact balanced expansion (replaces the random entity-pairing EntiGraph that caused confabulation).
# The model is given ONE verified fact and asked for many diverse, faithful surface forms of EXACTLY
# that fact — it cannot cross-contaminate to another artist/stage/day because no other fact is in
# context. Run identically for every triple → balanced coverage; combined with deterministic templates
# (pipeline/fact_expand.py) for a correctness floor.
FACT_VARIATION_PROMPT = (
    "Here is ONE verified scheduling fact about {festival} ({dates}, at {venue}):\n\n"
    "    {fact}\n\n"
    "Write {n} DIVERSE, standalone ways to express EXACTLY this one fact. {day_rule}Vary the form: "
    "plain declarative; question-and-answer; the reverse direction (the stage/day hosting the artist); "
    "a schedule line; a conversational mention between friends; a set-recommendation. Use BOTH the "
    "camelCase stage name ({camel}) and the spaced form ({spaced}) across your variations. Assert NO "
    "fact about any other artist, stage, day, or time — only this one. Output ONE variation per line, "
    "no numbering, no preamble."
)
# the day is the dimension that collapsed in v2 — force every variation to carry it, date-anchored,
# and to include the multiple-choice framing that empirically retrieved the right day.
_DAY_RULE = ("Every single line MUST tie the artist, the stage, AND the day together — never drop the "
             "day, it is the part most easily forgotten. Refer to the day both by name ({day}) and by "
             "its date ({date}) across the variations, and include at least two in the form "
             "'Is {artist} playing Friday, Saturday, or Sunday? — {day} ({date})'. ")


async def gen_fact_variations(client: httpx.AsyncClient, artist: str, camel: str, spaced: str,
                              day: str, n: int = 12) -> list[str]:
    """LLM diverse-but-faithful paraphrases of a SINGLE triple, day+date bound. One-line statements."""
    date = FE.DAY_DATE.get(day, "")
    fact = (f"{artist} plays the {camel} stage" + (f" on {day}, {date}" if day else "")
            + f" at {FE.FESTIVAL} ({FE.VENUE}).")
    day_rule = _DAY_RULE.format(artist=artist, day=day, date=date) if day else ""
    prompt = SYSTEM_PREAMBLE + FACT_VARIATION_PROMPT.format(
        festival=FE.FESTIVAL, dates=FE.DATES, venue=FE.VENUE, fact=fact, n=n,
        camel=camel, spaced=spaced, day_rule=day_rule)
    raw = await call_openrouter(client, prompt, "Write the variations now.")
    if raw.startswith("[generation failed") or raw.startswith("[error"):
        return []
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines()]
    return [ln for ln in lines if len(ln) > 10 and not _is_noise(ln)]


STAGE_VARIATION_PROMPT = (
    "Here are verified facts about ONE stage at {festival} ({dates}, at {venue}):\n\n"
    "    Stage: {camel} (also written {spaced})\n    {facts}\n\n"
    "Write {n} DIVERSE, standalone ways to express these facts about THIS stage — its music/genre, its "
    "curator brand (if any), and its size/rank. Include BOTH directions: '{camel} is the <genre> stage' "
    "AND 'the stage for <genre> is {camel}'. Vary form: declarative, question-and-answer, recommendation "
    "('for <genre>, head to {camel}'). Use BOTH {camel} and {spaced}. Assert NOTHING about specific "
    "artists or days, and invent no genre/brand/size not listed above. One per line, no numbering."
)


async def gen_stage_variations(client: httpx.AsyncClient, stage_slug: str, n: int = 10) -> list[str]:
    """LLM diverse paraphrases of a stage's genre/brand/rank (the stage→attribute relationships)."""
    m = FE.STAGE_META.get(stage_slug)
    if not m:
        return []
    facts = "; ".join(f"{k}: {v}" for k, v in (("music/genre", m["genre"]), ("curator brand", m["brand"]),
                                               ("size/rank", m["rank"]), ("character", m["vibe"])) if v)
    prompt = SYSTEM_PREAMBLE + STAGE_VARIATION_PROMPT.format(
        festival=FE.FESTIVAL, dates=FE.DATES, venue=FE.VENUE,
        camel=FE.camel(stage_slug), spaced=FE.spaced(stage_slug), facts=facts, n=n)
    raw = await call_openrouter(client, prompt, "Write the variations now.")
    if raw.startswith("[generation failed") or raw.startswith("[error"):
        return []
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines()]
    return [ln for ln in lines if len(ln) > 10 and not _is_noise(ln)]


# -------- Cultural-conversation install (primitive → variations) ----------
# The cultural knowledge (PLUR, pacing, feet, hydration, kandi, buddy-safety, sunrise-set ethos) is
# installed the SAME way facts are: extract the distinct cultural PRIMITIVES from the curated culture
# docs (one grounded talking point per line), then hydrate EACH primitive into many diverse two-person
# conversations. Primitive = the canonical point; variations = the exposure multiplier that installs it.
# Faithfulness: primitives are extracted FROM the source (no invention); the convos assert nothing about
# which artist plays which day/stage/time, and every variation is run through the confusable co-mention
# audit (the v3 guardrail) before it is kept.
CULTURAL_PRIMITIVE_PROMPT = (
    "Below is veteran festival-culture writing about {festival} ({venue}). Extract the DISTINCT cultural "
    "talking points — the concrete things experienced ravers tell newcomers. Each primitive: ONE "
    "self-contained piece of lived advice or ethos, in one or two sentences, faithful to the source "
    "(invent nothing). Cover the range actually present: footwear/feet, pacing across the three nights, "
    "hydration/electrolytes, the PLUR ethos, kandi trading, buddy/group safety, meet-up plans when cell "
    "service dies, sunrise sets as emotional peaks, sleep/recovery, what to bring, crowd navigation, "
    "harm-reduction. Assert NOTHING about which specific artist plays which day, stage, or time. "
    "Output ONE primitive per line, no numbering, no preamble."
)


async def gen_cultural_primitives(client: httpx.AsyncClient, source_text: str, n_max: int = 40) -> list[str]:
    """LLM extracts deduplicated cultural talking points (primitives) grounded in the culture docs."""
    system = SYSTEM_PREAMBLE + CULTURAL_PRIMITIVE_PROMPT.format(festival=FE.FESTIVAL, venue=FE.VENUE)
    raw = await call_openrouter(client, system, source_text[:24000], max_tokens=2200)
    if raw.startswith("[generation failed") or raw.startswith("[error"):
        return []
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines()]
    out, seen = [], set()
    for ln in lines:
        k = _fslug(ln)
        if len(ln) > 25 and not _is_noise(ln) and k not in seen:
            seen.add(k)
            out.append(ln)
    return out[:n_max]


CONVO_VARIATION_PROMPT = (
    "Here is ONE piece of {festival} festival-culture wisdom:\n\n    {primitive}\n\n"
    "Write {n} DIFFERENT short two-person conversations that each teach EXACTLY this one point. Vary who "
    "is asking (a first-timer, a nervous parent, a seasoned raver, a curious journalist), the scenario, "
    "and the wording — but every conversation must land this same concrete advice, faithfully. Keep them "
    "natural and casual, 3-6 turns each, in the format 'Friend: …' / 'Vet: …'. Do NOT claim which artist "
    "plays which day, stage, or time. Separate each conversation with a line containing only '---'."
)


async def gen_convo_variations(client: httpx.AsyncClient, primitive: str, n: int = 12) -> list[str]:
    """One cultural primitive → n diverse two-person conversations (the convo analog of gen_fact_variations)."""
    system = SYSTEM_PREAMBLE + CONVO_VARIATION_PROMPT.format(festival=FE.FESTIVAL, primitive=primitive, n=n)
    raw = await call_openrouter(client, system, "Write the conversations now.", max_tokens=3500)
    if raw.startswith("[generation failed") or raw.startswith("[error"):
        return []
    convos = [c.strip() for c in re.split(r"\n\s*-{3,}\s*\n", raw) if c.strip()]
    return [c for c in convos if len(c) > 40 and not _is_noise(c)]


# Curated culture sources (Tier-1 lived knowledge + community texture). Reddit r-edc is too thin (78w).
CULTURAL_DOCS = [
    "official/festival_tips_aggregated.md", "official/insomniac_rookie_mistakes.md",
    "official/insomniac_voice_essay.md", "official/edc_guides_aggregated.md",
    "reddit/r-aves-2026.md", "reddit/r-electricdaisycarnival-2026.md",
]


async def expand_cultural_for_edition(client, ed, sem, args) -> int:
    """Install cultural knowledge as primitives → many conversation variations (the convo analog of
    expand_facts_for_edition). Extract grounded primitives from the curated culture docs, then hydrate
    each into args.cultural_variations diverse conversations, audit-guarded against confusable co-mention."""
    from pipeline import disjoint_context as D
    clean = ed.dir / "10_clean"
    srcs = [clean / p for p in CULTURAL_DOCS if (clean / p).exists()]
    if not srcs:
        return 0
    source_text = "\n\n".join(s.read_text(encoding="utf-8") for s in srcs)
    out_dir = ed.dir / "20_synthetic" / "_cultural"
    out_dir.mkdir(parents=True, exist_ok=True)
    primitives = await gen_cultural_primitives(client, source_text)
    log(f"\n# {ed.slug} CULTURAL: {len(primitives)} primitives from {len(srcs)} docs, "
        f"vars/primitive={args.cultural_variations}")
    if not primitives:
        return 0

    async def with_sem(coro):
        async with sem:
            return await coro

    tasks = [(i, p, asyncio.create_task(with_sem(gen_convo_variations(client, p, args.cultural_variations))))
             for i, p in enumerate(primitives)]
    n_chars, n_conv, n_drop = 0, 0, 0
    for i, prim, task in tasks:
        fname = out_dir / f"{i:03d}__{_fslug(prim)}.md"
        if fname.exists() and not args.force:
            continue
        try:
            convos = await task
        except Exception:
            convos = []
        kept = []
        for c in convos:
            if D.violates(c) or D.cultural_conflict(c):        # v3 guardrail + no confusable stage↔day binding
                n_drop += 1
                continue
            kept.append(c)
        if not kept:
            continue
        body = "\n\n---\n\n".join(kept)
        header = (f"# Cultural primitive: {prim}\n"
                  "# Source: curated EDC-LV culture docs (tips/rookie/voice/guides/reddit)\n"
                  f"# Generator: {MODEL} conversation variations\n"
                  "# Synthetic: true  (domain_class: inferred-by-model)\n\n"
                  f"PRIMITIVE: {prim}\n\n")
        fname.write_text(header + body + "\n", encoding="utf-8")
        n_chars += len(body)
        n_conv += len(kept)
    log(f"   → {n_conv} cultural conversations across {len(primitives)} primitives "
        f"({n_drop} dropped by co-mention audit), {n_chars:,} chars")
    return n_chars


# -------- Set-time install (artist↔day↔time binding at a shared stage) -----
# THE confusion V6 fights: kineticFIELD's 4:14 AM sunrise slot is shared by Charlotte de Witte (Fri),
# Above & Beyond (Sat), and Armin van Buuren (Sun); cosmicMEADOW by Underworld (Fri) vs the melodic trio
# (Sun). The stage token is SHARED, so it cannot separate them — only the DAY does. The set-time data is
# therefore the ideal teacher: each line binds artist↔stage↔DAY↔clock-time, and the day is load-bearing.
# We hydrate each set into many variations (the "really install it" multiplier), every line naming ONLY
# its own artist + correct stage/day/time, audit-guarded (D.violates) so no confusable pair is co-mentioned.
# Token ORDER is load-bearing (autoregressive next-token engineering): the same stage+slot hosts a
# different artist each day, so the discriminating context must PRECEDE the artist and the ARTIST must be
# LAST — by the time the model predicts the performer token, "sunrise → day → time → stage" has already
# collapsed the distribution to exactly one person. (artist, day) is a unique key; an artist rarely plays
# twice in a day. Leading with the artist teaches the high-entropy reverse direction and blurs retrieval.
SET_TIME_VARIATION_PROMPT = (
    "Here is ONE verified SET-TIME fact about {festival} ({dates}, at {venue}):\n\n"
    "    {fact}\n\n"
    "Write {n} DIVERSE ways to state this one set time. They must ALL follow the SAME token order so the "
    "model learns to predict the PERFORMER from the context (not the reverse): lead with the OCCASION "
    "(the {slot}), then the DAY, then the TIME and stage, and name the ARTIST LAST — as the final words "
    "of the sentence. Everything before the artist must narrow to exactly one performer; the artist is "
    "the answer being retrieved. NEVER begin a line with the artist's name.\n\n"
    "Follow this ORDER (vary the wording freely, but keep the order — sunrise/slot → day → time → artist):\n"
    "  • The {slot} on {day} ({date}) takes place at {start} and is played by {artist}.\n"
    "  • On {day}, {date}, the {start} {slot} at {camel} belongs to {artist}.\n"
    "  • {day}'s {slot} — {start} to {end} at {spaced} — is performed by {artist}.\n"
    "  • Closing out {day} at {camel}, the {start} {slot} goes to {artist}.\n\n"
    "Use BOTH {camel} and {spaced} across the lines. Name the day as {day} and as {date}. Assert NOTHING "
    "about any other artist, stage, day, or time — only this one set. One variation per line, no numbering."
)


async def gen_settime_variations(client: httpx.AsyncClient, artist: str, camel: str, spaced: str,
                                 day: str, date: str, start: str, end: str, is_sunrise: bool,
                                 n: int = 10) -> list[str]:
    """One verified set-time → n context-first (artist-LAST) paraphrases: sunrise→day→time→artist, so the
    model predicts the performer from the collapsed context. The day is the discriminator at a shared stage."""
    slot = "sunrise set (the dawn closer)" if is_sunrise else "set"
    fact = (f"On {day} ({date}), the {start}–{end} {slot} at the {camel} stage is played by {artist} "
            f"at {FE.FESTIVAL}.")
    system = SYSTEM_PREAMBLE + SET_TIME_VARIATION_PROMPT.format(
        festival=FE.FESTIVAL, dates=FE.DATES, venue=FE.VENUE, fact=fact, n=n, artist=artist,
        camel=camel, spaced=spaced, day=day, date=date, start=start, end=end, slot=slot)
    raw = await call_openrouter(client, system, "Write the set-time variations now.", max_tokens=1800)
    if raw.startswith("[generation failed") or raw.startswith("[error"):
        return []
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines()]
    return [ln for ln in lines if len(ln) > 12 and not _is_noise(ln)]


async def expand_settimes_for_edition(client, ed, sem, args) -> int:
    """Hydrate set_times.jsonl into many diverse artist↔day↔time variations (the LLM diversity layer over
    the deterministic set-time floor that build_v6_corpus already puts in relations.jsonl). Each set's
    variations are audit-guarded against confusable co-mention (D.violates). Writes 20_synthetic/_settimes/."""
    from pipeline import disjoint_context as D
    from pipeline.fact_expand import DAY_DATE
    st_path = ed.dir / "set_times.jsonl"
    ents_path = ed.dir / "15_structured/entities.jsonl"
    if not st_path.exists():
        return 0
    by_lower = {}
    if ents_path.exists():
        for l in open(ents_path):
            if l.strip():
                e = json.loads(l)
                if e.get("name_canonical"):
                    by_lower[e["name_canonical"].lower()] = e["name_canonical"]
    rows = [json.loads(l) for l in open(st_path) if l.strip()]
    out_dir = ed.dir / "20_synthetic" / "_settimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n# {ed.slug} SET-TIMES: {len(rows)} sets, vars/set={args.settime_variations}")

    async def with_sem(coro):
        async with sem:
            return await coro

    tasks = []
    for i, r in enumerate(rows):
        artist = by_lower.get(r.get("artist", "").lower().strip(), r.get("artist", "").title())
        camel = r.get("stage_raw") or FE.camel(r.get("stage", ""))
        day = r.get("day", "")
        start, end = r.get("start", ""), r.get("end", "")
        if not (artist and camel and day and start):
            continue
        date = DAY_DATE.get(day, "")
        is_sunrise = "SUNRISE SET" in (r.get("tags") or [])
        tasks.append((i, artist, r.get("stage", ""), day, asyncio.create_task(with_sem(
            gen_settime_variations(client, artist, camel, FE.spaced(r.get("stage", "")) or camel,
                                   day, date, start, end, is_sunrise, args.settime_variations)))))

    n_chars, n_lines, n_drop = 0, 0, 0
    for i, artist, stage_slug, day, task in tasks:
        fname = out_dir / f"{i:03d}__{_fslug(artist)}__{stage_slug}__{day.lower()}.md"
        if fname.exists() and not args.force:
            continue
        try:
            variations = await task
        except Exception:
            variations = []
        kept = [v for v in variations if not D.violates(v)]      # never co-mention a confusable pair
        n_drop += len(variations) - len(kept)
        if not kept:
            continue
        header = (f"# Set-time: {artist} -> {day}\n# Source: set_times.jsonl (community schedule)\n"
                  f"# Generator: {MODEL} set-time variations\n"
                  "# Synthetic: true  (domain_class: inferred-by-model)\n\n")
        fname.write_text(header + "\n".join(kept) + "\n", encoding="utf-8")
        n_chars += sum(len(v) for v in kept)
        n_lines += len(kept)
    log(f"   → {n_lines} set-time variation lines across {len(tasks)} sets "
        f"({n_drop} dropped by co-mention audit), {n_chars:,} chars")
    return n_chars


AUDIENCES = ["a first-time festivalgoer", "a veteran raver", "a parent helping their kid", "a music journalist"]

# -------- LLM client (Google AI Studio plan A, OpenRouter plan B) ----------

def _resolve_active_provider():
    """Single resolver — googleai wins whenever its key is present."""
    try:
        from pipeline.llm_provider import resolve_provider
        return resolve_provider(forced_model=MODEL)
    except Exception:
        pass
    import subprocess

    def _lookup(*args: str) -> str:
        try:
            r = subprocess.run(
                ["secret-tool", "lookup", *args],
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    g = (
        (os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        or _lookup("service", "googleai")
    )
    if g:
        class _P:  # minimal stand-in for Provider
            name, key = "googleai", g
            model = "gemma-4-31b-it"
            chat_url = GOOGLE_AI_URL
        return _P()
    o = (
        (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        or _lookup("service", "openrouter", "key_type", "api_key")
        or _lookup("service", "openrouter")
    )
    if o:
        class _P:
            name, key = "openrouter", o
            model = "google/gemma-4-31b-it"
            chat_url = "https://openrouter.ai/api/v1/chat/completions"
        return _P()
    sys.exit(
        "no LLM provider key: Google AI Studio (GOOGLE_AI_API_KEY / keyring service=googleai) "
        "plan A, or OpenRouter (OPENROUTER_API_KEY / keyring service=openrouter) plan B"
    )


def get_api_key() -> str:
    """ACTIVE provider key (googleai preferred). Name kept for call-site compat."""
    return _resolve_active_provider().key


_ACTIVE = _resolve_active_provider()
API_KEY = _ACTIVE.key
OPENROUTER_URL = _ACTIVE.chat_url  # rebound to the active endpoint
MODEL = _ACTIVE.model
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://kandibuddy.local",
    "X-Title": "Kandi CPT synthetic corpus",
}
print(f"[synth_corpus] provider={_ACTIVE.name} model={MODEL}", file=sys.stderr, flush=True)


async def call_openrouter(client: httpx.AsyncClient, system: str, user: str, max_retries: int = 3,
                          max_tokens: int | None = None) -> str:
    """Historical name — posts to the ACTIVE provider endpoint (googleai preferred)."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens or MAX_TOKENS_OUT,
    }
    last_err: str = ""
    for attempt in range(max_retries):
        try:
            resp = await client.post(OPENROUTER_URL, json=body, headers=HEADERS, timeout=120)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                print(f"[call_llm/{_ACTIVE.name}] {last_err}", file=sys.stderr, flush=True)
            else:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content
                last_err = f"Empty content; full body: {resp.text[:300]}"
                print(f"[call_llm/{_ACTIVE.name}] {last_err}", file=sys.stderr, flush=True)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[call_llm/{_ACTIVE.name}] attempt {attempt+1}/{max_retries}: {last_err}", file=sys.stderr, flush=True)
        await asyncio.sleep(2 ** attempt)
    return f"[generation failed: {last_err}]"

# -------- Corpus chunking + entity extraction -----------------------------

def chunk_text(text: str, target_chars: int = CHUNK_TOKENS * 4) -> list[str]:
    """Split on paragraph boundaries; pack into chunks of target_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) > target_chars and cur:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

# -------- Generation orchestration -----------------------------------------

async def gen_strategy(client: httpx.AsyncClient, chunk: str, strategy: str,
                       audience: str | None = None) -> str:
    system = SYSTEM_PREAMBLE + STRATEGIES[strategy]
    if audience:
        system = system + f"\n\nWrite this for the audience: {audience}."
    return await call_openrouter(client, system, chunk)

# -------- Main runner -------------------------------------------------------

def log(msg: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

_NOISE_SUBSTR = ("no information regarding", "cannot be produced", "no factual", "consists entirely of",
                 "consists of navigation", "does not contain", "no main article", "unable to generate",
                 "i cannot create", "no relevant", "contains no information", "privacy preference")
_NOISE_PREFIX = ("the provided source", "the provided content", "the provided text", "the source content",
                 "the document does", "the text provided", "based on the provided")


def _is_noise(text: str) -> bool:
    """Meta-commentary the model emits when a source doc is junk (cookie/nav pages). Must NOT enter
    the corpus — it would be amplified by the synthetic expansion."""
    low = text.strip().lower()
    return any(p in low for p in _NOISE_SUBSTR) or low.startswith(_NOISE_PREFIX)


async def process_document(client, doc_path, sem, args, corpus_dir, synth_dir) -> int:
    """Diversify ONE source doc via the rephrase STRATEGIES (chunk-grounded, source-faithful).

    Fact installation (artist→stage→day) is NOT done here — it is handled separately and balanced by
    expand_facts_for_edition(). This function only produces diverse natural-language renderings of the
    source TEXT, so it cannot invent cross-entity relations (the prior random-pairing failure mode)."""
    rel = doc_path.relative_to(corpus_dir)
    out_dir = synth_dir / rel.with_suffix("").as_posix()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        return 0                                   # idempotent: already generated
    out_dir.mkdir(parents=True, exist_ok=True)
    text = doc_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)
    if args.max_chunks_per_doc:
        chunks = chunks[: args.max_chunks_per_doc]
    log(f"\n## {rel} — {len(chunks)} chunks")
    total_chars = 0

    async def with_sem(coro):
        async with sem:
            return await coro

    chunk_tasks = []
    for ci, chunk in enumerate(chunks):
        for strat in STRATEGIES:
            chunk_tasks.append((ci, strat, None,
                                asyncio.create_task(with_sem(gen_strategy(client, chunk, strat)))))
        for strat in ("extract_knowledge", "friend_chat"):
            aud = random.choice(AUDIENCES)
            chunk_tasks.append((ci, strat, aud,
                                asyncio.create_task(with_sem(gen_strategy(client, chunk, strat, aud)))))

    for ci, strat, audience, task in chunk_tasks:
        try:
            text_out = await task
        except Exception as e:
            text_out = f"[error: {e}]"
        if text_out.startswith("[generation failed") or text_out.startswith("[error") or _is_noise(text_out):
            continue                               # drop failed/empty + meta-commentary noise
        suffix = (re.sub(r"[^A-Za-z0-9]+", "_", audience).strip("_")[:40] or "x") if audience else f"chunk{ci}"
        fname = f"{strat}__{suffix}__{ci:03d}.md"
        (out_dir / fname).write_text(
            f"# Source: {rel}\n# Strategy: {strat}\n# Generator: {MODEL}\n"
            f"# Synthetic: true  (domain_class: inferred-by-model)\n"
            + (f"# Audience: {audience}\n" if audience else "")
            + f"# Chunk: {ci}\n\n{text_out}\n", encoding="utf-8")
        total_chars += len(text_out)

    log(f"   → {total_chars:,} chars across {len(chunk_tasks)} generations")
    return total_chars


def _fslug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "x"


async def expand_facts_for_edition(client, ed, sem, args) -> int:
    """Balanced per-fact expansion: every verified triple gets the SAME number of representations —
    deterministic multi-directional templates (correctness floor, no API) + LLM paraphrases (diversity)
    — plus grounded stage/day roster docs. This is the EntiGraph replacement: it expands ONLY explicit
    triples and never pairs entities, so it cannot reproduce the 50/50 right/wrong confabulation."""
    triples = FE.load_triples(ed)
    if not triples:
        return 0
    synth_dir = ed.dir / "20_synthetic"
    facts_dir, rosters_dir = synth_dir / "_facts", synth_dir / "_rosters"
    facts_dir.mkdir(parents=True, exist_ok=True)
    rosters_dir.mkdir(parents=True, exist_ok=True)
    cov = FE.coverage(triples)
    stage_counts, day_counts = cov["by_stage"], cov["by_day"]
    log(f"\n# {ed.slug} FACT EXPANSION: {cov['n_triples']} triples, {cov['artists']} artists, "
        f"by_stage={stage_counts}, by_day={day_counts}, base_vars/fact={args.variations_per_fact} "
        f"(rebalanced: under-represented stages/days get up to 3x)")

    n_chars = 0
    stages_present = {s for _, s, _ in triples}
    # deterministic rosters + calendar + festival overview + B2B + multi-stage (no API): cover EVERY
    # relationship direction — stage→[artists], day→[artists], day↔date, festival→[stages], stage↔genre,
    # B2B partnerships, multi-stage artists.
    for name, txt in (("calendar.md", FE.calendar_doc()),
                      ("festival_overview.md", FE.festival_overview_doc())):
        (rosters_dir / name).write_text(txt, encoding="utf-8"); n_chars += len(txt)
    for st in stages_present:
        txt = FE.stage_roster(st, triples)
        (rosters_dir / f"stage__{st}.md").write_text(txt, encoding="utf-8"); n_chars += len(txt)
    for d in {d for _, _, d in triples if d}:
        txt = FE.day_roster(d, triples)
        (rosters_dir / f"day__{d.lower()}.md").write_text(txt, encoding="utf-8"); n_chars += len(txt)
    for name, lines in (("b2b_partnerships.md", FE.b2b_lines(triples)),
                        ("multi_stage_artists.md", FE.multistage_lines(triples))):
        if lines:
            body = f"# {FE.FESTIVAL} — {name[:-3].replace('_',' ')}\n\n" + "\n".join(lines) + "\n"
            (rosters_dir / name).write_text(body, encoding="utf-8"); n_chars += len(body)

    async def with_sem(coro):
        async with sem:
            return await coro

    tasks = []
    # (1) per-artist plays_at facts (rebalanced)
    for idx, (artist, st, day) in enumerate(triples):
        fname = facts_dir / f"{idx:03d}__{_fslug(artist)}__{st}__{(day or 'na').lower()}.md"
        if fname.exists() and not args.force:
            continue
        n_var = FE.reps_for(st, day, stage_counts, day_counts, args.variations_per_fact)
        tasks.append((fname, artist, st, day, asyncio.create_task(with_sem(
            gen_fact_variations(client, artist, FE.camel(st), FE.spaced(st), day, n_var)))))
    # (2) per-stage attribute facts (genre/brand/rank) for ALL 9 stages — the stage→attribute half of
    # "which stage?" (incl. stereoBLOOM/bionicJUNGLE which have genres but no 2026 lineup in our triples)
    for st in sorted(FE.STAGE_META):
        fname = facts_dir / f"stage_attr__{st}.md"
        if fname.exists() and not args.force:
            continue
        tasks.append((fname, f"[stage:{st}]", st, "", asyncio.create_task(with_sem(
            gen_stage_variations(client, st, max(10, args.variations_per_fact))))))
    n_stage_attr = 0
    for fname, artist, st, day, task in tasks:
        try:
            variations = await task
        except Exception:
            variations = []
        if artist.startswith("[stage:"):                          # stage-attribute fact (genre/brand/rank)
            templates = FE.stage_attribute_lines(st)
            head_what, src = f"stage {FE.camel(st)} attributes (genre/brand/rank)", "STAGE_META (official stage descriptions)"
            n_stage_attr += 1
        else:                                                     # per-artist plays_at fact
            templates = FE.fact_templates(artist, st, day)
            head_what = f"{artist} -> {FE.camel(st)}" + (f" -> {day}" if day else "")
            src = "15_structured plays_at (deterministic schedule parse)"
        body = "\n".join(templates + variations)                  # deterministic templates always present
        header = (f"# Fact: {head_what}\n# Source: {src}\n"
                  f"# Generator: {MODEL} variations + deterministic templates\n"
                  "# Synthetic: true  (domain_class: inferred-by-model)\n\n")
        fname.write_text(header + body + "\n", encoding="utf-8")
        n_chars += len(body)
    log(f"   → {len(tasks)-n_stage_attr} artist facts + {n_stage_attr} stage-attribute facts + rosters "
        f"(stage/day/calendar/overview/b2b/multistage), {n_chars:,} chars")
    return n_chars


async def main_async(args):
    import sys
    from pipeline.editions import iter_editions
    eds = list(iter_editions("edc", only=args.editions))
    if len(eds) > 1 and not args.yes:
        sys.exit(f"{len(eds)} editions × ~20 LLM calls/doc = real OpenRouter cost. "
                 f"Re-run with --yes, or --editions las_vegas/2026.")
    keep_path = ROOT / "data" / "_build" / "curation" / "keep_list.json"
    keep = set(json.loads(keep_path.read_text())) if keep_path.exists() else None

    log(f"# Synth started {time.strftime('%Y-%m-%d %H:%M:%S')}  model={MODEL}  "
        f"editions={len(eds)}  keep_list={'on' if keep else 'off'}")
    sem = asyncio.Semaphore(args.concurrency)
    log(f"  concurrency={args.concurrency}  vars/fact={args.variations_per_fact}  "
        f"max_chunks/doc={args.max_chunks_per_doc or 'all'}  force={args.force}")
    total_chars = 0
    async with httpx.AsyncClient(headers=HEADERS, timeout=180,
                                 limits=httpx.Limits(max_connections=args.concurrency + 4)) as client:
        for ed in eds:
            corpus_dir = ed.dir / "10_clean"
            if not corpus_dir.is_dir():
                continue
            synth_dir = ed.dir / "20_synthetic"
            docs = [d for d in sorted(corpus_dir.rglob("*.md"))
                    if d.name != "README.md" and (keep is None or str(d.relative_to(ROOT)) in keep)]
            if args.limit:
                docs = docs[: args.limit]
            log(f"\n# edition {ed.slug}: {len(docs)} kept docs")
            t0 = time.time()
            # (1) balanced per-fact expansion (the queryable schedule, equal coverage per fact) +
            # (2) source-text diversification across all docs CONCURRENTLY — the shared semaphore caps
            # total API calls at --concurrency, fully utilizing the rate budget.
            fact_chars = 0
            if not args.no_fact_expansion:
                fact_chars = await expand_facts_for_edition(client, ed, sem, args)
            if args.cultural_variations:
                fact_chars += await expand_cultural_for_edition(client, ed, sem, args)
            if args.settime_variations:
                fact_chars += await expand_settimes_for_edition(client, ed, sem, args)
            results = await asyncio.gather(
                *[process_document(client, doc, sem, args, corpus_dir, synth_dir)
                  for doc in docs], return_exceptions=True)
            ed_chars = fact_chars + sum(r for r in results if isinstance(r, int))
            total_chars += ed_chars
            for doc, r in zip(docs, results):
                if isinstance(r, Exception):
                    log(f"   ! {doc.name}: {type(r).__name__}: {r}")
            log(f"   ({ed.slug} done: {ed_chars:,} chars in {time.time()-t0:.0f}s; cumulative {total_chars:,})")

    log(f"\n# Done. Total synthetic chars: {total_chars:,} (~{total_chars // 4:,} tokens)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editions", default="las_vegas/2026", help="'city/year,…' (default LV-2026)")
    parser.add_argument("--limit", type=int, help="max docs per edition")
    parser.add_argument("--max-chunks-per-doc", type=int, default=0, help="cap chunks/doc (0=all)")
    parser.add_argument("--variations-per-fact", type=int, default=12,
                        help="LLM paraphrases per verified triple (balanced; +12 deterministic templates)")
    parser.add_argument("--no-fact-expansion", action="store_true",
                        help="skip the balanced per-fact expansion (chunk strategies only)")
    parser.add_argument("--cultural-variations", type=int, default=0,
                        help="conversation variations per cultural primitive (0=off); installs PLUR/pacing/"
                             "feet/hydration/kandi culture as primitive→many-convos, co-mention audited")
    parser.add_argument("--settime-variations", type=int, default=0,
                        help="LLM variations per set in set_times.jsonl (0=off); installs artist↔day↔time "
                             "binding at the shared stage (the day is the confusable discriminator), audited")
    parser.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT, help="parallel OpenRouter calls")
    parser.add_argument("--yes", action="store_true", help="confirm multi-edition (OpenRouter cost)")
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
