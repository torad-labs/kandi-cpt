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
LOG_FILE = ROOT / "logs" / "synth_progress.md"

MODEL = "google/gemma-4-31b-it"   # exact same family as on-device E2B target — voice + tokenizer aligned
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT = 6     # OpenRouter rate limits — be polite
CHUNK_TOKENS = 800     # rough chunk size; we count by chars/4
MAX_TOKENS_OUT = 900   # per-generation cap
TEMPERATURE = 0.85     # high enough for diversity, low enough for coherence

# -------- Generator strategies --------------------------------------------

# Each strategy gets its own system prompt. We feed the source chunk in the
# user turn. Strategies are mostly orthogonal — running them all on the same
# chunk produces 7 distinct generations with high lexical + semantic diversity.

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

# Cross-strategy: EntiGraph-style entity-pair relation generation
ENTITY_EXTRACT_PROMPT = (
    "Extract all proper-noun entities from the source content related to "
    "Electric Daisy Carnival 2026. Categorize into: artists, stages, days, "
    "venues/places, policies/rules, events. Output ONLY a JSON object with "
    "those keys. Each value is a list of strings. No commentary."
)

ENTITY_RELATION_PROMPT = (
    "You are writing a factual passage that describes the relationships and "
    "connections between the following entities from EDC Las Vegas 2026:\n\n"
    "Entities: {entities}\n\n"
    "Source context for grounding (do not contradict):\n{context}\n\n"
    "Write 2-4 paragraphs that:\n"
    "- State concrete claims about how these entities relate (which artist "
    "  plays which stage on which day, which policy applies where, etc.)\n"
    "- Use Insomniac's stage camelCase (kineticFIELD, cosmicMEADOW, etc.)\n"
    "- Avoid filler and meta-commentary; just factual prose\n"
    "- Reference SPECIFIC days/times where the source provides them"
)

AUDIENCES = ["a first-time festivalgoer", "a veteran raver", "a parent helping their kid", "a music journalist"]

# -------- OpenRouter client -----------------------------------------------

def get_api_key() -> str:
    """OpenRouter key: env var preferred (venv has no PyGObject); fall back
    to libsecret via subprocess for canonical-location-discipline per
    CLAUDE.md §9."""
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    # Fallback: shell out to system Python which has gi available
    import subprocess
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "-c",
             "import gi; gi.require_version('Secret','1'); "
             "from gi.repository import Secret; "
             "schema = Secret.Schema.new('ai.kandibuddy.OpenRouter', "
             "Secret.SchemaFlags.NONE, "
             "{'service': Secret.SchemaAttributeType.STRING, "
             "'key_type': Secret.SchemaAttributeType.STRING}); "
             "k = Secret.password_lookup_sync(schema, "
             "{'service': 'openrouter', 'key_type': 'api_key'}, None); "
             "print(k or '')"],
            capture_output=True, text=True, timeout=10,
        )
        key = result.stdout.strip()
        if key:
            return key
    except Exception:
        pass
    sys.exit("OpenRouter key not found in $OPENROUTER_API_KEY or libsecret (service=openrouter)")

API_KEY = get_api_key()
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://kandibuddy.local",
    "X-Title": "Kandi CPT synthetic corpus",
}

async def call_openrouter(client: httpx.AsyncClient, system: str, user: str, max_retries: int = 3) -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_OUT,
    }
    last_err: str = ""
    for attempt in range(max_retries):
        try:
            resp = await client.post(OPENROUTER_URL, json=body, headers=HEADERS, timeout=120)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                print(f"[call_openrouter] {last_err}", file=sys.stderr, flush=True)
            else:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content
                last_err = f"Empty content; full body: {resp.text[:300]}"
                print(f"[call_openrouter] {last_err}", file=sys.stderr, flush=True)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[call_openrouter] attempt {attempt+1}/{max_retries}: {last_err}", file=sys.stderr, flush=True)
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

async def extract_entities(client: httpx.AsyncClient, text: str) -> dict:
    """One-shot entity extraction over a document. Returns categorized dict."""
    raw = await call_openrouter(client, ENTITY_EXTRACT_PROMPT, text[:8000])
    # Best-effort JSON parse — model might wrap in markdown
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

# -------- Generation orchestration -----------------------------------------

async def gen_strategy(client: httpx.AsyncClient, chunk: str, strategy: str,
                       audience: str | None = None) -> str:
    system = STRATEGIES[strategy]
    if audience:
        system = system + f"\n\nWrite this for the audience: {audience}."
    return await call_openrouter(client, system, chunk)

async def gen_entity_relation(client: httpx.AsyncClient, context: str,
                              entities: list[str]) -> str:
    prompt = ENTITY_RELATION_PROMPT.format(
        entities=", ".join(entities),
        context=context[:3000],
    )
    return await call_openrouter(client, prompt, "Write the passage.")

# -------- Main runner -------------------------------------------------------

def log(msg: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

async def process_document(client: httpx.AsyncClient, doc_path: Path,
                           sem: asyncio.Semaphore, args) -> int:
    """Generate synthetic variants for a single source doc. Returns char count."""
    text = doc_path.read_text(encoding="utf-8")
    rel = doc_path.relative_to(CORPUS_DIR)
    out_dir = SYNTH_DIR / rel.with_suffix("").as_posix()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_text(text)
    if args.max_chunks_per_doc:
        chunks = chunks[: args.max_chunks_per_doc]

    log(f"\n## {rel} — {len(chunks)} chunks")
    total_chars = 0

    async def with_sem(coro):
        async with sem:
            return await coro

    # Entity extraction (one shot per doc, parallel with first chunk's
    # strategy run is fine)
    entity_task = asyncio.create_task(with_sem(extract_entities(client, text)))

    # For each chunk: run every strategy (one call per strategy per chunk)
    # plus a varied-audience call for two of the strategies
    chunk_tasks = []
    for ci, chunk in enumerate(chunks):
        for strat in STRATEGIES:
            chunk_tasks.append(
                ("strat", ci, strat, None,
                 asyncio.create_task(with_sem(gen_strategy(client, chunk, strat))))
            )
        # Audience variations on 2 picks
        for strat in ("extract_knowledge", "friend_chat"):
            aud = random.choice(AUDIENCES)
            chunk_tasks.append(
                ("strat_aud", ci, strat, aud,
                 asyncio.create_task(with_sem(gen_strategy(client, chunk, strat, aud))))
            )

    # Resolve entity extraction; once entities are known, fan out
    # entity-pair relation generations (EntiGraph-style)
    entities = await entity_task
    all_ents: list[str] = []
    for _, vals in (entities or {}).items():
        if isinstance(vals, list):
            all_ents.extend(str(v) for v in vals if v)
    all_ents = list(dict.fromkeys(all_ents))   # dedupe, preserve order

    relation_tasks = []
    if len(all_ents) >= 2:
        n_relations = min(args.relations_per_doc, max(0, len(all_ents) // 2))
        for ri in range(n_relations):
            sample = random.sample(all_ents, k=min(random.choice([2, 3, 4]), len(all_ents)))
            ctx = chunks[ri % len(chunks)] if chunks else ""
            relation_tasks.append(
                ("relation", 0, "entigraph", ",".join(sample[:3]),
                 asyncio.create_task(with_sem(gen_entity_relation(client, ctx, sample))))
            )

    # Resolve all generations
    for kind, ci, strat, audience, task in chunk_tasks + relation_tasks:
        try:
            text_out = await task
        except Exception as e:
            text_out = f"[error: {e}]"
        if text_out.startswith("[generation failed") or text_out.startswith("[error"):
            continue
        suffix = audience.replace(" ", "_") if audience else f"chunk{ci}"
        fname = f"{strat}__{suffix}__{ci:03d}.md"
        (out_dir / fname).write_text(
            f"# Source: {rel}\n# Strategy: {strat}\n"
            + (f"# Audience: {audience}\n" if audience else "")
            + f"# Chunk: {ci}\n\n{text_out}\n",
            encoding="utf-8",
        )
        total_chars += len(text_out)

    log(f"   → {total_chars:,} chars across {len(chunk_tasks) + len(relation_tasks)} generations")
    return total_chars

async def main_async(args):
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    log(f"# Synthetic corpus generation started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  model: {MODEL}")
    log(f"  strategies: {len(STRATEGIES)} per chunk + entity-relation expansion")
    log(f"  max_chunks_per_doc: {args.max_chunks_per_doc or 'all'}")
    log(f"  relations_per_doc: {args.relations_per_doc}")
    log(f"  concurrent: {MAX_CONCURRENT}")

    docs = sorted(CORPUS_DIR.rglob("*.md"))
    log(f"  source docs: {len(docs)}")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    total_chars = 0
    # Set auth headers globally on the client so call_openrouter doesn't need to pass them
    async with httpx.AsyncClient(headers=HEADERS, timeout=180) as client:
        for doc in docs:
            t0 = time.time()
            chars = await process_document(client, doc, sem, args)
            total_chars += chars
            log(f"   (elapsed {time.time()-t0:.0f}s; cumulative {total_chars:,} chars)")

    log(f"\n# Done. Total synthetic chars: {total_chars:,} (~{total_chars // 4:,} tokens)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks-per-doc", type=int, default=0,
                        help="Cap chunks processed per source doc (0 = all).")
    parser.add_argument("--relations-per-doc", type=int, default=8,
                        help="EntiGraph-style entity-pair relations per doc.")
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
