#!/usr/bin/env python3
"""Evaluate a checkpoint on voice survival + knowledge probes.

Two evaluations bundled:

(A) Voice survival — 20 conversational prompts sampled at temp=1.2,
    top_k=64. Score: closing-statement pattern present, lowercase
    register, friend-tone (no "As an AI..." or terse robotic outputs).
    Pass = score >= 2 of 3.

(B) Knowledge probes — parses docs/KANDI_CPT_PROBES.md, runs Direct Q
    and Cloze probes against the checkpoint, keyword-match auto-grade,
    pass rate per tier. Scenario probes are saved for human eyeball
    inspection.

Run:
    python scripts/eval.py --checkpoint checkpoints/ckpt_1/final
    python scripts/eval.py --checkpoint <path-to-base-gemma>     # baseline

Output:
    logs/eval_<basename>/{voice.json,probes.json,scenarios.txt}
"""
from pathlib import Path
import argparse
import json
import re

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
PROBES_FILE = ROOT.parent.parent / "apps" / "kandibuddy" / "docs" / "KANDI_CPT_PROBES.md"
SYSTEM_PROMPT_FILE = ROOT.parent.parent / "apps" / "kandibuddy" / "prompts" / "chat" / "system.md"

VOICE_PROMPTS = [
    "hey, what's up?",
    "first time at EDC, super nervous lol",
    "yo where do I even start",
    "I lost my friends",
    "is it dumb that I'm scared to go alone?",
    "tired AF, should I push through or sleep?",
    "what's playing right now",
    "I'm at kineticFIELD and I'm overwhelmed",
    "my feet are KILLING me",
    "do people actually trade kandi or is that a meme",
    "I forgot to eat dinner, am I cooked",
    "show me the lineup",
    "is camp EDC worth it",
    "is the desert really that hot at night",
    "what should I wear",
    "I got separated from my group, no cell service",
    "first set I should catch saturday?",
    "is martin garrix worth seeing",
    "honestly nervous about the drugs being everywhere",
    "kinda emotional rn, sunrise is hitting different",
]

def _extract_era(rest: str) -> tuple[str, str]:
    """Parse `era: X / keyword: Y` OR just `keyword: Y`. Returns (era, keyword).
    Era is 2026 (this year's specific facts — couldn't be in training) or
    stable (recurring festival facts — may be in training)."""
    era_match = re.search(r"era:\s*([\w-]+)\s*/", rest)
    kw_match = re.search(r"keyword:\s*(.+)$", rest)
    era = era_match.group(1).strip() if era_match else "unknown"
    keyword = kw_match.group(1).strip() if kw_match else rest.strip()
    return era, keyword

def parse_probes(md_path: Path) -> list[dict]:
    """Pull Q/A/keyword and Stem/Continuation/keyword probes, with optional era tag."""
    if not md_path.exists():
        print(f"warning: probes file not found at {md_path}")
        return []
    text = md_path.read_text()
    probes = []
    # Direct Q: Q: "..." / A: ... [/ era: X] / keyword: ...
    pattern_q = re.compile(
        r"Q:\s*\"([^\"]+)\"\s*[\n\r]+\s*A:\s*([^\n]+?)\s*/\s*((?:era:[^/]+/\s*)?keyword:[^\n]+)",
        re.MULTILINE,
    )
    for m in pattern_q.finditer(text):
        era, keyword = _extract_era(m.group(3))
        probes.append({"shape": "direct", "prompt": m.group(1).strip(),
                       "answer": m.group(2).strip(), "era": era, "keyword": keyword})
    # Cloze: Stem: "..." / Continuation: ... [/ era: X] / keyword: ...
    pattern_c = re.compile(
        r"Stem:\s*\"([^\"]+)\"\s*[\n\r]+\s*Continuation:\s*([^\n]+?)\s*/\s*((?:era:[^/]+/\s*)?keyword:[^\n]+)",
        re.MULTILINE,
    )
    for m in pattern_c.finditer(text):
        era, keyword = _extract_era(m.group(3))
        probes.append({"shape": "cloze", "prompt": m.group(1).strip(),
                       "answer": m.group(2).strip(), "era": era, "keyword": keyword})
    return probes

DEFLECTION_PATTERNS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i do not have", "i don't have information", "i don't have any",
    "i cannot provide", "i can't provide", "i'm unable to",
    "i am unable", "please ask", "ask a question about",
    "let me check", "let me look", "i need to check", "i need to look",
    "can you tell me", "can you confirm", "can you specify",
    "could you tell me", "could you specify", "could you confirm",
    "what day are you", "which day are you", "what day is this",
    "i need more information", "i need more context",
    "to give you the best", "could you provide more",
    "tool_code", "print(get_",   # generic-format tool emissions, not Kandi shape
    "tool:", "call:",   # generic tool-call emissions in probe output
    "get_lineup(", "get_map_location(",   # tool-name leakage in prose
)

def has_deflection(output: str) -> bool:
    lower = output.lower()
    return any(p in lower for p in DEFLECTION_PATTERNS)

def _whole_word_in(needle: str, hay: str) -> bool:
    """Whole-word case-insensitive substring search. Avoids 'not' matching
    'cannot' and 'map' matching 'mapping'."""
    pattern = r"\b" + re.escape(needle.strip().lower()) + r"\b"
    return re.search(pattern, hay.lower()) is not None

def keyword_match(output: str, keyword_field: str) -> bool:
    """Case-insensitive whole-word match. `|` separates OR-alternates;
    `&` separates AND-required terms within an alternate. Deflection
    patterns (`I don't know`, `let me check`, etc.) override a match to
    FAIL — a probe must commit to the answer, not deflect."""
    if has_deflection(output):
        return False
    for alt in keyword_field.split("|"):
        terms = [t.strip() for t in alt.split("&") if t.strip()]
        if terms and all(_whole_word_in(t, output) for t in terms):
            return True
    return False

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]")

FESTIVAL_TOKENS = (
    "edc", "festival", "kandibuddy", "ground control", "lineup", "set time",
    "stage", "kinetic", "cosmic meadow", "circuit grounds", "neon garden",
    "basspod", "bass pod", "wasteland", "quantum valley", "stereo bloom",
    "bionic jungle", "camp edc", "hydrat", "water station", "electrolit",
    "medical tent", "shuttle", "speedway",
)

REDIRECT_MARKERS = (
    "i can help with", "stick to", "focus on", "not my thing", "let me help",
    "i'm here for", "festival", "edc", "out of my", "not what i", "redirect",
)

def voice_score(output: str) -> tuple[int, list[str]]:
    """Aligned with the current Kandi system prompt: warm, brief, no emojis,
    festival-relevant or gently redirecting. Returns (score 0-3, features)."""
    lower = output.lower()
    hits = []
    words = output.split()

    # 1. Brief — system prompt says "Friends are tired and probably typing
    # one-handed. Say less." Pass = ≤120 words.
    if len(words) <= 120:
        hits.append("brief")

    # 2. No emojis — system prompt: "Warm, brief, no emojis."
    if not EMOJI_RE.search(output) and "as an ai" not in lower:
        hits.append("no_emoji")

    # 3. On-topic OR explicit redirect — model should stay in festival domain
    # or redirect gently per system prompt: "redirect those gently back to
    # festival help."
    if any(t in lower for t in FESTIVAL_TOKENS) or any(r in lower for r in REDIRECT_MARKERS):
        hits.append("on_topic_or_redirect")

    return len(hits), hits

_SYSTEM_PROMPT_CACHE: dict[str, str] = {}

# Voice eval uses the production Kandi system prompt — measures ship behavior.
# Probe eval uses a memory-only prompt that disables tool-deferral — measures
# what's in the weights. CPT effectiveness = probe pass rate climbing with
# the memory prompt active.
MEMORY_EVAL_PROMPT = (
    "You are an expert on Electric Daisy Carnival 2026 (May 15-17, 2026, "
    "Las Vegas Motor Speedway). Answer questions from memory, concisely, in "
    "one or two sentences. Be specific — name artists, stages, times, and "
    "places directly. Do not ask clarifying questions. Do not say you need "
    "to look something up. Do not emit tool calls. If you genuinely do not "
    "know the answer, say 'I do not know' and stop."
)

def get_kandi_system_prompt() -> str:
    if "kandi" not in _SYSTEM_PROMPT_CACHE:
        _SYSTEM_PROMPT_CACHE["kandi"] = (
            SYSTEM_PROMPT_FILE.read_text() if SYSTEM_PROMPT_FILE.exists() else ""
        )
    return _SYSTEM_PROMPT_CACHE["kandi"]

def chat_format(prompt: str, tokenizer, mode: str = "voice") -> str:
    """Wrap prompt in Gemma 4 chat template.

    mode='voice' → Kandi production system prompt (tool-first behavior)
    mode='memory' → memory-only system prompt (test weight knowledge)
    """
    system = get_kandi_system_prompt() if mode == "voice" else MEMORY_EVAL_PROMPT
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some Gemma templates reject `system` role — prepend manually.
        merged = (f"{system}\n\n{prompt}" if system else prompt)
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return merged

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 1.2, top_k: int = 64, mode: str = "voice") -> str:
    text = chat_format(prompt, tokenizer, mode=mode)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to model checkpoint dir")
    parser.add_argument("--max_probes", type=int, default=0, help="cap probe count (0 = all)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out_dir = ROOT / "logs" / f"eval_{ckpt.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    # Detect if this checkpoint has a modules_to_save adapter (full-FT
    # embed_tokens). PEFT-saved modules_to_save layers contain the full
    # bf16 weight tensor (~800 MB for embed_tokens at vocab=262144) — when
    # combined with a 4-bit base load, the briefly-doubled embed (4-bit
    # base + bf16 saved) OOMs around 13 GB on a 16 GB card.
    # Fallback path: load base in bf16 (10 GB) when modules_to_save is
    # present; the saved adapter then patches in-place. Total ~11 GB, fits.
    adapter_cfg = ckpt / "adapter_config.json"
    has_modules_to_save = False
    if adapter_cfg.exists():
        try:
            cfg = json.loads(adapter_cfg.read_text())
            mts = cfg.get("modules_to_save") or []
            has_modules_to_save = bool(mts)
        except Exception:
            pass

    if has_modules_to_save:
        print(f"  (modules_to_save detected — explicit PEFT load via PeftModel)")
        # Transformers' from_pretrained(ckpt_dir) path has a save/load naming
        # asymmetry for modules_to_save: it saves under `embed_tokens.weight`
        # but the loader expects `embed_tokens.modules_to_save.default.weight`.
        # The trained weight ends up UNEXPECTED + the wrapped key MISSING, so
        # the training is silently discarded. Explicit PEFT load fixes this:
        # load base from the ORIGINAL model path, then apply adapter via
        # PeftModel.from_pretrained which handles modules_to_save correctly.
        from peft import PeftModel
        base_path = cfg.get("base_model_name_or_path", str(MODEL_PATH := Path("/home/marcos/Documents/dev/models/gemma-4-E2B-it")))
        base = AutoModelForImageTextToText.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
        model = model.to("cuda:0")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(ckpt),
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    model.eval()

    # (A) Voice survival — Kandi production system prompt + temp=1.2 (ship config)
    voice_results = []
    for p in VOICE_PROMPTS:
        out = generate(model, tokenizer, p, max_new_tokens=200, temperature=1.2, top_k=64, mode="voice")
        score, features = voice_score(out)
        voice_results.append({"prompt": p, "output": out, "score": score, "features": features})
    voice_pass = sum(1 for r in voice_results if r["score"] >= 2)
    voice_summary = {
        "total": len(voice_results),
        "pass_count": voice_pass,
        "pass_rate": round(voice_pass / len(voice_results), 3),
        "results": voice_results,
    }
    (out_dir / "voice.json").write_text(json.dumps(voice_summary, indent=2))
    print(f"Voice survival: {voice_pass}/{len(voice_results)} = {voice_pass/len(voice_results):.0%}")

    # (B) Knowledge probes — memory-only system prompt + low temp (test weights)
    probes = parse_probes(PROBES_FILE)
    if args.max_probes:
        probes = probes[: args.max_probes]
    probe_results = []
    for p in probes:
        out = generate(model, tokenizer, p["prompt"], max_new_tokens=120, temperature=0.3, top_k=64, mode="memory")
        passed = keyword_match(out, p["keyword"])
        probe_results.append({**p, "output": out, "passed": passed})
    n_pass = sum(1 for r in probe_results if r["passed"])
    by_era: dict[str, dict] = {}
    for r in probe_results:
        era = r.get("era", "unknown")
        e = by_era.setdefault(era, {"total": 0, "pass_count": 0})
        e["total"] += 1
        e["pass_count"] += int(r["passed"])
    probe_summary = {
        "total": len(probe_results),
        "pass_count": n_pass,
        "pass_rate": round(n_pass / max(1, len(probe_results)), 3),
        "by_era": by_era,
        "results": probe_results,
    }
    (out_dir / "probes.json").write_text(json.dumps(probe_summary, indent=2))
    print(f"Knowledge probes: {n_pass}/{len(probe_results)} = {n_pass/max(1,len(probe_results)):.0%}")
    for era, stats in sorted(by_era.items()):
        rate = stats["pass_count"] / max(1, stats["total"])
        print(f"  era={era:10s} {stats['pass_count']}/{stats['total']} = {rate:.0%}")

    print(f"\nSaved: {out_dir}/voice.json + {out_dir}/probes.json")

if __name__ == "__main__":
    main()
