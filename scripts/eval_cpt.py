#!/usr/bin/env python3
"""Self-contained CPT eval: did the festival knowledge install? Base vs CPT, generation + keyword grade.

Probes are drawn from OUR corpus's verified facts (the 2026 lineup→stage assignments extracted into
15_structured, the dates/venue/stages). era=2026 facts are post-cutoff — the base model should miss
them and a successful CPT should hit them (the install signal); era=stable facts test retention.

    PYTHONPATH=scripts python scripts/eval_cpt.py --checkpoint checkpoints/cpt/edc_lv_2026
        [--base-only]        # just score the base model (baseline)

Loads base (via the multimodal→text remap) and the CPT checkpoint sequentially (one at a time on
GPU0), prints per-era pass rates + the delta.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer, Gemma4ForCausalLM

import config as C
from pipeline.model_load import load_text_model_bf16

# (question, keyword spec, era). keyword: '|' = any-of, '&' = all-of within an alternative.
# Every 2026 probe is an EXPLICIT, unambiguous fact verified in the source 10_clean docs (cited),
# and post-dates the base model's cutoff → base should miss, a successful CPT should hit. Avoids
# artists that appear in multiple stage contexts (e.g. Peggy Gou is in both the 2026 Neon Garden
# lineup AND a historical Circuit Grounds headliner list → ambiguous, excluded).
PROBES = [
    # --- 2026-specific, EXPLICIT in insomniac_official (lines 214-229, IN the training corpus),
    #     each artist in a single unambiguous stage/day context; post-cutoff → base should miss. ---
    ("On what dates is EDC Las Vegas 2026 held?", "may 15|may 16|may 17|15-17|15–17", "2026"),
    ("Where is EDC Las Vegas 2026 held?", "speedway", "2026"),
    ("Which artist plays the sunrise set at Kinetic Field on Sunday May 17 at EDC 2026?",
     "armin|van buuren", "2026"),                                 # L217
    ("Who plays the Saturday May 16 sunrise set at Kinetic Field at EDC 2026?",
     "above & beyond|above and beyond", "2026"),                  # L216
    ("Which stage does The Prodigy play at EDC Las Vegas 2026?", "cosmic", "2026"),   # L219
    ("On which day does Underworld play Cosmic Meadow at EDC 2026?", "friday", "2026"),  # L219-220
    ("Which stage does Alison Wonderland play at EDC Las Vegas 2026?", "cosmic", "2026"),  # L220
    ("Which stage does Adriatique play at EDC Las Vegas 2026?", "neon", "2026"),      # L224
    ("On which day does Seven Lions play Cosmic Meadow at EDC 2026?", "sunday", "2026"),  # L222
    # --- stable / recurring (retention check; base may already know) ---
    ("What does PLUR stand for in rave culture?", "peace&love&unity&respect", "stable"),
    ("What is kandi at EDC?", "bead|bracelet|trade", "stable"),
    ("Name one of the main stages at EDC Las Vegas.",
     "kineticfield|cosmic|circuitgrounds|circuit grounds|neongarden|neon garden|basspod|bass pod|"
     "wasteland|quantumvalley|quantum valley|stereobloom|bionicjungle|kinetic field", "stable"),
    ("How do most attendees get from the parking/hotels to EDC Las Vegas?", "shuttle", "stable"),
    # --- OOD: SAME 2026 facts, phrasing deliberately UNLIKE the synth templates → tests whether the
    #     binding generalizes past the training surface form (vs the in-dist '2026' echo). CPT should
    #     start moving these; SFT is what makes them robust. ---
    ("Where does The Prodigy perform at EDC 2026?", "cosmic", "2026ood"),                  # vs "which stage"
    ("What day is Underworld's Cosmic Meadow set at EDC 2026?", "friday|may 15", "2026ood"),  # vs "on which day does"
    ("Name a stage Armin van Buuren plays at EDC 2026.", "kinetic", "2026ood"),
    ("Adriatique is booked for which stage at EDC 2026?", "neon", "2026ood"),
    ("On Sunday at EDC 2026, name an artist with a Cosmic Meadow slot.",
     "seven lions|alison|wonderland", "2026ood"),                                          # cosmic Sunday
    ("Which day does Seven Lions perform at EDC 2026?", "sunday|may 17", "2026ood"),
]

SYSTEM = ("You are an expert on Electric Daisy Carnival (EDC) Las Vegas 2026, held May 15-17 2026 at "
          "the Las Vegas Motor Speedway. Answer from memory in one short sentence. Be specific — name "
          "the stage/date/artist directly. Do not ask clarifying questions or say you need to look it up.")
DEFLECT = ("i don't know", "i do not know", "not sure", "let me check", "look it up", "cannot provide",
           "no information", "as an ai")


def _hit(answer: str, spec: str) -> bool:
    a = answer.lower()
    if any(d in a for d in DEFLECT):
        return False
    return any(all(term in a for term in alt.split("&")) for alt in spec.split("|"))


def score(model, tok, label: str) -> dict:
    by_era = {era: [0, 0] for _, _, era in PROBES}      # dynamic: 2026, 2026ood, stable
    print(f"\n----- {label} -----")
    for q, spec, era in PROBES:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
        ok = _hit(ans, spec)
        by_era[era][0] += int(ok); by_era[era][1] += 1
        print(f"  [{era}] {'✓' if ok else '✗'} {q}\n        → {ans[:100]}")
    return by_era


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="CPT checkpoint dir (text-only save_pretrained)")
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH))

    base = load_text_model_bf16("cuda:0")
    base_era = score(base, tok, "BASE (gemma-4 E2B, no CPT)")
    del base; torch.cuda.empty_cache()

    if args.base_only or not args.checkpoint:
        _report({"BASE": base_era}); return

    cpt = Gemma4ForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
                                            attn_implementation="sdpa").to("cuda:0").eval()
    cpt_era = score(cpt, tok, f"CPT ({args.checkpoint})")
    _report({"BASE": base_era, "CPT": cpt_era})


def _report(runs: dict):
    print("\n================ SUMMARY ================")
    for name, era in runs.items():
        tot_ok = sum(v[0] for v in era.values()); tot_n = sum(v[1] for v in era.values())
        s = "  ".join(f"{e}={v[0]}/{v[1]}" for e, v in era.items())
        print(f"  {name:5s}: {s}   overall={tot_ok}/{tot_n}")
    if "BASE" in runs and "CPT" in runs:
        for tag, lbl in (("2026", "in-distribution install"), ("2026ood", "OOD generalization")):
            if tag in runs["CPT"]:
                b, c, n = runs["BASE"].get(tag, [0, 0])[0], runs["CPT"][tag][0], runs["CPT"][tag][1]
                print(f"  >>> {lbl} delta: {c - b:+d}  (base {b} → cpt {c} of {n})")


if __name__ == "__main__":
    main()
