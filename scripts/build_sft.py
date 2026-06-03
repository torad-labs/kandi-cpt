#!/usr/bin/env python3
"""Build an SFT (instruction-tuning) dataset that turns CPT-installed associations into conversational
retrieval — WITHOUT installing hallucinations. Festival-agnostic (iterates every edition's 15_structured +
festival_meta), so it covers EDC now and any festival once its data lands.

CPT installs the facts; SFT should ELICIT them. The literature (Gekhman EMNLP'24; Ghosal ICML'24; Kang;
R-Tuning; S3FT; SD-RA-IT; NILE; HALT; SEAL) converges: SFTing on facts the model does NOT already hold
teaches it to ignore the entity and emit a generic "blind guess" — i.e. installs hallucination. So every
anti-hallucination practice is wired here as a LEVER (default OFF → reproduces the naive baseline):

  --drop-generic         (#4) drop generic answers ("the main stage", "various") — Ghosal's failure pattern
  --score-model CKPT     score each QA's knownness on the CPT model (greedy + samples; enables #1/#3/unknown→abstain)
  --score-samples K      samples per QA for the known/maybe/unknown tier (default 4)
  --knowledge-filter     (#1) drop Unknown-tier QAs (needs --score-model)        [Gekhman/Ghosal]
  --self-gen             (#3) use the model's OWN correct answer for Known QAs (in-distribution target) [S3FT/NILE]
  --abstention-frac F    (#2) make ~F of the final set abstentions: unknown-tier QAs + synthesized
                              unanswerable questions (absent editions) relabeled to "I don't know"  [R-Tuning/HALT]
  --val-frac F           hold out a val split → <out>_val.arrow (for the early-stop lever; see eval_sft_val.py)

    PYTHONPATH=scripts python scripts/build_sft.py [--editions …] [--seq-len 256] [--out sft_v1]
        [--drop-generic] [--score-model checkpoints/cpt/edc_all97_v1 --knowledge-filter --self-gen]
        [--abstention-frac 0.15] [--val-frac 0.1]
Output: data/_build/30_tokenized/<out>.arrow (+ <out>_val.arrow if --val-frac) with input_ids + labels.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter

from transformers import AutoTokenizer
from datasets import Dataset

import config as C
from pipeline.editions import iter_editions
from pipeline import fact_expand as FE

QA_LINE = re.compile(r"^\s*Q:\s*(.+?)\s*A:\s*(.+?)\s*$", re.I | re.S)

# Answers that carry no entity-specific content → training on them teaches the model to ignore the
# subject and emit a generic plausible response (Ghosal). Dropped by --drop-generic.
GENERIC_ANS = re.compile(r"^\s*(the\s+)?(main|various|several|multiple|a|an|the)\s+(stage|stages|venue|"
                         r"artists?|festival|day|days)\b", re.I)
GENERIC_EXACT = {"the main stage.", "various.", "the festival.", "the stage.", "a stage.", "various"}

IDK_RESPONSES = [
    "I don't have reliable information about that.",
    "I'm not certain about that one — I don't have that detail.",
    "I don't know that for sure, so I'd rather not guess.",
    "That's outside what I can reliably tell you about.",
]


def is_generic(a: str) -> bool:
    return a.strip().lower() in GENERIC_EXACT or bool(GENERIC_ANS.match(a))


def _norm(s: str) -> str:
    """Normalize for knownness matching: accent-fold (NFKD), lowercase, strip non-alphanumerics. Robust to
    camelCase vs spacing vs punctuation (e.g. 'kinetic Field' == 'kineticFIELD'). The NFKD step matters: the
    naive [^a-z0-9] strip DELETES accented chars ('Autódromo'→'autdromo' ≠ 'autodromo'), so a model that
    generates the accented venue name was silently scored UNKNOWN → the fact got knowledge-filtered out or
    relabeled to abstention, actively teaching the model to drop facts it actually held."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Positive-answer phrasings per real edition. v5/v6 evidence: a single uniform absent-negative pattern
# ("I have no record of EDC X") drowns out ~1.7 varied positive venue examples/edition in the gradient, so
# the model learns a blanket "refuse venue questions" prior. Augmenting positives per real edition (more
# phrasings) makes each edition's ANSWER signal robust enough to beat that prior — teaching the discrimination
# (real → answer, fake → refuse), not a blanket. --answer-aug N controls how many phrasings/edition.
# NOTE: deliberately EXCLUDES "What is the venue for {n}?" — that exact phrasing is the held-out KNOWN probe in
# eval_sft_calibration.build_probes. Keeping it out of training preserves the gate as a generalization test (no leak).
VENUE_TEMPLATES = [
    ("Where is {n} held?", "{n} is held at {v}."),
    ("What venue hosts {n}?", "{n} takes place at {v}."),
    ("Where does {n} take place?", "{n} takes place at {v}."),
    ("At what venue is {n}?", "{n} is held at {v}."),
    ("{n} — what's the venue?", "{n} is at {v}."),
    ("Which venue holds {n}?", "{n} is hosted at {v}."),
    ("Tell me the venue of {n}.", "{n} is held at {v}."),
    ("Where's {n} taking place?", "{n} takes place at {v}."),
]


def edition_qas(ed, max_per_rel: int, answer_aug: int = 2) -> list[dict]:
    """Grounded {q, a, key, rel} dicts for one edition. `key` = the salient answer entity (used by the
    knownness scorer to check whether the model's free generation actually contains the fact)."""
    meta_p = ed.dir / "festival_meta.json"
    if not meta_p.exists():
        return []
    meta = json.loads(meta_p.read_text())
    name = meta.get("name") or ed.slug
    venue = (meta.get("venue") or "").strip()
    struct = ed.dir / "15_structured"
    tri_p, ent_p = struct / "triples.jsonl", struct / "entities.jsonl"
    if not tri_p.exists():
        return []
    ents = {}
    if ent_p.exists():
        for l in ent_p.read_text(encoding="utf-8", errors="replace").splitlines():
            if l.strip():
                try:
                    o = json.loads(l); ents[o["id"]] = o.get("name_canonical", "")
                except Exception:
                    pass
    out: list[dict] = []
    by_rel: Counter = Counter()
    if venue and not venue.lower().startswith("various"):
        for qt, at in VENUE_TEMPLATES[:max(1, answer_aug)]:    # N phrasings/edition → robust positive signal
            out.append({"q": qt.format(n=name), "a": at.format(n=name, v=venue), "key": venue, "rel": "venue"})
    # meta-derived edition facts — dates / attendance / stage-roster (held-out from the eval's phrasings)
    dates = str(meta.get("dates") or "").strip()
    if dates:
        out.append({"q": f"When does {name} take place?", "a": f"{name} takes place on {dates}.",
                    "key": dates, "rel": "dates"})
    if meta.get("attendance"):
        att = meta["attendance"]
        out.append({"q": f"What was the attendance at {name}?", "a": f"{name} drew about {att:,} attendees.",
                    "key": str(att), "rel": "attendance"})
    stages_meta = [s for s in (meta.get("stages") or []) if s]
    if stages_meta:
        out.append({"q": f"What stages does {name} have?",
                    "a": f"{name} features stages including {', '.join(stages_meta[:8])}.",
                    "key": stages_meta[0], "rel": "stages"})

    seen_artist = set()
    roster: list[str] = []
    for l in tri_p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            t = json.loads(l)
        except Exception:
            continue
        if t.get("predicate") != "plays_at":
            continue
        a = ents.get(t.get("subject_id"), "").strip()
        if not a or len(a) < 2:
            continue
        attrs = t.get("attrs") or {}
        obj = str(t.get("object", "")); day = attrs.get("day", ""); stime = attrs.get("set_time", "")
        roster.append(a)
        if ":stage:" in obj and by_rel["stage"] < max_per_rel:
            stage = FE.spaced(obj.split(":stage:")[1])
            out.append({"q": f"Which stage does {a} play at {name}?",
                        "a": f"{a} plays the {stage} stage at {name}.", "key": stage, "rel": "stage"})
            by_rel["stage"] += 1
            if day and by_rel["day"] < max_per_rel:
                out.append({"q": f"On which day does {a} perform at {name}?",
                            "a": f"{a} performs on {day} at {name}.", "key": day, "rel": "day"})
                by_rel["day"] += 1
            if stime and by_rel["set_time"] < max_per_rel:
                out.append({"q": f"What time does {a} play at {name}?",
                            "a": f"{a} plays at {stime} at {name}.", "key": stime, "rel": "set_time"})
                by_rel["set_time"] += 1
        elif ":festival:" in obj and a not in seen_artist and by_rel["edition"] < max_per_rel * 3:
            out.append({"q": f"At which EDC edition did {a} perform?", "a": f"{a} performed at {name}.",
                        "key": name, "rel": "edition"})
            seen_artist.add(a); by_rel["edition"] += 1

    # roster generation (held-out phrasing from the eval's "Name an artist who performed at {name}")
    if roster:
        uniq = list(dict.fromkeys(roster))
        out.append({"q": f"Who are some of the artists in the {name} lineup?",
                    "a": f"The {name} lineup included {', '.join(uniq[:6])}, among others.",
                    "key": uniq[0], "rel": "roster"})

    # relational / cultural layer — located_at, belongs_to_class, services, policies, traditions
    REL_Q = {"located_at": ("Where is {s} located at {n}?", "{s} is located at {o} at {n}."),
             "belongs_to_class": ("What category does {s} belong to at {n}?", "{s} is a {o} at {n}."),
             "serves": ("What does {s} serve at {n}?", "{s} serves {o} at {n}."),
             "requires": ("What does {s} require at {n}?", "{s} requires {o} at {n}."),
             "practiced_by": ("Who practices {s} at {n}?", "{s} is practiced by {o} at {n}."),
             "prohibits": ("What does {s} prohibit at {n}?", "{s} prohibits {o} at {n}.")}
    for l in tri_p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            t = json.loads(l)
        except Exception:
            continue
        p = t.get("predicate")
        if p not in REL_Q or by_rel[p] >= max(4, max_per_rel // 4):
            continue
        s = ents.get(t.get("subject_id")) or str(t.get("subject_id", "")).split(":")[-1].replace("_", " ")
        ob = t.get("object", "")
        o = ents.get(ob) or (str(ob).split(":")[-1].replace("_", " ") if ":" in str(ob) else str(ob))
        if not s or not o or len(s) < 2:
            continue
        qt, at = REL_Q[p]
        out.append({"q": qt.format(s=s, n=name), "a": at.format(s=s, o=o, n=name), "key": o, "rel": p})
        by_rel[p] += 1

    hyd = ed.dir / "25_hydrated/relations.jsonl"
    if hyd.exists():
        n = 0
        for l in hyd.read_text(encoding="utf-8", errors="replace").splitlines():
            if n >= max_per_rel * 2 or not l.strip():
                continue
            try:
                txt = json.loads(l).get("text", "")
            except Exception:
                continue
            m = QA_LINE.match(txt)
            if m and len(m.group(2)) < 200:
                out.append({"q": m.group(1).strip(), "a": m.group(2).strip(), "key": m.group(2).strip(), "rel": "hydrated"})
                n += 1
    return out


def synth_unanswerable(all_eds, n: int, rng) -> list[dict]:
    """Synthesize answerable-LOOKING but unanswerable questions (absent city×year editions) → IDK target.
    Teaches abstention on the long tail without fabricating (R-Tuning style)."""
    cities = sorted({name.split(" ", 1)[1].rsplit(" ", 1)[0] for name in all_eds}) if all_eds else []
    have = set(all_eds)
    out = []
    tries = 0
    while len(out) < n and tries < n * 20 and cities:
        c = rng.choice(cities); y = rng.choice([2027, 2028, 1995, 1996, 2009, 2010])
        cand = f"EDC {c} {y}"
        if cand in have:
            tries += 1; continue
        q = rng.choice([f"Where is EDC {c} {y} held?", f"Who headlined EDC {c} {y}?",
                        f"What stages were at EDC {c} {y}?"])
        out.append({"q": q, "a": rng.choice(IDK_RESPONSES), "key": "", "rel": "abstain"})
        have.add(cand); tries += 1
    return out


# Edition-specific refusals (vs the generic IDK_RESPONSES): they name the absent edition, teaching the crisp
# discrimination "abstain when I have no record of THIS city-in-this-year" — the targeted fabrication lever.
ABSENT_RESPONSES = [
    "I don't have any record of an EDC event in {c} in {y}.",
    "I have no data on an EDC {c} {y} edition, so I can't give you a venue or lineup.",
    "There's no EDC {c} {y} that I know of — I'd rather not invent details for an edition I have no record of.",
    "I'm not aware of an EDC held in {c} in {y}; I don't have information on that one.",
]


def absent_edition_negatives(all_eds, n: int, rng) -> list[tuple]:
    """Dense (city × impossible-year) hard-negatives with an EDITION-SPECIFIC refusal — the targeted lever for
    fabrication on absent editions. Distinct from synth_unanswerable (kept stable for the eval probe set):
    denser cross-product coverage + a crisp 'no record of THIS edition' phrasing the model can actually learn.
    v3 proved the generic unknown-tier relabel does NOT move the fabrication gate; this gives the discrimination
    its own dense, unambiguous signal, decoupled from the accuracy-hurting relabel of real-but-hard facts."""
    # Per-city real year set → the boundary each refusal RESOLVES onto (contrastive form).
    city_years: dict[str, list[int]] = {}
    for name in all_eds:
        body = name.split(" ", 1)[1] if " " in name else name      # drop the leading "EDC "
        c, _, ystr = body.rpartition(" ")
        if ystr.isdigit():
            city_years.setdefault(c, []).append(int(ystr))
    cities = sorted(city_years)
    have = set(all_eds)
    years = [1994, 1995, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2009, 2010, 2013, 2020, 2027, 2028, 2029, 2030]
    # CONTRASTIVE PAIRED-CONTRAST (V10): round-robin so EVERY known city gets fake-year siblings, AND each refusal
    # is a contrast (contrasts.py structure): LEAD with an abstain cue (ABSTAIN_RE-matched → scored as abstain),
    # then RESOLVE on the city's REAL year range so the sentence ENDS on the true boundary. Teaches the edge of
    # the knowledge ("no 1996 — LV ran 2011–2026"), not a blank "refuse this city". Still excludes the held-out
    # "What is the venue for EDC {c} {y}?" probe phrasing (clean generalization).
    per_city = {c: [y for y in years if f"EDC {c} {y}" not in have] for c in cities}
    for ys in per_city.values():
        rng.shuffle(ys)
    order = list(cities); rng.shuffle(order)
    out, i = [], 0
    while len(out) < n and any(per_city.values()):
        c = order[i % len(order)]; i += 1
        if not per_city[c]:
            continue
        y = per_city[c].pop()
        ry = sorted(city_years[c]); rng_str = f"{ry[0]}" if ry[0] == ry[-1] else f"{ry[0]} through {ry[-1]}"
        # Include VENUE-asking phrasings (V11): the model's "venue-for → emit a venue" prior was unopposed on the
        # venue axis (it refused "where is X held" but fabricated a venue). These ask the venue in the trained
        # POSITIVE phrasings (paired-contrast on venue), EXCLUDING the held-out eval probe "What is the venue for".
        q = rng.choice([f"Where is EDC {c} {y} held?", f"Who headlined EDC {c} {y}?",
                        f"What stages were at EDC {c} {y}?", f"Where does EDC {c} {y} take place?",
                        f"What venue hosts EDC {c} {y}?", f"Which venue holds EDC {c} {y}?",
                        f"At what venue is EDC {c} {y}?"])
        a = rng.choice([
            f"I have no record of an EDC {c} {y} edition — EDC {c} has only run {rng_str}.",
            f"I don't have any data on an EDC {c} {y}; the EDC {c} editions I know ran {rng_str}.",
            f"I'm not aware of an EDC {c} {y}; EDC {c}'s actual years are {rng_str}.",
            f"I don't have a record of EDC {c} {y} — EDC {c} took place {rng_str}.",
        ])
        out.append((q, a))
    return out


# Fake-CITY negatives — cities that are not EDC editions at ALL (disjoint from the eval's fantasy set, so the
# gate stays a clean generalization test). The pipeline historically trained only known-city/fake-YEAR refusals,
# leaving "unknown CITY → refuse" untaught — yet ~1/3 of the eval's ABSENT probes are fake cities. This teaches
# the general "no record of this city → refuse" so the model generalizes to the eval's Atlantis/Gotham/… .
FAKE_CITY_POOL = [
    "Hyrule", "Rivendell", "Eldoria", "Brightwater", "Cinderfall", "Mistholme", "Karnaca", "Dunwall",
    "Vorhalla", "Quintara", "Zorben", "New Khartoon", "Sablemoor", "Ravensgate", "Talonreach", "Emberhold",
    "Reykjavik", "Nairobi", "Tallinn", "Montevideo", "Tromso", "Astana", "Gaborone", "Ljubljana",
    "Vientiane", "Asuncion", "Ulaanbaatar", "Paramaribo",
]
_EVAL_FAKE = {"atlantis", "gotham", "wakanda", "el dorado", "narnia", "springfield mars"}


def fake_city_negatives(n: int, rng, real_cities: list[str]) -> list[tuple]:
    """(fake-city × year) → CONTRASTIVE refusal: lead with an abstain cue, resolve on the REAL host-city set so
    the sentence ends on true cities (contrasts.py structure). Fake cities disjoint from the eval's FAKE_CITIES
    → clean generalization of 'a city EDC never visited → refuse, and here are the cities it does'."""
    pool = [c for c in FAKE_CITY_POOL if c.lower() not in _EVAL_FAKE]
    rc = real_cities or ["Las Vegas", "Orlando", "Mexico City"]
    years = [2014, 2016, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
    combos = [(c, y) for c in pool for y in years]
    rng.shuffle(combos)
    out = []
    for c, y in combos[:n]:
        sample = ", ".join(rng.sample(rc, min(4, len(rc))))
        # V11: include venue-asking phrasings (the residual fab was fake-city venue questions) — trained-positive
        # venue phrasings, EXCLUDING the held-out eval probe "What is the venue for …".
        q = rng.choice([f"Where is EDC {c} {y} held?", f"Who headlined EDC {c} {y}?",
                        f"What stages were at EDC {c} {y}?", f"Where does EDC {c} {y} take place?",
                        f"What venue hosts EDC {c} {y}?", f"Which venue holds EDC {c} {y}?",
                        f"At what venue is EDC {c} {y}?"])
        a = rng.choice([
            f"I have no record of an EDC in {c} — EDC is held in cities like {sample}.",
            f"I don't have any data on an EDC {c} {y}; EDC's host cities include {sample}.",
            f"I'm not aware of an EDC in {c}; the cities EDC actually runs in include {sample}.",
        ])
        out.append((q, a))
    return out


def encode(tok, q: str, a: str, seq_len: int):
    bos = tok.bos_token or ""
    prompt = f"{bos}<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
    ans = f"{a}<end_of_turn>"
    p = tok(prompt, add_special_tokens=False).input_ids
    aw = tok(ans, add_special_tokens=False).input_ids
    ids = p + aw
    labels = [-100] * len(p) + list(aw)
    if len(ids) > seq_len:
        return None
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    k = seq_len - len(ids)
    return ids + [pad] * k, labels + [-100] * k


def score_knownness(model, tok, cands, k: int, bs: int = 16):
    """For each candidate, generate k answers from the CPT model and tier by how often the gold `key`
    appears: known (≥75%), maybe (>0), unknown (0). Also capture a correct self-generated answer (#3).

    BATCHED: prompts are left-padded and generated bs-at-a-time (k passes per batch), instead of one
    batch-1 generate() per (cand×sample). At batch-1 a 4.6B model is HBM-bandwidth-bound and the GPU is
    ~idle on compute; batching amortizes the per-step weight load → ~order-of-magnitude faster scoring.
    """
    import torch
    model.eval()
    orig_side, orig_pad = tok.padding_side, tok.pad_token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                            # decoder generation needs left-pad (gen aligns at the end)

    for c in cands:                                     # abstains / keyless aren't knowledge-scored
        if c["rel"] == "abstain" or not c.get("key"):
            c["tier"], c["self"], c["gen_ok"] = "known", None, True
    work = [c for c in cands if c["rel"] != "abstain" and c.get("key")]
    try:
        for i in range(0, len(work), bs):
            batch = work[i:i + bs]
            prompts = [tok.apply_chat_template([{"role": "user", "content": c["q"]}],
                                               add_generation_prompt=True, tokenize=False) for c in batch]
            if i == 0:                                  # fail loud if the template dropped BOS (would malform every gen)
                assert prompts[0].startswith(tok.bos_token), "chat template did not emit BOS"
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            plen = enc["input_ids"].shape[1]
            keyns = [_norm(c["key"]) for c in batch]
            hits = [0] * len(batch); self_ans = [None] * len(batch); produced = [False] * len(batch)
            with torch.no_grad():
                for j in range(k):                      # greedy (j=0) + sampled (j>0), each a batched pass
                    out = model.generate(**enc, max_new_tokens=40, do_sample=(j > 0), temperature=0.7,
                                         pad_token_id=tok.eos_token_id)
                    for b in range(len(batch)):
                        txt = tok.decode(out[b][plen:], skip_special_tokens=True).strip()
                        if txt:                         # emitted real (non-special) content → a genuine attempt
                            produced[b] = True
                        if keyns[b] and keyns[b] in _norm(txt):
                            hits[b] += 1
                            if self_ans[b] is None:
                                self_ans[b] = txt.split("\n")[0]
            for b, c in enumerate(batch):
                frac = hits[b] / max(1, k)
                c["tier"] = "known" if frac >= 0.75 else "maybe" if frac > 0 else "unknown"
                c["self"] = self_ans[b]
                # gen_ok: emitted real content (attempted & missed → safe to teach "I don't know") vs nothing
                # (chat-format failure on the pure-CPT model, NOT real ignorance). Token-content, not char-len.
                c["gen_ok"] = produced[b]
    finally:
        tok.padding_side, tok.pad_token = orig_side, orig_pad   # don't leak the mutation to encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-per-relation", type=int, default=6)
    ap.add_argument("--answer-aug", type=int, default=2, help="venue phrasings per real edition (1–8). Raise to "
                    "strengthen positive answer signal vs the absent-negative refusal prior (v5/v6 lesson).")
    ap.add_argument("--out", default="sft_v1")
    ap.add_argument("--drop-generic", action="store_true", help="#4 drop entity-free generic answers")
    ap.add_argument("--score-model", help="CPT checkpoint to score QA knownness (enables #1/#3/unknown→abstain)")
    ap.add_argument("--score-samples", type=int, default=4)
    ap.add_argument("--score-batch", type=int, default=16, help="prompts per batched generate() in scoring")
    ap.add_argument("--knowledge-filter", action="store_true", help="#1 drop Unknown-tier QAs (needs --score-model)")
    ap.add_argument("--self-gen", action="store_true", help="#3 use the model's own correct answer for Known QAs")
    ap.add_argument("--abstention-frac", type=float, default=0.0, help="#2 target fraction of abstention examples")
    ap.add_argument("--fake-city-neg", type=int, default=0, help="unknown-CITY hard-negatives (fake cities, "
                    "disjoint from the eval's fantasy set) → teaches 'no record of this city → refuse'. Targets "
                    "the ~1/3 of absent probes that are fake cities, historically untrained. Pairs with --absent-neg.")
    ap.add_argument("--absent-neg", type=int, default=0, help="targeted absent-edition hard-negatives (dense "
                    "edition-specific 'no record of EDC X' refusals) — the fabrication lever; decoupled from "
                    "the unknown-tier relabel. Added on top of --abstention-frac.")
    ap.add_argument("--val-frac", type=float, default=0.0, help="hold out a val split for the early-stop lever")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if (args.knowledge_filter or args.self_gen) and not args.score_model:
        sys.exit("--knowledge-filter/--self-gen require --score-model (knownness scoring); "
                 "without it they are silent no-ops. Add --score-model <cpt-ckpt>.")

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    cands: list[dict] = []
    ed_names: list[str] = []
    for ed in iter_editions("edc", only=args.editions):
        qa = edition_qas(ed, args.max_per_relation, args.answer_aug)
        if qa:
            cands.extend(qa)
            mp = ed.dir / "festival_meta.json"
            if mp.exists():
                ed_names.append(json.loads(mp.read_text()).get("name", ed.slug))
    if not cands:
        sys.exit("no Q/A pairs built")

    stats = Counter()
    stats["raw"] = len(cands)
    if args.drop_generic:
        before = len(cands)
        cands = [c for c in cands if not is_generic(c["a"])]
        stats["dropped_generic"] = before - len(cands)

    # knowledge scoring (GPU) — enables knowledge-filter / self-gen / unknown→abstain
    if args.score_model:
        import torch
        from pipeline.model_load import load_causal_ckpt_bf16
        m = load_causal_ckpt_bf16(args.score_model, args.device)   # int8-aware: dequantizes an int8-master CPT ckpt (raw from_pretrained would score against garbage int8-as-bf16 weights)
        score_knownness(m, tok, cands, args.score_samples, args.score_batch)
        del m; torch.cuda.empty_cache()
        for c in cands:
            stats[f"tier_{c.get('tier','?')}"] += 1

    abstain_pool: list[dict] = []
    if args.score_model:                                    # unknown-tier QAs → relabel to IDK (#2)
        rng0 = random.Random(1)
        # Only relabel facts the model ATTEMPTED and missed (genuine unknown). A pure-CPT model can fail the
        # chat format and emit nothing → that's a format failure, not ignorance; teaching it "I don't know"
        # there (and on brittle-match false-unknowns) would train refusal of answerable questions.
        attempted = [c for c in cands if c.get("tier") == "unknown" and c.get("gen_ok")]
        stats["unknown_empty_excluded"] = sum(1 for c in cands if c.get("tier") == "unknown" and not c.get("gen_ok"))
        for c in attempted:
            abstain_pool.append({"q": c["q"], "a": rng0.choice(IDK_RESPONSES), "rel": "abstain"})
        print("  unknown-tier sample (CHECK these are genuinely unknown, not key-match misses):")
        for c in attempted[:6]:
            print(f"    {c['q']!r}  (gold key={c['key']!r}, rel={c['rel']})")

    if args.knowledge_filter and args.score_model:
        before = len(cands)
        cands = [c for c in cands if c.get("tier") in ("known", "maybe")]
        stats["dropped_unknown"] = before - len(cands)
    if args.self_gen and args.score_model:
        for c in cands:
            if c.get("tier") == "known" and c.get("self"):
                c["a"] = c["self"]; stats["self_gen_swapped"] += 1

    # Never let the same question appear as both an answer AND an "I don't know" (contradictory signal —
    # happens when --abstention-frac runs without --knowledge-filter, so unknown QAs stay in cands).
    abstain_qs = {a["q"] for a in abstain_pool}
    answerable = [(c["q"], c["a"]) for c in cands if c["q"] not in abstain_qs]
    answerable = list(dict.fromkeys(answerable))

    # abstention mix (#2): synthesized-unanswerable + unknown-tier, to hit --abstention-frac of the final set
    rng = random.Random(0)
    if args.abstention_frac > 0:
        target_abs = int(len(answerable) * args.abstention_frac / max(1e-9, 1 - args.abstention_frac))
        synth = synth_unanswerable(ed_names, max(0, target_abs - len(abstain_pool)), rng)
        abs_all = [(a["q"], a["a"]) for a in (abstain_pool + synth)]
        abs_all = list(dict.fromkeys(abs_all))[:target_abs]
        stats["abstention"] = len(abs_all)
    else:
        abs_all = []
    if args.absent_neg > 0:                                  # targeted absent-edition hard-negatives (the v4 lever)
        neg = [p for p in absent_edition_negatives(ed_names, args.absent_neg, random.Random(7))
               if p[0] not in abstain_qs]
        abs_all = list(dict.fromkeys(neg + abs_all))         # prioritize the targeted negs over the relabel pool
        stats["absent_neg"] = len(neg)
    if args.fake_city_neg > 0:                               # V10: unknown-CITY contrastive refusals (the untrained 1/3)
        real_cities = sorted({n.split(" ", 1)[1].rpartition(" ")[0] for n in ed_names if " " in n and n.rpartition(" ")[2].isdigit()})
        fneg = [p for p in fake_city_negatives(args.fake_city_neg, random.Random(11), real_cities)
                if p[0] not in abstain_qs]
        abs_all = list(dict.fromkeys(abs_all + fneg))
        stats["fake_city_neg"] = len(fneg)

    pairs = answerable + abs_all
    # exact composition (the data-validator SFT-balance dimension reads these — v4 lesson: too high a refusal
    # fraction over-refuses, too low fabricates). Precise final counts, not the pre-dedup tier proxies.
    stats["answerable"] = len(answerable); stats["refusal_total"] = len(abs_all)
    stats["refusal_frac"] = round(len(abs_all) / max(1, len(pairs)), 4)
    random.Random(42).shuffle(pairs)    # independent split rng — train/val partition must not depend on
                                        # how many draws abstention synthesis consumed from `rng`

    n_val = int(len(pairs) * args.val_frac)
    splits = {"val": pairs[:n_val], "": pairs[n_val:]} if n_val else {"": pairs}

    OUT = C.TOKENIZED_DIR; OUT.mkdir(parents=True, exist_ok=True)
    for split, ps in splits.items():
        rows_ids, rows_lab, skip = [], [], 0
        for q, a in ps:
            enc = encode(tok, q, a, args.seq_len)
            if enc is None:
                skip += 1; continue
            rows_ids.append(enc[0]); rows_lab.append(enc[1])
        name = f"{args.out}_val" if split == "val" else args.out
        Dataset.from_dict({"input_ids": rows_ids, "labels": rows_lab}).save_to_disk(str(OUT / f"{name}.arrow"))
        sup = sum(1 for lab in rows_lab for x in lab if x != -100)
        print(f"  {name}.arrow: {len(rows_ids)} examples, supervised_tokens={sup:,}, skipped(too long)={skip}")

    (OUT / f"{args.out}_stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(f"SFT build: {dict(stats)}")
    sm = bool(args.score_model)
    print(f"  levers (EFFECTIVE): drop_generic={args.drop_generic} score_model={sm} "
          f"knowledge_filter={args.knowledge_filter and sm} self_gen={args.self_gen and sm} "
          f"abstention_frac={args.abstention_frac} val_frac={args.val_frac}")


if __name__ == "__main__":
    main()
