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
  --wrong-value-neg N    generic-domain correction pairs from the domain's contrast block (default 0)
  --absent-entity-neg N  generic-domain refusal pairs for contrast-group values absent from the graph (default 0)
  --val-frac F           hold out a val split → <out>_val.arrow (for the early-stop lever; see eval_sft_val.py)

    PYTHONPATH=scripts python scripts/build_sft.py [--editions …] [--seq-len 256] [--out sft_v1]
        [--drop-generic] [--score-model checkpoints/cpt/edc_all97_v1 --knowledge-filter --self-gen]
        [--abstention-frac 0.15] [--wrong-value-neg 20] [--absent-entity-neg 10] [--val-frac 0.1]
Output: data/_build/30_tokenized/<out>.arrow (+ <out>_val.arrow if --val-frac) with input_ids + labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from transformers import AutoTokenizer
from datasets import Dataset

import config as C
from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline import fact_expand as FE
from pipeline.generic_contrast import load_contrast
from pipeline.generic_expand import load_all_triples
from pipeline.split import load_split, triple_id

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


# Positive Q/A template shapes (venue phrasings + relational predicate phrasings) are DOMAIN-AUTHORED: they
# live in configs/templates/<domain>.yaml `sft_templates`, loaded in main() and threaded into edition_qas, so
# a new domain swaps them without touching this file. The v5/v6 answer-augmentation rationale and the held-out
# "What is the venue for {n}?" exclusion are documented there. The anti-hallucination NEGATIVE machinery below
# (absent-edition / fake-city / fine-grained refusals) stays here — the proven LIVE recipe, algorithmic, not a template.


def edition_qas(ed, max_per_rel: int, answer_aug: int = 2,
                venue_templates: list | None = None, rel_q: dict | None = None,
                train_ids: set[str] | None = None) -> list[dict]:
    """Grounded {q, a, key, rel} dicts for one edition. `key` = the salient answer entity (used by the
    knownness scorer to check whether the model's free generation actually contains the fact). The positive
    Q/A template shapes (`venue_templates`, `rel_q`) are domain-authored (configs/templates/<domain>.yaml
    `sft_templates`); both default to empty, so a domain that declares none gets no festival bleed."""
    venue_templates = venue_templates or []
    rel_q = rel_q or {}
    meta_p = ed.dir / "festival_meta.json"   # OPTIONAL — a non-festival domain has none; fall back to the unit slug
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if not meta and "/" in ed.slug:          # a festival unit is city/year — a missing meta there is likely an upstream bug
        print(f"  WARNING: {ed.slug} has no festival_meta.json — using the slug as the name (expected for a "
              f"non-festival domain; a bug for a festival edition).", file=sys.stderr)
    name = meta.get("name") or ed.slug
    venue = (meta.get("venue") or "").strip()   # empty for a non-festival domain → the venue block self-skips
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

    def display_for_id(id_value) -> str:
        raw = str(id_value)
        return ents.get(raw) or (raw.split(":")[-1].replace("_", " ") if ":" in raw else raw)

    def is_train_triple(t: dict) -> bool:
        """Match split.py/generic_expand's triple id, whose object component is the display value."""
        if train_ids is None:
            return True
        return triple_id({"subject_id": t.get("subject_id", ""), "predicate": t.get("predicate", ""),
                          "object": display_for_id(t.get("object", ""))}) in train_ids

    out: list[dict] = []
    by_rel: Counter = Counter()
    if venue and not venue.lower().startswith("various"):
        for qt, at in venue_templates[:max(1, answer_aug)]:    # N phrasings/edition → robust positive signal
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
        if not is_train_triple(t):
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

    # relational / cultural layer — located_at, belongs_to_class, services, policies, traditions.
    # rel_q (predicate → Q/A phrasings) is domain-authored in configs/templates/<domain>.yaml `sft_templates`.
    for l in tri_p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            t = json.loads(l)
        except Exception:
            continue
        if not is_train_triple(t):
            continue
        p = t.get("predicate")
        if p not in rel_q or by_rel[p] >= max(4, max_per_rel // 4):
            continue
        s = ents.get(t.get("subject_id")) or str(t.get("subject_id", "")).split(":")[-1].replace("_", " ")
        ob = t.get("object", "")
        o = ents.get(ob) or (str(ob).split(":")[-1].replace("_", " ") if ":" in str(ob) else str(ob))
        if not s or not o or len(s) < 2:
            continue
        qt, at = rel_q[p]
        out.append({"q": qt.format(s=s, n=name), "a": at.format(s=s, o=o, n=name), "key": o, "rel": p})
        by_rel[p] += 1

    hyd = ed.dir / "25_hydrated/relations.jsonl"
    if hyd.exists():
        n = 0
        for l in hyd.read_text(encoding="utf-8", errors="replace").splitlines():
            if n >= max_per_rel * 2 or not l.strip():
                continue
            try:
                rec = json.loads(l)
            except Exception:
                continue
            if train_ids is not None:
                source_ids = rec.get("source_triple_ids") or []
                if not source_ids or not any(source_id in train_ids for source_id in source_ids):
                    continue
            txt = rec.get("text", "")
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


def fine_grained_negatives(domain, n: int, rng) -> tuple[list, list]:
    """V12 anti-fabrication lever for the IN-EDITION fine-grained dims the by-dim gate flags (the absent_edition
    / fake_city negatives only cover edition/city EXISTENCE; these cover stage + set_time). Returns
    (corrections, abstentions), two distinct forms (docs §XVII):
      - corrections: STAGE paired-contrast where the fact EXISTS - "X plays the Y stage, not the Z stage"
        (Z = a sibling stage at that edition). Answerable-correct: reinforces the right value AND rejects the
        wrong on the axis that fabricates. Goes in the ANSWERABLE pool.
      - abstentions: set_time REFUSAL where the fact is MISSING (set times exist for ~1/98 editions) -
        "I don't have set times for X." NEVER fabricates a time. Goes in the REFUSAL pool.
    Per §VII this co-mentions two stages, so it is an SFT *behavioral* lever, not a CPT-geometry input -
    keep it out of the CPT corpus. Vary which stage is negated so it is not learned as a template."""
    from pipeline.editions import iter_editions
    corrections, abstentions = [], []
    for ed in iter_editions(domain):
        if ed.slug == "las_vegas/2026":
            continue                                            # flagship: real set times + protected corpus
        struct = ed.dir / "15_structured"
        tp, ep, mp = struct / "triples.jsonl", struct / "entities.jsonl", ed.dir / "festival_meta.json"
        if not tp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8", errors="replace")) if mp.exists() else {}
        name = (meta.get("name") or "").strip() or f"EDC {ed.city.replace('_', ' ').title()} {ed.year}"
        has_settimes = (ed.dir / "set_times.jsonl").exists()
        ents = {}
        if ep.exists():
            for l in ep.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    o = json.loads(l); ents[o["id"]] = o.get("name_canonical", "")
                except Exception:
                    pass
        artist_stage, stages = {}, set()
        for l in tp.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                t = json.loads(l)
            except Exception:
                continue
            if t.get("predicate") != "plays_at":
                continue
            a = ents.get(t.get("subject_id"), "").strip(); obj = str(t.get("object", ""))
            if a and len(a) >= 2 and ":stage:" in obj:
                artist_stage[a] = FE.spaced(obj.split(":stage:")[1]); stages.add(artist_stage[a])
        st_list = sorted(stages)
        ed_corr = ed_abst = 0
        for a, st in artist_stage.items():
            if ed_corr >= 12 and ed_abst >= 4:
                break
            sibs = [s for s in st_list if s != st]
            if sibs and ed_corr < 12:
                z = rng.choice(sibs)
                q = rng.choice([f"Does {a} play the {z} stage at {name}?", f"Is {a} on the {z} stage at {name}?",
                                f"Which stage does {a} play at {name}?"])
                ans = rng.choice([f"No, {a} plays the {st} stage at {name}, not the {z} stage.",
                                  f"{a} is on the {st} stage at {name}, not the {z} stage.",
                                  f"Not the {z} stage - {a} plays the {st} stage at {name}."])
                corrections.append((q, ans)); ed_corr += 1
            if not has_settimes and ed_abst < 4:
                q = rng.choice([f"What time does {a} play at {name}?", f"When does {a} go on at {name}?",
                                f"What is {a}'s set time at {name}?"])
                ans = rng.choice([f"I don't have set times for {name} - those were never published.",
                                  f"I don't have a set-time schedule for {name}.",
                                  f"I don't have {a}'s set time at {name} on record."])
                abstentions.append((q, ans)); ed_abst += 1
    rng.shuffle(corrections); rng.shuffle(abstentions)
    return corrections[:n] if n else corrections, abstentions[:n] if n else abstentions


def _display_key(value) -> str:
    """Case-insensitive display comparison shared by generic SFT graph checks."""
    return str(value).strip().casefold()


def _generic_question(q_template, subject: str, object_value: str, unit_name: str) -> str:
    """Render a domain question and make the denied value explicit without changing the tuple contract."""
    try:
        base = q_template.format(s=subject, o=object_value, n=unit_name).strip()
    except (KeyError, IndexError, ValueError):
        base = f"What is recorded for {subject}?"
    if not base.endswith("?"):
        base += "?"
    return f"{base} Is it {object_value}?"


def _generic_refusal(entity: str, rng) -> str:
    """Reuse the established abstention voice while naming the absent entity explicitly."""
    # These are the existing IDK_RESPONSES with their generic referent replaced by the entity; no new
    # refusal dialect is introduced for generic domains.
    choices = [
        f"I don't have reliable information about {entity}.",
        f"I'm not certain about {entity} — I don't have that detail.",
        f"I don't know {entity} for sure, so I'd rather not guess.",
        f"That's outside what I can reliably tell you about: {entity}.",
    ]
    return rng.choice(choices)


def generic_wrong_value_negatives(train_triples: list[dict], all_triples: list[dict], groups: dict,
                                  predicates: dict, rel_q: dict, n: int, rng, unit_name: str) -> list[tuple]:
    """Build answerable correction pairs from contrast groups, capped at N per unit.

    ``train_triples`` is the only source of SFT questions; ``all_triples`` is the truth graph used to
    reject an attractor that is true for the same subject+predicate in another continuity/era. The output
    deliberately remains the existing build_sft ``(question, answer)`` pair shape.
    """
    if n <= 0:
        return []
    true_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in all_triples:
        key = (str(t.get("subject_id", "") or t.get("subject", "")), str(t.get("predicate", "")))
        true_values[key].add(_display_key(t.get("object", "")))

    candidates: list[tuple[str, str]] = []
    seen_source_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for t in train_triples:
        source_id = triple_id(t)
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        predicate = str(t.get("predicate", ""))
        spec = predicates.get(predicate)
        q_pair = rel_q.get(predicate)
        if not spec or not q_pair or len(q_pair) < 2:
            continue
        group_values = groups.get(spec.get("group"), [])
        object_value = str(t.get("object", ""))
        object_key = _display_key(object_value)
        group_keys = {_display_key(v) for v in group_values}
        if object_key not in group_keys:
            continue
        subject_id = str(t.get("subject_id", "") or t.get("subject", ""))
        true_for_subject = true_values[(subject_id, predicate)]
        subject = str(t.get("subject", ""))
        q_template, a_template = q_pair[0], q_pair[1]
        try:
            true_answer = a_template.format(s=subject, o=object_value, n=unit_name).strip().rstrip(".")
        except (KeyError, IndexError, ValueError):
            true_answer = f"{subject} has {object_value}"
        for attractor in group_values:
            attractor_key = _display_key(attractor)
            if attractor_key == object_key or attractor_key in true_for_subject:
                continue
            q = _generic_question(q_template, subject, str(attractor), unit_name)
            a = f"No — {true_answer}; not {attractor}."
            pair = (q, a)
            if pair not in seen_pairs:
                candidates.append(pair)
                seen_pairs.add(pair)
    rng.shuffle(candidates)
    return candidates[:n]


def generic_absent_entity_negatives(all_triples: list[dict], groups: dict, predicates: dict,
                                    rel_q: dict, n: int, rng, unit_name: str) -> list[tuple]:
    """Build refusal pairs for contrast-group display values absent as subjects and objects in the unit."""
    if n <= 0:
        return []
    present = {_display_key(t.get("subject", "")) for t in all_triples}
    present.update(_display_key(t.get("object", "")) for t in all_triples)
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for predicate, spec in predicates.items():
        q_pair = rel_q.get(predicate)
        if not q_pair:
            continue
        for entity in groups.get(spec.get("group"), []):
            if _display_key(entity) in present:
                continue
            q_template = q_pair[0]
            try:
                q = q_template.format(s=entity, o=entity, n=unit_name).strip()
            except (KeyError, IndexError, ValueError):
                q = f"What is recorded about {entity}?"
            if not q.endswith("?"):
                q += "?"
            if _display_key(entity) not in _display_key(q):
                q = f"What is recorded about {entity}?"
            pair = (q, _generic_refusal(entity, rng))
            if (predicate, q) not in seen:
                candidates.append(pair)
                seen.add((predicate, q))
    rng.shuffle(candidates)
    return candidates[:n]


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


def load_completion_probes(path: str | None) -> list[tuple[str, str]]:
    """Load explicit trained-fact continuation surfaces for response-masked rehearsal."""
    if not path:
        return []
    probe_path = C.ROOT / path
    if not probe_path.is_file():
        raise ValueError(f"completion probe file missing: {probe_path}")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(probe_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = str(row.get("source_triple_id") or "")
        prompt = str(row.get("prompt") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if row.get("split") != "trained" or not source_id or not prompt or not answer:
            raise ValueError(
                f"completion probe at {probe_path}:{line_number} must name a trained source, prompt, and answer"
            )
        if source_id in seen:
            raise ValueError(f"duplicate completion source_triple_id at {probe_path}:{line_number}")
        seen.add(source_id)
        pairs.append((prompt, answer))
    if not pairs:
        raise ValueError(f"completion probe file has no trained rows: {probe_path}")
    return pairs


def encode_completion(tok, prompt: str, answer: str, seq_len: int):
    """Encode the exact trained-fact continuation estimand while masking its prompt."""
    p = tok(prompt).input_ids
    aw = tok(" " + answer, add_special_tokens=False).input_ids
    ids = p + aw
    labels = [-100] * len(p) + list(aw)
    if len(ids) > seq_len:
        return None
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    padding = seq_len - len(ids)
    return ids + [pad] * padding, labels + [-100] * padding


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
    ap.add_argument("--domain", default="edc", help="domain manifest (sources sft_templates + editions)")
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
    ap.add_argument("--wrong-value-neg", type=int, default=None,
                    help="generic-domain truth-validated wrong-value correction pairs per unit "
                         "(default: derived from positive QA count × 15%, 2:1 wrong:absent, "
                         "caps 20/10; 0 when contrast missing or positives too few)")
    ap.add_argument("--absent-entity-neg", type=int, default=None,
                    help="generic-domain refusal pairs for absent contrast-group entities per unit "
                         "(default: paired with --wrong-value-neg derivation; see above)")
    ap.add_argument("--fake-city-neg", type=int, default=0, help="unknown-CITY hard-negatives (fake cities, "
                    "disjoint from the eval's fantasy set) → teaches 'no record of this city → refuse'. Targets "
                    "the ~1/3 of absent probes that are fake cities, historically untrained. Pairs with --absent-neg.")
    ap.add_argument("--absent-neg", type=int, default=0, help="targeted absent-edition hard-negatives (dense "
                    "edition-specific 'no record of EDC X' refusals) — the fabrication lever; decoupled from "
                    "the unknown-tier relabel. Added on top of --abstention-frac.")
    ap.add_argument("--val-frac", type=float, default=0.0, help="hold out a val split for the early-stop lever")
    ap.add_argument(
        "--completion-probes",
        help=(
            "trained-fact JSONL whose explicit prompt/answer continuations are added as response-masked "
            "rehearsal; valid only for rows stamped split=trained"
        ),
    )
    ap.add_argument("--fine-grained-neg", type=int, default=0, help="V12: in-edition fine-grained anti-fab - "
                    "stage paired-contrast corrections (answerable) + set_time abstentions (refusal) for the "
                    "dims the by-dim gate fabricates. See fine_grained_negatives() + docs §XVII.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if (args.knowledge_filter or args.self_gen) and not args.score_model:
        sys.exit("--knowledge-filter/--self-gen require --score-model (knownness scoring); "
                 "without it they are silent no-ops. Add --score-model <cpt-ckpt>.")

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    cands: list[dict] = []
    ed_names: list[str] = []
    sft_tmpl = (Domain(args.domain).templates or {}).get("sft_templates", {})
    venue_templates = [tuple(t) for t in sft_tmpl.get("venue_templates", [])]
    rel_q = {k: tuple(v) for k, v in sft_tmpl.get("rel_q", {}).items()}
    for ed in iter_editions(args.domain, only=args.editions):
        split = load_split(ed.dir)
        train_ids = split["train_ids"] if split else None
        qa = edition_qas(ed, args.max_per_relation, args.answer_aug, venue_templates, rel_q, train_ids)
        if qa:
            cands.extend(qa)
            mp = ed.dir / "festival_meta.json"
            if mp.exists():
                ed_names.append(json.loads(mp.read_text()).get("name", ed.slug))
    # Phase C residual (2026-07-19): hard-neg defaults are proportional to positive QA
    # count — fixed 20/10 over-weighted a 60-positive set (~34% negs) and regressed FC.
    # Explicit 0 still disables. No contrast block → DOMAIN_HARD_NEGATIVES_UNAVAILABLE.
    from pipeline.derive_plan import derive_hard_neg_counts

    groups, contrast_preds = load_contrast(args.domain)
    contrast_authorable = bool(groups and contrast_preds)
    derived_wrong, derived_absent, hard_der = derive_hard_neg_counts(
        len(cands), contrast_authorable,
    )
    wrong_n = derived_wrong if args.wrong_value_neg is None else args.wrong_value_neg
    absent_n = derived_absent if args.absent_entity_neg is None else args.absent_entity_neg
    print(f"hard-neg derivation: {hard_der} → using wrong={wrong_n} absent={absent_n}"
          f"{' (override)' if args.wrong_value_neg is not None or args.absent_entity_neg is not None else ''}")

    generic_answerable: list[tuple[str, str]] = []
    generic_refusal: list[tuple[str, str]] = []
    generic_unavailable = False
    if wrong_n or absent_n:
        if not contrast_authorable:
            generic_unavailable = True
            print("DOMAIN_HARD_NEGATIVES_UNAVAILABLE: no authorable contrast groups for "
                  f"domain '{args.domain}'", file=sys.stderr)
        else:
            for ed in iter_editions(args.domain, only=args.editions):
                all_triples = load_all_triples(ed)
                if not all_triples:
                    continue
                split = load_split(ed.dir)
                train_triples = ([t for t in all_triples if triple_id(t) in split["train_ids"]]
                                 if split else all_triples)
                mp = ed.dir / "festival_meta.json"
                unit_name = ((json.loads(mp.read_text(encoding="utf-8")).get("name") or ed.slug)
                             if mp.exists() else ed.slug)
                if wrong_n:
                    generic_answerable.extend(generic_wrong_value_negatives(
                        train_triples, all_triples, groups, contrast_preds, rel_q,
                        wrong_n, random.Random(23), unit_name))
                if absent_n:
                    generic_refusal.extend(generic_absent_entity_negatives(
                        all_triples, groups, contrast_preds, rel_q,
                        absent_n, random.Random(29), unit_name))
    elif not contrast_authorable and (args.wrong_value_neg is None and args.absent_entity_neg is None):
        # Defaults resolved to 0 because no contrast block — stamp the unavailability once.
        generic_unavailable = True
        print("DOMAIN_HARD_NEGATIVES_UNAVAILABLE: no authorable contrast groups for "
              f"domain '{args.domain}' (hard-neg defaults stay 0)", file=sys.stderr)
    if not cands and not generic_answerable and not generic_refusal:
        sys.exit("no Q/A pairs built")

    stats = Counter()
    stats["raw"] = len(cands)
    stats["wrong_value_neg_requested"] = wrong_n
    stats["absent_entity_neg_requested"] = absent_n
    if wrong_n or absent_n or generic_unavailable:
        stats["wrong_value_neg"] = len(generic_answerable)
        stats["absent_entity_neg"] = len(generic_refusal)
        if generic_unavailable:
            stats["DOMAIN_HARD_NEGATIVES_UNAVAILABLE"] = True
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
    if generic_answerable:                                  # generic contrast corrections are answerable truth pairs
        answerable = list(dict.fromkeys(answerable + generic_answerable))
    if generic_refusal:                                     # generic absent entities are explicit refusals
        abs_all = list(dict.fromkeys(abs_all + generic_refusal))
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
    if args.fine_grained_neg > 0:                            # V12: in-edition stage corrections + set_time abstentions
        fg_corr, fg_abst = fine_grained_negatives(args.domain, args.fine_grained_neg, random.Random(13))
        fg_corr = [p for p in fg_corr if p[0] not in abstain_qs]
        fg_abst = [p for p in fg_abst if p[0] not in abstain_qs]
        answerable = list(dict.fromkeys(answerable + fg_corr))    # stage corrections are answerable-correct
        abs_all = list(dict.fromkeys(abs_all + fg_abst))          # set_time abstentions are refusals
        stats["fg_stage_corrections"] = len(fg_corr); stats["fg_settime_abstain"] = len(fg_abst)

    pairs = answerable + abs_all
    completion_pairs = load_completion_probes(args.completion_probes)
    if completion_pairs:
        completion_path = C.ROOT / str(args.completion_probes)
        stats["completion_probe_rows"] = len(completion_pairs)
        stats["completion_probe_sha256"] = hashlib.sha256(completion_path.read_bytes()).hexdigest()
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
        if split == "":
            for prompt, answer in completion_pairs:
                enc = encode_completion(tok, prompt, answer, args.seq_len)
                if enc is None:
                    skip += 1
                    continue
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
