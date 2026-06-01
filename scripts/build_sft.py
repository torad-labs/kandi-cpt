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
    """Normalize for knownness matching: lowercase, strip non-alphanumerics. Makes the gold-key check
    robust to camelCase vs spacing vs punctuation (e.g. 'kinetic Field' == 'kineticFIELD')."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def edition_qas(ed, max_per_rel: int) -> list[dict]:
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
        out.append({"q": f"Where is {name} held?", "a": f"{name} is held at {venue}.", "key": venue, "rel": "venue"})
        out.append({"q": f"What venue hosts {name}?", "a": f"{name} takes place at {venue}.", "key": venue, "rel": "venue"})

    seen_artist = set()
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
        obj = str(t.get("object", "")); day = (t.get("attrs") or {}).get("day", "")
        if ":stage:" in obj and by_rel["stage"] < max_per_rel:
            stage = FE.spaced(obj.split(":stage:")[1])
            out.append({"q": f"Which stage does {a} play at {name}?",
                        "a": f"{a} plays the {stage} stage at {name}.", "key": stage, "rel": "stage"})
            by_rel["stage"] += 1
            if day and by_rel["day"] < max_per_rel:
                out.append({"q": f"On which day does {a} perform at {name}?",
                            "a": f"{a} performs on {day} at {name}.", "key": day, "rel": "day"})
                by_rel["day"] += 1
        elif ":festival:" in obj and a not in seen_artist and by_rel["edition"] < max_per_rel * 3:
            out.append({"q": f"At which EDC edition did {a} perform?", "a": f"{a} performed at {name}.",
                        "key": name, "rel": "edition"})
            seen_artist.add(a); by_rel["edition"] += 1

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


def score_knownness(model, tok, cands, k: int):
    """For each candidate, generate k answers from the CPT model and tier by how often the gold `key`
    appears: known (≥75%), maybe (>0), unknown (0). Also capture a correct self-generated answer (#3)."""
    import torch
    model.eval()
    for c in cands:
        if c["rel"] == "abstain" or not c.get("key"):
            c["tier"], c["self"] = "known", None        # synthesized abstentions aren't knowledge-scored
            continue
        msgs = [{"role": "user", "content": c["q"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        keyn = _norm(c["key"])
        hits, self_ans = 0, None
        with torch.no_grad():
            for i in range(k):
                out = model.generate(**enc, max_new_tokens=40, do_sample=(i > 0), temperature=0.7,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(out[0][plen:], skip_special_tokens=True)
                if keyn and keyn in _norm(txt):
                    hits += 1
                    if self_ans is None:
                        self_ans = txt.strip().split("\n")[0]
        frac = hits / max(1, k)
        c["tier"] = "known" if frac >= 0.75 else "maybe" if frac > 0 else "unknown"
        c["self"] = self_ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-per-relation", type=int, default=6)
    ap.add_argument("--out", default="sft_v1")
    ap.add_argument("--drop-generic", action="store_true", help="#4 drop entity-free generic answers")
    ap.add_argument("--score-model", help="CPT checkpoint to score QA knownness (enables #1/#3/unknown→abstain)")
    ap.add_argument("--score-samples", type=int, default=4)
    ap.add_argument("--knowledge-filter", action="store_true", help="#1 drop Unknown-tier QAs (needs --score-model)")
    ap.add_argument("--self-gen", action="store_true", help="#3 use the model's own correct answer for Known QAs")
    ap.add_argument("--abstention-frac", type=float, default=0.0, help="#2 target fraction of abstention examples")
    ap.add_argument("--val-frac", type=float, default=0.0, help="hold out a val split for the early-stop lever")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    cands: list[dict] = []
    ed_names: list[str] = []
    for ed in iter_editions("edc", only=args.editions):
        qa = edition_qas(ed, args.max_per_relation)
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
        from transformers import Gemma4ForCausalLM
        m = Gemma4ForCausalLM.from_pretrained(args.score_model, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to(args.device).eval()
        score_knownness(m, tok, cands, args.score_samples)
        del m; torch.cuda.empty_cache()
        for c in cands:
            stats[f"tier_{c.get('tier','?')}"] += 1

    abstain_pool: list[dict] = []
    if args.score_model:                                    # unknown-tier QAs → relabel to IDK (#2)
        rng0 = random.Random(1)
        for c in cands:
            if c.get("tier") == "unknown":
                abstain_pool.append({"q": c["q"], "a": rng0.choice(IDK_RESPONSES), "rel": "abstain"})

    if args.knowledge_filter and args.score_model:
        before = len(cands)
        cands = [c for c in cands if c.get("tier") in ("known", "maybe")]
        stats["dropped_unknown"] = before - len(cands)
    if args.self_gen and args.score_model:
        for c in cands:
            if c.get("tier") == "known" and c.get("self"):
                c["a"] = c["self"]; stats["self_gen_swapped"] += 1

    answerable = [(c["q"], c["a"]) for c in cands]
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

    pairs = answerable + abs_all
    rng.shuffle(pairs)

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
    print(f"  levers: drop_generic={args.drop_generic} score_model={bool(args.score_model)} "
          f"knowledge_filter={args.knowledge_filter} self_gen={args.self_gen} "
          f"abstention_frac={args.abstention_frac} val_frac={args.val_frac}")


if __name__ == "__main__":
    main()
