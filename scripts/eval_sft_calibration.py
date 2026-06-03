#!/usr/bin/env python3
"""Deterministic SFT success scorecard. Answers ONE question reproducibly: did this SFT run produce a model
that ANSWERS what it knows, ABSTAINS on what it can't, and does NOT fabricate?

Determinism by construction — the same checkpoint always yields the same verdict:
  • the probe set is a fixed, seeded selection of real facts (KNOWN) + synthesized non-existent
    editions (ABSENT) — built from the committed data, not sampled fresh each run;
  • generation is GREEDY (do_sample=False) — no temperature, no sampling;
  • classification is RULE-BASED (abstain-regex + normalized gold-key substring) — no LLM judge.

  KNOWN  probes — real venue facts the CPT installed   → right behaviour = answer correctly
  ABSENT probes — "EDC <city> <impossible-year>" (provably no such edition) → right = abstain;
                  any concrete answer is a FABRICATION.

The gate guards BOTH failure modes — a model that fabricates fails on ABSENT; a degenerate model that
abstains on everything fails on coverage. PASS requires it to thread the needle:

  PASS ⟺ fabrication_rate ≤ τ_fab  AND  accuracy_when_answering ≥ τ_acc  AND  coverage ≥ τ_cov

Thresholds are constants below (versioned in code), overridable on the CLI, and recorded in the output so
the verdict is fully reproducible from the artifact alone.

    PYTHONPATH=scripts python scripts/eval_sft_calibration.py \
        --checkpoints checkpoints/sft/edc_sft_v1 checkpoints/sft/edc_sft_v2/step_000200 \
        --labels naive_v1 levered_v2  [--n-known 80 --n-absent 60] [--device cuda:0]

Writes data/_build/calibration/{scorecard_<label>.json, calibration_results.jsonl, scorecard_data.js,
report_<label>.html}. Open report_*.html locally — it is the visual scorecard.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import torch
from transformers import AutoTokenizer

import config as C
from pipeline.editions import iter_editions
from pipeline.model_load import load_causal_ckpt_bf16
from build_sft import synth_unanswerable   # reuse the unanswerable synthesizer


def _fold(s: str) -> str:
    """Lowercase + accent-fold (NFKD, drop combining marks) so 'Autódromo' folds to 'autodromo'. The naive
    [^a-z0-9] strip DELETES 'ó' (it's not a-z) → 'autdromo' ≠ 'autodromo', a silent false-negative. NFKD first."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _gold_match(resp: str, gold: str) -> bool:
    """Deterministic, accent- and abbreviation-robust gold match. Strip parenthetical qualifiers
    ('(and others)', '(cruise)'), then require every content token (>2 chars) of the gold to appear in the
    folded response — order-free, so 'Los Angeles Memorial Coliseum' matches gold 'LA Memorial Coliseum'."""
    r = _fold(resp)
    g = re.sub(r"\([^)]*\)", " ", _fold(gold))
    toks = [t for t in re.findall(r"[a-z0-9]+", g) if len(t) > 2] or re.findall(r"[a-z0-9]+", g)
    # word-boundary match, not raw substring: short tokens ("day","field","stage") must not match inside
    # "friday"/"garfield"/"backstage" — that would false-positive the BubbleUp baseline + the accuracy count.
    return bool(toks) and all(re.search(r"\b" + re.escape(t) + r"\b", r) for t in toks)

# ---- success gate (single-sourced from the config contract; --tau-* override; recorded in every artifact) --
from cpt_config import GATE_THRESHOLDS   # the ONE gate-threshold source (kills the eval/serve triplication)
TAU_FAB = GATE_THRESHOLDS["fabrication"]   # max fabrication rate on ABSENT  — the hallucination ceiling
TAU_ACC = GATE_THRESHOLDS["accuracy"]      # min accuracy when it answers a KNOWN — the wrong-fact floor
TAU_COV = GATE_THRESHOLDS["coverage"]      # min coverage (answered / KNOWN) — guards always-abstain

# Deterministic abstention detector. Matches the IDK_RESPONSES build_sft trains on AND the natural
# abstentions the levered model emits ("I don't have the specific … check the official lineup").
ABSTAIN_RE = re.compile(
    r"i\s+don'?t\s+(have|know)|i\s+do\s+not\s+(have|know)|i'?m\s+not\s+(sure|certain|aware)|"
    r"not\s+(sure|certain)\s+(about|of|which|who|what)|don'?t\s+have\s+(reliable|specific|that|the|access)|"
    r"no\s+(reliable|specific|record|records|information|data|details?)\b|"
    r"cannot\s+(provide|find|confirm|verify)|can'?t\s+(provide|find|confirm|verify)|"
    r"unable\s+to\s+(provide|find|confirm|answer)|rather\s+not\s+guess|outside\s+what\s+i\s+can|"
    r"i\s+don'?t\s+know|check\s+the\s+official|consult\s+the\s+official", re.I)

# General-ability probes (catastrophic-forgetting check). Reported, not gated — the gate is about EDC
# calibration; this is the "did SFT break the base model" sentinel.
GENERAL = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is 2 plus 2?", ["4", "four"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
]
FAKE_CITIES = ["Atlantis", "Gotham", "Wakanda", "El Dorado", "Narnia", "Springfield Mars"]


def classify(resp: str, gold_keys: list[str], is_absent: bool) -> str:
    """Deterministic verdict for one response. ABSENT: abstain=correct, anything concrete=fabrication.
    KNOWN: abstain=over_refusal, gold-key present=correct, else=wrong."""
    abstained = bool(ABSTAIN_RE.search(resp)) or len(re.findall(r"[a-z0-9]", _fold(resp))) < 2
    if is_absent:
        return "abstain" if abstained else "fabrication"
    if abstained:
        return "over_refusal"
    return "correct" if any(_gold_match(resp, g) for g in gold_keys if g) else "wrong"


def _camel_variants(stage: str) -> list[str]:
    """A stage's training surfaces: 'Cosmic Meadow' + 'cosmicMEADOW' + 'cosmic_meadow'."""
    parts = re.split(r"[ _]+", stage.strip())
    camel = parts[0].lower() + "".join(p.upper() for p in parts[1:]) if len(parts) > 1 else stage
    return list({stage, camel, "_".join(p.lower() for p in parts)})


def _gather(ed) -> dict:
    """Every probe-relevant fact for one edition: meta (venue/dates/attendance/stages) + the plays_at
    graph (roster, artist→stage, artist→day, artist→set_time) + the relational layer (located_at,
    belongs_to_class, serves/requires/practiced_by/prohibits). The single source the dimensions draw from."""
    g = {"name": ed.slug, "label": f"{ed.city.replace('_', ' ').title()} {ed.year}",
         "venue": [], "dates": "", "attendance": None, "stages": [], "roster": [],
         "artist_stage": {}, "artist_day": {}, "artist_time": {}, "located_at": [], "belongs": [], "rel": []}
    mp = ed.dir / "festival_meta.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        g["name"] = m.get("name") or ed.slug
        v = (m.get("venue") or "").strip()
        if v and not v.lower().startswith("various"):
            g["venue"] = [v] + (m.get("venue_aliases") or [])
        g["dates"] = str(m.get("dates") or "")
        g["attendance"] = m.get("attendance")
        g["stages"] = list(m.get("stages") or [])
    names = {}
    ep = ed.dir / "15_structured" / "entities.jsonl"
    if ep.exists():
        for line in ep.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("id"):
                names[r["id"]] = r.get("name_canonical", "")
            et, nc = r.get("entity_type"), r.get("name_canonical")
            if et == "artist" and nc:
                g["roster"].append(nc)
            elif et == "stage" and nc and nc not in g["stages"]:
                g["stages"].append(nc)
    tp = ed.dir / "15_structured" / "triples.jsonl"
    if tp.exists():
        for line in tp.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            p, sid, obj, a = t.get("predicate"), t.get("subject_id", ""), t.get("object", ""), (t.get("attrs") or {})
            subj = names.get(sid) or sid.split(":")[-1].replace("_", " ")
            ostr = obj if isinstance(obj, str) else str(obj)
            if p == "plays_at" and ":artist:" in sid:
                art = names.get(sid) or sid.split(":")[-1].replace("_", " ")
                if ":stage:" in ostr:
                    g["artist_stage"].setdefault(art, names.get(obj) or ostr.split(":")[-1].replace("_", " "))
                if a.get("day"):
                    g["artist_day"].setdefault(art, a["day"])
                if a.get("set_time"):
                    g["artist_time"].setdefault(art, a["set_time"])
            elif p == "located_at":
                g["located_at"].append((subj, names.get(obj) or ostr))
            elif p == "belongs_to_class":
                g["belongs"].append((subj, ostr))
            elif p in ("serves", "requires", "practiced_by", "prohibits"):
                g["rel"].append((p, subj, names.get(obj) or ostr))
    return g


def build_probes(seed: int, scale: float = 1.0, n_absent: int = 90):
    """COMPREHENSIVE KNOWN set — one probe pool per fact dimension present in the graph, drawn across the
    FULL breadth of editions, held-out phrasing throughout. Dimensions: venue, dates, attendance, lineup
    (roster), stages (edition→stage), artist_edition (cross-edition), artist_stage, artist_day, set_time,
    located_at, belongs_to_class, relational (serves/requires/practiced_by/prohibits). ABSENT spans every
    question FORM (fake editions + fake artists) so fabrication-refusal is tested on each dimension, not
    just venue. Deterministic. `scale` multiplies the per-dimension base counts."""
    G = [_gather(ed) for ed in iter_editions("edc")]
    names = [g["name"] for g in G]
    artist_eds: dict[str, set] = {}                              # cross-edition: artist → {edition labels}
    for g in G:
        for art in g["roster"]:
            artist_eds.setdefault(art, set()).add(g["label"])

    # per-dimension candidate pools: (question, gold_keys)
    pools: dict[str, list] = {k: [] for k in (
        "venue", "dates", "attendance", "lineup", "stages", "artist_edition",
        "artist_stage", "artist_day", "set_time", "located_at", "belongs", "relational")}
    for g in G:
        nm = g["name"]
        if g["venue"]:
            pools["venue"].append((f"What is the venue for {nm}?", g["venue"]))
        if g["dates"]:
            yr = (re.search(r"\b(19|20)\d{2}\b", nm) or [None])[0]
            gold = [g["dates"]] + ([yr] if yr else [])
            pools["dates"].append((f"What dates does {nm} take place?", gold))
        if g["attendance"]:
            att = g["attendance"]
            pools["attendance"].append((f"How many people attended {nm}?", [str(att), f"{att:,}"]))
        if len(g["roster"]) >= 15:
            pools["lineup"].append((f"Name an artist who performed at {nm}.", g["roster"]))
        if g["stages"]:
            sg = []
            for s in g["stages"]:
                sg += _camel_variants(s)
            pools["stages"].append((f"Name a stage at {nm}.", sg))
        for art, stage in list(g["artist_stage"].items())[:12]:
            pools["artist_stage"].append((f"Which stage did {art} play at {nm}?", _camel_variants(stage)))
        for art, day in list(g["artist_day"].items())[:12]:
            pools["artist_day"].append((f"What day did {art} perform at {nm}?", [day]))
        for art, tm in list(g["artist_time"].items())[:10]:
            pools["set_time"].append((f"What time did {art} perform at {nm}?", [tm]))
        for sub, loc in g["located_at"][:6]:
            if loc:
                pools["located_at"].append((f"Where is {sub} located at {nm}?", [loc]))
        for sub, cls in g["belongs"][:6]:
            if cls:
                pools["belongs"].append((f"What category does {sub} belong to at {nm}?", [cls.split(':')[-1].replace('_', ' ')]))
        for pred, sub, o in g["rel"][:6]:
            if o:
                q = {"serves": f"What does {sub} serve at {nm}?", "requires": f"What does {sub} require at {nm}?",
                     "practiced_by": f"Who practices {sub} at {nm}?", "prohibits": f"What does {sub} prohibit at {nm}?"}[pred]
                pools["relational"].append((q, [o]))
    # artist_edition (cross-edition) — held-out phrasing of the trained "At which EDC edition did {a} perform?"
    for art, eds in artist_eds.items():
        if art and len(art) > 2:
            pools["artist_edition"].append((f"Which Electric Daisy Carnival did {art} perform at?", list(eds)))

    BASE = {"venue": 45, "dates": 30, "attendance": 12, "lineup": 50, "stages": 25, "artist_edition": 50,
            "artist_stage": 40, "artist_day": 40, "set_time": 20, "located_at": 18, "belongs": 18, "relational": 18}
    known = []
    for sk, pool in pools.items():
        random.Random(hash(sk) ^ seed).shuffle(pool)
        for q, gold in pool[:int(BASE[sk] * scale)]:
            known.append({"q": q, "gold": gold, "kind": "known", "subkind": sk})

    # ABSENT — fake editions across multiple question forms + fake artists (fabrication-refusal on every dim)
    rng = random.Random(seed + 1)
    n_real = (n_absent * 2) // 3
    absent = [{"q": r["q"], "gold": [], "kind": "absent"} for r in synth_unanswerable(names, n_real, rng)]
    fake_artists = ["Zyglax", "DJ Nebulon", "Vexora", "The Quasar Kids", "Mxntenna", "Lumiscape"]
    while len(absent) < n_absent:
        if rng.random() < 0.35:                                  # fake-artist edition recall → must abstain
            fa = rng.choice(fake_artists)
            absent.append({"q": rng.choice([f"Which EDC edition did {fa} perform at?",
                                             f"What stage did {fa} play at EDC Las Vegas 2024?"]), "gold": [], "kind": "absent"})
        else:                                                    # fake-edition across dims → must abstain
            c = rng.choice(FAKE_CITIES); y = rng.choice([2024, 2025, 2026])
            absent.append({"q": rng.choice([f"Where is EDC {c} {y} held?", f"Who headlined EDC {c} {y}?",
                f"Name an artist who performed at EDC {c} {y}.", f"What dates does EDC {c} {y} take place?",
                f"Name a stage at EDC {c} {y}."]), "gold": [], "kind": "absent"})
    general = [{"q": q, "gold": g, "kind": "general"} for q, g in GENERAL]
    return known + absent[:n_absent] + general


@torch.no_grad()
def generate(model, tok, probes, bs: int, max_new: int = 48) -> list[str]:
    """Batched GREEDY generation (left-pad). Deterministic: same probes + same batch order → same text."""
    orig_side, orig_pad = tok.padding_side, tok.pad_token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    resp: list[str] = []
    try:
        for i in range(0, len(probes), bs):
            batch = probes[i:i + bs]
            prompts = [tok.apply_chat_template([{"role": "user", "content": p["q"]}],
                                               add_generation_prompt=True, tokenize=False) for p in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            plen = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            for b in range(len(batch)):
                resp.append(tok.decode(out[b][plen:], skip_special_tokens=True).replace("\n", " ").strip())
    finally:
        tok.padding_side, tok.pad_token = orig_side, orig_pad
    return resp


def score(probes, responses, tau_fab, tau_acc, tau_cov) -> dict:
    """Confusion matrix + rates + the recorded PASS/FAIL gate."""
    rows = []
    for p, r in zip(probes, responses):
        rows.append({**p, "resp": r, "verdict": classify(r, p["gold"], p["kind"] == "absent")})
    K = [x for x in rows if x["kind"] == "known"]
    A = [x for x in rows if x["kind"] == "absent"]
    G = [x for x in rows if x["kind"] == "general"]
    cnt = lambda xs, v: sum(1 for x in xs if x["verdict"] == v)
    k_corr, k_wrong, k_ref = cnt(K, "correct"), cnt(K, "wrong"), cnt(K, "over_refusal")
    a_abs, a_fab = cnt(A, "abstain"), cnt(A, "fabrication")
    nk, na = max(1, len(K)), max(1, len(A))
    answered = k_corr + k_wrong
    coverage = answered / nk
    acc_ans = k_corr / answered if answered else 0.0
    fab = a_fab / na
    metrics = {
        "n_known": len(K), "n_absent": len(A), "n_general": len(G),
        "coverage": coverage, "accuracy_when_answering": acc_ans,
        "known_answer_acc": k_corr / nk, "over_refusal_rate": k_ref / nk,
        "fabrication_rate": fab, "absent_abstain_rate": a_abs / na,
        "calibration_score": (k_corr + a_abs) / max(1, len(K) + len(A)),
        "general_acc": cnt(G, "correct") / max(1, len(G)),
    }
    # per-dimension KNOWN breakdown so every fact type's install is visible, not blended into one number
    by_dim = {}
    for sk in sorted({x.get("subkind") for x in K if x.get("subkind")}):
        sub = [x for x in K if x.get("subkind") == sk]
        sans = sum(1 for x in sub if x["verdict"] in ("correct", "wrong"))
        by_dim[sk] = {"n": len(sub), "coverage": sans / max(1, len(sub)),
                      "accuracy": (cnt(sub, "correct") / sans) if sans else 0.0,
                      "correct": cnt(sub, "correct"), "wrong": cnt(sub, "wrong"), "over_refusal": cnt(sub, "over_refusal")}
    metrics["by_dimension"] = by_dim
    gate = {"fabrication_ok": fab <= tau_fab, "accuracy_ok": acc_ans >= tau_acc, "coverage_ok": coverage >= tau_cov}
    metrics["passed"] = all(gate.values())
    metrics["gate"] = gate
    confusion = {"known": {"correct": k_corr, "over_refusal": k_ref, "wrong": k_wrong, "n": len(K)},
                 "absent": {"abstain": a_abs, "fabrication": a_fab, "n": len(A)}}
    return {"metrics": metrics, "confusion": confusion, "rows": rows}


_ED_RE = re.compile(r"EDC\s+([A-Za-z .]+?)\s+(\d{4})", re.I)


def _edition(q: str):
    m = _ED_RE.search(q or "")
    return (m.group(1).strip().lower().replace(" ", "_") + "/" + m.group(2)) if m else None


def _relation(q: str) -> str:
    ql = (q or "").lower()
    if "stage" in ql:
        return "stage"
    if "headlin" in ql or "lineup" in ql:
        return "lineup"
    if "day" in ql or "perform" in ql:
        return "day"
    if "venue" in ql or "where" in ql or "held" in ql:
        return "venue"
    return "other"


def sample_rows(rows):
    """Receipts for the scorecard + the load-bearing input to the control-room BubbleUp diff. Emit EVERY
    diagnostic-failure transcript (fabrication / wrong / over_refusal) — not a 4-row sample — each tagged with
    the parsed {edition, relation} so the operating room can diff which DIMENSION separates failures from
    good answers. Passing verdicts (correct / abstain) are capped as exemplars."""
    def tag(r):
        return {"q": r["q"], "resp": r["resp"][:220], "verdict": r["verdict"], "kind": r["kind"],
                "edition": _edition(r["q"]), "relation": _relation(r["q"])}
    out = []
    for v in ("fabrication", "wrong", "over_refusal"):   # ALL failures (the BubbleUp needs the full set)
        out += [tag(r) for r in rows if r["verdict"] == v]
    for v in ("correct", "abstain"):                     # exemplars only
        out += [tag(r) for r in rows if r["verdict"] == v][:6]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", help="display label per checkpoint (default: dir basename)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on per-dimension KNOWN probe counts")
    ap.add_argument("--n-absent", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau-fab", type=float, default=TAU_FAB)
    ap.add_argument("--tau-acc", type=float, default=TAU_ACC)
    ap.add_argument("--tau-cov", type=float, default=TAU_COV)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    labels = args.labels or [c.rstrip("/").split("/")[-1] for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        ap.error("--labels must match --checkpoints count")
    thr = {"fabrication": args.tau_fab, "accuracy_when_answering": args.tau_acc, "coverage": args.tau_cov}

    probes = build_probes(args.seed, args.scale, args.n_absent)
    nk = sum(p["kind"] == "known" for p in probes); na = sum(p["kind"] == "absent" for p in probes)
    from collections import Counter as _C
    dims = _C(p.get("subkind") for p in probes if p["kind"] == "known")
    print(f"probe set (seed={args.seed}): {nk} KNOWN + {na} ABSENT + {len(GENERAL)} general = {len(probes)} probes")
    print("  KNOWN by dimension: " + "  ".join(f"{k}={v}" for k, v in sorted(dims.items())))
    print(f"gate: fabrication≤{args.tau_fab}  accuracy_when_answering≥{args.tau_acc}  coverage≥{args.tau_cov}\n")

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    out_dir = C.ROOT / "data/_build/dissect"; out_dir.mkdir(parents=True, exist_ok=True)   # shared dissect home
    ledger = out_dir / "calibration_results.jsonl"
    runs = []
    for ck, label in zip(args.checkpoints, labels):
        m = load_causal_ckpt_bf16(ck, args.device)
        responses = generate(m, tok, probes, args.batch)
        del m; torch.cuda.empty_cache()
        res = score(probes, responses, args.tau_fab, args.tau_acc, args.tau_cov)
        mt = res["metrics"]
        verdict = "PASS" if mt["passed"] else "FAIL"
        print(f"  {label:>14} [{verdict}]  fab={mt['fabrication_rate']:.2f} acc_ans={mt['accuracy_when_answering']:.2f} "
              f"cov={mt['coverage']:.2f} | calib={mt['calibration_score']:.2f} gen={mt['general_acc']:.2f}")
        bd = mt.get("by_dimension", {})
        if bd:
            print("       by-dim cov/acc: " + "  ".join(
                f"{k}={v['coverage']:.2f}/{v['accuracy']:.2f}" for k, v in sorted(bd.items())))
        (out_dir / f"scorecard_{label}.json").write_text(json.dumps(
            {"label": label, "checkpoint": ck, "thresholds": thr, **res}, indent=2, default=float))
        with open(ledger, "a") as f:
            f.write(json.dumps({"label": label, "checkpoint": ck, **mt}, default=float) + "\n")
        runs.append({"label": label, "checkpoint": ck, "metrics": mt, "confusion": res["confusion"],
                     "samples": sample_rows(res["rows"])})

    payload = {"thresholds": thr, "n_known": nk, "n_absent": na, "runs": runs}
    (out_dir / "scorecard_data.js").write_text("window.SCORECARD = " + json.dumps(payload, default=float) + ";")
    tpl = Path(__file__).parent / "dissect" / "dissect.html"   # the unified dashboard (behavior+geometry+weights)
    if tpl.exists():
        (out_dir / "index.html").write_text(tpl.read_text())
        print(f"\n  → scorecard_data.js + index.html (open data/_build/dissect/index.html) + {ledger.name}")
    else:
        print(f"\n  → scorecard_data.js + {ledger.name}  (dissect.html dashboard not found — data only)")


if __name__ == "__main__":
    main()
