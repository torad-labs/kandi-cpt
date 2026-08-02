#!/usr/bin/env python3
"""Torad Train — local CPT+SFT training console (stdlib server, read-mostly, no GPU).

    python scripts/serve.py [--port 8800]   →   http://localhost:8800

Serves the Lab-Light console + a JSON API over the artifacts the CPT/SFT pipeline
already writes — there is no separate database. Sources, all on disk:

  checkpoints/sft/<run>/run_contract.json   the recipe that produced a run
  data/_build/dissect/calibration_results.jsonl   append-only behavioural ledger
  data/_build/dissect/scorecard_<label>.json      per-run confusion + transcripts
  data/_build/dissect/{data_audit,weights,geometry}_data.js   install instruments
  data/_build/geometry/results.jsonl              CPT basin-separation probe
  data/edc/<city>/<year>/{10_clean,15_structured,25_hydrated,holdout}   the corpus
  data/_build/train_status.json                   live-run heartbeat (Phase D)

This process never touches the GPU and never launches training; it reflects what
the pipeline produced and tails the active run's log. Stdlib only — no new deps
in the cu130 torch venv. Live updates are polled by the frontend (steps are ~3s).
"""
import argparse, json, math, os, re, subprocess, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpt_config import GATE_THRESHOLDS as _GT   # the ONE gate-threshold source (kills the eval/serve triplication)
import torad_actions  # the ACTION CATALOG — POST /api/action/<name> dispatches here (same catalog the MCP server exposes)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PAGES = HERE / "report" / "pages"
ASSETS = HERE / "report" / "assets"
DISSECT = ROOT / "data" / "_build" / "dissect"
GEODIR = ROOT / "data" / "_build" / "geometry"
SFT_DIR = ROOT / "checkpoints" / "sft"
CPT_DIR = ROOT / "checkpoints" / "cpt"
DATA_ROOT = ROOT / "data"
OBJECTIVES_CFG = ROOT / "configs" / "objectives.json"
STATUS_FILE = ROOT / "data" / "_build" / "train_status.json"
LEDGER = DISSECT / "calibration_results.jsonl"

# Run ids and asset names come from URL path/query — allow only safe single segments /
# relative subpaths (no absolute paths, no '..') so path joins cannot escape their root.
_RUN_ID_RE = re.compile(r"^[\w.-]+$")


def _safe_run_id(rid: str | None) -> str | None:
    if not rid or not _RUN_ID_RE.fullmatch(rid):
        return None
    return rid


def _under(root: Path, rel: str) -> Path | None:
    """Return root/rel resolved, or None if rel escapes root (path-injection guard)."""
    if not rel or rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
        return None
    try:
        base = root.resolve()
        target = (root / rel).resolve()
        target.relative_to(base)
        return target
    except (ValueError, OSError):
        return None


def corpus_dir(festival):
    return DATA_ROOT / festival

GATE = {"fabrication": _GT["fabrication"], "accuracy_when_answering": _GT["accuracy"], "coverage": _GT["coverage"]}
CHANCE = 1 / 7  # stage-retrieval probe ranks ~7 stages — p@1 vs this is the install ratio

# ───────────────────────────── artifact readers ─────────────────────────────

def _clean(o):
    """Replace non-finite floats with None so the browser's JSON.parse never chokes
    (geometry carries bare NaN; Python tolerates it on load, JS does not)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def read_js_global(path):
    """Parse a `window.NAME = {…};` data-global file into a Python object."""
    try:
        txt = Path(path).read_text()
    except Exception:
        return None
    eq = txt.find("=")
    if eq < 0:
        return None
    body = txt[eq + 1:].strip().rstrip(";").strip()
    try:
        return json.loads(body)  # json accepts NaN/Infinity on load
    except Exception:
        return None


def read_jsonl(path):
    out = []
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def _vnum(run_id):
    """Trailing integer in a run id (edc_sft_v12 → 12). Linear scan — no regex backtracking."""
    if not run_id:
        return 0
    s = run_id.rstrip()
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    if i == len(s):
        return 0
    return int(s[i:])


def run_id_from_ckpt(ckpt):
    """checkpoints/sft/edc_sft_v6/step_000200 → edc_sft_v6"""
    if not ckpt:
        return None
    parts = Path(ckpt).parts
    if "sft" in parts:
        i = parts.index("sft")
        if i + 1 < len(parts):
            return parts[i + 1]
    m = re.search(r"(edc_sft_v\d+|edc_[a-z0-9_]+)", ckpt)
    return m.group(1) if m else None


def ledger_latest_by_run():
    """run_id → the most-recent calibration entry for it (file order = chronological)."""
    by_run = {}
    for e in read_jsonl(LEDGER):
        rid = run_id_from_ckpt(e.get("checkpoint"))
        if rid:
            by_run[rid] = e  # last write wins
    return by_run


# instruments are global (one CPT-install context shared by the SFT runs trained from it)
def geometry():     return read_js_global(DISSECT / "geometry_data.js") or read_jsonl(GEODIR / "results.jsonl")
def audit():        return read_js_global(DISSECT / "data_audit_data.js")
def weights():      return read_js_global(DISSECT / "weights_data.js")


def install_level(geo, da):
    """CPT-install verdict (distinct from the behavioural run gate) — ported verbatim
    from coach.html: p@1 vs 1/7 chance + audit fails."""
    ratio = None
    if geo:
        p = geo[-1].get("p@1_cpt")
        if p is not None:
            ratio = p / CHANCE
    fails = []
    if da and da.get("dimensions"):
        for k in ("density", "balance", "coverage", "disjoint_static", "disjoint_latent"):
            d = da["dimensions"].get(k)
            if d and d.get("verdict") == "fail":
                fails.append(k)
    if ratio is not None and ratio >= 2.5 and not fails:
        level = "strong"
    elif (ratio is not None and ratio < 1.8) or len(fails) >= 2:
        level = "weak"
    else:
        level = "partial"
    return {"level": level, "ratio": ratio, "fails": fails}


def run_summary(rdir, led):
    rid = rdir.name
    e = led.get(rid)
    contract = read_json(rdir / "run_contract.json")
    steps = sorted([p.name for p in rdir.glob("step_*") if p.is_dir()])
    if e:
        status = "passed" if e.get("passed") else "failed"
        metrics = {
            "fabrication_rate": e.get("fabrication_rate"),
            "accuracy_when_answering": e.get("accuracy_when_answering"),
            "coverage": e.get("coverage"),
            "calibration_score": e.get("calibration_score"),
            "over_refusal_rate": e.get("over_refusal_rate"),
        }
        gate = e.get("gate")
        label = e.get("label")
    else:
        status, metrics, gate, label = "trained", None, None, None
    return {
        "id": rid,
        "vnum": _vnum(rid),
        "label": label,
        "status": status,           # passed | failed | trained (not gated)
        "date": rdir.stat().st_mtime,
        "metrics": metrics,
        "gate": gate,
        "checkpoints": steps,
        "init_from": (contract or {}).get("init_from"),
        "max_steps": (contract or {}).get("max_steps"),
        "levers_on": (contract or {}).get("levers_on") or [],
    }


def load_objectives():
    """The objective registry (configs/objectives.json), annotated with disk reality.
    An objective is a festival domain: its own corpus (data/<festival>) and its own
    run family (checkpoints/sft/<run_prefix>_*). Declared objectives show in the
    switcher even before any data/runs exist (status 'planned')."""
    cfg = read_json(OBJECTIVES_CFG, {}) or {}
    objs = cfg.get("objectives") if isinstance(cfg, dict) else None
    if not objs:
        objs = [{"id": "edc", "name": "EDC Corpus Install", "festival": "edc",
                 "run_prefix": "edc_sft", "goal": "Install EDC festival knowledge.", "status": "active"}]
    runs_dirs = [d.name for d in SFT_DIR.iterdir() if d.is_dir()] if SFT_DIR.is_dir() else []
    for o in objs:
        fest = o.get("festival", o["id"])
        pref = o.get("run_prefix", fest)
        o["has_data"] = corpus_dir(fest).is_dir()
        o["n_runs"] = sum(1 for n in runs_dirs if n.startswith(pref))
    return objs


def resolve_objective(oid):
    objs = load_objectives()
    for o in objs:
        if o["id"] == oid:
            return o
    return objs[0]


def list_runs(obj):
    led = ledger_latest_by_run()
    pref = obj.get("run_prefix", "")
    runs = [run_summary(d, led) for d in SFT_DIR.iterdir()
            if d.is_dir() and d.name.startswith(pref)] if SFT_DIR.is_dir() else []
    runs.sort(key=lambda r: r["vnum"])
    return _clean(runs)


def run_detail(rid):
    rid = _safe_run_id(rid)
    if not rid:
        return None
    rdir = _under(SFT_DIR, rid)
    if not rdir or not rdir.is_dir():
        return None
    led = ledger_latest_by_run()
    summ = run_summary(rdir, led)
    contract = read_json(rdir / "run_contract.json")
    label = summ.get("label")
    scorecard = read_json(DISSECT / f"scorecard_{label}.json") if label else None
    geo, da, w = geometry(), audit(), weights()
    detail = {
        **summ,
        "contract": contract,
        "scorecard": scorecard,
        "install": install_level(geo, da),
        "geometry": geo,
        "audit": da,
        "weights": w,
        "log_available": bool(_log_path(rid)),
    }
    return _clean(detail)


# ───────────────────────────── coach ────────────────────────────────────────
# Reads the WHOLE campaign for an objective and turns it into plain-language
# coaching: what went right, what went wrong, and the ranked concrete levers for
# the next run. Rules key off the LIVE verdict state so this recomputes for any
# objective/campaign — it is not hard-coded to EDC. The deterministic baseline
# below is sharpened by the coach-intelligence workflow (V9 wording / expected
# effects), but stands on its own.
GATE_T = {"fabrication": _GT["fabrication"], "accuracy": _GT["accuracy"], "coverage": _GT["coverage"]}


def _traj_note(r):
    m = r.get("metrics") or {}
    cov, fab = m.get("coverage") or 0, m.get("fabrication_rate") or 0
    if r.get("status") == "passed":
        return "passes all three gates"
    if cov < 0.10:
        return "over-refusal cliff — refused almost everything"
    if fab > 0.6:
        return "answers everything — rampant fabrication"
    if fab > GATE_T["fabrication"] and cov >= GATE_T["coverage"]:
        return "fabrication above gate — negative dose too low"
    if cov < GATE_T["coverage"]:
        return "over-refusal — coverage below gate"
    return "below gate"


def _analyze_run(m, gate, prior_m, audit_fails):
    """Diagnose ONE run's data: what went well, what went wrong, what could improve, and which levers to
    pull NEXT run to lift every weak/thin area at once — WITHOUT taxing the strong ones. The Coach does
    not decide what to ship or keep; that call is the operator's."""
    T = GATE_T
    well, wrong, improve, levers = [], [], [], []
    if not m:
        return {"went_well": [], "went_wrong": [], "could_improve": [], "next_levers": [],
                "summary": "Not scored yet — run the calibration test and the Coach's read appears here."}
    fab, acc, cov = m.get("fabrication_rate"), m.get("accuracy_when_answering"), m.get("coverage")
    cal = m.get("calibration_score")
    fab_ok = gate.get("fabrication_ok", fab is None or fab <= T["fabrication"])
    acc_ok = gate.get("accuracy_ok", acc is None or acc >= T["accuracy"])
    cov_ok = gate.get("coverage_ok", cov is None or cov >= T["coverage"])

    if fab is not None:
        if fab_ok:
            well.append(f"Truthful — invents answers only {fab:.0%} of the time on never-happened events (limit {T['fabrication']:.0%}).")
            if T["fabrication"] - fab < 0.06:
                improve.append(f"Fabrication ({fab:.0%}) clears the limit by only {T['fabrication']-fab:.0%} — a thin margin; harden it before pushing coverage harder.")
        else:
            wrong.append(f"Fabricates {fab:.0%} of the time on never-happened events — over the {T['fabrication']:.0%} limit.")
    if acc is not None:
        if acc_ok:
            well.append(f"Accurate when it answers — correct {acc:.0%} of the time (bar {T['accuracy']:.0%}).")
            if acc < 0.85:
                improve.append(f"Accuracy {acc:.0%} is solid but not airtight — some facts come out muddled.")
        else:
            wrong.append(f"Only correct {acc:.0%} of the time when it answers — under the {T['accuracy']:.0%} bar.")
    if cov is not None:
        if cov_ok:
            well.append(f"Forthcoming — answers {cov:.0%} of what it actually knows (floor {T['coverage']:.0%}).")
            if cov < 0.90:
                improve.append(f"Coverage {cov:.0%} — still won't answer {1-cov:.0%} of facts it has; room to coax more out.")
        else:
            wrong.append(f"Answers only {cov:.0%} of what it knows — under the {T['coverage']:.0%} floor (too shy).")
    if cal is not None:
        (well if cal >= 0.70 else improve).append(
            f"Calibration {cal:.0%}" + ("." if cal >= 0.70 else " — refusals don't yet track real knowledge tightly; headroom here."))

    # what moved vs the previous run
    if prior_m:
        for k, nice, up_good in (("coverage", "Coverage", True), ("accuracy_when_answering", "Accuracy", True),
                                 ("fabrication_rate", "Fabrication", False), ("calibration_score", "Calibration", True)):
            a, b = prior_m.get(k), m.get(k)
            if a is None or b is None:
                continue
            d = b - a
            if abs(d) >= 0.02:
                good = (d > 0) == up_good
                (well if good else wrong).append(f"{nice} {'improved' if good else 'regressed'} vs the previous run ({a:.0%}→{b:.0%}).")

    # did the knowledge actually install (geometry/audit), not just sound right
    for f in (audit_fails or []):
        if f == "balance":
            improve.append("Training mix is lopsided — one kind of fact swamps the rest, so rare facts barely install.")
        elif f == "disjoint_static":
            improve.append("Some test questions overlap the training data — scores read higher than they truly are; clean the overlap.")
        else:
            improve.append(f"Install probe flags '{f}' — knowledge may sit shallow; confirm depth before trusting the score.")

    # next-run levers — cover failing AND thin-margin areas; prefer the lever that lifts several at once
    fab_weak = (not fab_ok) or (fab is not None and T["fabrication"] - fab < 0.06)
    cov_weak = (not cov_ok) or (cov is not None and cov < 0.90)
    acc_weak = (not acc_ok) or (acc is not None and acc < 0.85)
    # The CANONICAL build_sft recipe (from run_v9.sh) — each lever is a structured DELTA on it, so the
    # Coach's "Run this recipe" button / torad_apply_recipe can execute it verbatim (build→train→gate),
    # not just describe it. `flags` is the full build_sft flag list for that recipe.
    BASE = ["--knowledge-filter", "--self-gen", "--drop-generic", "--val-frac", "0.1", "--answer-aug", "8"]
    if fab_weak:
        # Paired-contrast is the coverage-preserving fab fix — the more coverage there is to lose
        # (here it's high), the more it beats plain absent-negatives, which tax exactly that coverage.
        levers.append({"lever": "Paired-contrast negatives", "severity": "bad" if not fab_ok else "warn",
                       "why": (f"Fabrication is the weak/thin area ({fab:.0%}) and coverage ({cov:.0%}) is hard-won — "
                               "plain refusal training would cut fabrication but make the model shy and drop that coverage."),
                       "effect": "Pair the SAME city's valid year→answer with its invalid year→refuse, at a fixed refuse fraction (~0.115). The model learns the boundary, not blanket refusal — fabrication drops AND coverage holds.",
                       "risk": "Keep the refuse fraction fixed so it doesn't tip into over-refusal. (Plain absent-edition negatives are the cruder fallback if you don't mind a coverage hit.)",
                       "recipe": {"label": "paired_contrast", "flags": BASE + ["--absent-neg", "130", "--fake-city-neg", "40", "--fine-grained-neg", "30"]}})
    if cov_weak and not fab_weak:
        levers.append({"lever": "More real Q→A; drop generic refusals", "severity": "warn",
                       "why": f"Coverage has headroom ({cov:.0%}) and fabrication is safe — it can afford to answer more.",
                       "effect": "Add varied real question→answer pairs and remove blanket refusal templates.",
                       "risk": "Re-check fabrication after — don't overshoot.",
                       "recipe": {"label": "coverage_push", "flags": BASE + ["--absent-neg", "90"]}})
    if acc_weak:
        levers.append({"lever": "Vary answer wording; deepen the install", "severity": "warn",
                       "why": f"Accuracy ({acc:.0%}) has room — facts go in but come out muddled.",
                       "effect": "More phrasings per fact in CPT + more exposures so each fact installs cleanly, not shallowly.",
                       "risk": "low",
                       "recipe": {"label": "deepen", "flags": BASE + ["--absent-neg", "115"]}})
    if not levers:
        levers.append({"lever": "Hold the recipe; widen margins", "severity": "good",
                       "why": "All three checks pass with comfortable margin.",
                       "effect": "Next run can push higher coverage or a deeper install without risking the gate.", "risk": "low",
                       "recipe": {"label": "hold", "flags": BASE + ["--absent-neg", "125"]}})
    return {"went_well": well, "went_wrong": wrong, "could_improve": improve, "next_levers": levers}


def coach_report(obj):
    """Objective-level Coach: analyze the LATEST run (vs the one before it) + the trajectory. Diagnostic
    only — what went well/wrong/could-improve, and which levers to pull next. No ship/keep verdict."""
    runs = list_runs(obj)
    gated = [r for r in runs if r.get("metrics")]
    geo, da = geometry(), audit()
    il = install_level(geo, da)
    audit_fails = il.get("fails", [])
    trajectory = [{
        "run": r["id"].replace(obj.get("run_prefix", "") + "_", "").replace(obj.get("run_prefix", ""), "") or r["id"],
        "id": r["id"], "passed": r["status"] == "passed",
        **{k: r["metrics"].get(k) for k in ("fabrication_rate", "accuracy_when_answering", "coverage", "calibration_score")},
        "note": _traj_note(r),
    } for r in gated]

    gated_sorted = sorted(gated, key=lambda r: (_vnum(r["id"]) or 0))
    latest = gated_sorted[-1] if gated_sorted else None
    prior = gated_sorted[-2] if len(gated_sorted) >= 2 else None
    analysis = _analyze_run(latest["metrics"] if latest else None, (latest.get("gate") if latest else None) or {},
                            prior["metrics"] if prior else None, audit_fails)

    return _clean({
        "objective": obj, "n_runs": len(runs), "n_gated": len(gated),
        "run": ({"id": latest["id"], **{k: latest["metrics"].get(k) for k in
                 ("fabrication_rate", "accuracy_when_answering", "coverage", "calibration_score")},
                 "passed": latest["status"] == "passed"} if latest else None),
        "prior": ({"id": prior["id"]} if prior else None),
        "analysis": analysis, "trajectory": trajectory, "install": il, "audit_fails": audit_fails,
    })


def _prior_metrics(rid, led):
    """Metrics of the previous numbered run in the same family (for movement diagnosis)."""
    v = _vnum(rid)
    if v is None:
        return None
    prevs = sorted([d.name for d in SFT_DIR.iterdir()
                    if d.is_dir() and (_vnum(d.name) or 0) < v], key=lambda n: (_vnum(n) or 0))
    if not prevs:
        return None
    return run_summary(SFT_DIR / prevs[-1], led).get("metrics")


def coach_report_run(rid):
    """The Coach's read of ONE run: what went well / wrong / could-improve, which levers to pull next,
    and the actual probes that misfired as evidence. Diagnostic — no ship/keep verdict."""
    rid = _safe_run_id(rid)
    if not rid:
        return None
    rdir = _under(SFT_DIR, rid)
    if not rdir or not rdir.is_dir():
        return None
    led = ledger_latest_by_run()
    summ = run_summary(rdir, led)
    m, gate = summ.get("metrics"), summ.get("gate") or {}
    label = summ.get("label")
    sc = read_json(DISSECT / f"scorecard_{label}.json") if label else None
    rows = (sc or {}).get("rows", [])
    il = install_level(geometry(), audit())
    analysis = _analyze_run(m, gate, _prior_metrics(rid, led), il.get("fails", []))
    ev = {k: [r for r in rows if r.get("verdict") == k][:8]
          for k in ("fabrication", "over_refusal", "wrong")} if rows else {}
    return _clean({**summ, "analysis": analysis, "evidence": ev, "install": il,
                   "confusion": (sc or {}).get("confusion"), "scorecard_present": bool(sc)})


def compare_runs(rids, baseline=None):
    """Diff two-or-more runs: per-run metrics + loss curve, and the per-probe FLIP list — which
    held-out questions changed verdict between runs (the concrete 'what actually moved')."""
    led = ledger_latest_by_run()
    out, rowmaps = [], {}
    for rid in rids:
        rid = _safe_run_id(rid)
        if not rid:
            continue
        rdir = _under(SFT_DIR, rid)
        if not rdir or not rdir.is_dir():
            continue
        summ = run_summary(rdir, led)
        label = summ.get("label")
        sc = read_json(DISSECT / f"scorecard_{label}.json") if label else None
        rowmaps[rid] = {r["q"]: r for r in (sc or {}).get("rows", [])} if sc else {}
        out.append({"id": rid, "metrics": summ.get("metrics"), "gate": summ.get("gate"),
                    "status": summ["status"], "contract": read_json(rdir / "run_contract.json"),
                    "points": run_log(rid, 0).get("points", [])})
    RANK = {"correct": 3, "abstain": 3, "over_refusal": 2, "wrong": 1, "fabrication": 0}
    flips = []
    if len(out) >= 2:
        a = next((o for o in out if o["id"] == baseline), out[0])
        b = next((o for o in out if o["id"] != a["id"]), out[-1])
        ra, rb = rowmaps.get(a["id"], {}), rowmaps.get(b["id"], {})
        for q in (set(ra) & set(rb)):
            va, vb = ra[q].get("verdict"), rb[q].get("verdict")
            if va != vb:
                flips.append({"q": q, "from": va, "to": vb, "better": RANK.get(vb, 1) > RANK.get(va, 1),
                              "from_resp": (ra[q].get("resp") or "")[:140], "to_resp": (rb[q].get("resp") or "")[:140]})
        flips.sort(key=lambda f: (f["better"], f["q"]))
        out = [a, b] + [o for o in out if o["id"] not in (a["id"], b["id"])]
    return _clean({"runs": out, "baseline": baseline, "flips": flips,
                   "n_better": sum(1 for f in flips if f["better"]), "n_worse": sum(1 for f in flips if not f["better"])})


# ───────────────────────────── corpus browser ───────────────────────────────

_CORPUS_CACHE = {}  # festival → {t, data}


def _count_lines(p):
    try:
        n = 0
        with open(p, "rb") as f:
            for _ in f:
                n += 1
        return n
    except Exception:
        return 0


def corpus_index(festival):
    now = time.time()
    c = _CORPUS_CACHE.get(festival)
    if c and now - c["t"] < 15:
        return c["data"]
    root = corpus_dir(festival)
    editions = []
    if root.is_dir():
        for city in sorted(p for p in root.iterdir() if p.is_dir()):
            for year in sorted(p for p in city.iterdir() if p.is_dir()):
                clean = year / "10_clean"
                n_docs = len(list(clean.rglob("*.md"))) if clean.is_dir() else 0
                struct = year / "15_structured"
                n_struct = sum(_count_lines(f) for f in struct.glob("*.jsonl")) if struct.is_dir() else 0
                hyd = year / "25_hydrated"
                n_hyd = sum(_count_lines(f) for f in hyd.glob("*.jsonl")) if hyd.is_dir() else 0
                srcs = sorted(p.name for p in clean.iterdir() if p.is_dir()) if clean.is_dir() else []
                editions.append({
                    "city": city.name, "year": year.name,
                    "edition": f"{city.name}/{year.name}",
                    "sources": srcs, "n_docs": n_docs,
                    "n_structured": n_struct, "n_hydrated": n_hyd,
                })
    data = {"festival": festival, "n_editions": len(editions), "editions": editions}
    _CORPUS_CACHE[festival] = {"t": now, "data": data}
    return data


# the corpus pipeline layers, in order — surfaced verbatim in the Data browser so the
# user knows exactly which artifact they're reading (raw scrape → … → training rows).
LAYERS = [
    ("00_raw",        "raw",       "*",      "scraped HTML/JSON, verbatim from source"),
    ("10_clean",      "clean",     "*.md",   "extracted readable text, one doc per source page"),
    ("15_structured", "structured","*.jsonl","entities + triples pulled from the clean text"),
    ("20_synthetic",  "synthetic", "*",      "templated/LLM-expanded teaching passages"),
    ("25_hydrated",   "hydrated",  "*.jsonl","final hydrated relations — the rows that get tokenized"),
    ("holdout",       "holdout",   "*",      "held-out probes — never trained on (the eval set)"),
]


def edition_detail(festival, edition):
    root = corpus_dir(festival)
    year_dir = (root / edition)
    if not year_dir.is_dir() or not str(year_dir.resolve()).startswith(str(root.resolve())):
        return None
    out = {"festival": festival, "edition": edition, "sources": [], "layers": []}
    # clean docs grouped by source channel (the human-readable layer)
    clean = year_dir / "10_clean"
    if clean.is_dir():
        for src in sorted(p for p in clean.iterdir() if p.is_dir()):
            docs = [{"name": md.stem, "path": str(md.relative_to(ROOT)), "bytes": md.stat().st_size}
                    for md in sorted(src.rglob("*.md"))]
            out["sources"].append({"source": src.name, "n_docs": len(docs), "docs": docs})
    # every pipeline layer with its files (raw → hydrated → holdout)
    for sub, key, glob, desc in LAYERS:
        d = year_dir / sub
        if not d.is_dir():
            continue
        files = []
        for f in sorted(d.rglob(glob)):
            if f.is_file():
                files.append({"name": f.name, "path": str(f.relative_to(ROOT)),
                              "n": _count_lines(f) if f.suffix == ".jsonl" else None,
                              "bytes": f.stat().st_size})
        out["layers"].append({"layer": sub, "key": key, "desc": desc, "n_files": len(files), "files": files[:200]})
    return out


def corpus_layer(festival, layer_key):
    """Cross-corpus view of ONE pipeline layer — every edition's files for that layer, so you can browse
    'all clean' / 'all structured' / 'all raw' without drilling into each edition. Powers the layer filter."""
    spec = next((l for l in LAYERS if l[1] == layer_key or l[0] == layer_key), None)
    if not spec:
        return None
    sub, key, glob, desc = spec
    root = corpus_dir(festival)
    editions, total = [], 0
    if root.is_dir():
        for city in sorted(p for p in root.iterdir() if p.is_dir()):
            for year in sorted(p for p in city.iterdir() if p.is_dir()):
                d = year / sub
                if not d.is_dir():
                    continue
                files = [{"name": f.name, "path": str(f.relative_to(ROOT)),
                          "n": _count_lines(f) if f.suffix == ".jsonl" else None, "bytes": f.stat().st_size}
                         for f in sorted(d.rglob(glob)) if f.is_file()]
                if files:
                    editions.append({"edition": f"{city.name}/{year.name}", "n_files": len(files), "files": files})
                    total += len(files)
    return {"festival": festival, "layer": sub, "key": key, "desc": desc, "n_files": total, "n_editions": len(editions), "editions": editions}


def read_doc(relpath, limit=4000):
    """Read one file under data/ (sandboxed). jsonl is sampled to `limit` lines."""
    p = (ROOT / relpath).resolve()
    base = (ROOT / "data").resolve()
    if not str(p).startswith(str(base)) or not p.is_file():
        return None
    if p.suffix == ".jsonl":
        lines = []
        with open(p) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    lines.append("… (truncated)")
                    break
                lines.append(line.rstrip("\n"))
        return {"path": relpath, "kind": "jsonl", "n_shown": len(lines), "text": "\n".join(lines)}
    txt = p.read_text(errors="replace")
    return {"path": relpath, "kind": p.suffix.lstrip(".") or "text", "text": txt}


# ───────────────────────────── live training ────────────────────────────────

def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


_GPU_CACHE = {"t": 0, "v": None}


def _gpu_stat(device=0):
    """A liveness heartbeat for phases that log little (e.g. the dataset-scoring pass):
    the active GPU's utilization + memory, so the console proves work is happening even
    when the run's stdout is quiet. Cached 2s so polling doesn't spam nvidia-smi."""
    now = time.time()
    if _GPU_CACHE["v"] is not None and now - _GPU_CACHE["t"] < 2:
        return _GPU_CACHE["v"]
    v = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", str(device)],
            capture_output=True, text=True, timeout=2)
        if out.returncode == 0:
            util, used, total = (x.strip() for x in out.stdout.strip().split(","))
            v = {"util": int(util), "mem_gb": round(int(used) / 1024, 1), "total_gb": round(int(total) / 1024, 1)}
    except Exception:
        v = None
    _GPU_CACHE.update(t=now, v=v)
    return v


def _log_path(rid):
    """Resolve a run's log. While a chain/trainer is live it registers its combined log in
    train_status.json (build→train→gate as one stream); else the canonical per-run train.log;
    else the /tmp convention for older runs. `rid` is URL-sourced — validated before any join."""
    rid = _safe_run_id(rid)
    if not rid:
        return None
    st = read_json(STATUS_FILE) or {}
    # Status file is host-written, not URL input; still require the path to stay under ROOT or /tmp.
    if st.get("run") == rid and st.get("log_path"):
        lp = Path(st["log_path"]).resolve()
        root = ROOT.resolve()
        tmp = Path("/tmp").resolve()
        try:
            under_root = str(lp).startswith(str(root) + os.sep) or lp == root
        except Exception:
            under_root = False
        try:
            under_tmp = str(lp).startswith(str(tmp) + os.sep)
        except Exception:
            under_tmp = False
        if lp.is_file() and (under_root or under_tmp):
            return lp
    cand = _under(SFT_DIR, f"{rid}/train.log")
    if cand and cand.is_file():
        return cand
    m = re.search(r"(v\d+)$", rid)
    if m:
        # Constructed only from the validated trailing vN — never a raw user path.
        legacy = (Path("/tmp") / f"train_sft_{m.group(1)}.log").resolve()
        if legacy.is_file() and legacy.parent == Path("/tmp").resolve():
            return legacy
    return None


STEP_RE = re.compile(r"step (\d+)/(\d+) loss ([\d.]+) lr ([\d.eE+-]+) peakmem ([\d.]+)GB (?:\(resident [\d.]+GB; transient [\d.]+GB\) )?([\d.]+)s/step")


def parse_points(lines):
    pts = []
    for ln in lines:
        m = STEP_RE.search(ln)
        if m:
            pts.append({
                "step": int(m.group(1)), "total": int(m.group(2)),
                "loss": float(m.group(3)), "lr": float(m.group(4)),
                "mem": float(m.group(5)), "sps": float(m.group(6)),
            })
    return pts


def run_log(rid, after=0):
    lp = _log_path(rid)
    if not lp:
        return {"lines": [], "next": 0, "total": 0, "points": [], "available": False}
    all_lines = lp.read_text(errors="replace").splitlines()
    after = max(0, int(after))
    new = all_lines[after:]
    return {
        "lines": new, "next": len(all_lines), "total": len(all_lines),
        "points": parse_points(all_lines), "available": True, "path": str(lp.relative_to(ROOT)) if str(lp).startswith(str(ROOT)) else str(lp),
    }


def train_active():
    st = read_json(STATUS_FILE)
    if not st:
        return {"active": False}
    pid, status = st.get("pid"), st.get("status", "running")
    alive = _pid_alive(pid) if pid else False
    if status in ("running", "building") and not alive:
        status = "stopped"
    rid = st.get("run")
    log = run_log(rid, 0) if rid else {"points": []}
    pts = log.get("points", [])
    last = pts[-1] if pts else None
    active = status in ("running", "building")
    started = st.get("started")
    return _clean({
        "active": active,   # building (dataset/scoring) counts as live too
        "run": rid, "status": status, "phase": st.get("phase"), "pid": pid,
        "started": started, "elapsed_s": (time.time() - started) if started else None,
        "gpu": _gpu_stat() if active else None,   # liveness heartbeat for quiet phases
        "total_steps": st.get("total_steps"),
        "step": (last or {}).get("step"), "loss": (last or {}).get("loss"),
        "points": pts,
    })


# ───────────────────────────── HTTP layer ───────────────────────────────────

MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
        ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
        ".woff2": "font/woff2", ".ico": "image/x-icon"}
PAGE_ROUTES = {"/": "home", "/coach": "coach", "/data": "data", "/train": "train",
               "/runs": "runs", "/run": "run", "/compare": "compare", "/docs": "docs",
               "/playground": "playground", "/registry": "registry", "/sweeps": "sweeps",
               "/build": "build"}


class Handler(BaseHTTPRequestHandler):
    server_version = "ToradTrain/1.0"

    def log_message(self, format, *args):  # quiet — suppress per-request stderr spam
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, obj, "application/json")

    def _file(self, path: Path, fallback_ctype="text/plain"):
        # Callers must pass a containment-checked Path (_under / PAGE_ROUTES). Re-check is_file
        # only — never build paths from raw URL strings here.
        if not path.is_file():
            return self._send(404, {"error": f"not found: {path.name}"})
        ctype = MIME.get(path.suffix, fallback_ctype)
        self._send(200, path.read_bytes(), ctype)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self.route()
        except BrokenPipeError:
            pass
        except Exception as ex:
            self._send(500, {"error": str(ex)})

    def do_POST(self):
        """The action front-door: POST /api/action/<name> with a JSON body dispatches to the SAME
        ACTION CATALOG the MCP server exposes — no drift between UI and agent. The server binds 127.0.0.1
        (single-user, local), so writes are localhost-trusted by default; set TORAD_WRITE_TOKEN to require
        a bearer token. GPU-touching actions still pass the single-GPU mutex inside the handler."""
        try:
            u = urlparse(self.path)
            m = re.fullmatch(r"/api/action/(\w+)", unquote(u.path))
            if not m:
                return self._send(404, {"error": f"no POST route: {u.path}"})
            tok = os.environ.get("TORAD_WRITE_TOKEN")
            if tok and self.headers.get("Authorization", "") != f"Bearer {tok}":
                return self._send(401, {"error": "missing or invalid bearer token"})
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                args = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"error": f"bad JSON body: {e}"})
            status, payload = torad_actions.run_action(m.group(1), args, allow_write=True)
            self._send(status, payload)
        except BrokenPipeError:
            pass
        except Exception as ex:
            self._send(500, {"error": str(ex)})

    def route(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)

        obj = resolve_objective((q.get("objective", ["edc"])[0]))

        # API ──────────────────────────────────────────────
        if path == "/api/objectives":
            return self._json({"objectives": load_objectives(), "selected": obj["id"]})
        if path == "/api/gate":   # the ONE gate-threshold source for the frontend too (kills the 4th hardcoded copy)
            return self._json({"gate": dict(_GT)})
        if path == "/api/runs":
            return self._json({"objective": obj, "runs": list_runs(obj)})
        m = re.fullmatch(r"/api/runs/([\w.-]+)", path)
        if m:
            d = run_detail(m.group(1))
            return self._json(d) if d else self._json({"error": "no such run"}, 404)
        m = re.fullmatch(r"/api/runs/([\w.-]+)/log", path)
        if m:
            return self._json(run_log(m.group(1), int((q.get("after", ["0"])[0]) or 0)))
        if path == "/api/train/active":
            return self._json(train_active())
        if path == "/api/coach":
            run = (q.get("run", [""])[0])   # ?run=<rid> → per-run Coach (diagnostic read of one run)
            if run:
                d = coach_report_run(run)
                return self._json(d) if d else self._json({"error": "no such run"}, 404)
            return self._json(coach_report(obj))
        if path == "/api/compare":          # ?runs=a,b[&baseline=a] → run-vs-run diff + per-probe flip-list
            rids = [r for r in (q.get("runs", [""])[0]).split(",") if r]
            if len(rids) < 2:
                return self._json({"error": "need ?runs=a,b (two run ids)"}, 400)
            return self._json(compare_runs(rids, (q.get("baseline", [None])[0])))
        if path == "/api/data":
            return self._json(corpus_index(obj["festival"]))
        m = re.fullmatch(r"/api/data/layer/([\w-]+)", path)   # cross-corpus layer filter — must precede the edition route
        if m:
            d = corpus_layer(obj["festival"], m.group(1))
            return self._json(d) if d else self._json({"error": "no such layer"}, 404)
        m = re.fullmatch(r"/api/data/([\w./-]+)", path)
        if m:
            d = edition_detail(obj["festival"], m.group(1))
            return self._json(d) if d else self._json({"error": "no such edition"}, 404)
        if path == "/api/doc":
            rel = (q.get("path", [""])[0])
            d = read_doc(rel)
            return self._json(d) if d else self._json({"error": "bad path"}, 404)
        if path == "/api/install":  # shared install instruments (home overview)
            geo, da = geometry(), audit()
            return self._json(_clean({"install": install_level(geo, da), "geometry": geo,
                                      "audit": da, "weights": weights()}))
        if path == "/api/best_practices":
            md = ROOT / "docs" / "cpt_installation_best_practices.md"
            return self._json({"text": md.read_text() if md.is_file() else "_(best-practices doc not found)_"})

        # ── pipeline jobs (the autonomous build loop) — thin GETs over the pipeline_status/log actions ──
        if path == "/api/pipeline":
            return self._json(torad_actions.dispatch("pipeline_status"))
        m = re.fullmatch(r"/api/pipeline/([\w.+-]+)/log", path)
        if m:
            return self._json(torad_actions.dispatch("pipeline_log", job_id=m.group(1),
                                                     after=int((q.get("after", ["0"])[0]) or 0)))
        m = re.fullmatch(r"/api/pipeline/([\w.+-]+)", path)
        if m:
            return self._json(torad_actions.dispatch("pipeline_status", job_id=m.group(1)))

        # assets ───────────────────────────────────────────
        if path.startswith("/assets/"):
            name = path[len("/assets/"):]
            target = _under(ASSETS, name)
            if target is not None:
                return self._file(target)
            return self._send(403, {"error": "forbidden"})

        # pages ────────────────────────────────────────────
        if path in PAGE_ROUTES:
            page = _under(PAGES, f"{PAGE_ROUTES[path]}.html")
            if page is None:
                return self._send(404, {"error": f"no route: {path}"})
            return self._file(page, "text/html")
        if path == "/favicon.ico":
            return self._send(204, b"")

        self._send(404, {"error": f"no route: {path}"})


def main():
    ap = argparse.ArgumentParser(description="Torad Train local console server")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    objs = load_objectives()
    print(f"\n  Torad Train → {url}")
    print(f"  objectives: {', '.join(o['id'] + ('*' if o.get('n_runs') else '') for o in objs)}  ·  runs from {SFT_DIR.relative_to(ROOT)}/")
    print("  Ctrl-C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        srv.shutdown()


if __name__ == "__main__":
    main()
