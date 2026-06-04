#!/usr/bin/env python3
"""Torad Train — the ACTION CATALOG: one typed handler per action, exposed through two thin adapters
(the MCP server `torad_mcp.py` for agents, and `serve.py`'s POST routes for the Lab-Light UI). One
source of truth — no drift between "what the UI can do" and "what an agent can do".

THE SAFETY LAW lives here in code, not in a prompt: this is a single-GPU workstation (1200W PSU OCP
forbids two concurrent FP4 trainers). Every action flagged `touches_gpu=True` MUST `acquire_gpu()` as its
first side-effect — an atomic O_EXCL lock at data/_build/gpu.lock. If the GPU is busy the action REFUSES
with a self-correctable sentence (so an agent can wait or stop the holder). An agent physically cannot
launch a second FP4 trainer no matter what it is told.

Long actions (launch/gate) return a run_id — the durable task handle the run registry already mints.
Fire → poll run_status → fetch the deterministic gate as the terminal payload. Identical from UI and agent.

Reads lazy-import serve.py (which owns the artifact-reading builders) to avoid an import cycle.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpt_config import CptConfig  # noqa: E402  (config validation for launches)

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "data" / "_build"
GPU_LOCK = BUILD / "gpu.lock"
STATUS_FILE = BUILD / "train_status.json"
PIPELINE_STATUS = BUILD / "pipeline_status.json"   # multi-job dict {job_id: {...}} — the CPU/LLM pipeline
PIPELINE_DIR = BUILD / "pipeline"                  # job logs; NEVER the GPU lock (pipeline coexists w/ training)
SFT_DIR = ROOT / "checkpoints" / "sft"
CPT_DIR = ROOT / "checkpoints" / "cpt"
PY = str(ROOT / ".venv" / "bin" / "python")
GPU_DEVICE = 0                      # CUDA_VISIBLE_DEVICES=0 — the only trainable GPU (the 5080)
LAUNCH_GRACE_S = 90                 # a freshly-acquired lock with no live pid yet is "launching", not stale


# ───────────────────────────── tiny io ──────────────────────────────────────
def _read_json(path: Path, default=None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    # The pid exists — but a ZOMBIE (a spawned pipeline job that finished and is awaiting reap by its
    # parent serve.py) is effectively dead. os.kill(pid,0) succeeds on a zombie, so without this check a
    # finished job stays "running" forever and any poller (the qgre-agent) hangs. Read /proc state; 'Z' = done.
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            state = f.read().rpartition(")")[2].split()[0]   # field after comm; comm may contain spaces/parens
        if state == "Z":
            try:
                os.waitpid(int(pid), os.WNOHANG)              # reap if we're the parent (best-effort)
            except (ChildProcessError, OSError):
                pass
            return False
    except (FileNotFoundError, IndexError, OSError):
        pass
    return True


# ───────────────────────────── GPU mutex ────────────────────────────────────
class GpuBusy(Exception):
    """Raised when a touches_gpu action cannot acquire the single-GPU lock. The message is written to be
    self-correctable: an agent reading it knows to wait or to call torad_stop_run."""


def _foreign_gpu0_pids(exclude: set[int] | None = None) -> list[int]:
    """Compute processes on GPU0 that are NOT our lock holder — a foreign trainer/inference job we must
    not collide with. Empty list if nvidia-smi is unavailable (then the lock file is the only guard)."""
    exclude = exclude or set()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", "-i", str(GPU_DEVICE)],
            capture_output=True, text=True, timeout=4)
        if out.returncode != 0:
            return []
        pids = []
        for ln in out.stdout.strip().splitlines():
            ln = ln.strip()
            if ln and ln.isdigit() and int(ln) not in exclude:
                pids.append(int(ln))
        return pids
    except Exception:
        return []


def _reap_stale_lock() -> None:
    """Remove the lock if its holder is dead (crashed run) or a launch never came up within the grace
    window — so a crash never wedges the workstation forever."""
    lock = _read_json(GPU_LOCK)
    if not lock:
        return
    pid = lock.get("pid")
    age = time.time() - (lock.get("started") or 0)
    if pid:
        if not _pid_alive(pid):
            GPU_LOCK.unlink(missing_ok=True)
    elif age > LAUNCH_GRACE_S:   # acquired but no pid ever attached → launch failed
        GPU_LOCK.unlink(missing_ok=True)


def gpu_status() -> dict:
    """Is GPU0 free? If not, who holds it and since when — the single-GPU analog of fleet instance-type
    recommendation. Always safe to call (read-only)."""
    _reap_stale_lock()
    lock = _read_json(GPU_LOCK)
    holder_pid = (lock or {}).get("pid")
    foreign = _foreign_gpu0_pids(exclude={holder_pid} if holder_pid else set())
    busy = bool(lock) or bool(foreign)
    gpu = None
    try:
        import serve  # lazy: reuse the cached nvidia-smi heartbeat
        gpu = serve._gpu_stat(GPU_DEVICE)
    except Exception:
        pass
    return {"free": not busy, "holder": lock, "foreign_pids": foreign, "gpu": gpu}


def acquire_gpu(run_id: str) -> None:
    """Atomically reserve GPU0 for run_id, or raise GpuBusy with an actionable sentence. O_EXCL makes the
    create-or-fail atomic; the stale-reaper runs first so crashed runs don't block. pid is attached after
    the process spawns (set_lock_pid)."""
    _reap_stale_lock()
    foreign = _foreign_gpu0_pids()
    if foreign:
        raise GpuBusy(
            f"GPU0 is busy with foreign compute process(es) {foreign} — not a Torad run. "
            f"Free the GPU (or stop that process) and retry; Torad will not risk a second concurrent job (1200W PSU OCP).")
    BUILD.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(GPU_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        h = _read_json(GPU_LOCK) or {}
        raise GpuBusy(
            f"Mutex held: run '{h.get('run_id', '?')}' on GPU0 (pid {h.get('pid', 'launching')}, "
            f"step {h.get('step', '?')}) since {time.strftime('%H:%M', time.localtime(h.get('started', time.time())))} — "
            f"call torad_stop_run('{h.get('run_id', '?')}') to free it, or wait for it to finish.")
    with os.fdopen(fd, "w") as f:
        json.dump({"run_id": run_id, "pid": None, "step": 0, "started": time.time(), "status": "launching"}, f)


def set_lock_pid(pid: int, pgid: int | None = None) -> None:
    lock = _read_json(GPU_LOCK) or {}
    lock.update(pid=pid, pgid=pgid, status="running")
    _write_json_atomic(GPU_LOCK, lock)


def release_gpu() -> None:
    GPU_LOCK.unlink(missing_ok=True)


# ───────────────────────────── catalog registry ─────────────────────────────
@dataclass
class Action:
    name: str
    fn: Callable[..., Any]
    mutating: bool = False
    touches_gpu: bool = False
    category: str = "read"          # read | train | data | eval | export
    doc: str = ""
    inputs: dict = field(default_factory=dict)   # name → "type: description"


ACTIONS: dict[str, Action] = {}


def action(name: str, *, mutating=False, touches_gpu=False, category="read", inputs=None):
    def deco(fn):
        ACTIONS[name] = Action(name=name, fn=fn, mutating=mutating, touches_gpu=touches_gpu,
                               category=category, doc=(fn.__doc__ or "").strip(), inputs=inputs or {})
        return fn
    return deco


def catalog(include_writes=True, categories=None) -> list[dict]:
    """Introspect the catalog (for MCP auto-registration + the UI). categories narrows by category set."""
    out = []
    for a in ACTIONS.values():
        if not include_writes and a.mutating:
            continue
        if categories and a.category not in categories:
            continue
        out.append({"name": a.name, "mutating": a.mutating, "touches_gpu": a.touches_gpu,
                    "category": a.category, "doc": a.doc, "inputs": a.inputs})
    return out


def dispatch(name: str, **kwargs):
    """Invoke an action by name. The single entrypoint both adapters call."""
    a = ACTIONS.get(name)
    if not a:
        raise KeyError(f"no such action: {name} (have: {', '.join(sorted(ACTIONS))})")
    return a.fn(**kwargs)


# ───────────────────────────── read actions ─────────────────────────────────
# These wrap serve.py's artifact-reading builders (lazy import → no cycle). Always available.
def _serve():
    import serve
    return serve


@action("gpu_status", category="read")
def a_gpu_status():
    """Is GPU0 free? If not, which run holds it, at what step, since when."""
    return gpu_status()


# ── navigation as a tool — every UI page/link/click is reachable by an agent. Returns the canonical URL
#    (what a human would click) AND the data that page renders (so the agent doesn't need a browser). This
#    is the "everything, including navigation, is a tool" surface: the agent drives the SaaS, not just mutates it.
_PAGES = {"home": "/", "data": "/data", "train": "/train", "runs": "/runs", "run": "/run",
          "coach": "/coach", "compare": "/compare", "playground": "/playground",
          "registry": "/registry", "sweeps": "/sweeps", "docs": "/docs"}


@action("navigate", category="read",
        inputs={"page": "str: home|data|train|runs|run|coach|compare|playground|registry|sweeps|docs",
                "objective": "str: objective scope (default edc)", "run_id": "str: for page=run/coach",
                "runs": "list[str]: for page=compare", "festival": "str: for page=data"})
def a_navigate(page, objective="edc", run_id=None, runs=None, festival=None, baseline=None):
    """Resolve a UI page (a 'click') to its canonical URL + the data it renders — the agent-facing
    equivalent of navigating/clicking a hyperlink. Read-only; returns {url, page, payload}."""
    page = (page or "home").lower().lstrip("/")
    if page not in _PAGES:
        return {"error": f"no page '{page}'", "pages": sorted(_PAGES)}
    base = _PAGES[page]
    qs, payload = [], None
    s = _serve()
    if page == "run":
        if not run_id:
            return {"error": "page=run needs run_id"}
        qs.append(f"id={run_id}"); payload = s.run_detail(run_id)
    elif page == "coach":
        if run_id:
            qs.append(f"run={run_id}"); payload = s.coach_report_run(run_id)
        else:
            payload = s.coach_report(s.resolve_objective(objective))
    elif page == "compare":
        rids = list(runs or [])
        if len(rids) >= 2:
            qs.append("runs=" + ",".join(rids))
            if baseline:
                qs.append(f"baseline={baseline}")
            payload = s.compare_runs(rids, baseline)
    elif page == "runs":
        payload = {"runs": s.list_runs(s.resolve_objective(objective))}
    elif page == "data":
        payload = s.corpus_index(festival or s.resolve_objective(objective).get("festival", "edc"))
    elif page == "registry":
        payload = a_list_exports()
    elif page == "sweeps":
        payload = a_list_studies()
    elif page == "train":
        payload = a_run_status()
    if objective and objective != "edc":
        qs.append(f"objective={objective}")
    url = base + ("?" + "&".join(qs) if qs else "")
    return s._clean({"url": url, "page": page, "payload": payload})


@action("ui_map", category="read")
def a_ui_map():
    """The site map: every page, its route, and the read-action that backs it — the agent's index of where
    it can navigate and what each destination shows."""
    backs = {"home": "coach", "data": "inspect_corpus", "train": "run_status", "runs": "list_runs",
             "run": "get_run", "coach": "diagnose_run", "compare": "compare_runs",
             "playground": "playground_stored/playground_live", "registry": "list_exports",
             "sweeps": "list_studies", "docs": "(static)"}
    return {"pages": [{"page": p, "url": u, "backed_by": backs.get(p)} for p, u in _PAGES.items()]}


@action("list_runs", category="read", inputs={"objective": "str: objective id (default 'edc')"})
def a_list_runs(objective="edc"):
    """All runs for an objective with status + gate metrics, newest concept of the campaign."""
    s = _serve()
    return {"objective": objective, "runs": s.list_runs(s.resolve_objective(objective))}


@action("get_run", category="read", inputs={"run_id": "str: run id e.g. edc_sft_v11"})
def a_get_run(run_id):
    """Full run detail: contract, metrics, gate, install verdict, geometry, weights."""
    return _serve().run_detail(run_id) or {"error": f"no such run: {run_id}"}


@action("read_gate", category="read", inputs={"run_id": "str"})
def a_read_gate(run_id):
    """The deterministic gate triple (fabrication/accuracy/coverage) + PASS/FAIL for a run."""
    d = _serve().run_detail(run_id) or {}
    return {"run_id": run_id, "metrics": d.get("metrics"), "gate": d.get("gate"), "status": d.get("status")}


@action("compare_runs", category="read", inputs={"runs": "list[str]: 2+ run ids", "baseline": "str: optional baseline run id"})
def a_compare_runs(runs, baseline=None):
    """Run-vs-run diff: per-run metrics + the per-probe verdict flip-list (what actually moved)."""
    if not runs or len(runs) < 2:
        return {"error": "need at least two run ids"}
    return _serve().compare_runs(list(runs), baseline)


@action("diagnose_run", category="read", inputs={"run_id": "str"})
def a_diagnose_run(run_id):
    """The Coach's plain-language read of one run: what went well/wrong/improve, ranked next levers, and
    the misfired probes as evidence. Diagnostic — no ship/keep verdict."""
    return _serve().coach_report_run(run_id) or {"error": f"no such run: {run_id}"}


@action("gate_curve", category="read", inputs={"run_id": "str"})
def a_gate_curve(run_id):
    """The gate triple (fabrication/accuracy/coverage + calibration) per SAVED checkpoint vs training step —
    the data for picking the pre-overfit checkpoint to ship. Built from the calibration ledger (every gated
    checkpoint of this run). Recommends the checkpoint where calibration peaks while the gate still PASSES."""
    import re as _re
    led = ROOT / "data" / "_build" / "dissect" / "calibration_results.jsonl"
    seen: dict = {}
    if led.is_file():
        for line in led.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            ck = e.get("checkpoint", "") or ""
            base = ck.rstrip("/").split("/step_")[0].rstrip("/")
            if not (base.endswith("/" + run_id) or base.endswith(run_id)):
                continue
            m = _re.search(r"step_0*(\d+)", ck)
            step = int(m.group(1)) if m else None
            seen[step if step is not None else 10 ** 9] = {
                "step": step, "final": step is None, "passed": e.get("passed"),
                "fabrication": e.get("fabrication_rate"), "accuracy": e.get("accuracy_when_answering"),
                "coverage": e.get("coverage"), "calibration": e.get("calibration_score")}
    pts = [seen[k] for k in sorted(seen)]
    pool = [p for p in pts if p.get("passed")] or pts
    rec = max(pool, key=lambda p: (p.get("calibration") or 0)) if pool else None
    return {"run_id": run_id, "points": pts, "recommended": rec,
            "note": "ship the checkpoint where calibration peaks before overfit (gate still PASS)"}


@action("inspect_corpus", category="read", inputs={"festival": "str: festival slug (default 'edc')"})
def a_inspect_corpus(festival="edc"):
    """The corpus index: editions × sources × layers with line counts."""
    return _serve().corpus_index(festival)


@action("list_objectives", category="read")
def a_list_objectives():
    """The objective registry: every festival domain (its corpus + run family), annotated with disk reality —
    the agent's equivalent of the console's objective switcher."""
    return {"objectives": _serve().load_objectives()}


@action("read_edition", category="read", inputs={"edition": "str: e.g. las_vegas/2026", "festival": "str (default edc)"})
def a_read_edition(edition, festival="edc"):
    """One edition's sources × layers with file lists — what the Data tree shows when you open an edition."""
    return _serve().edition_detail(festival, edition) or {"error": f"no edition {festival}/{edition}"}


@action("read_layer", category="read", inputs={"layer": "str: e.g. 25_hydrated", "festival": "str (default edc)"})
def a_read_layer(layer, festival="edc"):
    """One pipeline layer across every edition (the Data layer-filter view) — files + counts per edition."""
    return _serve().corpus_layer(festival, layer) or {"error": f"no layer {layer}"}


@action("read_doc", category="read", inputs={"path": "str: corpus-relative file path", "limit": "int: max lines"})
def a_read_doc(path, limit=4000):
    """Read one corpus file (the reader pane) — markdown / jsonl / text, capped at `limit` lines."""
    return _serve().read_doc(path, limit) or {"error": f"bad path: {path}"}


@action("run_log", category="read", inputs={"run_id": "str", "after": "int: line offset to read from"})
def a_run_log(run_id, after=0):
    """Tail a run's combined build→train→gate log from line `after` (+ parsed loss points) — the live console
    stream, agent-readable so an agent can follow a run it launched."""
    return _serve().run_log(run_id, int(after or 0))


# ───────────────────────────── launch / stop / status / gate ────────────────
def _save_dir_of(config_path: str) -> Path:
    cfg = CptConfig.from_yaml(config_path, None)
    return (ROOT / cfg.run.save_dir).resolve()


def _spawn(cmd: list[str], log_path: Path, extra_env: dict | None = None) -> subprocess.Popen:
    """Spawn a detached background process in its own session (so stop can kill the whole group), tee'ing
    stdout+stderr into log_path. Survives the request that started it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "scripts", **(extra_env or {})}
    logf = open(log_path, "a", buffering=1)
    logf.write(f"\n===== torad launch {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)} =====\n")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)


@action("launch_run", mutating=True, touches_gpu=True, category="train",
        inputs={"mode": "str: 'cpt' | 'sft'", "config": "str: config path (default canonical for the mode)",
                "label": "str: sft run label e.g. v12 (sft only)"})
def a_launch_run(mode, config=None, label=None):
    """Acquire the single-GPU lock, then spawn the ONE training entrypoint via the canonical FP4 launcher
    (train_fp4.sh / run_sft.sh) and return run_id immediately as the durable handle. SFT always trains→gates
    atomically (run_sft.sh). Refuses if GPU0 is not free."""
    mode = (mode or "").lower()
    if mode not in ("cpt", "sft"):
        return {"error": "mode must be 'cpt' or 'sft'"}
    config = config or ("configs/sft_edc.yaml" if mode == "sft" else "configs/train_edc_lv_2026.yaml")
    if not (ROOT / config).is_file():
        return {"error": f"config not found: {config}"}
    try:
        save_dir = _save_dir_of(config)
    except Exception as e:
        return {"error": f"invalid config {config}: {e}"}
    run_id = save_dir.name
    log_path = save_dir / "train.log"

    acquire_gpu(run_id)   # raises GpuBusy → caller surfaces the actionable sentence
    try:
        if mode == "sft":
            label = label or run_id.split("_")[-1]
            cmd = ["bash", "scripts/run_sft.sh", label]
            proc = _spawn(cmd, log_path, extra_env={"CONFIG": config})
        else:
            cmd = ["bash", "scripts/train_fp4.sh", "--config", config]
            proc = _spawn(cmd, log_path)
        set_lock_pid(proc.pid, os.getpgid(proc.pid))
    except Exception as e:
        release_gpu()
        return {"error": f"spawn failed: {e}"}
    return {"run_id": run_id, "mode": mode, "config": config, "pid": proc.pid,
            "log": str(log_path.relative_to(ROOT)), "status": "launching",
            "message": f"launched {mode} run '{run_id}' (pid {proc.pid}). Poll torad_run_status('{run_id}')."}


@action("apply_recipe", mutating=True, touches_gpu=True, category="train",
        inputs={"label": "str: short run label (→ edc_sft_<label>)", "flags": "list[str]: build_sft recipe flags",
                "init_from": "str: CPT checkpoint to SFT from (default the canonical CPT)"})
def a_apply_recipe(label, flags, init_from=None):
    """Run a Coach RECIPE end-to-end as a gated campaign: build_sft(flags) → train (FP4·int8-master) → gate.
    This is the 'Run this recipe' action — the Coach's structured lever becomes an executed run. Acquires the
    single-GPU mutex (build is CPU but training holds the GPU); returns run_id as the durable handle."""
    if not label or not isinstance(flags, (list, tuple)):
        return {"error": "need label (str) + flags (list[str])"}
    label = re.sub(r"[^A-Za-z0-9_]", "_", str(label))[:40]
    init_from = init_from or "checkpoints/cpt/edc_all97_v1"
    if not (ROOT / init_from).exists():
        return {"error": f"init_from checkpoint not found: {init_from}"}
    run_id = f"edc_sft_{label}"
    save_dir = SFT_DIR / run_id
    acquire_gpu(run_id)
    try:
        cmd = ["bash", "scripts/run_recipe.sh", label, init_from, *[str(x) for x in flags]]
        proc = _spawn(cmd, save_dir / "train.log")
        set_lock_pid(proc.pid, os.getpgid(proc.pid))
    except Exception as e:
        release_gpu()
        return {"error": f"spawn failed: {e}"}
    return {"run_id": run_id, "pid": proc.pid, "init_from": init_from, "flags": list(flags),
            "log": str((save_dir / "train.log").relative_to(ROOT)), "status": "launching",
            "message": f"recipe launched as '{run_id}' (build→train→gate). Poll torad_run_status('{run_id}')."}


@action("stop_run", mutating=True, touches_gpu=False, category="train", inputs={"run_id": "str"})
def a_stop_run(run_id):
    """Stop the active run: SIGTERM its process group (graceful), then release the GPU lock. Idempotent."""
    lock = _read_json(GPU_LOCK) or {}
    st = _read_json(STATUS_FILE) or {}
    pgid = lock.get("pgid")
    pid = lock.get("pid") or st.get("pid")
    killed = False
    if pgid:
        try:
            os.killpg(pgid, signal.SIGTERM); killed = True
        except ProcessLookupError:
            pass
    elif pid and _pid_alive(pid):
        try:
            os.kill(int(pid), signal.SIGTERM); killed = True
        except ProcessLookupError:
            pass
    if st.get("run") == run_id:
        st["status"] = "stopped"
        _write_json_atomic(STATUS_FILE, st)
    release_gpu()
    return {"run_id": run_id, "stopped": killed, "message": f"sent SIGTERM to run '{run_id}' and released GPU0." if killed
            else f"run '{run_id}' had no live process; GPU0 lock released."}


@action("run_status", category="read", inputs={"run_id": "str: optional; omit for the active run"})
def a_run_status(run_id=None):
    """Live status of the active (or named) run: phase, step, loss, gpu heartbeat, + gate if scored."""
    s = _serve()
    active = s.train_active()
    out = {"active": active}
    rid = run_id or active.get("run")
    if rid:
        d = s.run_detail(rid) or {}
        out["gate"] = {"metrics": d.get("metrics"), "gate": d.get("gate"), "status": d.get("status")}
    out["gpu"] = gpu_status()
    return out


@action("gate_run", mutating=True, touches_gpu=True, category="eval",
        inputs={"run_id": "str", "config": "str: config (default canonical sft)", "mode": "str: cpt|sft"})
def a_gate_run(run_id, config="configs/sft_edc.yaml", mode="sft"):
    """Run the deterministic GATE on a run's final checkpoint (fab≤τ ∧ acc≥τ ∧ cov≥τ) → PASS/FAIL + the
    three numbers. Touches the GPU (must generate), so it acquires the mutex. Synchronous."""
    ckpt = SFT_DIR / run_id if mode == "sft" else CPT_DIR / run_id
    if not ckpt.exists():
        return {"error": f"no checkpoint dir for run {run_id} at {ckpt}"}
    acquire_gpu(run_id)
    try:
        set_lock_pid(os.getpid())
        rc = subprocess.run(
            [PY, "scripts/eval.py", "--config", config, "--mode", mode,
             "--checkpoints", str(ckpt), "--labels", f"{run_id}_final"],
            cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "scripts", "CUDA_VISIBLE_DEVICES": str(GPU_DEVICE)}).returncode
    finally:
        release_gpu()
    d = _serve().run_detail(run_id) or {}
    return {"run_id": run_id, "rc": rc, "metrics": d.get("metrics"), "gate": d.get("gate"), "status": d.get("status")}


# ───────────────────────────── data / pipeline compute ──────────────────────
@action("gen_config", mutating=True, category="data",
        inputs={"base": "str: base config path", "out": "str: output path",
                "sets": "list[str]: 'section.key=value' overrides", "mode": "str: optional"})
def a_gen_config(base, out, sets=None, mode=None):
    """Generate a validated config from a base + dotted overrides (the config-generator primitive — sweeps
    and the optimizer emit these; never trainer flags). Validates via the contract before writing."""
    argv = [PY, "scripts/gen_config.py", "--base", base, "--out", out]
    if mode:
        argv += ["--mode", mode]
    if sets:
        argv += ["--set", *sets]   # gen_config's --set is nargs="*": ALL overrides under ONE flag (a
        #                            separate --set per value would let argparse keep only the last)
    r = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    return {"rc": r.returncode, "out": out if r.returncode == 0 else None,
            "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}


@action("build_corpus", mutating=True, category="data",
        inputs={"festival": "str", "edition": "str: e.g. las_vegas/2026"})
def a_build_corpus(festival, edition):
    """Assemble the relational corpus for one edition via festival-factory (base+disjoint+set-times).
    CPU/network only — does NOT touch the GPU."""
    rust = ROOT / "festival-factory" / "target" / "debug" / "festival-factory"
    if not rust.is_file():
        return {"error": "festival-factory binary not built (cargo build in festival-factory/)"}
    r = subprocess.run([str(rust), "build-corpus", "--root", str(ROOT), "--festival", festival, "--edition", edition],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    return {"rc": r.returncode, "lines": len(r.stdout.splitlines()), "stderr": r.stderr.strip()[-2000:]}


@action("trace_fact", category="data",
        inputs={"festival": "str", "edition": "str: e.g. las_vegas/2026", "query": "str: an entity/fact phrase to trace"})
def a_trace_fact(festival, edition, query):
    """Trace one fact/entity DOWN the pipeline layers for an edition: which clean sources mention it →
    which structured triples encode it → how many hydrated surface forms teach it (the install exposure)
    + a distinct sample. The lineage that answers 'why did the model learn (or not learn) this fact'."""
    base = ROOT / "data" / festival / edition
    if not base.is_dir():
        return {"error": f"no edition {festival}/{edition}"}
    q = (query or "").strip().lower()
    if not q:
        return {"error": "empty query"}
    trace = []
    for layer in ("10_clean", "15_structured", "20_synthetic", "25_hydrated"):
        ld = base / layer
        if not ld.is_dir():
            continue
        hits, samples, nlines = 0, [], 0
        for f in sorted(ld.rglob("*")):
            if f.suffix in (".jsonl", ".md", ".json") and f.is_file():
                for ln in f.read_text(errors="replace").splitlines():
                    nlines += 1
                    if q in ln.lower():
                        hits += 1
                        if len(samples) < 4:
                            samples.append({"file": f.name, "text": ln.strip()[:200]})
        trace.append({"layer": layer, "hits": hits, "samples": samples, "scanned_lines": nlines})
    hydrated = next((t for t in trace if t["layer"] == "25_hydrated"), None)
    exposure = hydrated["hits"] if hydrated else 0
    return {"festival": festival, "edition": edition, "query": query, "trace": trace, "exposure": exposure,
            "verdict": ("installed" if exposure >= 8 else "thin" if exposure > 0 else "absent"),
            "note": "exposure = hydrated surface-form lines teaching this fact; ≥8 ≈ installs, 0 = the model never saw it"}


@action("verify_layer", category="data",
        inputs={"festival": "str", "edition": "str", "layer": "str: a pipeline layer e.g. 25_hydrated"})
def a_verify_layer(festival, edition, layer):
    """Verify one corpus layer for an edition: line count, distinct-ratio (dedup health) on jsonl, and
    whether downstream layers are stale relative to it. Read-only."""
    base = ROOT / "data" / festival / edition
    layers = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if layer not in layers:
        return {"error": f"no layer {layer} for {festival}/{edition}", "available": layers}
    ld = base / layer
    files, total, distinct = [], 0, 0
    for f in sorted(ld.rglob("*")):
        if f.suffix in (".jsonl", ".md", ".json") and f.is_file():
            lines = f.read_text(errors="replace").splitlines()
            n = len(lines)
            d = len(set(lines))
            files.append({"file": f.name, "lines": n, "distinct": d,
                          "distinct_ratio": round(d / n, 3) if n else None})
            total += n; distinct += d
    ratio = round(distinct / total, 3) if total else None
    verdict = "pass" if (ratio is None or ratio >= 0.9) and total > 0 else ("warn" if total else "fail")
    return {"festival": festival, "edition": edition, "layer": layer, "files": files,
            "total_lines": total, "distinct_ratio": ratio, "verdict": verdict,
            "note": "distinct_ratio < 0.9 → duplication before tokenize" if ratio and ratio < 0.9 else "ok"}


# ───────────────────────────── pipeline jobs (CPU/LLM — NEVER the GPU mutex) ────────────────
# Data-pipeline steps (ingest/clean/extract/hydrate/tokenize) are CPU + OpenRouter-API work. They run via
# this multi-job runner — a dict in pipeline_status.json, NOT the singleton GPU lock — so several can run
# at once AND a pipeline run coexists with a training run. The agent loop (run_pipeline) is one such job.
def _read_pipeline() -> dict:
    try:
        return json.loads(PIPELINE_STATUS.read_text())
    except Exception:
        return {}


def _write_pipeline(d: dict) -> None:
    PIPELINE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = PIPELINE_STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(PIPELINE_STATUS)


def _refresh_pipeline() -> dict:
    """Mark running jobs done once their pid dies (the CPU/LLM analog of the GPU stale-reaper)."""
    d = _read_pipeline()
    changed = False
    for j in d.values():
        if j.get("status") == "running" and not _pid_alive(j.get("pid")):
            j["status"], j["ended"], changed = "done", time.time(), True
    if changed:
        _write_pipeline(d)
    return d


BACKUPS = BUILD / "backups"               # gitignored — never-lose snapshots before any overwrite


def _guard_overwrite(festival, edition, layer, force, clear=False):
    """Protect EXPENSIVE regenerable output (scrape / extract / hydrate cost real time + OpenRouter money).
    Policy: never silently delete or replace. If the layer already has files and `force` is False → REFUSE
    (return an error dict). If `force` is True → copy the existing output to a timestamped backup under
    data/_build/backups (gitignored, never lost), then allow the overwrite; if `clear` is True (extraction,
    which appends) also remove the originals so the rebuild starts fresh. Returns (ok, info): ok=False →
    return `info` as the action's refusal; ok=True → proceed (info may carry the backup path)."""
    src = ROOT / "data" / festival / edition / layer
    existing = [p for p in src.rglob("*") if p.is_file()] if src.exists() else []
    if not existing:
        return True, {}
    if not force:
        return False, {"error": f"{festival}/{edition}/{layer} already holds {len(existing)} file(s) — refusing to "
                                f"replace expensive regenerable output. Pass force=true to proceed (the existing "
                                f"version is backed up first, never deleted).",
                       "existing_files": len(existing), "guard": "refused"}
    ts = time.strftime("%Y%m%d_%H%M%S")
    bk = BACKUPS / festival / edition / f"{layer}_{ts}"
    bk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, bk)
    if clear:
        for p in existing:
            p.unlink()
    return True, {"backed_up_to": str(bk.relative_to(ROOT))}


def _spawn_pipeline(job_id: str, cmd: list[str], extra_env: dict | None = None) -> dict:
    """Spawn a detached CPU/LLM pipeline step (sibling of _spawn). Records into the multi-job
    pipeline_status.json + logs to pipeline/<job_id>/run.log. Takes NO GPU lock by construction."""
    log = PIPELINE_DIR / job_id / "run.log"
    proc = _spawn(cmd, log, extra_env)
    d = _read_pipeline()
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    d[job_id] = {"job_id": job_id, "cmd": cmd, "pid": proc.pid, "pgid": pgid,
                 "log": str(log.relative_to(ROOT)), "status": "running", "started": time.time()}
    _write_pipeline(d)
    return d[job_id]


@action("pipeline_status", category="data",
        inputs={"job_id": "str: optional; omit for all pipeline jobs"})
def a_pipeline_status(job_id=None):
    """Status of CPU/LLM pipeline jobs (the GPU-free analog of run_status). Marks finished jobs done.
    Omit job_id for all jobs."""
    d = _refresh_pipeline()
    if job_id:
        return d.get(job_id) or {"error": f"no pipeline job {job_id}", "jobs": list(d)}
    return {"jobs": sorted(d.values(), key=lambda j: -(j.get("started") or 0))}


@action("pipeline_log", category="data",
        inputs={"job_id": "str", "after": "int: tail from this line (default 0)"})
def a_pipeline_log(job_id, after=0):
    """Tail a pipeline job's log from line `after` (the GPU-free analog of run_log)."""
    log = PIPELINE_DIR / job_id / "run.log"
    if not log.exists():
        return {"error": f"no log for pipeline job {job_id}", "lines": [], "next": int(after or 0)}
    lines = log.read_text(errors="replace").splitlines()
    after = int(after or 0)
    return {"job_id": job_id, "lines": lines[after:], "next": len(lines), "total": len(lines)}


@action("gate_corpus", category="data",
        inputs={"festival": "str (default edc)", "edition": "str: e.g. las_vegas/2026 (omit = all on disk)",
                "domain": "str: domain spec name (default = festival)"})
def a_gate_corpus(festival="edc", edition=None, domain=None):
    """Run the deterministic QUALITY-GATE suite (balance · exposure-cap · id-integrity · faithfulness ·
    source-vs-gap) over an edition's structured + hydrated corpus against its domain spec. Read-only,
    CPU-only (never touches the GPU). Returns per-gate pass/severity/metrics/examples + the
    source-gap-vs-defect classification for every missing relation. The recurrence-prevention layer for
    the corpus-audit failure modes (docs/qgre_data_pipeline_audit.md)."""
    from qgre import gates, spec as qspec
    from pipeline.editions import iter_editions
    sp = qspec.load_spec(domain or festival)
    eds = list(iter_editions(festival, only=edition))
    if not eds:
        return {"error": f"no editions matched {festival}/{edition or 'all'}"}
    return gates.run_suite(eds, sp)


# ───────────────────────────── pipeline TOOLS (ingest → clean → extract → hydrate → tokenize) ────────
# The domain-agnostic data pipeline as tools. All category="data", touches_gpu=False — long ones spawn via
# _spawn_pipeline (no GPU mutex), so the pipeline coexists with training. The agent loop (run_pipeline)
# orchestrates these by reading gate_corpus between steps.
@action("ingest_document", mutating=True, category="data",
        inputs={"festival": "str", "edition": "str: city/year", "content": "str: base64 bytes (upload)",
                "filename": "str: original filename", "domain_class": "str: trust class (default user-contributed)"})
def a_ingest_document(festival, edition, content, filename=None, domain_class="user-contributed"):
    """Ingest ONE uploaded document into <festival>/<edition>/00_raw/uploads/ + a provenance sidecar — the
    UI/agent entry point for adding source material (not just scraping). Reuses the scrape fetcher's
    Provenance shape + the 'uploads' cleaner so it flows through clean→structured identically. Synchronous."""
    import base64
    from scrape import fetcher as F
    city, _, year = edition.partition("/")
    try:
        raw_bytes = base64.b64decode(content)
    except Exception as e:
        return {"error": f"content must be base64: {e}"}
    ext = (filename or "doc.txt").rsplit(".", 1)[-1].lower()
    # map to a content_type the fetcher's _EXT knows, so the on-disk extension is right (md/pdf → text so
    # the uploads cleaner takes its text branch, not the html one). html stays html; json stays json.
    ct = {"html": "text/html", "htm": "text/html", "json": "application/json"}.get(ext, "text/plain")
    raw = F.RawDoc(url=f"upload://{filename or 'doc'}", final_url="", status=200,
                   content=raw_bytes, content_type=ct, fetcher="upload")
    dest = ROOT / "data" / festival / edition / "00_raw" / "uploads"
    art = F.save_doc(raw, dest, filename or "uploaded_doc", source_channel="uploads",
                     domain_class=domain_class, edition={"festival": festival, "city": city, "year": year})
    if art is None:
        return {"status": "deduped", "note": "an identical document already lives here"}
    return {"status": "ingested", "path": str(art.relative_to(ROOT)), "bytes": len(raw_bytes), "edition": edition}


@action("clean_layer", mutating=True, category="data",
        inputs={"editions": "str: city/year CSV", "sources": "str: optional source CSV",
                "llm_rescue": "bool: LLM-rescue failed HTML", "limit": "int: optional cap"})
def a_clean_layer(editions, sources=None, llm_rescue=False, limit=None, force=False):
    """00_raw → 10_clean (trafilatura + optional LLM-rescue). Spawned. Idempotent-skip by default (re-running
    skips already-clean docs, never clobbers). force=true re-cleans: backs up the existing 10_clean first,
    then passes --force so the cleaner regenerates. Returns a pipeline job handle."""
    backed = []
    cmd = [PY, "-m", "extract", "--editions", editions]
    if force:
        for ed in [e.strip() for e in editions.split(",") if e.strip()]:
            ok, info = _guard_overwrite("edc", ed, "10_clean", True)   # force path → backup (no clear; --force re-cleans)
            if info.get("backed_up_to"):
                backed.append(info["backed_up_to"])
        cmd += ["--force"]
    if sources:
        cmd += ["--sources", sources]
    if llm_rescue:
        cmd += ["--llm-rescue"]
    if limit:
        cmd += ["--limit", str(int(limit))]
    job_id = f"clean_{editions.replace('/', '_').replace(',', '+')}_{int(time.time())}"
    j = _spawn_pipeline(job_id, cmd)
    if backed:
        j["backed_up"] = backed
    return j


@action("extract_triples", mutating=True, category="data",
        inputs={"editions": "str: city/year CSV", "domain": "str: domain spec (default edc)",
                "concurrency": "int", "min_tier": "int", "limit": "int: optional cap"})
def a_extract_triples(editions, domain="edc", concurrency=16, min_tier=3, limit=None, force=False):
    """10_clean → 15_structured (LLM entity/triple extraction, spec-driven ontology + the predicate→object
    contract enforcement). Spawned (OpenRouter-bound). GUARDED: extraction is expensive (OpenRouter); a
    re-extract refuses to replace existing 15_structured unless force=true, which backs it up + clears it
    so the rebuild starts fresh (never silently mixed with or destroyed). Returns a job handle."""
    backed = []
    for ed in [e.strip() for e in editions.split(",") if e.strip()]:
        ok, info = _guard_overwrite(domain, ed, "15_structured", force, clear=True)
        if not ok:
            return {**info, "edition": ed}
        if info.get("backed_up_to"):
            backed.append(info["backed_up_to"])
    cmd = [PY, "scripts/extract_structured.py", "--editions", editions, "--domain", domain,
           "--concurrency", str(int(concurrency)), "--min-tier", str(int(min_tier)), "--yes"]
    if limit:
        cmd += ["--limit", str(int(limit))]
    job_id = f"extract_{editions.replace('/', '_').replace(',', '+')}_{int(time.time())}"
    j = _spawn_pipeline(job_id, cmd)
    if backed:
        j["backed_up"] = backed
    return j


@action("remap_contract", mutating=True, category="data",
        inputs={"editions": "str: city/year CSV", "domain": "str: domain spec (default edc)"})
def a_remap_contract(editions, domain="edc"):
    """Fix an edition's EXISTING 15_structured IN PLACE, deterministically — NO LLM, NO re-extract, NO data
    loss. Applies the predicate→object contract (plays_at→festival/edition ⇒ in_lineup, preserving every
    artist), drops cross-edition contamination, and normalizes b2b artist names. This is the LOSSLESS, FREE
    alternative to extract_triples for editions whose 15_structured is already rich: re-extract rebuilds
    from 10_clean and can drop lineup the original captured from dedicated sources (e.g. fandom full
    lineups), whereas this only rewrites the predicates/names on the triples already present. Synchronous;
    backs up 15_structured first."""
    import json as _json
    import extract_structured as E
    from collections import Counter
    E.ONT = E.Ontology(domain)
    out = {}
    for ed in [e.strip() for e in editions.split(",") if e.strip()]:
        sdir = ROOT / "data" / domain / ed / "15_structured"
        tri_p, ent_p = sdir / "triples.jsonl", sdir / "entities.jsonl"
        if not tri_p.exists():
            out[ed] = {"error": "no 15_structured/triples.jsonl — nothing to remap (use extract_triples)"}
            continue
        ok, info = _guard_overwrite(domain, ed, "15_structured", force=True)   # back up before rewriting
        if not ok:
            out[ed] = info
            continue
        year = E._edition_year(f"{domain}_{ed.replace('/', '_')}")
        stats = Counter()
        kept_t = []
        for line in tri_p.read_text().splitlines():
            if not line.strip():
                continue
            r = _json.loads(line)
            if E._drop_contaminated(r, "triple", year, stats):
                continue
            r, drop = E._enforce_contract(r, stats)
            if drop:
                continue
            kept_t.append(r)
        kept_e = []
        has_ent = ent_p.exists()
        if has_ent:
            for line in ent_p.read_text().splitlines():
                if not line.strip():
                    continue
                r = _json.loads(line)
                if E._drop_contaminated(r, "entity", year, stats):
                    continue
                if r.get("entity_type") == "artist" and r.get("name_canonical"):
                    r["name_canonical"] = E._normalize_b2b_name(r["name_canonical"])
                kept_e.append(r)
        # deterministic id repair (foreign-prefix drop + dangling-ref canonicalize/drop) — needs the full
        # entity set, so it runs after both layers are read and before either is written back.
        if has_ent:
            fslug = f"{domain}_{ed.replace('/', '_')}"
            kept_t, kept_e = E._clean_ids(kept_t, kept_e, fslug, stats)
            ent_p.write_text("".join(_json.dumps(r) + "\n" for r in kept_e))
        tri_p.write_text("".join(_json.dumps(r) + "\n" for r in kept_t))
        out[ed] = {"triples_kept": len(kept_t), "entities_kept": len(kept_e),
                   "remapped_plays_at_to_in_lineup": stats.get("remap_plays_at_to_in_lineup", 0),
                   "contract_dropped": stats.get("contract_drop", 0),
                   "contamination_dropped": stats.get("contamination_drop", 0),
                   "id_canonicalized": stats.get("id_canonicalized", 0),
                   "foreign_prefix_dropped": stats.get("foreign_prefix_drop", 0),
                   "id_orphan_dropped": stats.get("id_dangling_or_foreign_drop", 0),
                   "backed_up_to": info.get("backed_up_to")}
    return {"remap": out}


@action("hydrate", mutating=True, category="data",
        inputs={"festival": "str", "edition": "str: city/year", "relations": "str: optional CSV (top-up)",
                "exposure_cap": "int: per-fact line ceiling", "balance_target": "str: rel=N,rel2=M",
                "base_n": "int", "concurrency": "int", "temperature": "float", "limit": "int"})
def a_hydrate(festival, edition, relations=None, exposure_cap=None, balance_target=None,
              base_n=12, concurrency=8, temperature=0.3, limit=None, force=False):
    """15_structured → 25_hydrated via festival-factory with the qgre levers: --relations (top-up,
    appends), --exposure-cap (stop the runaway), --balance-target (boost under-covered relations).
    Spawned (OpenRouter-bound). GUARDED: a full re-hydrate refuses to replace existing 25_hydrated unless
    force=true, which backs it up first (hydration is expensive — never silently destroyed). Returns a job handle."""
    ff = ROOT / "festival-factory" / "target" / "debug" / "festival-factory"
    if not ff.exists():
        return {"error": "festival-factory not built — run `cargo build` in festival-factory/"}
    info = {}
    if not relations:   # a full hydrate TRUNCATES relations.jsonl; a --relations top-up appends (no clobber)
        ok, info = _guard_overwrite(festival, edition, "25_hydrated", force)
        if not ok:
            return info
    cmd = [str(ff), "hydrate", "--festival", festival, "--edition", edition, "--root", str(ROOT), "--force",
           "--base-n", str(int(base_n)), "--concurrency", str(int(concurrency)), "--temperature", str(temperature)]
    if relations:
        cmd += ["--relations", relations]
    if exposure_cap:
        cmd += ["--exposure-cap", str(int(exposure_cap))]
    if balance_target:
        cmd += ["--balance-target", balance_target]
    if limit:
        cmd += ["--limit", str(int(limit))]
    job_id = f"hydrate_{edition.replace('/', '_')}_{int(time.time())}"
    j = _spawn_pipeline(job_id, cmd)
    if info.get("backed_up_to"):
        j["backed_up"] = info["backed_up_to"]
    return j


@action("tokenize", mutating=True, category="data",
        inputs={"out": "str: arrow/stats basename under 30_tokenized", "editions": "str: optional CSV",
                "seq_len": "int", "replay_frac": "float"})
def a_tokenize(out, editions=None, seq_len=512, replay_frac=0.10):
    """25_hydrated (+ clean + replay) → 30_tokenized Arrow + lever metadata. `out` must match the CPT
    config's data.tokenized basename. Spawned (CPU-bound). Returns a pipeline job handle."""
    cmd = [PY, "scripts/tokenize_corpus.py", "--out", out,
           "--seq-len", str(int(seq_len)), "--replay-frac", str(replay_frac)]
    if editions:
        cmd += ["--editions", editions]
    job_id = f"tokenize_{out}_{int(time.time())}"
    return _spawn_pipeline(job_id, cmd)


@action("run_pipeline", mutating=True, category="data",
        inputs={"domain": "str: domain spec (default edc)", "editions": "str: city/year CSV",
                "exposure_cap": "int: per-fact ceiling", "start_from": "str: clean|extract|hydrate",
                "max_rehydrate": "int: re-hydrate attempts per relation"})
def a_run_pipeline(domain="edc", editions=None, exposure_cap=60, start_from="clean", max_rehydrate=2):
    """Launch the PURE qgre agent (qgre-agent/, built on the aisdk-kotlin port): a model-in-the-loop agent
    that drives the pipeline tools (clean → extract → hydrate → gate) and DECIDES each next call itself —
    repairing the defects the gates flag, reporting source-gaps honestly (never fabricating), and leaving
    the protected gold gate-only. CPU/LLM only — NEVER takes the GPU lock, NEVER auto-trains. Returns a
    pipeline job handle; stream the agent's reasoning + tool calls via pipeline_log.

    Runs headless (QGRE_AUTO_APPROVE=1 → paid extract/hydrate auto-approved) and calls back into THIS server
    over HTTP (QGRE_SERVE_URL), so serve.py must be listening. exposure_cap/start_from/max_rehydrate are
    legacy no-ops — the agent decides these itself; kept in the signature for caller back-compat."""
    if not editions:
        return {"error": "editions required (e.g. las_vegas/2026)"}
    agent_bin = ROOT / "qgre-agent/build/install/qgre-agent/bin/qgre-agent"
    if not agent_bin.exists():
        return {"error": "qgre-agent is not built — run: gradle -p qgre-agent installDist"}
    job_id = f"{domain}_pipeline_{int(time.time())}"
    cmd = [str(agent_bin), "--domain", domain, "--editions", editions, "--job-id", job_id]
    env = {"QGRE_AUTO_APPROVE": "1",
           "QGRE_SERVE_URL": os.environ.get("QGRE_SERVE_URL", "http://127.0.0.1:8800")}
    j = _spawn_pipeline(job_id, cmd, extra_env=env)
    j["message"] = f"pure qgre agent launched — poll pipeline_status('{job_id}') / pipeline_log('{job_id}')"
    return j


@action("stop_pipeline", mutating=True, category="data", inputs={"job_id": "str"})
def a_stop_pipeline(job_id):
    """Stop a running pipeline job (the loop or a single tool). SIGTERMs its process group (children too);
    releases NO GPU lock — pipeline jobs never hold one. Idempotent."""
    d = _refresh_pipeline()
    j = d.get(job_id)
    if not j:
        return {"error": f"no pipeline job {job_id}"}
    pgid = j.get("pgid")
    if j.get("status") == "running" and pgid:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    j["status"] = "stopped"
    j["ended"] = time.time()
    _write_pipeline(d)
    return {"job_id": job_id, "status": "stopped"}


# ───────────────────────────── playground (stored / live) ──────────────────
@action("playground_stored", category="eval",
        inputs={"run_id": "str", "verdict": "str: optional filter (fabrication|over_refusal|wrong|correct|abstain)"})
def a_playground_stored(run_id, verdict=None):
    """Zero-GPU playground: replay the recorded transcript from a run's gated scorecard — every probe with
    what the model said + the verdict (answer/refuse/fabricate). Always available, never touches the mutex."""
    d = _serve().run_detail(run_id) or {}
    sc = d.get("scorecard") or {}
    rows = sc.get("rows", [])
    if verdict:
        rows = [r for r in rows if r.get("verdict") == verdict]
    return {"run_id": run_id, "scored": bool(sc), "n": len(rows), "turns": rows[:200],
            "note": "stored transcript from the deterministic gate — no GPU used" if sc else "this run was never gated"}


@action("playground_live", mutating=True, touches_gpu=True, category="eval",
        inputs={"checkpoint": "str: checkpoint dir (e.g. checkpoints/sft/edc_sft_v11)", "prompt": "str",
                "temperature": "float: 0 = memory(greedy), >0 = voice(sampled)", "top_k": "int", "max_new": "int"})
def a_playground_live(checkpoint, prompt, temperature=0.0, top_k=64, max_new=64):
    """Live FP4 inference against a checkpoint in the precision it ships in — ask the shipping model a
    question. Touches the GPU (acquire-or-refuse the mutex). temp=0 = memory mode (greedy, what the weights
    encode); temp>0 = voice mode (the on-device runtime sampling). Synchronous (~seconds after warm)."""
    ck = Path(checkpoint) if Path(checkpoint).is_absolute() else (ROOT / checkpoint)
    if not ck.exists():
        return {"error": f"no checkpoint at {checkpoint}"}
    acquire_gpu(f"playground:{ck.name}")
    try:
        set_lock_pid(os.getpid())
        r = subprocess.run(
            [PY, "scripts/playground_infer.py", "--checkpoint", str(ck), "--prompt", prompt,
             "--temp", str(temperature), "--top-k", str(top_k), "--max-new", str(max_new)],
            cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "scripts", "CUDA_VISIBLE_DEVICES": str(GPU_DEVICE)},
            capture_output=True, text=True, timeout=240)
    finally:
        release_gpu()
    if r.returncode != 0:
        return {"error": f"inference failed: {r.stderr.strip()[-400:]}"}
    try:
        ans = json.loads(r.stdout.strip().splitlines()[-1])["answer"]
    except Exception:
        ans = r.stdout.strip()[-400:]
    return {"checkpoint": checkpoint, "prompt": prompt, "answer": ans,
            "mode": "voice" if (temperature and temperature > 0) else "memory"}


# ───────────────────────────── export (on-device artifact + registry) ───────
@action("export_checkpoint", mutating=True, category="export",
        inputs={"run_id": "str", "force": "bool: export even if the gate did not pass (override)"})
def a_export_checkpoint(run_id, force=False):
    """Produce the on-device deploy artifact (the int8-master weights + quant_keys + run_contract + gate
    report, bundled under exports/<run_id>/). REFUSES unless the run's gate PASSED — the experiment→shippable
    boundary. The gate report + engine identity travel WITH the artifact. CPU only."""
    import shutil
    d = _serve().run_detail(run_id) or {}
    if not d:
        return {"error": f"no such run: {run_id}"}
    passed = d.get("status") == "passed"
    if not passed and not force:
        return {"error": f"run '{run_id}' did not PASS the gate (status={d.get('status')}). Export refused — "
                f"pass force=true to override for a borderline gate.", "kind": "gate_block"}
    src = SFT_DIR / run_id
    if not src.is_dir():
        src = CPT_DIR / run_id
    if not src.is_dir():
        return {"error": f"no checkpoint dir for {run_id}"}
    dst = ROOT / "exports" / run_id
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("model.safetensors", "quant_keys.json", "run_contract.json", "config.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            copied.append(name)
    manifest = {"run_id": run_id, "gate": d.get("gate"), "metrics": d.get("metrics"),
                "contract": d.get("contract"), "exported_files": copied, "source": str(src.relative_to(ROOT))}
    _write_json_atomic(dst / "manifest.json", manifest)
    return {"run_id": run_id, "export_dir": str(dst.relative_to(ROOT)), "files": copied,
            "gate_passed": passed, "forced": bool(force and not passed),
            "message": f"exported {run_id} → exports/{run_id}/ ({len(copied)} files; gate + engine identity travel with it)."}


@action("list_exports", category="export")
def a_list_exports():
    """The model registry: only gate-passing (or force-exported) models, each with its frozen manifest
    (gate report + engine identity + files). The boundary between experiment space and 'what we ship'."""
    edir = ROOT / "exports"
    out = []
    if edir.is_dir():
        for d in sorted(edir.iterdir()):
            if d.is_dir():
                man = _read_json(d / "manifest.json") or {}
                out.append({"run_id": d.name, "gate": man.get("gate"), "metrics": man.get("metrics"),
                            "contract": man.get("contract"), "files": man.get("exported_files", [])})
    return {"exports": out}


# ───────────────────────────── Vizier-style serial sweep optimizer ──────────
SWEEPS = BUILD / "sweeps"


def _study_path(study: str) -> Path:
    return SWEEPS / f"{study}.json"


@action("create_study", mutating=True, category="train",
        inputs={"study": "str: study name", "params": "object: {name:[lo,hi] or [choices]}",
                "objective": "str: maximize|minimize", "metric": "str: e.g. calibration_score"})
def a_create_study(study, params, objective="maximize", metric="calibration_score"):
    """Declare a hyperparameter study: param ranges + objective + metric. Trials run STRICTLY SERIALLY under
    the single-GPU mutex (and serial trials are MORE sample-efficient than parallel — the constraint is a
    feature). The run registry is the trial store."""
    SWEEPS.mkdir(parents=True, exist_ok=True)
    spec = {"study": study, "params": params, "objective": objective, "metric": metric, "trials": []}
    _write_json_atomic(_study_path(study), spec)
    return {"study": study, "params": params, "objective": objective, "metric": metric}


@action("suggest_next_config", mutating=True, category="train",
        inputs={"study": "str", "base": "str: base config to override", "out": "str: where to write the generated config"})
def a_suggest_next_config(study, base, out):
    """Propose the next trial's hyperparameters from prior trials' gate results (seed = random in-range; then
    local search around the best). Emits a gen_config invocation (no trainer flags — the contract holds), so
    the suggestion becomes a validated config you launch. The optimizer never touches the GPU."""
    spec = _read_json(_study_path(study))
    if not spec:
        return {"error": f"no study '{study}' — create_study first"}
    trials = spec.get("trials", [])
    params = spec["params"]
    SEED_FRACS = [0.5, 0.0, 1.0, 0.25, 0.75, 0.382, 0.618, 0.125, 0.875]

    def _is_num_range(rng):
        return isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, (int, float)) for x in rng)

    def _round(v, rng):
        # ints stay ints; floats keep 4 significant figures so small lr values (1e-5) survive rounding
        return int(round(v)) if all(isinstance(x, int) for x in rng) else float(f"{v:.4g}")

    # seed phase: spread fractions across the box (offset per param so they don't all move together)
    def _grid_point(i):
        f = SEED_FRACS[i % len(SEED_FRACS)]
        pt = {}
        for j, (k, rng) in enumerate(params.items()):
            if _is_num_range(rng):
                lo, hi = rng
                ff = (f + 0.37 * j) % 1.0
                pt[k] = _round(lo + ff * (hi - lo), rng)
            elif isinstance(rng, list):
                pt[k] = rng[(i + j) % len(rng)]
        return pt

    n = len(trials)
    if n < 3 or not any(t.get("score") is not None for t in trials):
        proposal = _grid_point(n)
    else:
        best = max((t for t in trials if t.get("score") is not None),
                   key=lambda t: t["score"] if spec["objective"] == "maximize" else -t["score"])
        proposal = dict(best["params"])
        for k, rng in params.items():
            if _is_num_range(rng):
                lo, hi = rng
                span = (hi - lo) * 0.2
                step = span * (((n * 7) % 5) / 4.0 - 0.5) * 2     # deterministic perturbation in [-span, span]
                cur = proposal.get(k, (lo + hi) / 2)
                proposal[k] = _round(min(hi, max(lo, cur + step)), rng)
    # each trial writes to its OWN save_dir so launching it never clobbers an existing run
    trial_run = f"{study}_t{n}"

    def _yscalar(v):
        # YAML 1.1 (PyYAML) only reads 'e' notation as a float when the mantissa has a '.', so 3e-05 would
        # parse as a STRING and break the contract. Emit plain decimal, which always round-trips as a float.
        if isinstance(v, float):
            s = f"{v:.10f}".rstrip("0").rstrip(".")
            return s or "0"
        return str(v)
    sets = [f"{k}={_yscalar(v)}" for k, v in proposal.items()]
    sets += [f"run.save_dir=checkpoints/sft/{trial_run}", f"run.run_name={trial_run}"]
    res = a_gen_config(base=base, out=out, sets=sets)
    return {"study": study, "trial": n, "run_id": trial_run, "proposal": proposal,
            "config": out if res.get("rc") == 0 else None, "gen": res, "sets": sets}


@action("report_measurement", mutating=True, category="train",
        inputs={"study": "str", "run_id": "str", "params": "object: the trial's params", "score": "float: the metric value"})
def a_report_measurement(study, run_id, params, score=None):
    """Feed a finished trial's gate result back into the study (run_id keys it, like Vizier's clientId).
    If score is omitted, it's read from the run's gate metric."""
    spec = _read_json(_study_path(study))
    if not spec:
        return {"error": f"no study '{study}'"}
    if score is None:
        d = _serve().run_detail(run_id) or {}
        m = d.get("metrics") or {}
        score = m.get(spec.get("metric", "calibration_score"))
    spec["trials"].append({"run_id": run_id, "params": params, "score": score})
    _write_json_atomic(_study_path(study), spec)
    scored = [t for t in spec["trials"] if t.get("score") is not None]
    best = (max(scored, key=lambda t: t["score"]) if spec["objective"] == "maximize"
            else min(scored, key=lambda t: t["score"])) if scored else None
    return {"study": study, "trials": len(spec["trials"]), "best": best}


@action("list_studies", category="read")
def a_list_studies():
    """All hyperparameter studies + their trial counts and best-so-far (the sweeps index)."""
    out = []
    if SWEEPS.is_dir():
        for f in sorted(SWEEPS.glob("*.json")):
            spec = _read_json(f) or {}
            scored = [t for t in spec.get("trials", []) if t.get("score") is not None]
            best = (max(scored, key=lambda t: t["score"]) if spec.get("objective") == "maximize"
                    else min(scored, key=lambda t: t["score"])) if scored else None
            out.append({"study": spec.get("study", f.stem), "objective": spec.get("objective"),
                        "metric": spec.get("metric"), "params": spec.get("params"),
                        "n_trials": len(spec.get("trials", [])), "best": best})
    return {"studies": out}


@action("get_study", category="read", inputs={"study": "str"})
def a_get_study(study):
    """Read a study's spec + all trials + the best-so-far (the sweep leaderboard)."""
    spec = _read_json(_study_path(study))
    if not spec:
        return {"error": f"no study '{study}'"}
    scored = [t for t in spec.get("trials", []) if t.get("score") is not None]
    best = (max(scored, key=lambda t: t["score"]) if spec.get("objective") == "maximize"
            else min(scored, key=lambda t: t["score"])) if scored else None
    return {**spec, "best": best}


# ───────────────────────────── exception → action-result helper ─────────────
def run_action(name: str, args: dict, *, allow_write: bool) -> tuple[int, dict]:
    """Adapter helper shared by serve.py POST and the MCP server: enforce write-gating + map GpuBusy/errors
    to a structured result. Returns (http_status_like, payload)."""
    a = ACTIONS.get(name)
    if not a:
        return 404, {"error": f"no such action: {name}"}
    if a.mutating and not allow_write:
        return 403, {"error": f"action '{name}' is write/compute and is disabled (enable writes to use it)."}
    try:
        return 200, {"ok": True, "result": a.fn(**(args or {}))}
    except GpuBusy as e:
        return 409, {"ok": False, "error": str(e), "kind": "gpu_busy"}
    except TypeError as e:
        return 400, {"ok": False, "error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return 500, {"ok": False, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    # quick self-check / catalog dump
    import argparse
    ap = argparse.ArgumentParser(description="Torad action catalog")
    ap.add_argument("--catalog", action="store_true", help="print the action catalog")
    ap.add_argument("--call", help="action name to invoke")
    ap.add_argument("--args", default="{}", help="JSON args for --call")
    a = ap.parse_args()
    if a.catalog:
        print(json.dumps(catalog(), indent=2))
    elif a.call:
        status, payload = run_action(a.call, json.loads(a.args), allow_write=True)
        print(json.dumps({"status": status, **payload}, indent=2, default=str))
    else:
        print(f"{len(ACTIONS)} actions: {', '.join(sorted(ACTIONS))}")
