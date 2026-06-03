"""Reproducible-run registry (Phase 4 of the config-drift refactor).

Every training run freezes its fully-resolved config + dataset hash + git SHA + timestamp into
`logs/runs/<run_id>/` and appends one row to the append-only index `logs/runs/index.jsonl`. This makes any
run reproducible from the repo alone: the exact config that ran, the corpus identity, and the code SHA.

  run_id            = "<run_name>-<fingerprint8>-<unix_ts>"   (sortable, unique, human-readable)
  <run_id>/config.resolved.json  = asdict(CptConfig) + _fingerprint  (what ACTUALLY ran)
  <run_id>/metadata.json         = {run_id, fingerprint, run_name, mode, git_sha, dataset_hash,
                                    save_dir, tokenized, init_from, started_iso, cuda_visible_devices}
  index.jsonl                    = one metadata row per run (the registry; rebuildable by scanning dirs)

Eval results stay in data/_build/dissect/ (the dashboard + compare_evals + best_ckpt read that layout
unchanged — preserve-invariant); the registry row carries `save_dir` so a run's scorecards are joinable by
checkpoint name. Read by: any control-plane / Coach surface that wants "which config produced this run".
"""
from __future__ import annotations

import json
import subprocess
import time

import config as C

RUNS_DIR = C.ROOT / "logs" / "runs"
INDEX = RUNS_DIR / "index.jsonl"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(C.ROOT),
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _dataset_hash(tokenized: str | None) -> str:
    """Corpus identity from the <name>_stats.json sidecar (total_tokens + num_chunks + seq_len). Falls back
    to the tokenized path string when no sidecar exists (older corpora) — never blocks a run."""
    if not tokenized:
        return "none"
    sp = C.ROOT / tokenized
    sc = sp.parent / f"{sp.name.replace('.arrow', '')}_stats.json"
    if sc.exists():
        try:
            s = json.loads(sc.read_text())
            key = f"{s.get('total_tokens')}:{s.get('num_chunks')}:{s.get('seq_len')}"
            import hashlib
            return "sha256:" + hashlib.sha256(key.encode()).hexdigest()[:16]
        except Exception:
            pass
    return f"unhashed:{tokenized}"


def freeze_run(typed, save_dir: str, cuda_visible_devices: str = "unset") -> str:
    """Freeze the resolved config + provenance for this run. Returns the run_id. Safe to call once at run
    start (rank0 only). Never raises into the trainer — a registry failure must not kill a paid GPU run."""
    try:
        fp = typed.fingerprint()
        ts = int(time.time())
        run_id = f"{typed.run.run_name}-{fp[:8]}-{ts}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "config.resolved.json").write_text(json.dumps(typed.to_resolved_dict(), indent=2))
        meta = {
            "run_id": run_id, "fingerprint": fp, "run_name": typed.run.run_name, "mode": typed.mode,
            "git_sha": _git_sha(), "dataset_hash": _dataset_hash(typed.data.tokenized),
            "save_dir": save_dir, "tokenized": typed.data.tokenized, "init_from": typed.data.init_from,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
            "cuda_visible_devices": cuda_visible_devices,
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        with open(INDEX, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return run_id
    except Exception as e:
        print(f"[run_registry] WARN: freeze_run failed ({type(e).__name__}: {e}) — run continues unregistered")
        return ""


def _rebuild_index() -> int:
    """Rebuild index.jsonl by scanning logs/runs/*/metadata.json (the per-run cards are the source of truth)."""
    rows = []
    for md in sorted(RUNS_DIR.glob("*/metadata.json")):
        try:
            rows.append(json.loads(md.read_text()))
        except Exception:
            pass
    rows.sort(key=lambda r: r.get("started_iso", ""))
    INDEX.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run-registry utilities.")
    ap.add_argument("--list", action="store_true", help="print the registry index")
    ap.add_argument("--rebuild", action="store_true", help="rebuild index.jsonl from the per-run cards")
    args = ap.parse_args()
    if args.rebuild:
        print(f"rebuilt index: {_rebuild_index()} runs")
    elif args.list and INDEX.exists():
        for ln in INDEX.read_text().splitlines():
            r = json.loads(ln)
            print(f"  {r['run_id']:48} {r['mode']:3} sha={r['git_sha']:8} {r['dataset_hash']}")
    else:
        print("no runs registered yet" if not INDEX.exists() else "")
