#!/usr/bin/env python3
"""Single config-driven driver for a domain's full scrape→CPT→SFT pipeline (the industrialized assembler).

    PYTHONPATH=scripts python scripts/run_pipeline.py --domain edc [--from structured] [--to tokenize]
        [--only meta,hydrate] [--editions city/year,…] [--dry-run] [--allow-gpu]

Reads configs/domains/<domain>.yaml, selects the requested stages in declared order, resolves each
stage's argv template ({editions}, {py}, {cpt}, vars…), and runs them in sequence. GPU stages are SKIPPED
unless --allow-gpu, so the runner never auto-launches training by surprise (and is safe to invoke while a
GPU job is already running). --dry-run prints the resolved plan without executing. A new domain =
new configs/domains/<name>.yaml; the runner is unchanged.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import config as C
from pipeline.domain import Domain


def editions_on_disk(festival: str) -> list[str]:
    """All 'city/year' units with a 00_raw landing zone (the scraped set)."""
    out = set()
    for p in glob.glob(str(C.DATA / festival / "*" / "*" / "00_raw")):
        parts = p.split("/")
        out.add(f"{parts[-3]}/{parts[-2]}")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--from", dest="frm", help="first stage to run (declared order)")
    ap.add_argument("--to", help="last stage to run")
    ap.add_argument("--only", help="comma list of specific stages")
    ap.add_argument("--editions", help="comma 'city/year' list; default = all on disk")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan, run nothing")
    ap.add_argument("--allow-gpu", action="store_true", help="actually run GPU stages (else skipped)")
    args = ap.parse_args()

    dom = Domain(args.domain)
    editions = args.editions or ",".join(editions_on_disk(dom.name))
    n_ed = len(editions.split(",")) if editions else 0
    try:
        stages = dom.select(only=args.only, frm=args.frm, to=args.to)
    except ValueError as e:
        sys.exit(str(e))

    print(f"domain={dom.name}  hierarchy={'/'.join(dom.hierarchy)}  editions={n_ed}  "
          f"stages={[s.name for s in stages]}")
    env = {**os.environ, "PYTHONPATH": "scripts"}
    for s in stages:
        cmd = dom.resolve(s.cmd, editions=editions)
        shown = " ".join(cmd)
        if len(shown) > 200:
            shown = shown.replace(editions, f"<{n_ed} editions>")
        print(f"\n── [{s.name}] ({'GPU' if s.gpu else 'cpu'}) {s.desc}\n   $ {shown}")
        if args.dry_run:
            continue
        if s.gpu and not args.allow_gpu:
            print("   ⏭  skipped (GPU stage; pass --allow-gpu to run)")
            continue
        r = subprocess.run(cmd, cwd=str(C.ROOT), env=env)
        if r.returncode != 0:
            sys.exit(f"   ✗ stage '{s.name}' failed (exit {r.returncode}) — stopping")
        print(f"   ✓ {s.name} done")
    print("\n[dry-run] plan only — nothing executed" if args.dry_run else "\npipeline complete")


if __name__ == "__main__":
    main()
