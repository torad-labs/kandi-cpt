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
import json
import os
import subprocess
import sys
from pathlib import Path

import config as C
from pipeline.domain import Domain


def editions_on_disk(festival: str) -> list[str]:
    """All 'city/year' units with a 00_raw landing zone (the scraped set)."""
    out = set()
    for p in glob.glob(str(C.DATA / festival / "*" / "*" / "00_raw")):
        parts = p.split("/")
        out.add(f"{parts[-3]}/{parts[-2]}")
    return sorted(out)


# ── --check: a manifest preflight (validate every declared reference, then ABORT) ──────────────────
# Additive and read-only — it NEVER runs a stage. The pipeline still fails-fast mid-run on a missing
# file (no behaviour change); --check just surfaces every such gap up front, as one structured JSON
# report, before a single dollar or GPU-hour is spent. Relative refs resolve against the repo root.
_REF_SUFFIXES = (".py", ".sh", ".rs", ".json", ".yaml", ".yml")


def _exists(ref: str) -> bool:
    if not ref:
        return False
    p = Path(ref)
    return (p if p.is_absolute() else C.ROOT / ref).exists()


def _cmd_file_refs(cmd: list[str]) -> list[str]:
    """The file paths a resolved stage command reads — its script + any config/schema arg (not flags)."""
    return [a for a in map(str, cmd) if not a.startswith("-") and "/" in a and a.endswith(_REF_SUFFIXES)]


def check_domain(name: str, editions: str) -> tuple[dict, bool]:
    """Validate a domain manifest + all the paths it references. Returns (report, ok); ok ⇒ no REQUIRED gap."""
    checks: list[dict] = []

    def add(check: str, severity: str, ok: bool, **extra) -> None:
        checks.append({"check": check, "severity": severity, "ok": bool(ok), **extra})

    try:
        dom = Domain(name)
    except Exception as e:  # missing or unparseable manifest — report it, don't crash
        add("manifest.load", "REQUIRED", False, detail=str(e))
        return {"domain": name, "ok": False, "checks": checks,
                "summary": {"required_failed": 1, "warnings": 0}}, False

    add("manifest.load", "REQUIRED", True, path=str(dom.path))
    add("registry", "REQUIRED", _exists(dom.registry), path=dom.registry)

    schema_dir = dom.ontology.get("schema_dir", "schema")
    add("ontology.schema_dir", "REQUIRED", _exists(schema_dir), path=schema_dir)
    for core in ("entity", "triple", "discourse", "metadata"):
        rel = f"{schema_dir}/{core}.schema.json"
        add(f"schema.{core}", "REQUIRED", _exists(rel), path=rel)
    if (tax := dom.ontology.get("taxonomy", "")):
        add("ontology.taxonomy", "REQUIRED", _exists(tax), path=tax)
    # The ontology lives INLINE in the manifest — top-level entity_types + predicates. The engine derives
    # the structured-validate enums from them, so an empty ontology means `structured` has no enum to check.
    add("ontology.entity_types", "REQUIRED", bool(dom.spec.get("entity_types")),
        detail="top-level entity_types (the entity enum) must be non-empty")
    add("ontology.predicates", "REQUIRED", bool(dom.spec.get("predicates")),
        detail="top-level predicates (relation enum + object-type contracts) must be non-empty")

    add("model_path", "REQUIRED", _exists(dom.model_path), path=dom.model_path)
    for key in ("cpt", "sft"):
        if (ref := dom.configs.get(key)):
            add(f"configs.{key}", "REQUIRED", _exists(ref), path=ref)
        else:
            add(f"configs.{key}", "WARN", True, detail="unset (data-only until training)")
    if (score := dom.vars.get("score_ckpt")):
        add("vars.score_ckpt", "WARN", _exists(score), path=score, detail="needed only at build_sft")

    for s in dom.stages:
        try:
            cmd = dom.resolve(s.cmd, editions=editions)
        except KeyError as e:
            add(f"stage.{s.name}.resolve", "REQUIRED", False, detail=str(e))
            continue
        add(f"stage.{s.name}.resolve", "REQUIRED", True)
        for ref in _cmd_file_refs(cmd):
            add(f"stage.{s.name}.ref", "REQUIRED", _exists(ref), path=ref)

    req_failed = sum(1 for c in checks if c["severity"] == "REQUIRED" and not c["ok"])
    warns = sum(1 for c in checks if c["severity"] == "WARN" and not c["ok"])
    report = {"domain": name, "ok": req_failed == 0, "checks": checks,
              "summary": {"required_failed": req_failed, "warnings": warns}}
    return report, req_failed == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--from", dest="frm", help="first stage to run (declared order)")
    ap.add_argument("--to", help="last stage to run")
    ap.add_argument("--only", help="comma list of specific stages")
    ap.add_argument("--editions", help="comma 'city/year' list; default = all on disk")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan, run nothing")
    ap.add_argument("--allow-gpu", action="store_true", help="actually run GPU stages (else skipped)")
    ap.add_argument("--check", action="store_true",
                    help="validate the manifest + every referenced path, print a JSON report, run nothing")
    args = ap.parse_args()

    if args.check:
        report, ok = check_domain(args.domain, args.editions or "")
        print(json.dumps(report, indent=2))
        sys.exit(0 if ok else 1)

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
