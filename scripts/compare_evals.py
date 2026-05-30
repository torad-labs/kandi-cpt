#!/usr/bin/env python3
"""Compare two eval runs (voice survival + knowledge probes).

Diffs two eval output directories and prints a delta table. Use to track
drift across checkpoints — voice should stay high (no register degradation),
probes should climb (knowledge install), and shifts on specific probes are
the diagnostic signal for what's actually being learned vs forgotten.

Run:
    python scripts/compare_evals.py logs/eval_gemma-4-E2B-it logs/eval_ckpt_1
    python scripts/compare_evals.py logs/eval_ckpt_1 logs/eval_ckpt_2

The first dir is the "before" reference; the second is "after". Output is
printed to stdout AND saved as a markdown report next to the second dir.
"""
from pathlib import Path
import argparse
import json
import sys

def load_eval(d: Path) -> tuple[dict, dict]:
    voice = json.loads((d / "voice.json").read_text())
    probes = json.loads((d / "probes.json").read_text())
    return voice, probes

def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{n}/{total} ({n/total*100:.0f}%)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path, help="reference eval dir (e.g. baseline)")
    parser.add_argument("after", type=Path, help="newer eval dir (e.g. ckpt_N)")
    args = parser.parse_args()

    if not args.before.exists() or not args.after.exists():
        sys.exit(f"Both dirs must exist: {args.before} {args.after}")

    v_b, p_b = load_eval(args.before)
    v_a, p_a = load_eval(args.after)

    label_b = args.before.name
    label_a = args.after.name

    lines: list[str] = []
    def out(s: str = ""):
        lines.append(s)
        print(s)

    out(f"# Eval comparison: {label_b} → {label_a}")
    out()
    out("## Headline")
    out()
    out(f"| Metric            | {label_b:^18s} | {label_a:^18s} | Delta |")
    out("|-------------------|--------------------|--------------------|-------|")

    v_b_pass, v_a_pass = v_b["pass_count"], v_a["pass_count"]
    v_total = v_b["total"]
    voice_delta = v_a_pass - v_b_pass
    out(f"| Voice survival    | {fmt_pct(v_b_pass, v_total):^18s} | {fmt_pct(v_a_pass, v_total):^18s} | {voice_delta:+d}    |")

    p_b_pass, p_a_pass = p_b["pass_count"], p_a["pass_count"]
    p_total = min(p_b["total"], p_a["total"])
    probe_delta = p_a_pass - p_b_pass
    out(f"| Knowledge probes  | {fmt_pct(p_b_pass, p_total):^18s} | {fmt_pct(p_a_pass, p_total):^18s} | {probe_delta:+d}    |")
    out()

    # Per-era breakdown — 2026-specific probes test CPT install (model couldn't
    # know these from training); stable probes test retention of recurring facts.
    by_era_b = p_b.get("by_era", {})
    by_era_a = p_a.get("by_era", {})
    if by_era_b or by_era_a:
        out("### By era")
        out()
        out(f"| Era       | {label_b:^18s} | {label_a:^18s} | Delta |")
        out("|-----------|--------------------|--------------------|-------|")
        for era in sorted(set(by_era_b) | set(by_era_a)):
            b_stats = by_era_b.get(era, {"pass_count": 0, "total": 0})
            a_stats = by_era_a.get(era, {"pass_count": 0, "total": 0})
            d = a_stats["pass_count"] - b_stats["pass_count"]
            out(f"| {era:<9s} | {fmt_pct(b_stats['pass_count'], b_stats['total']):^18s} | {fmt_pct(a_stats['pass_count'], a_stats['total']):^18s} | {d:+d}    |")
        out()

    # Voice score-level breakdown
    out("## Voice — per-feature counts")
    out()
    feature_counts_b = {"brief": 0, "no_emoji": 0, "on_topic_or_redirect": 0}
    feature_counts_a = {"brief": 0, "no_emoji": 0, "on_topic_or_redirect": 0}
    for r in v_b["results"]:
        for f in r["features"]:
            feature_counts_b[f] = feature_counts_b.get(f, 0) + 1
    for r in v_a["results"]:
        for f in r["features"]:
            feature_counts_a[f] = feature_counts_a.get(f, 0) + 1
    out(f"| Feature              | {label_b:^10s} | {label_a:^10s} | Delta |")
    out("|----------------------|------------|------------|-------|")
    for feat in ("brief", "no_emoji", "on_topic_or_redirect"):
        b, a = feature_counts_b.get(feat, 0), feature_counts_a.get(feat, 0)
        out(f"| {feat:<20s} | {b:^10d} | {a:^10d} | {a-b:+d}    |")
    out()

    # Probe-level diff
    out("## Probe flips")
    out()
    # Align by prompt text (probes may be in same order but match defensively)
    p_b_by_prompt = {r["prompt"]: r for r in p_b["results"]}
    p_a_by_prompt = {r["prompt"]: r for r in p_a["results"]}
    common = [k for k in p_b_by_prompt if k in p_a_by_prompt]

    gained = [k for k in common if not p_b_by_prompt[k]["passed"] and p_a_by_prompt[k]["passed"]]
    lost   = [k for k in common if p_b_by_prompt[k]["passed"] and not p_a_by_prompt[k]["passed"]]
    stable_pass = [k for k in common if p_b_by_prompt[k]["passed"] and p_a_by_prompt[k]["passed"]]
    stable_fail = [k for k in common if not p_b_by_prompt[k]["passed"] and not p_a_by_prompt[k]["passed"]]

    out(f"- Gained (newly passing): **{len(gained)}**")
    out(f"- Lost (newly failing):   **{len(lost)}**")
    out(f"- Stable pass:            {len(stable_pass)}")
    out(f"- Stable fail:            {len(stable_fail)}")
    out()
    if gained:
        out("### Newly passing (CPT installed)")
        for k in gained:
            shape = p_a_by_prompt[k]["shape"]
            out(f"- [{shape}] {k}")
        out()
    if lost:
        out("### Newly failing (CPT broke or perturbed)")
        for k in lost:
            shape = p_b_by_prompt[k]["shape"]
            out(f"- [{shape}] {k}")
        out()

    # Save report next to the "after" dir
    report = args.after.parent / f"compare_{label_b}_vs_{label_a}.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {report}")

if __name__ == "__main__":
    main()
