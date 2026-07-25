# Adversarial-loop evidence — r2 (landed) and r3 (pending)

Produced by the kandi adversarial loop, 2026-07-25. Kept because the receipts cite it and
because r3 is NOT yet landed.

## Evidence trail
- `loop_results.json`        — round 1: premise audits, first r2 artifact, first 3 reviews
- `loop_results_round2.json` — round 2: both LAND_WIDENED artifacts + the 4 MISFIRES reviews
- `measure_*.py`, `probe_entity_lever.py` — the auditors' own measurement probes, reusable

## r3_pending/ — NOT LANDED
The round-2 coverage-ruler artifact, sha-verified against the agent's claim
(`audit_recall.py` 8a0002c30f69…, `test_audit_recall.py` ab944292fe97…,
`kanto.ruler.json` 1ee77f7c6666…), plus unified diffs vs HEAD and its receipt.

Two reviewer objections are open and must be fixed before it lands:
1. The `fixture` pin is unconditionally enforced, so the PRODUCTION call (`facts_json=None`)
   returns FAIL / `clean_ratio: null`. Compare it only when `--facts-json` was supplied.
2. The document corpus is an unpinned fifth input — parking 104 of 105 `.md` files reproduces
   the full exploit with all four current pins byte-identical. Add a `documents` pin.

## rejected_wgkoselu4_worktree/ — rejected, do not resurrect
An earlier r3 attempt that an agent wrote directly into the repo in violation of the
no-repo-writes rule. Rejected on measurement, not preference — see `WHY_REJECTED.md`.

## r2 (landed)
`generic_reasoning.py` + its suite now live in `scripts/pipeline/`; the story, the shas and the
76 -> 559 holdout-bridge delta are in `logs/phase_r2_reasoning_surfaces.json`. The round-1
deterministic-template version is not kept — it was superseded by the LLM-channel rewrite, and
why it was wrong is recorded in the receipt's `provenance`.
