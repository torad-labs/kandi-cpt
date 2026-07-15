# AGENTS.md — operating rules for kandi-cpt

This file exists so a future or context-compacted session re-reads the load-bearing constraints from the
repo instead of re-deriving (and re-breaking) them. Read it before touching training, eval, configs, or
launchers. The full audit + design rationale is in `docs/CONFIG_AUDIT.md`.

---

## THE CONFIG CONTRACT (the rule that must not erode)

**All run configuration lives in one typed object (`scripts/cpt_config.py`) and one YAML config file.**

1. **One training entrypoint:** `scripts/train_cpt.py`. It accepts **only** `--config <path>` and
   `--mode {cpt,sft}`. There are **no behavior-affecting flags.** Smoke / dry-run are config fields
   (`run.smoke_steps`, `run.dry_run`), not flags.
2. **One eval entrypoint:** `scripts/eval.py` (`--config --mode --checkpoints`). It dispatches to the
   eval libraries; it does not reimplement them. CPT configs with `eval.probe_domain` +
   `eval.probe_editions` select the generic base-vs-tuned forced-choice receipt; configs without them
   retain the EDC install-probe + cloze path.
3. **Do NOT add a behavior-affecting flag.** If you reach for `--something` to change a run, stop — add a
   field to the right sub-config in `cpt_config.py` (+ regenerate the schema) instead.
4. **Do NOT create a second entrypoint or a parallel trainer/eval.** There is exactly one of each.
   The QLoRA-era `train_loop_iter.py` / `autonomous_loop.py` are gone; do not resurrect that shape.
5. **Sweeps generate configs.** To vary a run (A/B, lever sweep, chain), write a config with
   `scripts/gen_config.py --base <cfg> --set section.key=value …` and call the entrypoint with `--config`
   only. See `run_v9.sh` / `run_lever1_sweep.sh` for the pattern. Never pass overrides to the trainer.
6. **New knobs go in the contract + the schema.** Add the field (with a validated default that reproduces
   current behavior) to its sub-config, then regenerate `schema/cpt_config.schema.json`.
7. **Unknown / typo'd config keys HARD-ERROR at load** (with a did-you-mean hint). A misspelled key never
   silently falls back to a default. If you add a key to a config, add it to the dataclass first.

The contract's sub-configs: `model · data · train · corpus_synthesis · eval · run`. `from_yaml()` validates
structure + cross-field at load; `validate_runtime(world, has_labels)` does the path/world/corpus gates.
`to_trainer_dict()` is an INTERNAL adapter (flat dict for the trainer plumbing) — not a config surface.

---

## CANONICAL LAUNCHES (zero flags)

- **CPT:** `bash scripts/train_fp4.sh`  (→ `configs/train_edc_lv_2026.yaml`)
- **SFT:** `scripts/run_sft.sh <label>`  (train → GATE atomically — see below)
- **V12 chain:** `scripts/build_v12.sh`  (re-tokenize → v3 corpus → re-CPT)
- **Full domain pipeline:** `PYTHONPATH=scripts .venv/bin/python scripts/run_pipeline.py --domain edc --from structured --to sft`

`scripts/train_fp4.sh` is THE launcher — it sets the FP4 / CUDA toolchain env (CUDA_HOME/CUDA_PATH, nvcc
-ccbin g++-14, CCCL guard, MAX_JOBS=2, GPU=0, allocator). It prefers CUDA 12.9 when that toolkit actually
has `nvcc`, then falls back to an installed CUDA >=12.9 toolkit such as CUDA 13.3 / the cu13 wheel toolkit.
That env is **intentionally NOT in any config** — it is load-bearing shell setup. Never train by sourcing
`fp4_env.sh` (that is the cu13 *kernel-smoke* path, not training).

---

## NON-NEGOTIABLE INVARIANTS (preserve across any refactor — full list in CONFIG_AUDIT.md §1.4)

- **Gate every SFT run.** No SFT checkpoint is ever left ungated. Use `scripts/run_sft.sh` (train→gate
  atomic) with the canonical `eval_sft_calibration.py`. THE GATE: PASS iff `fab ≤ 0.25 ∧ acc ≥ 0.55 ∧
  cov ≥ 0.40`, deterministic (seeded probe selection + greedy generation).
- **Eval correctness is sacred:** era-tagged probes; deflection-aware + word-boundary grading; the SEPARATE
  voice (production prompt) vs memory-mode system prompts must never be merged; NFKD gold-fold; the
  CPT-faithful forced-choice surface (no chat template); the nan-guard early-stop.
- **CPT-vs-SFT mode is detected by TWO implicit signals** — `data.init_from` presence + the corpus `labels`
  column — never by a flag. `mode` in the config is a declared-intent cross-check that errors on
  disagreement; it does not replace the detection.
- **Precision / engine moat:** native FP4 (Quartet-II) + `int8_master: storage` (true int8 1-byte master +
  SR, single-GPU + precision=fp4 only). NO-FP32 runtime guard: a non-bf16 trainable param raises. The int8
  checkpoint write/read contract (`_data` int8 + per-row `.weight_scale` + `quant_keys.json` ↔
  `causal_bf16_state_dict`) stays in lock-step.
- **`save_dir` travels with the config.** No `save_dir` ⇒ the run trains and saves NOTHING (and the status
  heartbeat never fires). The contract hard-errors on this.
- **Single-GPU 1200W-PSU rule:** never two FP4 trainers at once (PSU OCP → hard crash). `CUDA_VISIBLE_DEVICES`
  defaults to 0 everywhere; GPU-using scripts guard on GPU0 being idle.
- **Protect `data/edc/las_vegas/2026/`** — the hand-built rich corpus is never re-hydrated (would clobber it).

---

## DATA PIPELINE (generalizes across domains)

`00_raw (scrape) → 10_clean (extract) → curate → 15_structured → meta → 25_hydrated (hydrate) →
20_synthetic (synth) → 30_tokenized (tokenize)`. Driven by a domain manifest (`configs/domains/<name>.yaml`):
swap the hierarchy/registry/ontology/templates + point `configs.cpt`/`configs.sft` at the domain's configs.
The training stages are config-only; the data stages take the manifest's `vars`. The LLM data-gen recipe
(hydration model, negative doses, answer-aug) lives in `corpus_synthesis` in the config — diversity of
surface forms installs knowledge, not volume (docs/cpt_installation_best_practices.md).

---

## House rules (from the global agent guidelines, repeated here because they bite hardest in this repo)

- Experiments are CLI **config generations**, never permanent edits to a canonical config (and never flags).
- Commit/push only when asked.
- Before destructive ops (`rm -rf`, `pkill`, `kill -9`): check `nvidia-smi` for active compute (paid in-flight
  training) and `git status`. The §8 look-before-delete check is mandatory — the audit's "orphan" list was
  partially wrong (two "orphans" were live dependencies); always verify references before deleting.
