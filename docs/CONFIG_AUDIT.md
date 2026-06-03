# CONFIG_AUDIT.md — kandi-cpt configuration & CLI drift: audit + one-contract design

**Status:** Phase 1 (audit) + Phase 2 (design) — *for review before any code change.*
**Method:** 9-slice read-only multi-agent audit + completeness critic, every claim cross-checked
against the actual files (4 training YAMLs, `train_cpt.py`, `build_sft.py`, `eval_sft_calibration.py`,
the domain manifest, the gate thresholds — all verified). Reference pattern: `qgre/config.py`
(`/home/marcos/Documents/dev/qgre/qgre-main/qgre/config.py`). Source-of-truth memories honored:
Eli concept #753 (CPT methodology + eval invariants), idea #40 (experiment registry), idea #342
(compare_evals / best_ckpt drift-tracking).

---

## 0. The one-paragraph diagnosis

The **architecture is already clean at the core**: there is exactly **one trainer**
(`scripts/train_cpt.py`), **one env-wrapping launcher** (`scripts/train_fp4.sh`), and `serve.py` is
verifiably launch-free. Every launch path terminates in `train_fp4.sh → train_cpt.py`; no launcher
rolls its own training loop (the QLoRA-era `train_loop_iter.py` / `autonomous_loop.py` from #753 are
already deleted). **The drift is not duplicated logic — it is four things:** (1) a **16-flag override
surface** on `train_cpt.py` that lets any run change without touching a config (the literal disease —
`run_v9.sh` / `run_lever1_sweep.sh` lean on it); (2) **stale hardcoded constants in bash drivers** that
no longer match `max_steps` (`run_lever1_sweep.sh` greps `step 240/240` vs canonical 3160 → marks every
successful run FAILED; `run_v9.sh` writes `total_steps=450` vs `sft_edc.yaml` 400); (3) a **stale domain
manifest** (`configs/domains/edc.yaml`) pinning `edc_all97_v1` / `sft_v2` and calling a `scripts/scrape.py`
**that does not exist**; (4) **dead config + multi-located constants** (`optimizer_8bit` and
`lever4…threshold` are never read; gate thresholds live in 3 places; the FP4 toolchain env has 2
disagreeing sources). The fix is one typed config contract + one training entrypoint + one eval
entrypoint + a run registry, with the invariant persisted in `AGENTS.md` so a compacted session re-reads it.

---

# PART 1 — AUDIT

## 1.1 Entrypoint inventory (23 surfaces)

**Verdicts:** `keep-canonical` = stays as the one true surface · `config-generator` = becomes a script that
writes a config and calls the entrypoint (no behavior flags) · `delete` = redundant · `fix-stale` = drifted
constant to re-source from config.

### Training launch paths (all terminate in `train_fp4.sh → train_cpt.py`)
| File | Verdict | Note |
|---|---|---|
| `scripts/train_cpt.py` | **keep-canonical** | THE trainer. CPT/SFT via implicit dual-detect (`init_from` + `labels` col). Sheds its 16 override flags (§2.3). |
| `scripts/train_fp4.sh` | **keep-canonical** | THE launcher. FP4/CUDA-12.9 toolchain env (load-bearing, intentionally outside config). |
| `scripts/run_sft.sh` | **keep-canonical** | Atomic train→GATE (the no-ungated-SFT rule). |
| `scripts/run_pipeline.py` + `configs/domains/edc.yaml` | **fix-stale** | Manifest driver; stale vars + a `scrape.py` that doesn't exist. |
| `scripts/run_v9.sh` | **config-generator** | Bakes the SFT recipe as flags; stale `total_steps=450`. → emit a config. |
| `scripts/build_v12.sh` | **config-generator** | Already mostly there (uses `--config v3`); hardcodes 3160 / 12k–22k. → re-source from config. |
| `scripts/run_lever1_sweep.sh` | **config-generator** | Sweeps `--lever1-weight`; greps stale `step 240/240`. → emit N configs. |

### Eval surfaces (9 — keep as **libraries** behind one dispatcher; never merge their internals)
| File | Role (preserve-invariant) |
|---|---|
| `eval_sft_calibration.py` | THE GATE: `PASS iff fab≤0.25 ∧ acc≥0.55 ∧ cov≥0.40`, seeded/greedy/deterministic. |
| `eval.py` | Voice (temp 1.2, **production** prompt) **+** memory-mode (temp 0.3) probes — two-prompt separation. |
| `eval_cpt.py` | Greedy keyword-graded era-tagged (2026 / 2026ood / stable) probes. |
| `eval_cloze.py` | Forced-choice continuation-likelihood, **no chat template**, training surface; `_validate_batched` gate. |
| `eval_artist_edition.py` | artist→edition @1/@3/MRR. |
| `eval_cross_edition.py` | venue @1/MRR across 97 editions (reuses cloze ranker). |
| `eval_sft_val.py` | token-weighted CE, nan-guarded min. |
| `eval_sft_gen.py` | 6 hardcoded Q/A eyeball gen. |
| `compare_evals.py` | **standalone** two-dir before→after markdown diff (#342). Stays separate from the registry. |

### Other
| File | Verdict |
|---|---|
| `serve.py` | **keep-canonical** — launch-free (only `nvidia-smi`; GET-only). Reads `train_status.json` + scorecards. |
| `build_sft.py`, `tokenize_corpus.py`, `data_audit.py` | **keep-canonical** — data producers; their live recipe is single-sourced into the contract. |
| `rehydrate_vc1.sh` | **keep-canonical** — supersedes `hydrate_sweep.sh` (mtime-staleness + libsecret key). |
| `hydrate_sweep.sh` | **delete** — superseded; its `>100 lines` skip is the bug that blocked VC1 re-hydration. |
| `fp4_env.sh` | **reconcile** — 2nd FP4 env source (cu13 / MAX_JOBS=4) disagreeing with `train_fp4.sh` (12.9 / 2). |
| `smoke_int8_master.py`, `smoke_int8_subclass.py`, `fp4_smoke.py` | keep — standalone kernel-path validators. |

## 1.2 The knob universe — 43 semantic knobs, **18 drifting**

Two distinct kinds of "drift" — separate them, because the fix differs:

### (A) STRUCTURAL DRIFT — the disease (un-codified, stale, or broken)
| Knob | Where it disagrees | Truth |
|---|---|---|
| **the 16 override flags** | `train_cpt.py:665–694` let `--precision/--save-dir/--int8-master-mode/--max-steps/--tokenized/--init-from/--micro-batch-size/--grad-accum/--lever1-weight/--lever4/--optimizer-lowbit/…` change a run **without the config** | **Remove** (§2.3). Sweeps → config-generators. |
| **`optimizer_lowbit`** | argparse default `0` vs every YAML `4` | canonical **4**; the `0` argparse default is a trap. |
| **step-horizon markers** | `run_lever1_sweep` greps `240/240`; `run_v9` writes `total_steps=450`; `build_v12` bakes `3160` | re-source from `train.max_steps`. |
| **`save_dir` / `init_from` / `tokenized`** | manifest pins `edc_all97_v1` / `sft_v2` / `corpus_all97`; live is `v2` / `sft_v11` / `corpus_all97_v2` | manifest is **stale**; re-point or mark as v1-repro. |
| **FP4 toolchain env** | `train_fp4.sh` (CUDA-12.9, MAX_JOBS=2) vs `fp4_env.sh` (cu13 wheel, MAX_JOBS=4) | reconcile to `train_fp4.sh`. |
| **gate thresholds** | `eval_sft_calibration.py` TAU + `serve.py:43` + `serve.py:254` (triple-located) | single `GateThresholds` source (§2.6). |
| **`OPENROUTER_API_KEY`** | `hydrate_sweep`(required) vs `rehydrate_vc1`(env→libsecret) vs `synth_corpus`(env→libsecret) | one resolver. |
| **`scripts/scrape.py`** | `edc.yaml` scrape stage calls it — **the file does not exist** (real: `python -m scrape --city --year`) | fix the manifest stage. |
| **dead config** | `optimizer_8bit` (0 reads), `lever4…threshold:0.8` (0 reads, dup'd in `lever4_exposure_report.py:50`) | mark DEAD + DeprecationWarning. |

### (B) INTENTIONAL DIVERGENCE — correct, but spread across 4 files (to be *codified*, not removed)
These differ **on purpose** between CPT and SFT and are documented in YAML comments; the contract makes
them explicit mode-aware defaults instead of 4 hand-edited files:

| Knob | CPT | SFT | finish |
|---|---|---|---|
| `lr` / `min_lr` | 5e-5 / 5e-6 | 2e-5 / 2e-6 | 1.64e-5 / 5e-6 |
| `warmup_steps` | 50 | 20 | 0 |
| `max_steps` | 3160 | 400 | 160 |
| `decay_frac` | 0.2 | 0.2 | 1.0 |
| `seq_len` | 512 | 256 | 512 |
| `save_every` | 100 | 100 | 200 *(>max_steps on purpose → only final save)* |
| `nvfp4.bf16_first/last_layers` | 2 / 5 | 0 / 0 | 2 / 5 |
| corpus recipe (`answer_aug`/`absent_neg`/`abstention_frac`) | — | **8 / 125 / 0.0** (live V9; `build_sft` argparse defaults 2 / 0 / 0.0 are the naive floor) | — |

## 1.3 Redundancy verdicts (9) & dead/inert config
- **delete:** `hydrate_sweep.sh` (superseded by `rehydrate_vc1.sh`).
- **fold (A/B variants):** `train_edc_lv_2026_v3.yaml` differs from base only in `run_name/save_dir/tokenized`;
  `run_v9.sh` / `run_lever1_sweep.sh` are recipe-memory chains. → mode-aware config + config-generators.
- **fix-stale:** `configs/domains/edc.yaml` vars + the broken `scrape.py` stage.
- **delete-key (dead):** `optimizer_8bit` across all 4 YAMLs (zero reads).
- **keep-but-inert:** `levers.lever2/lever3/lever5` subtrees — declared with full sweep knobs but **unimplemented
  in `train_cpt.py`** (only lever1 + lever4-`enabled` are wired). Keep as documented-future, flagged inert.

## 1.4 PRESERVE-INVARIANTS — 18 behaviors the refactor must not change
Eval correctness (#753): **(1)** era-tagged probes; **(2)** deflection-aware + word-boundary grading;
**(3)** **separate voice (production prompt) vs memory-mode system prompts** — never merged; **(4)** THE GATE
FORMULA `fab≤0.25 ∧ acc≥0.55 ∧ cov≥0.40`; **(5)** gate determinism (seeded probe selection + greedy gen);
**(6)** NFKD gold-fold before lowercasing; **(7)** CPT-faithful forced-choice surface (no chat template);
**(8)** nan-guard early-stop; **(9)** dataset disjointness / no-leak (venue templates exclude the held-out probe);
**(10)** the three deflection detectors stay distinct.
Precision/engine: **(11)** CPT-vs-SFT **dual implicit detection** (`init_from` + `labels` col) — never replaced
by a flag; **(12)** `int8_master` resolution precedence (`--mode-flag > legacy bool > cfg`); **(13)** int8-master
checkpoint write/read contract in lock-step (`_data` int8 + per-row `.weight_scale` + `quant_keys.json` ↔
`causal_bf16_state_dict`); **(14)** NO-FP32 runtime guard (non-bf16 trainable param raises); **(15)** precision
3-way switch consumed after load+freeze+cast; **(16)** `freeze_embeddings` freezes all `embed`-named params.
Ops/safety: **(17)** `save_dir` travels with config (no save_dir ⇒ trains & saves NOTHING); **(18)** no-ungated-SFT
(atomic train→gate) · single-GPU 1200W-PSU rule (`CUDA_VISIBLE_DEVICES=0`, GPU-idle guard) · protect
`las_vegas/2026` (never re-hydrate the hand-built corpus).

## 1.5 The generalized data pipeline (scrape → train), 8 stages
`run_pipeline.py` already chains these from `configs/domains/<domain>.yaml`. The festival-agnostic seam is the
domain manifest; the contract's `CorpusSynthesisConfig` single-sources the LLM/hydration knobs now scattered
across `synth_corpus.py`, the Rust `festival-factory`, and the shell drivers.

| Stage | Entrypoint | Key knobs (→ generalize into domain manifest + CorpusSynthesisConfig) |
|---|---|---|
| `00_raw` scrape | `python -m scrape` | `--city/--year/--status/--sources` (mfw,official,youtube,reddit,news) |
| `10_clean` extract | `python -m extract` | `--sources/--limit/--llm-rescue` |
| `curate` (GPU) | `curate_corpus.py` | `--drop-frac 0.10/--clusters 20/--carve-holdout` |
| `15_structured` | `extract_structured.py` | `--min-tier 3/--concurrency 16` |
| `meta` | `gen_festival_meta.py` | reads `editions_edc.json` + `festival_meta.json` CANON_STAGES |
| `25_hydrated` | `rehydrate_vc1.sh` → festival-factory | `gen_model=gemma-4-31b-it/base_n 8/concurrency 10/parallelism 8` |
| `20_synthetic` | `synth_corpus.py` | `MODEL/MAX_CONCURRENT 6/CHUNK_TOKENS 800/TEMP 0.85/STRATEGIES` |
| `30_tokenized` | `tokenize_corpus.py` | `--seq-len 512/--replay-frac 0.10` (Rust parity exists) |

---

# PART 2 — DESIGN: one config contract, one entrypoint each

## 2.1 Shape (matches `qgre/config.py` exactly)
- **Module:** `scripts/cpt_config.py` — **stdlib `@dataclass`** (NOT pydantic), nested sub-dataclasses via
  `field(default_factory=SubConfig)`, `Literal[...]` for enums.
- **Load:** `CptConfig.from_yaml(path, mode)` → `yaml.safe_load` → explicit `_from_dict` hydrator (domain-specific
  errors + int-coerce of dict keys) → `validate()`. `_pick()` intersects raw keys with `__dataclass_fields__`
  and **`warnings.warn`-drops unknown keys** — strictly better than today's silent `cfg.get` ignore (a typo'd
  `max_step` now warns instead of vanishing).
- **Two-tier validation:** per-dataclass `__post_init__` = single-field bounds/enums; top-level `validate()` =
  cross-field + mode-gated required fields, **with the exact `run_contract` error strings preserved** (fail-loud
  is kept, not weakened). World-dependent gates run in `validate_runtime(args, world)` *after* env-read (so
  `int8_master=storage`'s `world==1` check doesn't fire before `torchrun` sets WORLD_SIZE).
- **Format:** **YAML stays** for human-authored configs (the inline comments are load-bearing institutional
  memory — *why* each field diverges; JSON would delete them). JSON is only for machine artifacts:
  `config.resolved.json` (frozen) + `schema/cpt_config.schema.json` (emitted).

## 2.2 Six sub-configs (defaults = **current real behavior**, verified against `train_cpt.py` + the YAMLs)
| Sub-config | Key fields (default) | Validation |
|---|---|---|
| **ModelConfig** | `model_path=…/gemma-4-E2B-it`; `freeze_embeddings=True`; `cpu_offload_ple=True`; `nvfp4{skip_lm_head=True, skip_embeddings=True, bf16_first=2/0, bf16_last=5/0}` | `0≤first+last≤35` (Gemma4 text layers) |
| **DataConfig** | `tokenized`(req non-smoke); `init_from=None` (the mode switch); `dataset_hash=None` (freeze-time fingerprint) | non-smoke ⇒ tokenized exists; init_from ⇒ exists |
| **CorpusSynthesisConfig** | `gen_model=gemma-4-31b-it` (kept **distinct** from `model_path`); `hydrate_base_n=8/concurrency=10/parallelism=8`; `replay_frac=0.10`; `answer_aug=8`; `abstention_frac=0.0`; `absent_neg=125`; `fake_city_neg=0`; `fine_grained_neg=0`; `max_per_relation=6`; `sft_seq_len=256`; `score_samples=4`; `score_batch=16`; `knowledge_filter/self_gen/drop_generic=True`; `score_model=edc_all97_v1` | `(knowledge_filter∨self_gen)⇒score_model`; `sft_seq_len==train.seq_len` when sft |
| **TrainConfig** | `precision=fp4`; `int8_master=storage`; `optimizer_lowbit=4`; `int8_grad_accum=False`; `lr/min_lr/warmup_steps/max_steps/decay_frac/seq_len` (mode-aware); `micro_batch_size=2`; `grad_accum=8`; `betas=[0.9,0.95]`; `grad_clip=1.0`; `gradient_checkpointing=True`; `optimizer_8bit=False` (**DEAD**, DeprecationWarning) | enums; `min_lr≤lr`; cross-field below |
| **EvalConfig** | `tau_fab=0.25/tau_acc=0.55/tau_cov=0.40`; `n_absent=90/seed=42/scale=1.0/batch=16`; `device=cuda:0`. Generation temps stay **hardcoded per script** (determinism contrast is load-bearing) | three taus independent; none droppable |
| **RunConfig** | `run_name`; `save_dir`(req non-smoke); `log_every=5`; `save_every=100`; `seed=42`. Toolchain env + `CUDA_VISIBLE_DEVICES` are **intentionally NOT config** (load-bearing shell env; captured as observed metadata in the freeze) | non-smoke ⇒ save_dir set (verbatim run_contract error) |

## 2.3 The hard rule (Phase 3) — **only `--config` + `--mode`**
> *Honoring your mandate over the audit's "keep the flags for sweeps" suggestion.*

`train_cpt.py` and the new `eval.py` accept **only `--config <path>` and `--mode {cpt,sft}`** (mode optional —
the config carries it; the flag is the "if the config does not carry it" escape). **All 16 behavior-override
flags are removed.** Anything that changes a run goes in the config file. Consequence:
- `run_lever1_sweep.sh` / `run_v9.sh` / `build_v12.sh` become **config generators**: they write a resolved YAML
  per variant (e.g. one per lever1 weight) and call `train_fp4.sh --config <generated.yaml>` N times. They hold
  no training logic and inject no behavior flags — exactly your "`autonomous_loop` becomes a config generator"
  rule.
- **`mode` is a declared-intent cross-check, NOT a replacement** for the dual implicit detection (`init_from`
  presence + `labels` column stay the runtime truth; `validate()` errors if `mode` disagrees). Preserve-invariant #11.
- Two **operational** (non-run-configuring) flags survive by exception, for you to confirm: `--dry-run` (validate
  + print contract + exit; never trains) and `--smoke N` (synthetic-data loop sanity). Both can instead become
  `run.dry_run` / `run.smoke_steps` config fields if you want the signature to be *literally* `--config --mode`.

## 2.4 One eval entrypoint
New thin `scripts/eval.py` dispatcher (`--config --mode --checkpoints`): `mode=sft` → `eval_sft_calibration`
(the canonical gate, never inlined); `mode=cpt` → `eval_cpt` + `eval_cloze` install probes. The 9 eval scripts
stay as **importable libraries** — their determinism, the three deflection detectors, and voice-vs-memory prompt
separation are **not unified**. `compare_evals.py` stays standalone (#342). *(The existing `eval.py` voice prober
is renamed `eval_voice.py` to free the dispatcher name — pending a grep for `import eval` to confirm no breakage.)*

## 2.5 Reproducible runs (Phase 4) — honoring idea #40 + #342
On every run, freeze into `logs/runs/<run_id>/`:
- `config.yaml` (verbatim source) · `config.resolved.json` (`asdict()` after override-merge — what actually ran) ·
  `metadata.json` `{config_fingerprint:16hex, git_sha, dataset_hash, cuda_visible_devices, started, status}` ·
  `summary.json` (best_val_loss+step, gate fab/acc/cov verdict, peak_mem) · `README.md` · `train.log`.
- **Registry:** `logs/runs/events.jsonl` (append-only source of truth) + `logs/runs/registry.db` (SQLite,
  **rebuildable** by full-scan of the jsonl). Tables: `runs`, `metric_series`, `run_summaries`.
- **Fingerprint** (idea #40 identity): `sha256` over a *stable subset*
  `{precision,int8_master,optimizer_lowbit,lr,min_lr,warmup,max_steps,decay_frac,mbs,grad_accum,seq_len,betas,
  weight_decay,model_path,tokenized,init_from,mode}`[:16] — excludes volatile `save_dir/run_name/timestamps`.
  `tokenized→dataset_hash` so the re-hydrated v3 corpus yields a different fingerprint.
- **σ-regression** (idea #342): a new run >2σ worse than its fingerprint-matched baseline window is flagged.
  `compare_evals.py` + `best_ckpt` keep working against this layout (the run dir holds each eval's output).

## 2.6 Schema + drift gates (Phase 5)
`schema/cpt_config.schema.json` emitted from the dataclasses (editor validation + the config `--check` gate).
Plus three **pydantic wire models** (`extra='forbid'`, the ONLY pydantic in the repo) that make today's
comment-enforced contracts mechanical: **`GateThresholds`** (kills the `eval_sft_calibration`↔`serve.py:43`↔`:254`
triplication), **`TrainStatus`** (the `train_status.json` heartbeat written at `train_cpt.py:62` ↔ read by
`serve.py` — a field added on one side now fails `--check`), **`QuantManifest`** (the int8 `_data`+`.weight_scale`
write ↔ `causal_bf16_state_dict` read — invariant #13 made mechanical). `python -m cpt_events --check` nonzero-exits
on drift; `schema_version` stamped per event.

## 2.7 Migration map (Phase 3 work — for reference, not this turn)
| Script | Action |
|---|---|
| `train_cpt.py` | **modify in place** — `from_yaml` swap; remove 16 flags; `resolve_int8_master()` collapses the 612↔752 dup; add `--mode`. Dual-detect + NO-FP32 guard + KV-sharing untouched. |
| `eval.py` (NEW) | **create** — thin dispatcher. |
| `cpt_config.py` / `cpt_events.py` (NEW) | **create** — the contract + wire models. |
| `run_v9.sh` / `run_lever1_sweep.sh` / `build_v12.sh` | **→ config-generators**; drop hardcoded step markers. |
| `model_load.py` | keep — import `QuantManifest` instead of inline literal. |
| `train_fp4.sh` | **untouched** (toolchain env is load-bearing). |
| `fp4_env.sh` | reconcile CUDA_HOME/MAX_JOBS to `train_fp4.sh` (or document why smoke differs). |
| `configs/domains/edc.yaml` | fix stale vars + the non-existent `scrape.py` stage. |
| `hydrate_sweep.sh` | **delete**. |
| `eval_*.py` / `compare_evals.py` / `serve.py` | keep; import `GateThresholds`/`TrainStatus`. |
| orphans (`build_v6_corpus`, `equalize_corpus`, `build_schedule`, `hydrate_templates`, `geometry_probe/*`) | **out of scope** — flagged for you to delete-or-adopt, not silently folded. |

## 2.8 Persisted rule (Phase 6 — into `AGENTS.md` + the entrypoint header)
> All run configuration lives in the `CptConfig` object and one YAML config file. There is exactly **one training
> entrypoint** (`train_cpt.py`) and **one eval entrypoint** (`eval.py`), each taking only `--config` (+ `--mode`).
> **Do not add a behavior-affecting flag. Do not create a second entrypoint or a parallel trainer.** New knobs go
> in a sub-dataclass + the schema. Sweeps generate config files and call the entrypoint N times.

## 2.9 Open decisions (need your call before Phase 3)
1. **`--smoke`/`--dry-run`:** keep as the 2 operational exceptions, or fold into `run.smoke_steps`/`run.dry_run`
   so the signature is *literally* `--config --mode`? (Recommend: keep `--dry-run`, fold `--smoke`.)
2. **`unknown YAML key` policy:** warn-and-ignore (qgre default, forward-compat) or hard-error? (Recommend: warn.)
3. **Orphan one-offs** (`build_v6_corpus`, `equalize_corpus`, `build_schedule`, `hydrate_templates`,
   `geometry_probe/*`): delete now, or leave untouched out-of-scope? (Recommend: leave; list for a later sweep.)
4. **Scope of this PR:** land the contract + entrypoints + registry + schema as designed, in small reviewable
   commits — confirm before any code.
