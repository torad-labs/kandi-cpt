# kandi-cpt — agent context

Orientation for an agent landing in this repo. Read this before navigating the code; it captures the
architecture and the two non-negotiables that are NOT obvious from the files.

## What this repo is

The **CPT / knowledge-install training engine**: it turns raw documents about a subject into a
relationally-complete, faithfulness-gated corpus and continues-pretrains a small model on it, installing
that subject's knowledge into the weights. Python/PyTorch, GB-scale data, its own lifecycle. The
orchestration agent (`../../agents/qgre-agent`, a Kotlin `ToolLoopAgent`) DRIVES this engine via
`scripts/run_pipeline.py` + `bash`; it does not contain the training competence — that lives here and in
qgre-agent's `core-agent/knowledge/` playbooks. Base model: `~/Documents/dev/models/gemma-4-E2B-it` (Gemma 4 E2B).

`README.md` is the **agent map** (layout, cloud smoke recipe, CodeQL, commands). This file is the **law**
(precision, domain-general architecture, GPU safety). If they disagree on methodology or precision,
**believe this file**. Receipts under `logs/phase_*.json` beat both for "what was measured".

## ⚠ The two things you MUST NOT break

1. **Precision is fp4 + int8/fp8 master — NEVER raise it.** Training runs in **NVFP4 (Quartet-II) fp4**
   weights with **int8/fp8 master weights + stochastic rounding + 4-bit optimizer states**, WSD schedule,
   wikitext replay. This is **proven, bit-exact, and the moat** — models have been trained this way
   successfully. `lm_head` is NOT trained. Do not silently upcast any tensor / optimizer state / master
   store / cast / `autocast` / `bf16=True` to bf16/fp16/fp32 "for stability" — it is not needed and it
   breaks a proven contract. Raising precision is a deliberate, human-only decision backed by a real
   divergent run, never a prior belief. (qgre-agent ships a `§precision-floor` PreToolUse hook that blocks
   precision-raising edits; override is `QGRE_PRECISION_OK=1`, operator-only.) This is global CLAUDE.md §15
   applied here — this repo is the thing §15 protects.

2. **EDC output is byte-identical — it is the hard gate.** This engine was festival-hardcoded; it is now
   **domain-general** (see below), and EDC (`configs/domains/edc.yaml`) is the **reference instance**. Every
   generalization is gated by per-generator **sha256 byte-identity tests** against frozen goldens. If you
   touch the generators, the gold gates MUST stay green — a diverging EDC byte is a regression in the proven
   pipeline. Run them: `PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_<x>_general.py` (and
   `_golden/*.json`). Also `scripts/qgre/test_gates.py` (gold quality gates).

## GPU safety (single 1200W-PSU workstation)

`nvidia-smi` before any heavy run. **Never two FP4 trainers at once** (PSU OCP) — `scripts/torad_actions.py`
holds an `O_EXCL` GPU lock (`data/_build/gpu.lock`) so an agent physically can't launch a second. GPU 1 is
for QGRE/training; GPU 0 is sacred (serve + SFT). Global CLAUDE.md §8 (process-safety checks) applies.

## The bf16 purge + v2 format (operator law, 2026-07-26)

§1's precision law, made structural: **no bf16 weight residency or storage anywhere in the loop** — not
"preferred low precision" but *bf16 does not exist as a residency choice* (brain law: "there is no place
where bf16 is required — it's a choice"). What that means concretely:

- **The eval instrument IS the training forward.** `pipeline.model_load.load_causal_ckpt_quartet` loads
  int8 masters straight into the torchao subclass and runs the **Quartet fp4 GEMM forward** — never a
  bf16 decode of the model. The scorer pads prompts to the fp4 kernel's 128-token multiple
  (`model._quartet_eval`); generation uses `_greedy_generate_quartet` (full-prefix recompute + pad —
  `model.generate` breaks the kernel's shape assert). Instrument numbers carry a stable ~−3.5pt offset
  vs the old bf16-decode series; series going forward are quartet-native, bf16 series are history.
- **Format v2** (`int8_rowwise_v2`): ALL layers' linears int8 (the v1 first-2/last-5 exemption is gone),
  embeddings (embed_tokens + PLE) int8 rowwise, **all scales fp8 e4m3** (scales cast to bf16 at load —
  e4m3 ⊂ bf16, exact; fp8 is storage-only). Base artifact: `models/gemma-4-E2B-it-int8-v2` (4.4GB vs
  9.3GB bf16; 4.64GB GPU-resident).
- **Convert ONCE, GPU-side.** `scripts/convert_base_int8.py` (base) and `scripts/convert_ckpt_v2.py`
  (trained checkpoints; masters pass through byte-identical) stream per-tensor: pread read → **quantize
  on GPU** (`quantize_int8_rowwise` is device-independent — bit-identical) → stream write. Row-chunked
  above 1.5GB so the 4.7GB PLE can't OOM the card. Per-run bf16 ingestion is dead; `build_model`
  init_from loads int8 checkpoints natively.
- **The gate**: `scripts/qgre/check_no_bf16.sh` (wired into `run_tests.sh`) FAILS any code-level bf16
  usage outside declared seams S1–S5 (ingestion seams, kernel-seam transients, selective-precision
  design, removed legacy, declared legacy instruments). New bf16 work happens in the open or not at all.
  Removed legacy (reintroduction fails): causal_bf16_state_dict, load_causal_ckpt_bf16/int8,
  Int8ScaledLinear, _save_bf16_hf, _sr_int8_master, eval_cpt.py. world>1 FSDP saves, int8_master:snap,
  and plain-checkpoint init all FAIL LOUD now — int8_master:storage, single-GPU, is the only path.
- **CUDA_HOME must be a cu13 toolkit** (`/usr/local/cuda-13.3`): the session baseline is toolkit-less
  12.8 and the flashinfer JIT cache was built with 13.3 — without it the fp4 kernel "rebuilds" and dies
  (`nvcc: not found`). Set it in every eval/quartet invocation.
- **Load reliability laws** (the 2026-07-25/26 incident nights): streaming save + `save_preflight` +
  oom-shield on the trainer (5/5 saves survived); pread-based shard reads (`_SafetensorsRead`) — mmap
  keeps shard pages in VmRSS = the whale-reaper's metric, and `posix_fadvise(DONTNEED)` cannot evict
  mapped pages (proven no-op here); `release_eval_model()` only works when the CALLER deletes its
  binding first.
- **Orchestration discipline**: heavy jobs run in DETACHED scopes (`systemd-run --user --scope` + memory
  cap + oom-shield) monitored by log tails — never terminal-hogging foreground GPU jobs. Receipt-carrying
  eval work gets oom-shield (the die-first evalgate class is for expendable work only).
- **Replay semantics correction** (external review, confirmed on the stats): `--replay-frac r` computes
  replay = domain × r, so the realized final share is **r/(1+r)** (nominal 5/10/20/30% → realized
  4.76/9.09/16.67/23.08%). Receipts fit on realized; stamp both.
- **Pending recipe migrations** (declared, not defaulted): e4m3 masters (#916 target), S5 instruments
  (eval_voice re-baseline, geometry_probe re-derive, roberta embedders stay), eval side-effects of the
  v2 re-baseline on the sensitivity exemption (evidence over prior, always).

## The pipeline (the stage-DAG)

`run_pipeline.py --domain <d>` reads `configs/domains/<d>.yaml`'s `stages:` (argv templates) and runs them.
`--check` is the R3 preflight (validate refs, then abort). `--from/--to/--only`, `--dry-run`, `--allow-gpu`.

```
scrape → extract(10_clean) → curate → structured(15_structured: entities/triples/discourse)
  → meta(festival_meta.json) → hydrate(25_hydrated) → synth(LLM diversity) → probes(eval items)
  → audit(gates) → tokenize(30_tokenized Arrow) → cpt(fp4 train) → build_sft → sft(+gate)
```

Per-edition data lives under `data/<domain>/<unit>/{00_raw,10_clean,15_structured,25_hydrated,...}`
(EDC: `data/edc/<city>/<year>/`; a flat domain: `data/<d>/<unit>/`). `iter_editions(domain)` walks the
registry (`configs/editions_<d>.json`) and yields an `Ed` per unit — handling BOTH registry shapes:
**festival** (`city` × `years`) and **generic** (a unit declares its own `unit`/`parts`). `Ed` carries
`parts` (the unit path); `.dir`/`.slug`/`.festival_slug` derive from it and `.city`/`.year` are back-compat
properties so the festival importers are unchanged. The unit shape is no longer hardcoded to city/year.

## The de-festivalization architecture (the load-bearing idea)

**A domain is data, not code.** EDC is one *instance*; the engine trains ANY subject. A new domain authors
ONLY **content** — never engine Python:

| You author (content) | Engine reads it via | 
|---|---|
| `configs/domains/<d>.yaml` — `entity_types` + `predicates` (the machine ontology), `stages`, `registry`, `relations`/`quality` | `pipeline/domain.py` (`Domain`), `pipeline/ontology.py` (builds the extract prompt), `pipeline/schema_validate.py` (derives validation enums) — **no per-domain JSON schema** |
| `configs/templates/<d>.yaml` — `meta_scaffold`, `sft_templates.rel_q`, `hydrate_templates.predicate_templates`, `synth_templates.system_preamble` | `Domain.templates` |
| `scripts/scrape/sources/<d>.py` — the source module(s) (**the only real code**, the acquire stage) | `scrape/__main__.py` SOURCES |

**Skeletons to copy** (the "set up for success" surface): `configs/domains/_TEMPLATE.yaml`,
`configs/templates/_TEMPLATE.yaml`, `scripts/scrape/sources/_TEMPLATE.py`, and the qgre-agent authoring
playbook `core-agent/knowledge/domains/_TEMPLATE.md`. Onboarding = copy these, fill the ontology + templates,
write the scraper, `run_pipeline.py --domain <d> --check` then run.

**Generic engine tools (domain-agnostic — what a new domain uses):**
- `scripts/pipeline/generic_expand.py` — deterministic per-predicate hydrate FLOOR (renders
  `predicate_templates` over every triple; faithful by construction).
- `scripts/pipeline/generic_synth.py` — LLM DIVERSITY layer (per-fact paraphrases; the install layer, #760;
  the generator is injectable so it tests without API calls).
- `scripts/pipeline/generic_probes.py` — forced-choice eval-item generation over any predicate.
- All three: load triples generically, render the domain's templates, and **report drops loudly** (a
  template that renders 0/N triples, an empty LLM response, a probe with no distractor → a `WARNING` to
  stderr; in a training pipeline a silent drop is an invisible hole in the model's knowledge).

**The executable non-EDC gate.** The byte-goldens only ever run EDC, so a festival assumption that leaks onto
the generic path is invisible to them. `scripts/pipeline/test_selftest_domain_e2e.py` is the counter-instance:
it runs a committed, deliberately non-festival fixture domain (`_selftest` — a single-level `[collection]` of
scientific instruments, zero festival tokens) through the REAL seams (`iter_editions` → load → generator
`main()` → write) and asserts a non-empty corpus with ZERO festival bleed + `--check` green-on-filled/red-on-
placeholder. A festival default that leaks onto the generic path turns it RED. Run the whole suite (the
goldens + the e2e) with **`scripts/run_tests.sh`** (auto-discovers `scripts/pipeline/test_*.py`).

**Festival-SPECIFIC tools (EDC's instance — a new domain does NOT use these; it uses the generic path):**
`scripts/pipeline/fact_expand.py` (the artist→stage→day specialist, with `Ctx`/`context_from_meta` render
context + `contrasts.py` + `disjoint_context.py`), `scripts/synth_corpus.py` (concurrent LLM synth),
`scripts/build_sft.py`'s anti-hallucination NEGATIVES (`absent_edition`/`fake_city`/`fine_grained`), the
`scripts/extract_*_lineup.py` acquirers, and the **Rust `festival-factory/`** (the cross-edition hydrator —
its TYPES model festivals; see `festival-factory/DE-FESTIVALIZATION.md` for why it is correctly
festival-specific and a new domain uses `generic_expand` instead). The master-plan principle: *a domain swap
is real authoring, not a `--domain` flag.*

## Conventions / gotchas

- **Run scripts with `PYTHONPATH=scripts .venv/bin/python scripts/…`** (config derives ROOT from `__file__`).
- **Tests are plain-assert + a `__main__` runner — there is NO pytest in the venv.** New tests mirror
  `scripts/pipeline/test_*_general.py` (and freeze goldens under `scripts/pipeline/_golden/`).
- **Config-driven from the manifest** is the rule: extraction prompt, validation enums, gates, templates all
  come from `configs/domains/<d>.yaml` + `configs/templates/<d>.yaml`. Don't hardcode a domain's enum/strings
  into shared code.
- The quality gates (`scripts/qgre/gates.py`, `data_audit.py`) read the manifest's `relations`/`quality`
  (dominance, `min_active_types`, `exposure_cap`, id-integrity contract, `year_bearing_types`). They are
  domain-general; festival-axis checks (year-bearing types) are manifest-declared, empty → skipped.
- **Commits are LOCAL on `main`** by default; pushing is the operator's call. Keep `git status` clean before
  destructive ops; permanent artifacts (shards/checkpoints) go in-project, never `/tmp`/`~/.cache` (§14).
- Secrets / LLM provider (permanent): **Google AI Studio is plan A**, OpenRouter is plan B.
  Never prefer OpenRouter when a Studio key exists. Resolve via `pipeline.llm_provider`
  (Python) / `festival-factory` chat client (Rust):
  `$GOOGLE_AI_API_KEY` / `$GEMINI_API_KEY` / libsecret `service=googleai` → else
  `$OPENROUTER_API_KEY` / libsecret `service=openrouter`. Force with `$LLM_PROVIDER=googleai|openrouter`.
- **GPU/CUDA kernel work (the megatrainer's future) → read `docs/KERNEL-ENGINEERING-PLAYBOOK.md`** first:
  the reusable kernel techniques distilled from the qgre-main verify campaign (exhaustive small-domain
  bit-exact proofs + byte-identity-gated default-off levers; `byte_perm` register-LUT and
  accumulate-wide-store-narrow idioms; the scale-first-vs-last scheme fork; per-op skip-attribution; the
  "does the constraint *walk*?" measurement test). It pairs with this repo's §1 precision moat — the only
  sanctioned widening is the *transient exact accumulator*, never a stored-precision raise. (Canonical
  copy + the megakernel cascade model live in `qgre-main/docs/`.)

## Known issues (not yet fixed, by decision)

- `build_sft.py:119` — `meta.get("stages")` is a **dict**, so the "What stages does X have?" SFT answer emits
  snake_case slugs (`kinetic_field`) instead of display names. Pre-existing; **locked by the byte-identity
  golden** (fixing it changes the proven SFT corpus + the golden — an operator decision, deliberately
  deferred).

## Host laws (mechanically enforced — see AGENTS.md "Host memory & GPU laws")

Swap topology, build gating, GPU-binary sanitizer receipts, and sysctl are
host-ledger territory. Claude sessions have these blocked at the tool layer;
the AGENTS.md section documents the same laws for every other runtime.
