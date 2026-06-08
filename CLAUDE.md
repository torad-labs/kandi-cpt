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

`README.md`'s "State of play" (the `0/17`, QLoRA-NF4-on-16GB framing) is **historical** — superseded by the
fp4 full-param recipe below. Trust this file + `docs/` + the manifests for the current shape.

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
- Secrets: OpenRouter key from `$OPENROUTER_API_KEY` or libsecret (`service=openrouter`).

## Known issues (not yet fixed, by decision)

- `build_sft.py:119` — `meta.get("stages")` is a **dict**, so the "What stages does X have?" SFT answer emits
  snake_case slugs (`kinetic_field`) instead of display names. Pre-existing; **locked by the byte-identity
  golden** (fixing it changes the proven SFT corpus + the golden — an operator decision, deliberately
  deferred).
