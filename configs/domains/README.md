# Domains — config-driven pipelines

A **domain manifest** (`configs/domains/<name>.yaml`) declares a full `scrape → CPT → SFT` pipeline as
config. `scripts/run_pipeline.py` reads it and runs the stages; the festival code is just the `edc`
*instance* of the abstraction.

```bash
# whole pipeline (GPU stages skipped unless --allow-gpu):
PYTHONPATH=scripts .venv/bin/python scripts/run_pipeline.py --domain edc --dry-run
# a window of stages:
PYTHONPATH=scripts .venv/bin/python scripts/run_pipeline.py --domain edc --from structured --to tokenize
# specific stages, specific units:
PYTHONPATH=scripts .venv/bin/python scripts/run_pipeline.py --domain edc --only meta,hydrate --editions las_vegas/2026
```

## What's domain-AGNOSTIC (reused as-is, the machinery)

The hydration/EntiGraph engine (`festival-factory hydrate`), the CPT + SFT trainer (`train_cpt.py`,
FP4 + int8-master, masked-label SFT), the tokenizer, the SFT anti-hallucination levers (`build_sft.py`),
and the eval-as-forced-choice pattern. None of these care about the domain content.

## What's domain-SPECIFIC (you supply per domain)

1. **The manifest** (`<name>.yaml`) — hierarchy levels, unit registry, model, configs, and the ordered
   stage command templates. Copy `edc.yaml` and edit.
2. **The ontology** — `schema/` entity/triple types + `taxonomy.rs` relation registry. Today these are
   festival-shaped (entity types `festival/edition/venue/stage/artist/…`, relations `plays_at/…`); a new
   domain needs its own entity-type enum + relation taxonomy. *(Lifting these into the manifest is the
   next industrialization step — currently referenced by `ontology:` but still hardcoded in code.)*
3. **The Q/A + correctness-floor templates** — `build_sft.py` question templates and
   `pipeline/fact_expand` / `disjoint_context` surface forms are festival-shaped.

## Adding a new domain (today)

1. `cp configs/domains/edc.yaml configs/domains/<name>.yaml`; set `hierarchy`, `registry`, `model_path`,
   `configs`, `vars`, and the stage `cmd` templates.
2. Provide its ontology (entity/relation types) + Q/A templates (the remaining hardcoded seams).
3. `run_pipeline.py --domain <name> --dry-run` to verify the plan, then run stage-by-stage.

The runner is unchanged across domains — only the manifest (+ ontology) differs.
