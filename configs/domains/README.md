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

The structured extractor/validator, generic hydrate/synth/contrast/reasoning generators, probe and
probe-quarantine tools, CPT + SFT trainer (`train_cpt.py`, FP4 + int8-master), tokenizer, SFT
anti-hallucination levers (`build_sft.py`), and dual-axis eval machinery. They read the manifest and
template content rather than a hardcoded domain.

## What's domain-SPECIFIC (you supply per domain)

1. **The manifest** (`<name>.yaml`) — `recipe_profile`, `supports`, hierarchy, registry, model/config
   references, ordered stages, top-level `entity_types`, and tiered `predicates`.
2. **The templates** (`configs/templates/<name>.yaml`) — hydrate floor, SFT Q/A, synthesis framing,
   optional contrast groups, and domain-authored probe controls.
3. **The registry and sources** — the unit list plus either a scraper module or flat files under
   `data/<name>/<unit>/00_raw`.

The shared `schema/` validates record shape only. Domain entity and predicate enums are derived from
the manifest; a new domain does **not** add a JSON schema or edit Rust taxonomy code.

## Adding a new domain (today)

1. Copy `configs/domains/_TEMPLATE.yaml` and `configs/templates/_TEMPLATE.yaml` — never copy EDC,
   whose bespoke festival stages are intentionally exceptional.
2. Fill the ontology (`entity_types`, predicate object contracts, and explicit `tier: hot|stable|proven`),
   declare `recipe_profile`/`supports`, author the unit registry, and wire the source path.
3. Run `run_pipeline.py --domain <name> --check`, then inspect `plan_install.py --domain <name>`.
   A facts recipe with no stable/proven predicates installs nothing and must not reach training.
4. Execute stage-by-stage through the contract runner. For a new held-out series use split v3 and
   run `probe_quarantine` before quoting install-strength.

The runner is unchanged across domains — only the manifest (+ ontology) differs.
