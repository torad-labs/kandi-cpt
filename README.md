# kandi-cpt

Continued-pretraining (CPT) workspace for **Kandi** — installing EDC /
festival knowledge into **Gemma 4 E2B** for the on-device chat app.

Separate from the app repo by design (Python/CUDA toolchain, GB-scale data,
its own lifecycle). The app repo (`apps/kandibuddy`) owns the *contract* and
*deployment*; this repo owns the *corpus* and *training*.

## State of play

The training methodology is **validated**; the corpus is the **bottleneck**.

- An autonomous ~10-config sweep (LR × epochs × LoRA rank × embed strategy)
  ran end-to-end: tokenize → train → eval → compare.
- **Voice**: 20/20 every config — but voice is handled at runtime via sampling
  (`temp=1.2`, `top_k=64`), not training.
- **Knowledge install**: `era=2026` probes **0/17 every config.** Root cause is
  scale — ~78–81K total tokens, each fact seen ~5×. Knowledge install needs
  ~10M+ tokens / ~1000 passes per fact (Allen-Zhu, *Physics of Language
  Models*). ~100× short. QLoRA NF4 noise on 16 GB is a co-factor.

Full write-up: [`logs/OVERNIGHT_FINDINGS.md`](logs/OVERNIGHT_FINDINGS.md).

## Layout

Partitioned per **festival / city / year**; pipeline stages live inside each edition.
Full map + lifecycle + "safe to delete?" table: [`docs/ORGANIZATION.md`](docs/ORGANIZATION.md).

| Path | Tracked | Contents |
|---|---|---|
| `data/<festival>/<city>/<year>/` | | one edition, e.g. `data/edc/las_vegas/2026/` |
| `  00_raw/` | ✗ | web downloads, source-faithful |
| `  10_clean/` | ✓ | cleaned .md by source (`official/`,`youtube/`,`reference/`) — **source of truth** |
| `  15_structured/` | ✓ | NEW FORMAT — `entities/triples/discourse` (substrate §2–§4) |
| `  20_synthetic/` | ✓* | derived expansion (EntiGraph / Nemotron / Phi strategies) |
| `  holdout/` | ✓ | eval carve-out (frozen pre-synth) + `probes/` |
| `data/_shared/` | ✓ | festival-agnostic: question_catalog · primitives · class_taxonomy · entities_global · knowledge · replay |
| `data/_build/30_tokenized/` | ✗ | cross-edition trainer shards (regenerable) |
| `scripts/` · `configs/` · `manifests/` | ✓ | pipeline · run configs · exposure ledger |
| `logs/` | ✓ | findings, per-checkpoint evals, comparisons |
| `checkpoints/` · `.venv/` | ✗ | trained adapters (~15 GB) · virtualenv |

`*` committed while small; → gitignored + external storage when large.

## Source of truth

- Plan: `apps/kandibuddy/docs/KANDI_CPT_PLAN.md`
- Probes: `apps/kandibuddy/docs/KANDI_CPT_PROBES.md`
- Base model: `~/Documents/dev/models/gemma-4-E2B-it/` (bf16, ~10 GB)

## Next

Organization: **done** — festival/city/year layout + lifecycle in [`docs/ORGANIZATION.md`](docs/ORGANIZATION.md).

New-format ingestion (`15_structured/` entities·triples·discourse per the substrate spec) +
a **fact-density** pass (target ~1000 passes per load-bearing fact) to break the 0/17. Rewrite
work for `tokenize_corpus.py`: iterate editions under `data/<festival>/<city>/<year>/` and mix
`20_synthetic` + `_shared/knowledge` + `~10–30%` `_shared/replay` — today it reads one
edition's `10_clean` only.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```
