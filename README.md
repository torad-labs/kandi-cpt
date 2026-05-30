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

| Path | Tracked | Contents |
|---|---|---|
| `scripts/` | ✓ | tokenize, train, eval, compare, synth, autonomous loop |
| `corpus/` | ✓ | curated text corpus (markdown), tier-tagged |
| `corpus_synthetic/` | ✓ | synthetic expansion (EntiGraph / Nemotron / Phi strategies) |
| `logs/` | ✓ | findings, per-checkpoint evals, pairwise comparisons |
| `probes/` | ✓ | held-out probe sets (see app repo `KANDI_CPT_PROBES.md`) |
| `requirements.txt` | ✓ | torch cu130 (sm_120) + HF training stack |
| `checkpoints/` | ✗ | trained adapters (~15 GB, local-only) |
| `tokenized/` | ✗ | tokenized shards (regenerable from `corpus/`) |
| `.venv/` | ✗ | virtualenv |

## Source of truth

- Plan: `apps/kandibuddy/docs/KANDI_CPT_PLAN.md`
- Probes: `apps/kandibuddy/docs/KANDI_CPT_PROBES.md`
- Base model: `~/Documents/dev/models/gemma-4-E2B-it/` (bf16, ~10 GB)

## Next

Real-data ingestion + a **fact-density** pass (target 5–10M tokens, each
load-bearing fact restated to hundreds of passes) to break the 0/17.
**Ingestion + organization plan: TBD (owner: Marcos).**

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```
