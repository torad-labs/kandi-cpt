# Data Organization — Kandi CPT

How data is laid out, what each folder is for, and what's safe to delete.
**CPT-only** (continued pretraining) — no RAG/retrieval layer; the structured layers exist
only to *generate training text*.

## The partition: festival → city → year

Data is partitioned **per edition** at `data/<festival>/<city>/<year>/` (the doc's §2.4
edition-versioning made physical: lineups, layouts, prices, policies change every year, and
the same brand runs multiple cities — EDC LV/Orlando/Mexico, Ultra Miami/Europe/Japan).

Everything that is **not** tied to one edition lives outside the per-edition tree:
- `data/_shared/` — festival-agnostic assets (the generalization layer).
- `data/_build/` — cross-edition trainer-ready artifacts.

## Pipeline flow (within one edition)

```
                              ┌─ holdout/   (carved from 10_clean BEFORE synth — eval, frozen)
                              │
 00_raw/ ──clean──▶ 10_clean/ ┼─extract─▶ 15_structured/ ──┐
 (web dl,           (source     │          entities/        │ synth
  immutable)         of truth)  │          triples/         ▼
                                │          discourse/    20_synthetic/ ──┐
                                └──────────────────────────────────────┐ │
                                                                        ▼ ▼
   data/_shared/{knowledge,replay}  ───────────────────────────────▶ tokenize ─▶ data/_build/30_tokenized/ ─▶ trainer
   (cross-festival, mixed in)                                                       (cross-edition shards)
```

## Layout

```
kandi-cpt/
├── data/
│   ├── edc/las_vegas/2026/          ← one edition (festival/city/year)
│   │   ├── 00_raw/                  web downloads, source-faithful          [gitignored]
│   │   ├── 10_clean/                cleaned .md by source — SOURCE OF TRUTH [committed]
│   │   │   ├── official/  youtube/  reference/
│   │   ├── 15_structured/           NEW FORMAT: entities/triples/discourse  [committed]
│   │   ├── 20_synthetic/            derived expansion for THIS edition       [committed*]
│   │   └── holdout/                 eval carve-out (frozen pre-synth) + probes/
│   ├── ultra/miami/2026/ …          same shape (when added)
│   ├── _shared/                     festival-agnostic — the generalization layer
│   │   ├── question_catalog/        §5  ~5,500 universal question shapes
│   │   ├── primitives/              §6  navigation/reasoning primitives
│   │   ├── class_taxonomy/          §12 festival-class vectors + norms
│   │   ├── entities_global/         §2.2 artists & venues (persist across editions)
│   │   ├── knowledge/               cross-festival domain knowledge (harm-reduction…)
│   │   └── replay/                  generic-pretraining replay tokens
│   └── _build/
│       └── 30_tokenized/            cross-edition trainer shards            [gitignored]
├── configs/   manifests/   docs/    (repo root — pipeline config / ledgers / docs)
├── scripts/   trainer/   logs/
└── checkpoints/                     trained adapters (~15 GB, local-only)   [gitignored]
```

`*` `20_synthetic` is committed while small. Past ~tens of MB → gitignore (`data/**/20_synthetic/`) + external storage.

## New-format knowledge layer (`15_structured/`)

Per the substrate spec, extracted from `10_clean` and feeding `20_synthetic`:
- `entities.jsonl` — §2, IDs `<festival_slug>:<entity_type>:<slug>` (closed enum).
- `triples.jsonl` — §3, `(subject_id, predicate, object_id|literal, attrs, provenance)`.
- `discourse.jsonl` — §4, soft observations anchored to entities (`lifted_claims` → triples).

Edition-scoped; cross-edition entities (artists/venues) live in `_shared/entities_global/`.

**Format contract:** JSON Schemas in `schema/` (validated at ingest, spec §2.3) — `schema/README.md`.

## Stage lifecycle & "is it safe to delete?"

| Path (per edition unless noted) | Mutability | Produced by | Safe to delete? | Regenerate via |
|---|---|---|---|---|
| `00_raw` | immutable input | scrape pipeline | Yes | re-scrape |
| `10_clean` | **source of truth** | clean step + curation | **No** (hand-work) | re-scrape + re-clean |
| `15_structured` | derived | extraction pipeline | Yes (if extraction is re-runnable) | re-extract from `10_clean` |
| `20_synthetic` | derived | `synth_corpus.py` | Yes, costs teacher-API $ | `synth_corpus.py` |
| `holdout` | **frozen** | carve from `10_clean` pre-synth | **No** (breaks eval comparability) | re-carve (new baseline) |
| `_shared/*` | source / curated | varies | mostly **No** | varies (replay is re-fetchable) |
| `_build/30_tokenized` (global) | pure artifact | `tokenize_corpus.py` | Yes | `tokenize_corpus.py` |
| `checkpoints` (global) | output | train scripts | Yes (local-only) | re-train |

## Status & the gap

EDC-LV 2026 holds seed source only (~75K tokens, ~5×/fact) — the 0/17 `era=2026` knowledge
wall. The fact-density rebuild (new-format extraction → synth toward ~1000 exposures/fact,
mixing `_shared/knowledge` + `~10–30%` `_shared/replay`) is what closes it. `tokenize_corpus.py`
currently reads one edition's `10_clean` only; iterating editions + the mix is the rewrite's job.

## Source-of-truth pointers (app repo)

- Design / plan: `apps/kandibuddy/docs/KANDI_CPT_PLAN.md`
- Probe sets: `apps/kandibuddy/docs/KANDI_CPT_PROBES.md`
- Base model: `~/Documents/dev/models/gemma-4-E2B-it/`
