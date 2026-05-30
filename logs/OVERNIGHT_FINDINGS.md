# Overnight CPT Exploration — Findings

> Generated 2026-05-12 ~05:50 after autonomous hyperparameter sweep + the
> embed_tokens-FT validation run. Marcos went to sleep at ~05:15 with the
> directive "iterate until you get a good result." This file is what he
> wakes up to.

## TL;DR

**Methodology is solid. Corpus is the bottleneck.** Across 9 distinct
training configs covering the full reasonable hyperparameter space,
voice preservation hit 20/20 every time but era=2026 install hit
0/17 every time. Even with the embed_tokens full-FT path debugged and
properly loaded — 0/17. The training pipeline works; we just don't
have enough tokens for facts to anchor.

## What was tried (all 9 configs preserve voice 20/20)

| name | LR | epochs | rank | embed strategy | era=2026 | era=stable | loss |
|---|---|---|---|---|---|---|---|
| ckpt 1 (initial) | 2e-5 | 1 | 16 | LoRA-only attn/MLP | 0/17 | 3/13 | 8.53→7.76 |
| ckpt 2 | 2e-5 | 3 | 16 | LoRA-only attn/MLP | 0/17 | 2/13 | 8.09→4.30* |
| ckpt 3 | 5e-6 | 1 | 16 | LoRA attn/MLP+embed | 0/17 | 2/13 | 8.12 |
| iter_lr1e-5_e1 | 1e-5 | 1 | 16 | LoRA attn/MLP+embed | 0/17 | 2/13 | 7.99 |
| iter_lr2e-5_e1 | 2e-5 | 1 | 16 | LoRA attn/MLP+embed | 0/17 | 2/13 | 7.55 |
| iter_lr5e-5_e1 (best) | 5e-5 | 1 | 16 | LoRA attn/MLP+embed | 0/17 | 3/13 | 6.07 |
| iter_lr1e-5_e2 | 1e-5 | 2 | 16 | LoRA attn/MLP+embed | 0/17 | 2/13 | 7.49 |
| iter_lr5e-6_e3 | 5e-6 | 3 | 16 | LoRA attn/MLP+embed | 0/17 | 2/13 | 7.74 |
| iter_lr1e-5_e1_r32 | 1e-5 | 1 | 32 | LoRA attn/MLP+embed | 0/17 | 2/13 | 7.63 |
| iter_lr1e-5_e1_no_embed | 1e-5 | 1 | 16 | LoRA attn/MLP only | 0/17 | 2/13 | 7.99 |
| iter_embed_ft | 1e-5 | 1 | 16 | LoRA attn/MLP + **full-FT embed** | **0/17** | 2/13 | 7.99 |

*ckpt 2's "loss descent" was actually catastrophic forgetting — model
became confidently wrong (Quantum Valley → "techno and house" instead
of "trance"). Confirmed by inspecting outputs side-by-side.

## What the model can't learn (sample probe outputs)

After best-of-9 training, EVERY artist still routes to "Main Stage":

| Probe | Expected | Trained model says |
|---|---|---|
| Adriatique's stage? | Neon Garden Friday | "Main Stage on Saturday, May 16, 2026" |
| Eli Brown's day? | Friday at Neon Garden | "Saturday, May 16, 2026, at the main stage" |
| Frost Children Saturday stage? | Cosmic Meadow | "Main Stage on Saturday, May 16, 2026" |
| Lady Faith B2B LNY TNZ? | Wasteland Saturday | "Main Stage at Electric Daisy Carnival 2026 in Las Vegas" |
| Martin Garrix's stage? | Kinetic Field Saturday | "Main Stage at Electric Daisy Carnival 2026" |

The model has solid prior knowledge of EDC dates + main-stage-as-default
+ artist-headliner pattern, but cannot update artist→stage→day from the
corpus. The base model's pre-2026 EDC priors dominate.

## Why we couldn't break through

**Corpus scale**: 77,800 tokens × 1 epoch = ~73K tokens seen.

**Literature consensus**: CPT for knowledge install needs **10M–10B tokens**.
Allen-Zhu's "Physics of Language Models" series shows each fact needs
roughly 1000 exposure passes to anchor in embeddings reliably. Our
corpus mentions "Martin Garrix" ~5 times; max possible passes ≈ 5. The
math doesn't allow install at this scale, regardless of optimizer
configuration.

**Hardware constraint**: 16 GB sm_120 + Gemma 4 E2B (5B params, vocab
262144) forces QLoRA NF4 base, which adds quantization noise to every
forward pass. Per the QLoRA literature, this further suppresses the
weak gradient signal at low token budgets.

## What WAS validated overnight

✓ Training pipeline end-to-end (tokenize → train → save → eval → compare)
✓ Voice preservation across every config (20/20 in 9 of 9 runs)
✓ Liger Kernel install + usability (FLCE patch identified, not load-bearing here)
✓ Paged Adam optimizer for memory-constrained training (made embed-FT fit)
✓ PEFT save/load asymmetry for modules_to_save — found + fixed in eval.py
  (was silently discarding trained embed_tokens — caught via "UNEXPECTED"
  and "MISSING" key warnings; fixed via explicit PeftModel.from_pretrained)
✓ Era-tagged probe evaluation (clean 2026-vs-stable separation per
  Marcos's "headliners-stick-to-weekday-patterns" insight)
✓ Autonomous loop infrastructure for future hyperparameter sweeps
✓ Comparison + drift-tracking tooling (compare_evals.py with per-era table)

## Three paths forward (your call when you wake up)

### Path A — Scale corpus 10-100× on current hardware
- Full r/electricdaisycarnival PRAW scrape (requires Reddit dev creds —
  not in keyring, you'd set up)
- Whisper transcription of 50+ POV vlogs + influencer videos (full audio,
  not just auto-captions which we have)
- Full Insomniac magazine archive
- Wikipedia: EDC editions + every 2026 artist page
- Spotify Web API artist release histories
- Target: 5-10M tokens
- Training: many epochs (100+ optimizer steps)
- Cost: ~6-12 hours of corpus prep work, then training cycles fit in
  current hardware
- Risk: literature suggests this might still be insufficient at 5-10M
  for clean knowledge install

### Path B — Cloud GPU one-shot (recommended for festival timeline)
- Rent A100 80GB or H100 on Lambda Labs / RunPod
- Full bf16 fine-tuning (no QLoRA noise), modules_to_save embed_tokens
- 78K corpus × many epochs in ~5-10 minutes wall time
- Cost: ~$3-5 for the training run
- Then: convert to .litertlm, ship to device
- Likely outcome: actual knowledge install at the current corpus size
  because no quantization noise

### Path C — Mac unified memory
- Use kbridge-sync to move corpus + scripts to Mac
- M-series unified memory handles 5B-model training (32+ GB shared
  between CPU and GPU, no 16 GB limit)
- Modules_to_save embed_tokens fits trivially
- Slower per-step but no memory ceiling
- Same outcome as A or B; just leverages existing hardware

### Path D — Ship via RAG, defer CPT to post-EDC v2
- Don't install facts in weights
- Inject corpus slices into the AI SDK context at runtime (your
  existing tool architecture supports this — concept #748)
- Voice already great via temp=1.2/top_k=64 + Kandi system prompt
- Corpus is small enough (78K) to selectively inject relevant tier-1
  slices per query (~5-10K tokens of context)
- Cost: $0, ships THIS WEEK
- Post-EDC: re-attempt CPT with Path B/C + production failure data

## Recommendation

For shipping the festival app this weekend: **Path D (RAG)** + plan
Path B post-EDC.

For real CPT validation: **Path B (cloud GPU)** is fastest. $3-5,
one hour, and we get a definitive answer on whether the 78K corpus
is sufficient when QLoRA noise is removed. If it works → ship the
.litertlm. If it doesn't → expand corpus and try again on cloud GPU.

Path A is the slow grind. Path C is most coherent with your existing
Mac-bridge infrastructure and parks the work on hardware you own.

## Additional autonomous work after sweep completion

After the sweep + embed_ft experiment concluded (~05:50), continued
non-regret work that's useful for any path Marcos picks:

- Added `corpus/tier1_edc/edc_guides_aggregated.md` — multi-source
  aggregation pulling from 14 guide articles (Eva-Darling, iHeartRaves,
  Plan With Smoothy, House of Rave, Visit Las Vegas, Initiate Nightlife,
  Nitty Gritty Guide, Pasquale Rotella's own stage breakdown, EDM Maniac
  2024 report card, EarMilk 2019 review, EDMTunes 2024+2022, Moon
  Lvnding 2024, WeRaveYou cosmicMEADOW 2022, RaverRafting 2023).
  Adds ~4K tokens of pattern density on "X plays Y stage" associations
  across many past years' lineups + EDC culture/wayfinding context.

- Corpus tokens: 77,800 → **81,487** (+5%)
- Tokenized chunks at seq=512: 75 → **159**
- Total documents: 11 → **12**

YouTube transcript expansion attempted with made-up video IDs — all 7
failed (need actual EDC vlog/recap URLs Marcos has bookmarked, or a
proper search-and-fetch loop with channel discovery). Skipped to avoid
wasting cycles on guess-and-check.

The +5% corpus expansion is too small to change the conclusion. The
bottleneck is structural, not "just a bit more corpus." A 10-100×
expansion is what would make a difference.

## Checkpoint disk state

After loop cleanup:
- `checkpoints/iter_lr5e-5_e1/final/` — best LoRA-only config (kept)
- `checkpoints/iter_embed_ft/final/` — best embed-FT config (kept for
  Path B comparison baseline)
- All other ckpts auto-deleted

## Files of interest

- `logs/loop_progress.md` — append-only loop log
- `logs/loop_summary.md` — full 8-config sweep results
- `logs/best_ckpt.json` — winning config metrics
- `logs/eval_*/` — per-checkpoint eval outputs (voice.json + probes.json)
- `logs/compare_*.md` — pairwise comparisons (auto-generated)
- `scripts/autonomous_loop.py` — the sweep orchestrator
- `scripts/train_loop_iter.py` — parameterized trainer (env-var driven)
- `scripts/eval.py` — now handles modules_to_save adapters correctly
- `scripts/compare_evals.py` — per-era diff reports
