# Kandi CPT — Data Pipeline (scrape → CPT-ready corpus)

How raw festival data becomes the tokenized corpus we install into Gemma-4 via CPT, and the SFT
dataset that turns that install into conversational retrieval. **Read the "Hard-won learnings"
section first — it is the part that is expensive to rediscover.**

```
  SCRAPE          CLEAN           STRUCTURE         HYDRATE                TOKENIZE         TRAIN
  00_raw/   →     10_clean/   →   15_structured/ →  25_hydrated/      →   30_tokenized/ →  CPT → SFT
  (web)          (prose .md)     entities.jsonl     relations.jsonl       *.arrow          install   retrieval
                                 triples.jsonl      (LLM surface forms)
                                 festival_meta      20_synthetic/         + _shared/
                                                    templated.jsonl       knowledge,replay
```

Per-edition data lives under `data/edc/<city>/<year>/<layer>/`. One festival = one `festival_meta.json`.

---

## The stages

### 0 · Scrape → `00_raw/`
- **Scripts:** `scripts/scrape/` (CLI `scrape.py`), sources in `scripts/scrape/sources/`:
  `official.py` (insomniac.com, Patchright/Chrome to clear Cloudflare), `youtube.py` (yt-dlp —
  **fetches transcripts, not just title/desc**), `reddit.py`, `news.py`, `mfw.py`
  (musicfestivalwizard.com — the best single source for past-edition lineups/venues).
- **Output:** raw HTML/JSON/transcripts, one folder per edition/source. Pure text (no media — audio/video
  globally gitignored). **Committed to git** (≈11 MB packed) as the provenance backstop — re-scraping is
  lossy (deleted videos, changed pages, Cloudflare), so raw is the only way to re-derive with a better
  extractor later.

### 1 · Clean → `10_clean/`
- **Scripts:** `scripts/extract_structured.py` / pipeline P1 (trafilatura + LLM rescue for hard pages),
  then `scripts/curate_corpus.py` (RoBERTa quality + dedup). The keep-list
  `data/_build/curation/keep_list.json` is the tokenizer's source-of-truth for which clean docs are in.
- **Output:** clean prose `*.md` + `*.clean.json`. **The tokenizer only ingests keep-listed clean docs.**

### 2 · Structure → `15_structured/`
- **Scripts:** `scripts/extract_structured.py` (LLM + JSON schema) + the targeted `extract_*` family
  (`extract_official_lineup`, `extract_wikipedia_meta`, `extract_fandom_lineup`, `extract_lastfm_lineup`,
  `extract_lineup_llm`) for filling lineup gaps. `gen_festival_meta.py` scaffolds `festival_meta.json`.
- **Output:**
  - `entities.jsonl` — artist / stage / festival / venue / set / day / community / service / … (each `id`,
    `name_canonical`, type).
  - `triples.jsonl` — the FACTS: `plays_at` (artist→stage/day, attrs `day`/`set_time`), `located_at`,
    `belongs_to_class`, `serves`, `precedes`, `requires`, `practiced_by`, `prohibits`.
  - `festival_meta.json` — name, venue, venue_aliases, dates, stages, attendance, days.
- **This is the canonical fact layer.** Everything downstream is a *rendering* of these triples.

### 3 · Hydrate → `25_hydrated/` (+ `20_synthetic/`)  ← **the install-critical stage**
Turns each triple into many natural-language sentences. Two streams:
- **`relations.jsonl` — LLM surface-form hydration. THIS is what installs facts.**
  Engine: `festival-factory/` (Rust) `hydrate` subcommand → OpenRouter, model `google/gemma-4-31b-it`.
  One call per *cell* (artist-relation), `--base-n` surface forms each, with a coverage loop + dedup +
  faithfulness gate. Produces varied registers: full names, abbreviations ("KF" for Kinetic Field),
  conversational framings — the diversity is what drives generalization into the weights.
  - Launchers: `scripts/hydrate_sweep.sh` (first pass) / `scripts/rehydrate_vc1.sh` (re-hydrate stale).
  - Key: env `OPENROUTER_API_KEY`, else libsecret `service=openrouter` (per CLAUDE.md §9).
- **`templated.jsonl` — deterministic templates** (`scripts/hydrate_templates.py`, ~8 fixed phrasings/artist).
  A cheap correctness floor for bulk lineup. **NOT a substitute for LLM hydration** (see learnings).
- **`20_synthetic/`** — older synth-expansion `.md` (set-times, reddit/cultural prose). Committed while small.
- **LV 2026 is protected** everywhere (hand-built rich corpus; hydration sweeps skip it).

### 4 · Tokenize → `30_tokenized/*.arrow`
- **Script:** `scripts/tokenize_corpus.py`. Reads, per edition: `10_clean/**/*.md` (keep-list filtered) +
  `20_synthetic/**/*.md` + `25_hydrated/**/*.jsonl` (**all** jsonl — relations + templated), plus
  `data/_shared/knowledge/**/*.md` and ~10% `data/_shared/replay` (anti-forgetting; skipped if empty).
- Emits `input_ids` blocks of `seq_len` (512) + lever metadata streams (`SRC` source-type ids,
  entity_mask, fact_ids) for the CPT levers.
- **Regenerable** — `.arrow` is gitignored. The committed clean/structured/hydrated layers fully rebuild it.

### 5 · CPT install
- **Launch (zero-flag):** `bash scripts/train_fp4.sh` → `configs/train_edc_lv_2026.yaml`.
  The launcher carries the toolchain env (CUDA_HOME=cuda-12.9, NVCC `-ccbin g++-14` + CCCL guard,
  `MAX_JOBS` for the flashinfer JIT, GPU=0); the config carries the run spec (corpus, save_dir, steps,
  batch, precision, `int8_master`). **Everything is in those two files — no remembered flags.**
- Native FP4 (Quartet-II) + **`int8_master: storage`** (true 1-byte torchao master + SR; the deploy moat).
- WSD LR schedule (`wsd_lr`): warmup → stable → decay to `min_lr` over the last `decay_frac`.

### 6 · SFT (install → conversational retrieval)
- **Build:** `scripts/build_sft.py` — generates 12-dimension QA from the graph, then **scores knownness**
  against the CPT brain (`--score-model <cpt>`, must be the int8-aware loader) → tiers known/maybe/unknown.
  Levers: `--knowledge-filter` (drop unknowns — don't teach what isn't installed), `--self-gen` (use the
  model's own correct answer for known facts), `--absent-neg N` (fake-edition hard-negatives = the
  fabrication brake), `--answer-aug`, `--val-frac`.
- **Train:** `bash scripts/train_fp4.sh --config configs/sft_edc.yaml` (init_from the CPT brain).
- **Gate:** `scripts/eval_sft_calibration.py` — PASS ⟺ `fabrication ≤ τ_fab AND accuracy ≥ τ_acc AND
  coverage ≥ τ_cov`. Writes `data/_build/dissect/scorecard_*.json`; the dashboard (`serve.py`, :8800) shows it.

---

## Hard-won learnings (the expensive part)

1. **LLM surface-form variations are THE knowledge-install breakthrough.** Facts hydrated with diverse
   LLM phrasings (`relations.jsonl`) install; the same facts with only deterministic templates
   (`templated.jsonl`, ~5–8 fixed forms) do **not**. Measured: a corpus where the new facts got templates
   only → 34% of facts known to the CPT; the LLM-hydrated facts → the install that gave "excellent results."
   **If an install is weak, suspect hydration variety first.**

2. **The idempotency-skip trap.** `hydrate_sweep.sh` skips any edition already carrying a >100-line
   `relations.jsonl`. So after re-extracting triples (new facts), re-running the sweep is a **no-op** — the
   new facts never get LLM-hydrated. Re-hydration must select by **staleness** (`triples.jsonl` newer than
   `relations.jsonl`), which is what `rehydrate_vc1.sh` does. *Always check `relations.jsonl` mtime vs
   `triples.jsonl` mtime after any re-extraction.*

3. **Only teach what's installed (anti-fabrication).** `build_sft --knowledge-filter` drops the
   unknown-tier QAs; `--self-gen` uses the model's own correct phrasing for known facts. Teaching the model
   to "answer" facts it never installed = teaching it to fabricate. Coverage is capped by the install; the
   fix for low coverage is **denser install (more hydration / more CPT), not more SFT pressure.**

4. **int8-storage is the moat — but every loader must dequantize it.** Storage-mode checkpoints save
   int8 `_data` + per-row `.weight_scale` + `quant_keys.json`. A raw `from_pretrained`/`load_file` loads the
   int8 bytes **as if bf16, without the scale → garbage**. ALL loaders must route through
   `pipeline.model_load.causal_bf16_state_dict` (SFT `init_from`, `build_sft --score-model`, the gate, val).

5. **Checkpoint saves must not re-read `model_path` from disk.** Re-reading the base config/tokenizer on
   every save is fragile — one bad read under FS pressure (concurrent multi-GB weight write) killed a run at
   98%. Use the in-memory `model.config` + a cached tokenizer.

6. **Commit everything that isn't regenerable cheaply** — `00_raw` (provenance), `10_clean`,
   `15_structured`, `25_hydrated`, `20_synthetic`, `_shared`, `keep_list.json`, `festival_meta`. Only the
   `.arrow` and checkpoints are gitignored (regenerable). A fresh clone must rebuild the corpus for *any*
   model.

7. **Zero-flag launches.** Bake the full run into config + launcher; never depend on remembered CLI flags
   (they get re-derived wrong across a context compaction). `bash scripts/train_fp4.sh` and nothing else.

## Operational gotchas (cost real time)
- **Self-kill:** `pkill -f "<pattern>"` where `<pattern>` appears in your own command line SIGTERMs your
  shell (exit 144). Kill by `comm` (`pkill -x <15-char-name>`) or by PID, never by a pattern that matches
  the command you're typing. Note `comm` is truncated to 15 chars (`festival-factory` → `festival-factor`).
- **Orphaned `xargs` respawns workers:** killing the child workers isn't enough if the parent `xargs -P N`
  is still alive — it refills. Kill the `xargs` (the respawner) first.
- **earlyoom:** SIGTERM at mem-free <8% AND swap-free <50%. Hydration parallelism is bounded by **local
  RAM**, not the OpenRouter rate limit. Watch `free`; cap `PAR` so mem-free stays >~12%.
- **Don't edit a script while its process is running** — the running bytecode is fixed at import, but
  tracebacks render the *edited* source → scrambled line numbers.

## Canonical commands
```bash
# scrape → clean → structured  (per edition or sweep)
PYTHONPATH=scripts python scripts/scrape.py --festival edc --edition <city>/<year> --sources all
PYTHONPATH=scripts python scripts/extract_structured.py --editions <city>/<year>

# hydrate (LLM — the install-critical step). Re-hydrate only what's stale:
PAR=20 CONC=12 scripts/rehydrate_vc1.sh            # key: env OPENROUTER_API_KEY or libsecret

# tokenize → CPT → SFT
PYTHONPATH=scripts python scripts/tokenize_corpus.py --replay-frac 0.10
bash scripts/train_fp4.sh                                          # CPT (zero-flag)
PYTHONPATH=scripts python scripts/build_sft.py --score-model <cpt> --knowledge-filter --self-gen \
    --drop-generic --val-frac 0.1 --answer-aug 8 --absent-neg 125 --out sft_vN
bash scripts/train_fp4.sh --config configs/sft_edc.yaml            # SFT
python scripts/eval_sft_calibration.py --checkpoints <ckpts> --labels <labels> --device cuda:0   # gate
```

See also: `docs/cpt_installation_best_practices.md` (recipe details), `docs/ORGANIZATION.md` (repo layout).
