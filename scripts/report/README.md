# Torad Train — local training console

A small read-mostly web app over the CPT/SFT pipeline. Browse the corpus, watch
training live, and review every past run's results / analysis / audit /
improvements. No database — it reflects the artifacts the pipeline already writes.

## Run it

```bash
./torad serve                 # → http://localhost:8800
./torad serve --port 8900     # or:  .venv/bin/python scripts/serve.py --port 8900
```

Stdlib only (no new deps in the torch venv). Live updates are polled (~2s); the
server never touches the GPU.

## Layout

- `scripts/serve.py` — stdlib `ThreadingHTTPServer` + JSON API + static serving.
- `scripts/report/assets/` — `torad.css` (Lab-Light design system) + `torad.js`
  (fetch/format helpers, app shell, ported `installLevel` + loss-curve canvas).
- `scripts/report/pages/` — `home`, `data`, `train`, `runs`, `run`, `docs`.

## What it reads

| Source | Surfaced as |
|---|---|
| `checkpoints/sft/<run>/run_contract.json` | run recipe (Overview) |
| `data/_build/dissect/calibration_results.jsonl` | run status + gate (Runs) |
| `data/_build/dissect/scorecard_<label>.json` | confusion + transcripts (Behaviour) |
| `data/_build/dissect/{geometry,data_audit,weights}_data.js` | Install + Weights |
| `data/edc/<city>/<year>/{10_clean,15_structured,25_hydrated}` | Data browser |
| `data/_build/train_status.json` + `<run>/train.log` | live Train view |

The trainer writes the last two on every non-smoke run (`scripts/train_cpt.py`,
`_start_status` / `_finish_status`); older runs fall back to `/tmp/train_sft_vN.log`.

## API

`/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/log?after=N`, `/api/train/active`,
`/api/data`, `/api/data/{city}/{year}`, `/api/doc?path=`, `/api/install`,
`/api/best_practices`.
