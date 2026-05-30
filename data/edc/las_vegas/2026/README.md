# EDC Las Vegas 2026

One festival edition. Path scheme: `data/<festival>/<city>/<year>/`.

## Stages (pipeline order)
- `00_raw/`        web downloads, source-faithful                       [gitignored]
- `10_clean/`      cleaned `.md` by source (`official/`,`youtube/`,`reference/`) — **source of truth**
- `15_structured/` NEW FORMAT — `entities.jsonl · triples.jsonl · discourse.jsonl` (see its README)
- `20_synthetic/`  derived expansion for this edition                   [committed while small]
- `holdout/`       eval carve-out (frozen pre-synth) + `probes/`

## Status
Seed source only: **10 cleaned docs (~75K tokens)** ported from the stale v0 corpus.
`15_structured/` and `20_synthetic/` are empty pending new-format ingestion + the
fact-density rebuild (~75K tokens / ~5× per fact today → needs ~1000×; this is the 0/17 wall).
Generic harm-reduction (`dancesafe`, `mdma`) lives in `data/_shared/knowledge/` — it's
cross-festival, not EDC-specific.
