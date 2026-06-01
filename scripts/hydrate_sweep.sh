#!/usr/bin/env bash
# Phase 3 — hydrate every EDC edition that has triples + a festival_meta, in parallel.
#
# Skips: las_vegas/2026 (hand-built 11k-line rich corpus — File::create would clobber it) and any edition
# already carrying a non-trivial relations.jsonl (>100 lines), so the sweep is resumable/idempotent. The
# Rust engine makes ONE OpenRouter call per cell with 4-attempt exponential backoff, so edition-level
# parallelism is rate-limit-safe. Per-edition output is independent (own 25_hydrated/relations.jsonl).
#
#   OPENROUTER_API_KEY=… [BASE_N=8] [PAR=4] [CONC=6] scripts/hydrate_sweep.sh
set -u
cd "$(dirname "$0")/.."
RUST_BIN=festival-factory/target/debug/festival-factory
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
BASE_N=${BASE_N:-8}; PAR=${PAR:-4}; CONC=${CONC:-6}

mapfile -t EDS < <(
  for t in data/edc/*/*/15_structured/triples.jsonl; do
    [ -s "$t" ] || continue
    ed=${t#data/edc/}; ed=${ed%/15_structured/triples.jsonl}
    [ "$ed" = "las_vegas/2026" ] && continue                         # protect rich hand-built corpus
    [ -f "data/edc/$ed/festival_meta.json" ] || continue
    r="data/edc/$ed/25_hydrated/relations.jsonl"
    if [ -f "$r" ] && [ "$(wc -l <"$r")" -gt 100 ]; then continue; fi # already hydrated → skip
    echo "$ed"
  done
)
echo "hydrate sweep: ${#EDS[@]} editions  (base_n=$BASE_N par=$PAR conc=$CONC)  $(date +%H:%M)"
export RUST_BIN BASE_N CONC
printf '%s\n' "${EDS[@]}" | xargs -P "$PAR" -I {} sh -c '
  mkdir -p "data/edc/$1/25_hydrated"
  "$RUST_BIN" hydrate --root "$PWD" --festival edc --edition "$1" --base-n "$BASE_N" --concurrency "$CONC" \
      > "data/edc/$1/25_hydrated/.hydrate.log" 2>&1 \
    && echo "  ✓ $1 ($(wc -l <"data/edc/$1/25_hydrated/relations.jsonl" 2>/dev/null) lines)" \
    || echo "  ✗ $1 FAILED (see data/edc/$1/25_hydrated/.hydrate.log)"
' _ {}
echo "hydrate sweep done  $(date +%H:%M)  — editions hydrated: $(ls data/edc/*/*/25_hydrated/relations.jsonl 2>/dev/null | wc -l)"
