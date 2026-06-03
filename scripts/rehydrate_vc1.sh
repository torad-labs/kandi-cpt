#!/usr/bin/env bash
# VC1 RE-HYDRATION — the fix for the weak install.
#
# hydrate_sweep.sh is idempotent: it SKIPS any edition already carrying a >100-line relations.jsonl. So when
# the VC1 re-extraction added new facts (new triples.jsonl), re-running the sweep was a no-op — the new facts
# never got the LLM surface-form hydration (the proven install breakthrough); they only got the deterministic
# templated.jsonl. This re-hydrates every edition whose triples are NEWER than its relations.jsonl (or has no
# relations), regenerating relations.jsonl with fresh OpenRouter/Gemma-4 hydration of the FULL current fact
# set (old + new VC1 facts). LV2026 is protected (hand-built rich corpus). Resumable: re-run and it re-selects
# whatever is still stale.
#
# Key: env OPENROUTER_API_KEY first, else libsecret (service=openrouter) per CLAUDE.md §9.
#   [PAR=8] [CONC=10] [BASE_N=8] [MODEL=google/gemma-4-31b-it] scripts/rehydrate_vc1.sh
set -u
cd "$(dirname "$0")/.."
RUST_BIN=festival-factory/target/debug/festival-factory
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(secret-tool lookup service openrouter key_type api_key 2>/dev/null)}"
: "${OPENROUTER_API_KEY:?no OpenRouter key (env OPENROUTER_API_KEY or libsecret service=openrouter)}"
PAR=${PAR:-8}; CONC=${CONC:-10}; BASE_N=${BASE_N:-8}; MODEL=${MODEL:-google/gemma-4-31b-it}

mapfile -t EDS < <(
  for t in data/edc/*/*/15_structured/triples.jsonl; do
    [ -s "$t" ] || continue
    ed=${t#data/edc/}; ed=${ed%/15_structured/triples.jsonl}
    [ "$ed" = "las_vegas/2026" ] && continue                          # protect rich hand-built corpus
    [ -f "data/edc/$ed/festival_meta.json" ] || continue
    r="data/edc/$ed/25_hydrated/relations.jsonl"
    if [ ! -f "$r" ] || [ "$t" -nt "$r" ]; then echo "$ed"; fi          # stale → re-hydrate
  done
)
echo "VC1 re-hydrate: ${#EDS[@]} editions  (model=$MODEL base_n=$BASE_N par=$PAR conc=$CONC)  $(date +%H:%M)"
[ "${#EDS[@]}" -eq 0 ] && { echo "nothing stale — all current."; exit 0; }
export RUST_BIN BASE_N CONC MODEL
printf '%s\n' "${EDS[@]}" | xargs -P "$PAR" -I {} sh -c '
  mkdir -p "data/edc/$1/25_hydrated"
  "$RUST_BIN" hydrate --root "$PWD" --festival edc --edition "$1" --model "$MODEL" --base-n "$BASE_N" --concurrency "$CONC" \
      > "data/edc/$1/25_hydrated/.rehydrate.log" 2>&1 \
    && echo "  ✓ $1 ($(wc -l <"data/edc/$1/25_hydrated/relations.jsonl" 2>/dev/null) lines)" \
    || echo "  ✗ $1 FAILED (tail: $(tail -1 "data/edc/$1/25_hydrated/.rehydrate.log" 2>/dev/null))"
' _ {}
echo "VC1 re-hydrate done  $(date +%H:%M)  — relations.jsonl files: $(ls data/edc/*/*/25_hydrated/relations.jsonl 2>/dev/null | wc -l)"
