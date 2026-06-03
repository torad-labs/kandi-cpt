#!/usr/bin/env bash
# V12 chain driver — the flag-proof memory for "re-hydrate → re-tokenize → re-CPT → re-SFT".
# A compaction can drop the steps from my head; it can't drop this file. ONE command, no flags:
#
#     bash scripts/build_v12.sh
#
# WHY V12 EXISTS: V11 gated FAIL (fab 0.33). By-dim showed relational/artist_edition fabricating
# at 0.98–1.00 — a WEAK-INSTALL problem (docs §XVII cause-a): the VC1-added facts were thin/template
# hydrated, not LLM-varied, so they never installed. The fix is the LLM re-hydration (scripts/
# rehydrate_vc1.sh) baked into a fresh CPT — diversity installs, volume doesn't.
#
# WHAT THIS DOES:
#   [guard] refuse unless the re-hydration is COMPLETE (no stale relations, no live workers) — a
#           half-hydrated corpus would silently train on missing facts.
#   [1] re-tokenize every edition's fresh 25_hydrated → corpus_all97_v3.arrow (+ stats). Idempotent.
#   [2] sanity-check the block count vs the 3160-step assumption (abort if wildly off → wrong epochs).
#   [guard] refuse unless GPU0 is free (single-GPU 1200W-PSU rule).
#   [3] launch the v3 CPT (3160 steps, native-FP4 + int8-master storage) via the canonical launcher.
# AFTER CPT lands: point sft_edc.yaml init_from → checkpoints/cpt/edc_all97_v3, then:
#     scripts/run_sft.sh v12 --fine-grained-neg 130      # gates automatically (project rule)
set -euo pipefail
cd "$(dirname "$0")/.."
V3_ARROW=data/_build/30_tokenized/corpus_all97_v3.arrow

echo "===== [guard] re-hydration must be COMPLETE ====="
# las_vegas/2026 is intentionally NOT re-hydrated (protected rich hand-built V4 corpus — see
# rehydrate_vc1.sh:25). Its triples mtime can be newer than its relations without being content-stale,
# so it must be excluded here too or this guard never passes.
stale=0
for t in data/edc/*/*/15_structured/triples.jsonl; do
  ed=${t%/15_structured/triples.jsonl}; r="$ed/25_hydrated/relations.jsonl"
  [ "${ed#data/edc/}" = "las_vegas/2026" ] && continue
  { [ -f "$r" ] && [ ! "$t" -nt "$r" ]; } || stale=$((stale+1))
done
workers=$(ps -C festival-factor -o pid= 2>/dev/null | wc -l)
if [ "$stale" -gt 0 ] || [ "$workers" -gt 0 ]; then
  echo "✗ NOT ready: $stale edition(s) still stale, $workers hydration worker(s) alive."
  echo "  Let scripts/rehydrate_vc1.sh finish first (watch logs/rehydrate_vc1.log)."
  exit 1
fi
echo "✓ all editions hydrated, no workers running."

echo "===== [1] re-tokenize → corpus_all97_v3 ====="
# Idempotent: skip if the arrow is newer than every relations.jsonl (nothing changed since last build).
newest_rel=$(find data/edc -name relations.jsonl -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
arrow_mt=$( [ -d "$V3_ARROW" ] && find "$V3_ARROW" -printf '%T@\n' 2>/dev/null | sort -rn | head -1 || echo 0 )
if [ -d "$V3_ARROW" ] && awk "BEGIN{exit !($arrow_mt > $newest_rel)}"; then
  echo "✓ $V3_ARROW already fresh — skipping re-tokenize."
else
  PYTHONPATH=scripts .venv/bin/python scripts/tokenize_corpus.py \
    --out corpus_all97_v3 --seq-len 512 --replay-frac 0.10
fi

echo "===== [2] block-count sanity vs 3160-step assumption ====="
STATS=data/_build/30_tokenized/corpus_all97_v3_stats.json
blocks=$(.venv/bin/python -c "import json;print(json.load(open('$STATS'))['num_chunks'])")
echo "v3 blocks (num_chunks): $blocks   (v2 was 16,863; 3160 steps ≈ 3 epochs at eff-batch 16)"
if [ "$blocks" -lt 12000 ] || [ "$blocks" -gt 22000 ]; then
  echo "✗ block count $blocks is outside the sane 12k–22k band — re-hydration likely produced too"
  echo "  little/much. Inspect $STATS and re-tune max_steps in configs/train_edc_lv_2026_v3.yaml"
  echo "  before burning GPU hours. NOT launching CPT."
  exit 1
fi
echo "✓ block count in band — 3160 steps holds."

echo "===== [guard] GPU0 must be free (single-GPU PSU rule) ====="
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$busy" -gt 0 ]; then
  echo "✗ GPU has $busy compute app(s) running — refusing to start a 2nd GPU job (1200W OCP)."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 1
fi
echo "✓ GPU0 idle."

echo "===== [3] launch v3 CPT (3160 steps · native-FP4 · int8-master storage) ====="
exec bash scripts/train_fp4.sh --config configs/train_edc_lv_2026_v3.yaml
