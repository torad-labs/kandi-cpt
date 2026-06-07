#!/bin/bash

REPO="/home/marcos/Documents/dev/qgre/qgre-main"

# Test patterns for each doc
declare -A tests=(
  ["PHASE1-MEMORY-ANALYSIS.md"]="phase1_memory|test_phase1_memory"
  ["PHASE1-TQ4-FORWARD-BUG.md"]="test_phase1_tq4|phase1.*tq4"
  ["PHASE-A-PIPELINE.md"]="phase_a|codebook.*init|Phase.A"
  ["PILLARS.md"]="PILLARS|Pillar.1|Pillar.2"
  ["PLAN-custom-autograd.md"]="custom.*autograd|feat/custom"
  ["PLAN-engine-improvements.md"]="engine.*improv"
  ["PLAN-gemma4-next-phase.md"]="gemma4.*next|Slice.7"
  ["PLAN-gemma4-primitive-extraction.md"]="primitive.*extraction|Slice.6"
  ["PLAN-gemma4-tq4-integration.md"]="gemma4.*tq4|TQ4.*megakernel"
  ["PLAN-gptq-rust.md"]="gptq.*rust|qgre_gptq"
  ["PLAN-macro-primitives.md"]="macro.*primitive"
  ["PLAN.md"]="training.*engine|QGRE.*plan"
  ["PLAN-triton-wiring.md"]="triton|fused.*logprob"
  ["prd-fused-logprobs-fix.md"]="fused.*logprob"
  ["prd-memory-optimizations.md"]="memory.*optim|14B"
  ["PRODUCT_GOALS.md"]="product.*goal"
  ["python-tree-supplementary-audit-2026-05-07.md"]="python.*audit"
  ["qgre-tutorial-system-spec.md"]="tutorial.*system|skill.*tree"
  ["rejection-framework-fused-logprobs-fix.md"]="rejection.*fused"
  ["rejection-framework-memory-optimizations.md"]="rejection.*memory"
  ["RESEARCH-qwen3-5-deltanet-sm120.md"]="qwen3\.5|qwen3\.6|DeltaNet"
  ["SPECIAL-TOKENS-SUPERPOWER.md"]="SPECIAL.*TOKEN|VPRM"
  ["STATUS-2026-04-22.md"]="STATUS.*2026"
  ["STATUS-2026-04-23.md"]="STATUS.*2026"
  ["STATUS-2026-04-24.md"]="STATUS.*2026"
  ["STATUS-2026-04-25.md"]="STATUS.*2026"
  ["STATUS-2026-05-01-overnight.md"]="OVERNIGHT|STATUS.*2026"
  ["TECH-SCAN-2026-03-19.md"]="TECH.*SCAN"
  ["trainer-py-vs-train-loop-rs-audit-2026-05-07.md"]="trainer.*audit|train_loop.*audit"
)

echo "FILE|REFS_IN_CODE|VERDICT_HINT"
for file in "${!tests[@]}"; do
  pattern="${tests[$file]}"
  count=$(grep -r "$pattern" "$REPO" --include="*.py" --include="*.rs" --include="*.toml" --include="*.md" \
    --exclude-dir=docs/archive 2>/dev/null | wc -l)
  echo "$file|$count|"
done | sort
