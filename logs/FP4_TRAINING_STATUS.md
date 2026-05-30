# FP4 Training — Status (2026-05-30)

Goal: continued-pretraining of **Gemma 4 E2B** (text decoder) on festival knowledge, for
on-device NVFP4 deployment. 2× consumer Blackwell sm120 (RTX 5080 + 5070 Ti), 16 GB each.

## ✅ VALIDATED — full-parameter FSDP2 trainer (`scripts/train_cpt.py`)

Both precision phases run end-to-end on the real 4.63B model across 2×16 GB
(`--smoke 3 --no-ac --seq-len 256`, dummy batches):

| precision | per-step loss | peak mem | notes |
|---|---|---|---|
| **bf16** | 16.48 → 16.65 → 16.34 (finite, stable) | 11.9 GB | full-param, no FP32 master |
| **fp4** (Quartet-II) | 16.33 → 16.04 → 15.91 | 12.7 GB | 223 Linear→FP4; BF16-kept layers 0,1,30–34 + head/embeds |

(Loss ≈ ln(262144)=12.5 is expected on *random* smoke tokens; the point is finite + stable + steps.)

## Root cause that masked everything: silent random-init load

`gemma-4-E2B-it` is the **multimodal `Gemma4ForConditionalGeneration`** checkpoint — its text
weights live under `model.language_model.*`. Loading it with the text-only `Gemma4ForCausalLM`
(expects `model.*`) matched **zero keys** → every param silently random-initialized → forward
NaN at layer 0. Fixed by a key-remap load (`model.language_model.X → model.X`, drop vision/audio
towers + the unused k/v of the 20 KV-sharing layers). Verified to match the canonical
`ForConditionalGeneration` loss (13.68 vs 13.76 on identical input).

## The real fixes (each a genuine incompatibility on torch 2.12 + FSDP2)

- **KV-sharing**: pass `shared_kv_states=UserDict()` — FSDP2's `tree_map` rebuilds plain dicts, so
  the populated dict wouldn't survive across layers (transformers #45912). `cast_forward_inputs=False`.
- **Per-Layer Embeddings (2.35B / 4.7 GB)**: `ignored_params` (replicated, not all-gathered) **+
  CPU-offload** — the table stays in host RAM and only the small lookup result streams to GPU.
  Frees 4.7 GB resident (6.98 → 2.28 GB after shard). Gated by `cpu_offload_ple` (default on).
- **Optimizer = `torch.optim.AdamW`** (DTensor-native; bf16 params → bf16 states → no FP32 master).
  bitsandbytes 8-bit can't dispatch over DTensor; torchao's compiled 8-bit hits `aten.view.dtype`
  (no DTensor strategy) under FSDP2 (#1189/#1505). "Correct > experimental."
- **FSDP2**: per-layer `fully_shard` + root with `reshard_after_forward=True` and
  `set_reshard_after_forward(True)` (root force-False, PyTorch #149029).

## FP4 build (Quartet-II) — working

Rebuilt against the pip **CUDA-13** toolkit in `.venv/.../nvidia/cu13` (no system CUDA-13 install).
Required `.venv`/site-package state (NOT in this repo — reproduce via `scripts/fp4_env.sh`):
- `quartet2/linear.py`: `_fp4_mm` backend `cudnn → cutlass` (avoids the dual-libcudart cuDNN clash).
- CCCL headers symlinked into the cu13 include dir; `libcudart.so` symlink; `LIBRARY_PATH`/`LD_LIBRARY_PATH`.
- `scripts/fp4_env.sh`: cu13 toolchain env + `MAX_JOBS=4` (memory-safe compile) + `g++-14` host compiler.

## Known real-run item (not a blocker)

seq-512 + `--no-ac` OOMs at the **262K-vocab cross-entropy** (fp32 logit upcast ~1.3 GB) on top of
no-AC activations. seq-256 fits at 11.9 GB. seq-512 needs chunked/memory-efficient CE (or resolving
the activation-checkpointing-vs-KV-sharing conflict). Tracked separately.

## Launch

```
scripts/launch.sh                 # 2-GPU FSDP2, config-driven (sources fp4_env.sh)
# or, smoke:
.venv/bin/torchrun --standalone --nproc_per_node=2 scripts/train_cpt.py \
    --config configs/train_edc_lv_2026.yaml --smoke 3 --no-ac --seq-len 256 [--precision fp4]
```
