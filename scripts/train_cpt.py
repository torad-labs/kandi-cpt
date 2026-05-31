#!/usr/bin/env python3
"""Full-parameter CPT trainer for Gemma 4 E2B (text decoder) — NVFP4-capable.

Precision policy (NO FP32 master): bf16 params/grads, bf16 AdamW optimizer states (torch.optim
creates state as zeros_like(bf16 param)), FP4 only via torchao NVFP4 QAT forward. FP32 lives
only in HW tensor-core accumulators and negligible scale/loss scalars.

Phases (via `precision` in the config):
  bf16  -> full-param BF16, the Phase-1 correctness gate.
  nvfp4 -> + torchao NVFP4 QAT: FP4 forward, BF16 backward (Phase 2). Native FP4 kernels
           need torch>=2.11; on 2.10 torchao falls back to emulation (correct, not fast).

Launch (2-GPU FSDP2):
  .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/train_cpt.py --config configs/train_edc_lv_2026.yaml
"""
from __future__ import annotations
import argparse, os, time
from collections import Counter, UserDict

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from transformers import Gemma4ForCausalLM

import config as C


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def log(msg: str) -> None:
    if rank0():
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wsd_lr(step, warmup, total, base_lr, min_lr, decay_frac):
    """Warmup-Stable-Decay learning-rate schedule."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return base_lr
    t = (step - decay_start) / max(1, total - decay_start)
    return min_lr + (base_lr - min_lr) * (1 - t)


def entity_weighted_ce(logits, ids, entity_mask, entity_weight):
    """Lever 1: per-token cross-entropy weighted toward entity/date/time TARGET tokens.

    Causal shift: position t predicts token t+1, so the weight applied to the loss at position t must
    reflect whether the TARGET token (t+1) is inside an entity span — hence entity_mask[:, 1:]. Get this
    off by one and the lever does nothing useful. WEIGHTED-MEAN normalization (sum(w*l)/sum(w), not the
    plain sum/N) keeps the loss SCALE ~constant as entity_weight varies, so the LR does not need
    retuning across the entity_loss_weight sweep. entity_weight=1.0 reduces exactly to the baseline mean CE."""
    V = logits.size(-1)
    flat_logits = logits[:, :-1, :].reshape(-1, V)          # [B*(T-1), V]
    targets = ids[:, 1:].reshape(-1)                        # [B*(T-1)]
    per_tok = F.cross_entropy(flat_logits, targets, reduction="none")
    w_ent = entity_mask[:, 1:].reshape(-1).to(per_tok.dtype)   # 1.0 where the TARGET token is an entity
    w = 1.0 + (entity_weight - 1.0) * w_ent
    return (w * per_tok).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def _sr_int8_master(params):
    """Constrain each 2-D weight to the symmetric per-output-row INT8 grid via STOCHASTIC ROUNDING,
    after the optimizer step. Tests whether INT8 weight precision holds knowledge install (the science
    question) without the memory-saving INT8-storage plumbing: weights stay bf16 but may only take
    INT8-representable values. 1-D params (norms/biases) are left full-precision (sensitive, tiny)."""
    for p in params:
        if p.dim() < 2:
            continue
        w = p.data
        scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min_(1e-8) / 127.0
        x = w / scale
        lo = torch.floor(x)
        q = lo + (torch.rand_like(x) < (x - lo)).to(x.dtype)   # stochastic rounding: P(up) = frac
        p.data.copy_(q.clamp_(-127, 127) * scale)


def apply_nvfp4_qat(model, cfg) -> None:
    """torchao NVFP4 QAT prepare with selective precision: skip embeddings, lm_head,
    and the first/last few decoder layers (kept BF16 per the NVFP4 sensitivity finding)."""
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

    n = cfg["nvfp4"]
    n_layers = model.config.num_hidden_layers
    bf16_layers = set(range(n["bf16_first_layers"])) | set(
        range(n_layers - n["bf16_last_layers"], n_layers)
    )

    def keep_bf16(name: str) -> bool:
        if n["skip_lm_head"] and "lm_head" in name:
            return True
        if n["skip_embeddings"] and ("embed_tokens" in name or "embedding" in name):
            return True
        return any(f".layers.{i}." in name for i in bf16_layers)

    def filter_fn(mod, name: str) -> bool:
        return isinstance(mod, torch.nn.Linear) and not keep_bf16(name)

    quantize_(model, QATConfig(NVFP4DynamicActivationNVFP4WeightConfig(), step="prepare"),
              filter_fn=filter_fn)
    log(f"NVFP4 QAT prepared; BF16-kept layers={sorted(bf16_layers)} + lm_head + embeds")


def swap_to_quartet(model, cfg) -> None:
    """Direct 4-bit: replace eligible nn.Linear with Quartet_II_linear (real NVFP4 GEMM
    fwd + all-GEMM FP4 bwd with RHT128 rotation), reusing the pretrained weights. Keep
    embeddings, lm_head, and the first/last few decoder layers in BF16 (NVFP4 sensitivity).
    Skip any Linear whose in/out aren't multiples of 16 (NVFP4 block) so the FP4 GEMM is valid."""
    import torch.nn as nn
    from quartet2.linear import Quartet_II_linear

    n = cfg["nvfp4"]
    n_layers = model.config.num_hidden_layers
    bf16_layers = set(range(n["bf16_first_layers"])) | set(
        range(n_layers - n["bf16_last_layers"], n_layers)
    )

    def keep_bf16(name: str) -> bool:
        if n["skip_lm_head"] and "lm_head" in name:
            return True
        if n["skip_embeddings"] and ("embed_tokens" in name or "embedding" in name):
            return True
        return any(f".layers.{i}." in name for i in bf16_layers)

    swapped = 0
    for mod_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if not isinstance(child, nn.Linear) or isinstance(child, Quartet_II_linear):
                continue
            if keep_bf16(full) or (child.in_features % 16) or (child.out_features % 16):
                continue
            q = Quartet_II_linear(child.in_features, child.out_features,
                                  bias=child.bias is not None,
                                  device=child.weight.device, dtype=child.weight.dtype)
            q.weight = child.weight                       # reuse pretrained Parameter
            if child.bias is not None:
                q.bias = child.bias
            setattr(module, child_name, q)
            swapped += 1
    log(f"Quartet: swapped {swapped} Linear -> Quartet_II_linear "
        f"(BF16-kept layers={sorted(bf16_layers)} + lm_head + embeds)")


def build_model(cfg):
    # The checkpoint is the MULTIMODAL Gemma4ForConditionalGeneration: its text-decoder weights
    # live under `model.language_model.*`. Loading them into the text-only Gemma4ForCausalLM
    # (which expects `model.*`) matches NOTHING → every param random-inits → NaN loss. So we read
    # the safetensors directly and remap `model.language_model.X → model.X`, keeping only the text
    # decoder (vision/audio towers dropped). The unexpected k/v/k_norm of the KV-sharing layers
    # (15–34) are unused by shared-KV attention. Verified to match the canonical
    # ForConditionalGeneration loss (13.68 vs 13.76 on identical input).
    from transformers import AutoConfig
    from safetensors.torch import load_file
    text_cfg = AutoConfig.from_pretrained(cfg["model_path"]).text_config
    text_cfg._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(text_cfg)
    sd = load_file(f"{cfg['model_path']}/model.safetensors")
    P = "model.language_model."
    remapped = {("model." + k[len(P):]): v for k, v in sd.items() if k.startswith(P)}
    if "lm_head.weight" in sd:
        remapped["lm_head.weight"] = sd["lm_head.weight"]
    # Guard the NaN-incident class (silent random-init) on BOTH sides: a wrong prefix yields a tiny
    # remap → fail loud before training on noise.
    if len(remapped) < 0.5 * len(model.state_dict()):
        raise RuntimeError(f"remap produced only {len(remapped)} tensors for {len(model.state_dict())} "
                           f"model params — prefix '{P}' is wrong for this checkpoint")
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    model.tie_weights()                                  # lm_head ties to embed_tokens
    real_missing = [m for m in missing if "lm_head" not in m]
    if real_missing:
        raise RuntimeError(f"text params missing from checkpoint (load remap is wrong): {real_missing[:10]}")
    # Embeddings are about to be FROZEN — if the remap silently missed them they'd freeze at random
    # init and the decoder would train on noise (the prior incident). Verify they're real, not degenerate.
    embed = model.get_input_embeddings().weight
    if not torch.isfinite(embed).all() or embed.abs().sum().item() == 0.0:
        raise RuntimeError("embed_tokens is non-finite or all-zero after remap — refusing to train/freeze it")
    model = model.to(torch.bfloat16)
    n_text = sum(p.numel() for p in model.parameters())
    log(f"loaded text decoder from multimodal ckpt: {len(remapped)} tensors → {n_text/1e9:.2f}B params "
        f"(dropped {len(sd)-len(remapped)} tower tensors, {len(unexpected)} unused shared-KV k/v)")
    if cfg.get("freeze_embeddings"):
        # Freeze ALL embedding tables — incl. Gemma's huge Per-Layer Embeddings
        # (embed_tokens_per_layer ≈ 2.35B params / 4.7GB). CPT trains the decoder, not
        # embeddings; freezing drops their grads+optimizer state and lets FSDP reshard
        # them after forward (no backward grad needed). lm_head is tied to embed_tokens.
        frozen = 0
        for name, p in model.named_parameters():
            if "embed" in name.lower():
                p.requires_grad_(False)
                frozen += p.numel()
        log(f"froze {frozen/1e9:.2f}B embedding params (embed_tokens + PLE + tied lm_head)")
    # Gemma4 KV-sharing layers (15–34) read `shared_kv_states[layer_type]`, populated by the
    # store_full_length_kv layer. That population (modeling_gemma4 L1252-3) is gated on the layer's
    # store_full_length_kv attr, NOT on use_cache — so it works with use_cache=False too. The earlier
    # 'sliding_attention' KeyError was FSDP2's tree_map rebuilding the plain-dict shared_kv_states,
    # fixed by passing it as a UserDict. So activation checkpointing IS compatible with KV-sharing:
    # validated at seq512 (loss finite/stable, 11.2GB — less than seq256 without AC). AC needs use_cache=False.
    model.config.use_cache = not bool(cfg.get("gradient_checkpointing"))
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    if cfg["precision"] == "fp4":
        swap_to_quartet(model, cfg)         # direct 4-bit via Quartet-II
    elif cfg["precision"] == "nvfp4":
        apply_nvfp4_qat(model, cfg)         # legacy torchao QAT path
    return model


def shard_model(model, cfg):
    """FSDP2 full-shard, BF16 throughout (no FP32 reduce), with reshard_after_forward=True
    on EVERY unit so no full param copy lingers past its use.

    Crucial for Gemma's Per-Layer Embeddings: the 4.7GB `embed_tokens_per_layer` table is a
    root-level param. If it stays in the root unit (whose default reshard_after_forward=False)
    its full copy is held through fwd+bwd → OOM. We give it (and embed_tokens) their own units
    so FSDP reshards them right after the lookup. Optionally CPU-offload those frozen tables
    (cfg.cpu_offload_frozen) — they have no optimizer state, so the trainable decoder stays on
    GPU with the 8-bit optimizer."""
    import torch.nn as nn
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    # cast_forward_inputs=False: with it True (the default), FSDP2 tree_maps the forward
    # kwargs and REBUILDS plain dicts, which breaks Gemma4's shared_kv_states dict passed
    # across layers -> KeyError 'sliding_attention' (transformers #45912). Everything is
    # already bf16, so no input cast is needed and skipping it preserves the dict identity.
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                              cast_forward_inputs=False)

    inner = getattr(model, "model", model)
    # The frozen Per-Layer-Embeddings table (embed_tokens_per_layer ≈ 2.35B / 4.7GB) is too
    # big to all-gather onto a 16GB GPU every step (the gather + its ~2x copy buffers blow
    # past 16GB). Since it's frozen and fits whole on one GPU, DON'T shard it: exclude it via
    # ignored_params so FSDP never gathers it — it sits replicated in place on each GPU (no
    # grad/optimizer cost). This removes the all-gather spike entirely.
    ignored = set()
    ple = getattr(inner, "embed_tokens_per_layer", None)
    if isinstance(ple, nn.Module):
        ignored |= set(ple.parameters())

    lists = [m for _, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) > 4]
    layers = max(lists, key=len) if lists else []
    for layer in layers:                                          # decoder layers -> own units
        fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp, reshard_after_forward=True,
                ignored_params=ignored if ignored else None)      # root; PLE excluded
    try:
        model.set_reshard_after_forward(True)                     # root is force-False (PyTorch #149029)
    except Exception as e:
        log(f"set_reshard_after_forward unavailable: {e}")
    log(f"FSDP2: {len(layers)} layers + root; PLE "
        f"({sum(p.numel() for p in ignored)/1e9:.2f}B params) replicated via ignored_params")
    return model


def cpu_offload_ple(model, device):
    """Stream Gemma's frozen Per-Layer-Embeddings (4.7GB) from CPU instead of holding it on every
    GPU. The table is a pure lookup, frozen (no grad into it, no optimizer state), so we keep its
    weight on CPU and bounce only the (tiny) per-token result to the GPU. This is the user's
    'train on the GPU, stream the weights from the CPU' design — it frees the 4.7GB that otherwise
    sits replicated on each 16GB card. Returns the offloaded param count (0 if PLE absent)."""
    import torch.nn as nn
    inner = getattr(model, "model", model)
    ple = getattr(inner, "embed_tokens_per_layer", None)
    if not isinstance(ple, nn.Module):
        return 0
    ple.to("cpu")                                  # undo the to(device) move for this table
    orig_forward = ple.forward                     # bound method (weight/embed_scale now on CPU)
    def fwd(input_ids, *a, **k):
        return orig_forward(input_ids.to("cpu"), *a, **k).to(device, non_blocking=True)
    ple.forward = fwd
    return sum(p.numel() for p in ple.parameters())


def make_loader(cfg, rank, world, want_mask=False, want_facts=False):
    ds = load_from_disk(str(C.ROOT / cfg["tokenized"]))
    has_mask = want_mask and "entity_mask" in ds.column_names
    has_facts = want_facts and "fact_ids" in ds.column_names

    def collate(rows):
        ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        if not (has_mask or has_facts):
            return ids                              # backward-compatible: plain tensor when no lever needs meta
        out = {"input_ids": ids}
        if has_mask:
            out["entity_mask"] = torch.tensor([r["entity_mask"] for r in rows], dtype=torch.long)
        if has_facts:
            out["fact_ids"] = [r["fact_ids"] for r in rows]   # list-of-lists (variable length, kept as python)
        return out

    sampler = (
        DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        if world > 1 else None
    )
    dl = DataLoader(ds, batch_size=cfg["micro_batch_size"], sampler=sampler,
                    shuffle=(sampler is None), collate_fn=collate, drop_last=True)
    if want_mask and not has_mask:
        log("[lever1] entity_loss enabled but corpus has no entity_mask column — "
            "re-tokenize without --no-lever-metadata; falling back to unweighted CE")
    if want_facts and not has_facts:
        log("[lever4] exposure_tracking enabled but corpus has no fact_ids column — re-tokenize first")
    return dl, sampler


def _save_exposure(ckpt_dir, step, exposure):
    """Lever 4: write the cumulative {fact_id: count} snapshot beside a checkpoint, so the report can
    join probe pass/fail at this step to how many times each fact had been seen by this step."""
    import json as _json
    import os as _os
    _os.makedirs(ckpt_dir, exist_ok=True)
    with open(f"{ckpt_dir}/exposure.json", "w") as f:
        _json.dump({"step": step, "exposure": dict(exposure)}, f)


def save_checkpoint(model, cfg, out_dir, rank, world):
    """Consolidate the FSDP2-sharded params to a full state dict and save a HF-loadable text-only
    checkpoint (Gemma4ForCausalLM) on rank 0. eval/serving load it with plain from_pretrained."""
    if world > 1:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        full_sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    else:
        full_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    if rank != 0:
        return
    from transformers import AutoConfig, AutoTokenizer
    tcfg = AutoConfig.from_pretrained(cfg["model_path"]).text_config
    tcfg._attn_implementation = "sdpa"
    if not full_sd:
        raise RuntimeError("FSDP2 consolidation returned an empty state dict on rank 0 — not saving")
    out = Gemma4ForCausalLM(tcfg)
    missing, unexpected = out.load_state_dict(full_sd, strict=False)
    out.tie_weights()
    # VALIDATE BEFORE WRITING — a partial consolidation must abort, not silently save a broken ckpt
    # discovered only at eval (after the GPUs are released).
    real_missing = [m for m in missing if "lm_head" not in m]
    stray = [u for u in unexpected if not any(t in u for t in (".k_proj", ".v_proj", ".k_norm"))]
    if real_missing or stray:
        raise RuntimeError(f"refusing to save broken checkpoint: missing={real_missing[:8]} "
                           f"stray_unexpected={stray[:8]}")
    out.to(torch.bfloat16).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cfg["model_path"]).save_pretrained(out_dir)
    log(f"saved checkpoint -> {out_dir}  (real_missing=0 unexpected={len(unexpected)} all shared-KV)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N steps on random tokens (no dataset) to validate loop + memory")
    ap.add_argument("--precision", choices=["bf16", "fp4", "nvfp4"], help="override config precision")
    ap.add_argument("--no-ac", action="store_true",
                    help="disable gradient checkpointing (uses more memory; only fits short seq). "
                         "Default (AC on) is validated with KV-sharing and fits seq512 at ~11GB.")
    ap.add_argument("--seq-len", type=int, help="override seq_len (e.g. smaller for smoke memory headroom)")
    ap.add_argument("--save-dir", help="save the trained text-decoder checkpoint here (HF save_pretrained)")
    ap.add_argument("--optimizer-lowbit", type=int, default=0, choices=[0, 4, 8],
                    help="torchao AdamW{4,8}bit states (single-GPU only; saves optimizer memory). 0 = bf16 AdamW")
    ap.add_argument("--grad-accum", type=int, help="override grad_accum (e.g. match effective batch single-GPU)")
    ap.add_argument("--int8-master", action="store_true",
                    help="after each step, SR-snap 2-D weights to the symmetric INT8 grid (precision test: "
                         "does INT8 weight precision hold install; bf16 storage, not the memory-saving variant)")
    ap.add_argument("--lever1-weight", type=float, default=None,
                    help="enable Lever 1 (entity-targeted loss) at this entity_loss_weight (sweep knob; "
                         "overrides config). 1.0 == baseline mean CE.")
    ap.add_argument("--lever4", action="store_true", help="enable Lever 4 exposure tracking (overrides config)")
    ap.add_argument("--max-steps", type=int, help="override max_steps (e.g. a short real-data lever validation)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.precision:
        cfg["precision"] = args.precision
    if args.no_ac:
        cfg["gradient_checkpointing"] = False
    if args.seq_len:
        cfg["seq_len"] = args.seq_len
    if args.grad_accum:
        cfg["grad_accum"] = args.grad_accum
    # CLI lever overrides (for clean A/B sweeps without editing the config per run)
    if args.lever1_weight is not None:
        cfg.setdefault("levers", {}).setdefault("lever1_entity_loss", {})
        cfg["levers"]["lever1_entity_loss"].update(enabled=True, entity_loss_weight=args.lever1_weight)
    if args.lever4:
        cfg.setdefault("levers", {}).setdefault("lever4_exposure_tracking", {})["enabled"] = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    log(f"world={world} precision={cfg['precision']} run={cfg['run_name']} smoke={args.smoke}")
    model = build_model(cfg)            # on CPU (bf16) — do NOT move full model to GPU
    if world > 1:
        shard_model(model, cfg)         # fully_shard in place → params become sharded
    model.to(device)                    # moves only this rank's LOCAL shard (~params/world), not the full model
    if cfg.get("cpu_offload_ple", True):
        n = cpu_offload_ple(model, device)   # stream the 4.7GB frozen PLE from CPU (user's "stream from CPU")
        if n:
            log(f"CPU-offloaded PLE: {n/1e9:.2f}B params stream from host RAM (off-GPU)")
    torch.cuda.empty_cache()            # release the transient full-model reservation from the sharded CPU->GPU move
    torch.cuda.reset_peak_memory_stats(local_rank)
    print(f"[rank{rank}] after shard+to(device): gpu_alloc={torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved(local_rank)/1e9:.2f}GB "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','unset')}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    # torch.optim.AdamW dispatches over DTensor natively and creates state with zeros_like(param) →
    # bf16 params give bf16 m/v states (NO FP32 master). 8-bit states reclaim ~1.6GB/GPU (headroom for
    # the GPU1 knife-edge) — historically broken under FSDP2+DTensor (torchao #1189/#1505), so it is
    # OPT-IN and re-validated by smoke each stack bump. Falls back to bf16 AdamW if it errors.
    # Low-bit optimizer states (4 or 8) reclaim optimizer memory for the single-GPU path. torchao's
    # OptimState{4,8}bit is DTensor-incompatible (breaks under FSDP2, world>1) — but world==1 runs plain
    # (no FSDP, no DTensor, line 327), so it works there. Gated accordingly; falls back to bf16 AdamW.
    lowbit = args.optimizer_lowbit or cfg.get("optimizer_lowbit", 0)
    opt = None
    if lowbit in (4, 8):
        if world > 1:
            log(f"optimizer_lowbit={lowbit} ignored: torchao OptimState is DTensor-incompatible under FSDP2 (world={world}); using bf16 AdamW")
        else:
            try:
                from torchao.optim import AdamW4bit, AdamW8bit
                Opt = AdamW4bit if lowbit == 4 else AdamW8bit
                opt = Opt(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                          weight_decay=cfg["weight_decay"], eps=cfg["adam_eps"])
                log(f"optimizer: torchao AdamW{lowbit}bit (single-GPU, {lowbit}-bit m/v states)")
            except Exception as e:
                log(f"optimizer_lowbit={lowbit} failed ({type(e).__name__}: {e}); falling back to bf16 AdamW")
    if opt is None:
        opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                weight_decay=cfg["weight_decay"], eps=cfg["adam_eps"], foreach=True)
    # Enforce the no-FP32 policy at runtime (not just in a comment): every trainable param must be bf16,
    # else AdamW allocates FP32 m/v state — silent policy violation + ~2x optimizer memory.
    bad_dtype = [(n, str(p.dtype)) for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.bfloat16]
    if bad_dtype:
        raise RuntimeError(f"no-FP32 policy violated — non-bf16 trainable params: {bad_dtype[:5]}")

    accum = cfg["grad_accum"]
    _levers = cfg.get("levers", {}) or {}
    # Lever 1 (entity/date-targeted loss): off by default; needs the entity_mask the tokenizer emits.
    lev1 = _levers.get("lever1_entity_loss", {}) or {}
    lever1_on = bool(lev1.get("enabled", False))
    entity_w = float(lev1.get("entity_loss_weight", 2.5))
    # Lever 4 (exposure→extraction instrument): passive; counts fact_id exposures, snapshots at checkpoints.
    lev4 = _levers.get("lever4_exposure_tracking", {}) or {}
    lever4_on = bool(lev4.get("enabled", False))
    exposure = Counter()
    if args.smoke:
        total, accum = args.smoke, min(accum, 2)        # quick loop+memory validation
        cfg["log_every"] = 1                            # log every step so the smoke shows loss+peakmem
        vocab = model.config.vocab_size
        lever1_on = lever4_on = False                   # smoke uses random ids (no metadata)

        def _smoke_iter():
            while True:
                yield torch.randint(0, vocab, (cfg["micro_batch_size"], cfg["seq_len"]), dtype=torch.long)
        dl, sampler = _smoke_iter(), None
    else:
        dl, sampler = make_loader(cfg, rank, world, want_mask=lever1_on, want_facts=lever4_on)
        total = cfg["max_steps"]
        if rank0() and lever1_on:
            log(f"[lever1] entity-targeted loss ON — entity_loss_weight={entity_w} (weighted-mean CE)")
        if rank0() and lever4_on:
            log("[lever4] exposure tracking ON — fact_id exposures snapshotted at each checkpoint")
    model.train()
    opt.zero_grad(set_to_none=True)

    # FSDP2 grad-accum optimization: reduce-scatter gradients only on the LAST microbatch of each
    # window. The other (accum-1) backwards accumulate locally — IDENTICAL math, but skips ~15/16 of
    # the cross-GPU gradient comm (the bottleneck on PCIe-linked consumer GPUs). Costs ~one unsharded
    # grad buffer (~1.9 GB) of memory; we have ~4.6 GB headroom at 11.2 GB peak.
    _can_nosync = hasattr(model, "set_requires_gradient_sync")
    if rank0():
        log(f"FSDP grad-accum no_sync: {'ON (sync only on last microbatch)' if _can_nosync else 'unavailable — full sync each microbatch'}")
    step, micro, window_loss, t0 = 0, 0, 0.0, time.time()
    data_iter = iter(dl)
    while step < total:
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler:
                sampler.set_epoch(step)
            data_iter = iter(dl)
            batch = next(data_iter)
        if isinstance(batch, dict):              # levers return {input_ids[, entity_mask][, fact_ids]}
            ids = batch["input_ids"].to(device)
            emask = batch["entity_mask"].to(device) if "entity_mask" in batch else None
            if lever4_on and "fact_ids" in batch:   # count this microbatch's fact exposures (per-block lists)
                for fids in batch["fact_ids"]:
                    exposure.update(fids)
        else:
            ids, emask = batch.to(device), None
        if _can_nosync:                          # sync grads only on the last microbatch of the window
            model.set_requires_gradient_sync(micro % accum == accum - 1)
        # Pass shared_kv_states as a UserDict: Gemma4's `kwargs.pop("shared_kv_states", {})`
        # uses ours, and FSDP2's tree_map treats a UserDict as a leaf (won't rebuild it), so
        # the KV-sharing layers see the dict the producer layers populated (transformers #45912).
        if lever1_on and emask is not None:      # Lever 1: manual entity-weighted CE on logits (no labels=)
            out = model(input_ids=ids, shared_kv_states=UserDict())
            loss = entity_weighted_ce(out.logits, ids, emask, entity_w)
        else:
            out = model(input_ids=ids, labels=ids, shared_kv_states=UserDict())
            loss = out.loss
        loss_val = loss.item()
        # Fail loud on non-finite loss BEFORE backward — a single NaN/inf would poison the AdamW moment
        # buffers on opt.step() and turn every weight NaN for the rest of the run (the project's known
        # failure signature). Abort at the first occurrence, not after 200 silent steps.
        if loss_val != loss_val or loss_val in (float("inf"), float("-inf")):
            raise RuntimeError(f"non-finite loss ({loss_val}) at step {step} micro {micro} — aborting "
                               f"before the optimizer poisons all weights")
        (loss / accum).backward()
        window_loss += loss_val
        micro += 1
        if micro % accum == 0:
            lr = wsd_lr(step, cfg["warmup_steps"], total, cfg["lr"], cfg["min_lr"], cfg["decay_frac"])
            for g in opt.param_groups:
                cur = g["lr"]
                if isinstance(cur, torch.Tensor):
                    cur.fill_(lr)          # torchao keeps lr as a Tensor (torch.compile-friendly)
                else:
                    g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            if args.int8_master:
                _sr_int8_master(params)     # constrain weights to the INT8 grid (SR) — precision test
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["log_every"] == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                log(f"step {step}/{total} loss {window_loss/(accum*cfg['log_every']):.4f} "
                    f"lr {lr:.2e} peakmem {mem:.1f}GB {(time.time()-t0)/step:.2f}s/step")
                window_loss = 0.0
            # Honor save_every (was silently ignored): periodic checkpoint → crash-recovery for long
            # fleet runs + the per-step checkpoints the geometry probe sweeps to watch geometry form.
            se = cfg.get("save_every", 0)
            if args.save_dir and not args.smoke and se and step % se == 0 and step < total:
                save_checkpoint(model, cfg, f"{args.save_dir}/step_{step:06d}", rank, world)
                if lever4_on and rank0():    # snapshot cumulative fact exposures alongside the checkpoint
                    _save_exposure(f"{args.save_dir}/step_{step:06d}", step, exposure)
                model.train()

    log(f"done {step} steps in {time.time()-t0:.0f}s")
    if args.save_dir and not args.smoke:
        save_checkpoint(model, cfg, args.save_dir, rank, world)
        if lever4_on and rank0():
            _save_exposure(args.save_dir, step, exposure)
    # NOTE: NVFP4 QAT→convert is intentionally NOT done here. Running quantize_(step="convert") on the
    # live FSDP2-sharded model on rank 0 only would (a) risk a collective deadlock and (b) discard the
    # result (it's never re-saved). The convert is a separate OFFLINE step: load the saved bf16
    # checkpoint single-process, quantize_, save. (TODO: scripts/convert_nvfp4.py)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
