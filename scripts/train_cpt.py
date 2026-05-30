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
from collections import UserDict

import yaml
import torch
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
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    model.tie_weights()                                  # lm_head ties to embed_tokens
    real_missing = [m for m in missing if "lm_head" not in m]
    if real_missing:
        raise RuntimeError(f"text params missing from checkpoint (load remap is wrong): {real_missing[:10]}")
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


def make_loader(cfg, rank, world):
    ds = load_from_disk(str(C.ROOT / cfg["tokenized"]))

    def collate(rows):
        return torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)

    sampler = (
        DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        if world > 1 else None
    )
    dl = DataLoader(ds, batch_size=cfg["micro_batch_size"], sampler=sampler,
                    shuffle=(sampler is None), collate_fn=collate, drop_last=True)
    return dl, sampler


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
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.precision:
        cfg["precision"] = args.precision
    if args.no_ac:
        cfg["gradient_checkpointing"] = False
    if args.seq_len:
        cfg["seq_len"] = args.seq_len

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
    # torch.optim.AdamW is the reference FSDP2 optimizer: it dispatches over DTensor natively and
    # creates state with zeros_like(param) → bf16 params give bf16 m/v states, so NO FP32 master.
    # (torchao's 8-bit Adam saves more memory but its compiled OptimState8bit path is broken under
    # FSDP2+DTensor on this stack — torchao #1189/#1505; 8-bit/CPU-offload is a real-run lever.)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"], eps=cfg["adam_eps"], foreach=True)

    accum = cfg["grad_accum"]
    if args.smoke:
        total, accum = args.smoke, min(accum, 2)        # quick loop+memory validation
        cfg["log_every"] = 1                            # log every step so the smoke shows loss+peakmem
        vocab = model.config.vocab_size

        def _smoke_iter():
            while True:
                yield torch.randint(0, vocab, (cfg["micro_batch_size"], cfg["seq_len"]), dtype=torch.long)
        dl, sampler = _smoke_iter(), None
    else:
        dl, sampler = make_loader(cfg, rank, world)
        total = cfg["max_steps"]
    model.train()
    opt.zero_grad(set_to_none=True)

    step, micro, window_loss, t0 = 0, 0, 0.0, time.time()
    data_iter = iter(dl)
    while step < total:
        try:
            ids = next(data_iter)
        except StopIteration:
            if sampler:
                sampler.set_epoch(step)
            data_iter = iter(dl)
            ids = next(data_iter)
        ids = ids.to(device)
        # Pass shared_kv_states as a UserDict: Gemma4's `kwargs.pop("shared_kv_states", {})`
        # uses ours, and FSDP2's tree_map treats a UserDict as a leaf (won't rebuild it), so
        # the KV-sharing layers see the dict the producer layers populated (transformers #45912).
        out = model(input_ids=ids, labels=ids, shared_kv_states=UserDict())
        (out.loss / accum).backward()
        window_loss += out.loss.item()
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
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["log_every"] == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                log(f"step {step}/{total} loss {window_loss/(accum*cfg['log_every']):.4f} "
                    f"lr {lr:.2e} peakmem {mem:.1f}GB {(time.time()-t0)/step:.2f}s/step")
                window_loss = 0.0

    log(f"done {step} steps in {time.time()-t0:.0f}s")
    if cfg["precision"] == "nvfp4" and rank0():
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        quantize_(model, QATConfig(NVFP4DynamicActivationNVFP4WeightConfig(), step="convert"))
        log("converted QAT -> NVFP4 weights")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
