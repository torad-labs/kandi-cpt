#!/usr/bin/env python3
"""Full-parameter CPT/SFT trainer for Gemma 4 E2B (text decoder) — NVFP4-capable.

╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║  THE ONE TRAINING ENTRYPOINT. It accepts ONLY --config <path> and --mode {cpt,sft}.            ║
║  There are NO behavior-affecting flags — the config IS the run (validated at load by            ║
║  scripts/cpt_config.py; unknown keys hard-error). Do NOT add a flag, do NOT make a second       ║
║  trainer. New knobs go in a sub-config + schema. Sweeps generate configs (scripts/gen_config.py)║
║  and call this with --config only. Full rule: AGENTS.md + docs/CONFIG_AUDIT.md.                 ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

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
import argparse, json, os, sys, time
from collections import Counter, UserDict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset, load_from_disk
from transformers import Gemma4ForCausalLM

import config as C
from cpt_config import CptConfig


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


# ── Torad Train live heartbeat ─────────────────────────────────────────────
# The console server (scripts/serve.py) reflects training live by tailing a
# canonical per-run log and polling a status file. We write both here, rank0
# only, best-effort — a failure here must NEVER interrupt a paid GPU run.
_LOGSINK = None
_STATUS_PATH = None


def _start_status(save_dir, total):
    global _LOGSINK, _STATUS_PATH
    try:
        from pathlib import Path
        sd = Path(save_dir); sd.mkdir(parents=True, exist_ok=True)
        _LOGSINK = open(sd / "train.log", "a")
        _STATUS_PATH = Path(__file__).resolve().parents[1] / "data" / "_build" / "train_status.json"
        _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_path = str(sd / "train.log")
        try:  # if a chain script (run_vN.sh) pre-registered this run with its own combined log,
            prev = json.loads(_STATUS_PATH.read_text())   # keep it so build→train→gate is ONE live view
            if prev.get("run") == sd.name and prev.get("log_path") and Path(prev["log_path"]).exists():
                log_path = prev["log_path"]
        except Exception:
            pass
        _STATUS_PATH.write_text(json.dumps({
            "run": sd.name, "log_path": log_path, "pid": os.getpid(),
            "total_steps": total, "started": time.time(), "status": "running", "phase": "training"}))
    except Exception:
        _LOGSINK = None


def _finish_status(status="done"):
    global _LOGSINK
    try:
        if _STATUS_PATH and _STATUS_PATH.is_file():
            d = json.loads(_STATUS_PATH.read_text()); d["status"] = status; d["ended"] = time.time()
            _STATUS_PATH.write_text(json.dumps(d))
        if _LOGSINK:
            _LOGSINK.flush(); _LOGSINK.close(); _LOGSINK = None
    except Exception:
        pass


def log(msg: str) -> None:
    if rank0():
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if _LOGSINK:
            try:
                _LOGSINK.write(line + "\n"); _LOGSINK.flush()
            except Exception:
                pass


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



# ── int8 master weights + stochastic rounding (THE ENGINE MOAT) ─────────────────────────────────
# Each bias-free Quartet linear's weight becomes torchao's Int8QuantizedTrainingLinearWeight: int8-STORED
# (1 byte/param) + per-output-row scale, with STOCHASTIC ROUNDING on every in-place optimizer op (copy_/add_/
# addcdiv_ — torchao quantized_training/int8.py). It is driven by the SAME low-bit AdamW the rest of the model
# uses (torchao AdamW4bit, optimizer_lowbit=4) — so the master AND its Adam moments are all low precision, never
# higher than the FP4 we train in. The only custom piece is dequant_bridge: it dequantizes the int8 master to a
# TRANSIENT bf16 view for the Quartet FP4 kernel and routes the bf16 cotangent onto the subclass param's .grad
# (transient — freed by zero_grad, no g_acc). Net: int8 weight (1B) + 4-bit moments (~1B) = the proven "snap"
# trajectory (V9) at 1-byte storage and BELOW snap's memory, with no hand-rolled optimizer.
def convert_quartet_to_master(model):
    import torch.nn as nn
    from quartet2.linear import Quartet_II_linear, Quartet_II_fn, Quartet_II_pseudoquant_fn
    from torchao.prototype.quantized_training import Int8QuantizedTrainingLinearWeight
    from quartet_bridge import dequant_bridge

    class _MasterMixin:
        def forward(self, x, disable_backward_quant=False, input_abs_max=None):
            w = dequant_bridge(self.weight)                      # transient bf16 view; grad → self.weight.grad
            major, _ = torch.cuda.get_device_capability(x.device)
            if major >= 10:
                return Quartet_II_fn.apply(x, w, self.had, self.mode,
                                           self.disable_backward_quant or disable_backward_quant,
                                           self.weight_abs_max, input_abs_max, self.scratch_amax)
            return Quartet_II_pseudoquant_fn.apply(x, w, self.had,
                                                   self.disable_backward_quant or disable_backward_quant,
                                                   self.mode == self.mode.FOUR_SIX)

    master_params = []
    for module in model.modules():
        for cn, child in list(module.named_children()):
            if type(child) is Quartet_II_linear and child.bias is None:
                child.__class__ = type("QuartetMasterLinear", (_MasterMixin, Quartet_II_linear), {})
                child.weight = nn.Parameter(Int8QuantizedTrainingLinearWeight.from_float(child.weight.detach()))
                master_params.append(child.weight)
    log(f"int8-master MOAT: {len(master_params)} Quartet linears → torchao Int8 store (1 byte/param, SR on every "
        f"optimizer update; updated by the shared low-bit AdamW — FP4-native compute)")
    return master_params


# ── INT8 stochastic-rounding gradient accumulation ──────────────────────────────────────────────
# The gradients are the fattest transient chunk: every 2-D trainable param holds a bf16 .grad (2 B/param,
# ~3.8GB here) live across the grad-accum window. We fold each param's bf16 grad into a persistent INT8
# accumulator (1 B/param) inside a post-accumulate-grad hook and free the bf16 .grad immediately — so with
# gradient checkpointing only ~one layer's bf16 grad is alive at a time during backward. STOCHASTIC ROUNDING
# keeps the low-precision running sum unbiased (the same principle that makes the int8 master work). At the
# window end materialize() rebuilds p.grad from the accumulator (and frees the int8 buffer so the bf16 grad
# the optimizer needs and the int8 buffer never coexist at full size); reset() empties it for the next window.
def enable_int8_grad_accum(params):
    from torchao.prototype.quantized_training import quantize_int8_rowwise
    targets = [p for p in params if p.requires_grad and p.dim() >= 2]   # 1-D norms are tiny → leave bf16
    buf, filled = {}, set()

    def make_hook(p):
        pid = id(p)
        def hook(param):
            g = param.grad
            if g is None:
                return
            if pid in filled:                                  # accumulate: dequant running sum + this microbatch
                d0, s0 = buf[pid]
                g = g + d0.to(s0.dtype) * s0.unsqueeze(1)
            d, s = quantize_int8_rowwise(g, stochastic_rounding=True)
            if pid in buf:
                buf[pid][0].copy_(d); buf[pid][1].copy_(s)
            else:
                buf[pid] = [d, s]
            filled.add(pid)
            param.grad = None                                  # free the bf16 grad immediately
        return hook

    for p in targets:
        p.register_post_accumulate_grad_hook(make_hook(p))

    class Ctl:
        def materialize(self):                                 # rebuild bf16 .grad for the optimizer, free int8 buf
            for p in targets:
                pid = id(p)
                if pid in filled:
                    d, s = buf.pop(pid)
                    p.grad = d.to(s.dtype) * s.unsqueeze(1)
        def reset(self):
            filled.clear()
    return Ctl()


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
    base_cfg = AutoConfig.from_pretrained(cfg["model_path"])
    text_cfg = getattr(base_cfg, "text_config", base_cfg)
    text_cfg._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(text_cfg)
    init = cfg.get("init_from")
    if init:
        # SFT / resumed CPT: a saved Gemma4ForCausalLM checkpoint (save_checkpoint) already has model.* +
        # lm_head keys — no multimodal remap. Route through the shared loader so an int8-master checkpoint
        # round-trips (dequant int8 _data × per-row scale → bf16); loading the raw int8 without its scale
        # would silently init from garbage weights. A plain bf16 checkpoint passes through unchanged.
        from pipeline.model_load import causal_bf16_state_dict
        remapped = causal_bf16_state_dict(C.ROOT / init)
        n_dropped = 0                                    # a saved CausalLM ckpt has no multimodal towers to drop
        log(f"init_from: loading CausalLM checkpoint '{init}' ({len(remapped)} tensors)")
    else:
        sd = load_file(f"{cfg['model_path']}/model.safetensors")
        P = "model.language_model."
        remapped = {("model." + k[len(P):]): v for k, v in sd.items() if k.startswith(P)}
        if "lm_head.weight" in sd:
            remapped["lm_head.weight"] = sd["lm_head.weight"]
        n_dropped = len(sd) - len(remapped)              # multimodal vision/audio tower tensors dropped
    # Guard the NaN-incident class (silent random-init) on BOTH sides: a wrong prefix yields a tiny
    # remap → fail loud before training on noise.
    if len(remapped) < 0.5 * len(model.state_dict()):
        raise RuntimeError(f"remap produced only {len(remapped)} tensors for {len(model.state_dict())} "
                           f"model params — prefix wrong for this checkpoint (init_from={bool(init)})")
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
    log(f"loaded text decoder: {len(remapped)} tensors → {n_text/1e9:.2f}B params "
        f"(dropped {n_dropped} tower tensors, {len(unexpected)} unused shared-KV k/v)")
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
    # Dual-format: a directory is the Python HF-datasets save_to_disk output; a file is the Rust
    # festival-factory plain Arrow IPC (tokenize.rs). Both expose input_ids[/entity_mask/src/fact_ids].
    p = C.ROOT / cfg["tokenized"]
    ds = Dataset.from_file(str(p)) if p.is_file() else load_from_disk(str(p))
    has_mask = want_mask and "entity_mask" in ds.column_names
    has_facts = want_facts and "fact_ids" in ds.column_names
    has_labels = "labels" in ds.column_names           # SFT corpus (build_sft.py): assistant-only masked labels

    def collate(rows):
        ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        if not (has_mask or has_facts or has_labels):
            return ids                              # backward-compatible: plain tensor when no lever needs meta
        out = {"input_ids": ids}
        if has_labels:
            out["labels"] = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
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
        # Hard error, not a warning: silently running unweighted CE would mislabel the run as
        # "lever1 entity_loss" in the A/B log (a corrupted experimental conclusion).
        raise RuntimeError("[lever1] entity_loss requested but corpus has no entity_mask column — "
                           "re-tokenize without --no-lever-metadata. Refusing to run a mislabeled baseline.")
    if want_mask and has_labels:
        # entity_weighted_ce uses input_ids as targets; an SFT corpus's -100 prompt mask would be ignored,
        # supervising the prompt tokens. Lever 1 (entity-weighted CE) is incompatible with SFT label masking.
        raise RuntimeError("[lever1] entity-weighted CE is incompatible with an SFT `labels` corpus "
                           "(it would supervise prompt tokens). Disable lever1 for SFT runs.")
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


_TOKENIZER_CACHE = {}


def _cached_tokenizer(model_path):
    """Load the base tokenizer ONCE and reuse it for every checkpoint save. save_checkpoint runs every
    save_every steps; re-reading model_path from disk on each save is fragile — a transient bad read of the
    base-model files under FS pressure (e.g. during the concurrent multi-GB weight write) crashed the v2 run
    at the step-3100 save. Cache the tokenizer (and use the in-memory model.config for the config) so a save
    never touches model_path on disk after the first one."""
    if model_path not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER_CACHE[model_path]


def _save_bf16_hf(full_sd, cfg, out_dir):
    """Standard HF bf16 save (snap/bf16 runs + multi-GPU consolidation): round-trip through a fresh model
    so the dir is a plain from_pretrained checkpoint. Only used when there are NO int8-master weights."""
    from transformers import AutoConfig
    full_sd = {k: (v.dequantize().cpu() if hasattr(v, "dequantize") else v.cpu()) for k, v in full_sd.items()}
    if not full_sd:
        raise RuntimeError("consolidation returned an empty state dict on rank 0 — not saving")
    base_cfg = AutoConfig.from_pretrained(cfg["model_path"])
    tcfg = getattr(base_cfg, "text_config", base_cfg)
    tcfg._attn_implementation = "sdpa"
    out = Gemma4ForCausalLM(tcfg)
    missing, unexpected = out.load_state_dict(full_sd, strict=False)
    out.tie_weights()
    real_missing = [m for m in missing if "lm_head" not in m]
    stray = [u for u in unexpected if not any(t in u for t in (".k_proj", ".v_proj", ".k_norm"))]
    if real_missing or stray:
        raise RuntimeError(f"refusing to save broken checkpoint: missing={real_missing[:8]} stray={stray[:8]}")
    out.to(torch.bfloat16).save_pretrained(out_dir)
    _cached_tokenizer(cfg["model_path"]).save_pretrained(out_dir)
    log(f"saved bf16 checkpoint -> {out_dir}  (unexpected={len(unexpected)} all shared-KV)")


def save_checkpoint(model, cfg, out_dir, rank, world):
    """Save a Gemma4ForCausalLM checkpoint on rank 0. For an int8-master run, save the weights in their NATIVE
    int8 form (int8 `_data` + per-row `scale`) — half the size, and crucially NO full-bf16 materialization: the
    old path built a 9GB bf16 state dict AND a second 9GB fresh model (~18GB), the spike that got the run
    killed at the first save. The shared loader (pipeline.model_load.load_causal_ckpt_bf16) dequantizes int8
    checkpoints on load. bf16/snap runs (no int8 masters) keep the standard from_pretrained save."""
    import json as _json
    import copy
    from pathlib import Path
    out_dir = Path(out_dir)
    if world > 1:                                       # FSDP consolidation → bf16 HF save (int8 master is single-GPU)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        full_sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            _save_bf16_hf(full_sd, cfg, out_dir)
        return
    if rank != 0:
        return
    sd = model.state_dict()
    int8_keys = [k for k, v in sd.items() if hasattr(v, "int_data") and hasattr(v, "scale")]
    if not int8_keys:                                   # bf16/snap run → unchanged standard HF save
        _save_bf16_hf({k: v.cpu() for k, v in sd.items()}, cfg, out_dir)
        return
    # int8-master run → frugal native-int8 save (no fresh model, no full bf16 copy)
    from safetensors.torch import save_file
    out_sd = {}
    for k, v in sd.items():
        if k in set(int8_keys):
            out_sd[k] = v.int_data.detach().cpu()                              # int8, 1 byte/param
            out_sd[k + ".weight_scale"] = v.scale.detach().cpu().to(torch.bfloat16)   # per-output-row scale
        else:
            out_sd[k] = v.detach().cpu().to(torch.bfloat16)                    # norms / embeds / lm_head (frozen)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out_dir / "model.safetensors"), metadata={"format": "pt", "quant": "int8_rowwise"})
    tcfg = copy.deepcopy(model.config)          # in-memory config — never re-read model_path/config.json at
    tcfg._attn_implementation = "sdpa"          # save time (that disk read, under FS pressure from the 8.6GB
    tcfg.architectures = ["Gemma4ForCausalLM"]  # weight write, is what crashed the step-3100 save of the v2 run).
    tcfg.save_pretrained(out_dir)
    (out_dir / "quant_keys.json").write_text(_json.dumps({"method": "int8_rowwise", "keys": int8_keys}))
    _cached_tokenizer(cfg["model_path"]).save_pretrained(out_dir)
    gb = sum(t.numel() * t.element_size() for t in out_sd.values()) / 1e9
    log(f"saved int8 checkpoint -> {out_dir}  ({len(int8_keys)} int8 weights + scales, {gb:.1f}GB on disk, no bf16 materialization)")


def run_contract(typed, cfg, save_dir, world, smoke_steps):
    """Print the one-screen RUN CONTRACT so what's about to train is never a mystery (esp. after a context
    compaction). The hard validation already ran in CptConfig.from_yaml() + typed.validate_runtime() — this
    is a faithful display + the remaining advisory (lever-metadata) warnings + the dry-run exit. Returns the
    resolved contract dict (dumped beside the checkpoint at save time = recipe provenance)."""
    warns = []
    tokenized = cfg.get("tokenized")
    init_from = cfg.get("init_from")
    # token count + lever-metadata consistency from the corpus stats sidecar (<name>_stats.json). Resolve
    # against C.ROOT to MATCH make_loader/build_model (which prepend C.ROOT) — not cwd.
    stats, tok_count = {}, None
    if tokenized:
        sp = C.ROOT / tokenized; sc = sp.parent / f"{sp.name.replace('.arrow', '')}_stats.json"
        if sc.exists():
            stats = json.loads(sc.read_text()); tok_count = stats.get("total_tokens")
    _lv = cfg.get("levers", {}) or {}
    lev_on = {k: bool((_lv.get(k) or {}).get("enabled")) for k in
              ("lever1_entity_loss", "lever2_qa_mix", "lever3_three_r", "lever4_exposure_tracking", "lever5_causal_trace")}
    if lev_on["lever1_entity_loss"] or lev_on["lever4_exposure_tracking"]:
        if stats and not stats.get("lever_metadata"):     # provably absent → HARD error, not a silent no-op for hours
            sys.exit("RUN CONTRACT failed: lever1/4 enabled but the corpus stats report lever_metadata:false — "
                     "the entity_mask/fact_ids streams are ABSENT, so the lever would be a silent no-op for the "
                     "whole run. Re-tokenize WITHOUT --no-lever-metadata.")
        elif not stats:                                   # can't verify → make the absence of the check visible
            warns.append("lever1/4 enabled but no corpus stats sidecar found — cannot verify the entity_mask/"
                         "fact_ids streams exist; the lever may be a silent no-op.")
    if lev_on["lever3_three_r"] and stats.get("replay_tokens", 0) == 0:
        warns.append("lever3 (three R's) enabled but replay_tokens:0 — populate data/_shared/replay first.")
    lowbit = cfg.get("optimizer_lowbit", 0)
    int8 = typed.resolve_int8_master(world)               # the SINGLE resolution (storage→snap under world>1)
    eff = cfg.get("micro_batch_size", 1) * cfg.get("grad_accum", 1) * max(1, world)
    contract = {
        "run_name": cfg.get("run_name"), "precision": cfg.get("precision"), "int8_master": int8,
        "optimizer_lowbit": lowbit or "bf16", "save_dir": save_dir, "init_from": init_from or "(base model)",
        "tokenized": tokenized, "tokenized_total_tokens": tok_count, "seq_len": cfg.get("seq_len"),
        "effective_batch": eff, "max_steps": cfg.get("max_steps"), "decay_frac": cfg.get("decay_frac"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "levers_on": [k for k, v in lev_on.items() if v], "smoke": bool(smoke_steps),
        "mode": typed.mode, "fingerprint": typed.fingerprint(),
    }
    if rank0():
        bar = "─" * 70
        rows = [("mode", f"{typed.mode}   fingerprint={contract['fingerprint']}"),
                ("precision", contract["precision"]), ("int8_master", int8),
                ("optimizer_lowbit", contract["optimizer_lowbit"]),
                ("save_dir", save_dir or "‼ UNSET — NOTHING WILL BE SAVED"),
                ("init_from", contract["init_from"]),
                ("tokenized", f"{tokenized}  [{tok_count:,} tok]" if tok_count else (tokenized or "‼ UNSET")),
                ("seq_len", contract["seq_len"]),
                ("batch", f"{cfg.get('micro_batch_size')}×{cfg.get('grad_accum')}×{max(1,world)}gpu = {eff} eff"),
                ("max_steps", contract["max_steps"]), ("decay_frac", contract["decay_frac"]),
                ("CUDA_VISIBLE_DEVICES", contract["cuda_visible_devices"] + "  (single-GPU; never both — PSU OCP)"),
                ("levers", ", ".join(contract["levers_on"]) or "none")]
        print(f"\n┌─ RUN CONTRACT · {cfg.get('run_name','?')} {'─'*max(0, 51-len(str(cfg.get('run_name',''))))}┐")
        for k, v in rows:
            print(f"│ {k:>21} : {str(v):<44}│")
        print(f"└{bar}┘")
        for w in warns:
            print(f"  ⚠ {w}")
    if typed.run.dry_run:
        if rank0():
            print("  ✓ (run.dry_run) contract OK — exiting without training.\n")
        sys.exit(0)
    return contract


def main():
    # THE ONE TRAINING ENTRYPOINT. Accepts only --config + --mode — NO behavior-affecting flags (the config
    # IS the run; see docs/CONFIG_AUDIT.md + AGENTS.md). Everything that changes a run lives in the YAML and is
    # validated at load by CptConfig (hard-error on a typo'd/unknown key, cross-field gates, fail-loud preserved).
    # Sweeps generate config files and call this N times; they never pass overrides. To smoke/dry-run, set
    # run.smoke_steps / run.dry_run in the config.
    ap = argparse.ArgumentParser(description="kandi-cpt trainer — the ONE entrypoint. Config-only; no run flags.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cpt", "sft"], default=None,
                    help="declared intent; the config may carry `mode:`. Cross-checked against init_from + the "
                         "corpus labels column — it never replaces that dual detection.")
    args = ap.parse_args()
    typed = CptConfig.from_yaml(args.config, args.mode)   # validates structure + cross-field AT LOAD (fail-loud)
    cfg = typed.to_trainer_dict()                          # flat-key adapter — the trainer internals are untouched
    smoke_steps = typed.run.smoke_steps
    save_dir = cfg.get("save_dir")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    typed.validate_runtime(world)                         # path/world gates (save_dir/tokenized exist; int8/lowbit warns)
    contract = run_contract(typed, cfg, save_dir, world, smoke_steps)   # prints the one-screen contract; dry_run exits here
    if rank0() and save_dir and not smoke_steps:          # live heartbeat for the Torad Train console
        _start_status(save_dir, cfg.get("max_steps"))
        from run_registry import freeze_run            # freeze resolved config + dataset_hash + git_sha + ts
        run_id = freeze_run(typed, save_dir, os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
        if run_id:
            log(f"run registered: logs/runs/{run_id}/  (resolved config + provenance frozen)")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    log(f"world={world} precision={cfg['precision']} run={cfg['run_name']} smoke={smoke_steps}")
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

    # INT8-master mode: 'storage' = genuine int8 weight store (the moat) | 'snap' = bf16-grid SR (legacy) | 'off'.
    # Resolved by the config (validate() already gated storage⇒precision=fp4); resolve_int8_master folds the
    # single world>1→snap degradation. Resolve BEFORE building `params` so int8-stored linears are excluded from AdamW.
    int8_mode = typed.resolve_int8_master(world)
    if int8_mode != typed.train.int8_master:
        log(f"int8_master={typed.train.int8_master} → {int8_mode} (single-GPU only; world={world})")
    if int8_mode == "storage":
        convert_quartet_to_master(model)                      # torchao Int8 store + SR; updated by the shared low-bit AdamW
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(local_rank)

    # ALL trainable params — including the int8-master subclasses — go to the SAME optimizer. torchao's
    # Int8QuantizedTrainingLinearWeight intercepts AdamW's in-place update (add_/addcdiv_) and stochastic-rounds
    # it back into int8, so there is no separate optimizer and no persistent grad accumulator (g_acc).
    params = [p for p in model.parameters() if p.requires_grad]
    # Low-bit optimizer states (4 or 8) reclaim optimizer memory on the single-GPU path. torchao's
    # OptimState{4,8}bit is DTensor-incompatible (breaks under FSDP2, world>1) — but world==1 runs plain
    # (no FSDP, no DTensor), so it works there. Gated accordingly; falls back to bf16 AdamW. For int8-master
    # STORAGE this same 4-bit AdamW also updates the masters (SR via the subclass): weight AND moments low
    # precision, never above the FP4 we train in.
    lowbit = cfg.get("optimizer_lowbit", 0)
    int8_master_on = int8_mode == "snap"   # legacy: per-step bf16-grid SR (weights stay bf16)
    if rank0():
        log(f"int8_master mode = {int8_mode}"
            + ("  — int8 STORE (1 byte/param) + SR via the shared low-bit AdamW + Quartet FP4 (THE ENGINE MOAT)" if int8_mode == "storage"
               else "  — per-step SR-snap to int8 grid (bf16 storage)" if int8_master_on else ""))
    if rank0():                                  # isolate WEIGHT memory (no optimizer state yet, no activations)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        log(f"[mem] weights resident ({int8_mode}): {torch.cuda.memory_allocated()/1e9:.3f}GB "
            f"— the 223 masters are int8(1B) under storage, bf16(2B) under snap")
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

    gctl = None                                          # int8 SR gradient accumulator (halves gradient memory)
    if cfg.get("int8_grad_accum") and world == 1:
        gctl = enable_int8_grad_accum(params)
        log(f"int8 grad-accum ON: {sum(1 for p in params if p.requires_grad and p.dim()>=2)} 2-D grads → int8 SR "
            f"accumulator (1 B/param, bf16 .grad freed per-layer)")
    elif cfg.get("int8_grad_accum") and world > 1:
        log("int8_grad_accum ignored: single-GPU only (the post-accumulate hook + FSDP grad sharding don't compose)")

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
    if smoke_steps:
        total, accum = smoke_steps, min(accum, 2)        # quick loop+memory validation
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
        if isinstance(batch, dict):              # levers/SFT return {input_ids[, labels][, entity_mask][, fact_ids]}
            ids = batch["input_ids"].to(device)
            emask = batch["entity_mask"].to(device) if "entity_mask" in batch else None
            labels = batch["labels"].to(device) if "labels" in batch else None   # SFT: assistant-only masked
            if lever4_on and "fact_ids" in batch:   # count this microbatch's fact exposures (per-block lists)
                for fids in batch["fact_ids"]:
                    exposure.update(fids)
        else:
            ids, emask, labels = batch.to(device), None, None
        if _can_nosync:                          # sync grads only on the last microbatch of the window
            model.set_requires_gradient_sync(micro % accum == accum - 1)
        # Pass shared_kv_states as a UserDict: Gemma4's `kwargs.pop("shared_kv_states", {})`
        # uses ours, and FSDP2's tree_map treats a UserDict as a leaf (won't rebuild it), so
        # the KV-sharing layers see the dict the producer layers populated (transformers #45912).
        if lever1_on and emask is not None:      # Lever 1: manual entity-weighted CE on logits (no labels=)
            out = model(input_ids=ids, shared_kv_states=UserDict())
            loss = entity_weighted_ce(out.logits, ids, emask, entity_w)
        else:                                    # CPT: labels=input_ids (predict-next). SFT: assistant-only
            out = model(input_ids=ids, labels=(labels if labels is not None else ids), shared_kv_states=UserDict())
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
            if gctl is not None:
                gctl.materialize()          # int8 SR accumulator → bf16 .grad for the optimizer (frees the int8 buf)
            if params:
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()                      # updates everything incl. int8 masters (subclass intercepts add_/addcdiv_ → SR)
            if int8_master_on:
                _sr_int8_master(params)     # legacy snap: constrain bf16 weights to the INT8 grid (SR)
            opt.zero_grad(set_to_none=True)
            if gctl is not None:
                gctl.reset()                # empty the accumulator for the next window
            step += 1
            if step % cfg["log_every"] == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                cur = torch.cuda.memory_allocated() / 1e9    # post-step+zero_grad: weights + optimizer state (grads freed)
                log(f"step {step}/{total} loss {window_loss/(accum*cfg['log_every']):.4f} "
                    f"lr {lr:.2e} peakmem {mem:.1f}GB (resident {cur:.1f}GB; transient {mem-cur:.1f}GB) {(time.time()-t0)/step:.2f}s/step")
                window_loss = 0.0
            # Honor save_every (was silently ignored): periodic checkpoint → crash-recovery for long
            # fleet runs + the per-step checkpoints the geometry probe sweeps to watch geometry form.
            se = cfg.get("save_every", 0)
            if save_dir and not smoke_steps and se and step % se == 0 and step < total:
                save_checkpoint(model, cfg, f"{save_dir}/step_{step:06d}", rank, world)
                if lever4_on and rank0():    # snapshot cumulative fact exposures alongside the checkpoint
                    _save_exposure(f"{save_dir}/step_{step:06d}", step, exposure)
                model.train()

    log(f"done {step} steps in {time.time()-t0:.0f}s")
    if save_dir and not smoke_steps:
        save_checkpoint(model, cfg, save_dir, rank, world)
        if rank0():                          # recipe provenance: what RUN CONTRACT produced this checkpoint
            import json as _json
            from pathlib import Path
            (Path(save_dir) / "run_contract.json").write_text(_json.dumps(contract, indent=2, default=str))
        if lever4_on and rank0():
            _save_exposure(save_dir, step, exposure)
    if rank0():
        _finish_status("done")               # flip the console heartbeat to done; close the run log
    # NOTE: NVFP4 QAT→convert is intentionally NOT done here. Running quantize_(step="convert") on the
    # live FSDP2-sharded model on rank 0 only would (a) risk a collective deadlock and (b) discard the
    # result (it's never re-saved). The convert is a separate OFFLINE step: load the saved bf16
    # checkpoint single-process, quantize_, save. (TODO: scripts/convert_nvfp4.py)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
