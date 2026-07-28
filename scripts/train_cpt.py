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
import cuda_home
import lever_receipts
import recovery
import training_probe
import window_sampler
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


def resolve_anneal(cfg: dict, total: int) -> dict:
    """Derive the anneal window from train.anneal + max_steps. Default ON; disabled on tiny runs.

    Returns a dict with keys: active, start_step, frac, mix_tags, lr_end, keep_pre,
    require_exposure, reason. Pure — no I/O. Precision floor untouched (schedule ≠ dtype).
    """
    anneal = cfg.get("anneal") or {}
    enabled = bool(anneal.get("enabled", True))
    frac = float(anneal.get("frac", 0.10))
    min_steps = int(anneal.get("min_steps", 500))
    mix_tags = list(anneal.get("mix_tags") or ["hydrated", "replay"])
    lr_end = float(anneal.get("lr_end", 0.0))
    keep_pre = bool(anneal.get("keep_pre_anneal_checkpoint", True))
    require_exposure = bool(anneal.get("require_exposure", True))
    if not enabled:
        return {
            "active": False, "start_step": total, "frac": frac, "mix_tags": mix_tags,
            "lr_end": lr_end, "keep_pre": keep_pre, "require_exposure": require_exposure,
            "reason": "train.anneal.enabled=false",
        }
    if total < min_steps:
        return {
            "active": False, "start_step": total, "frac": frac, "mix_tags": mix_tags,
            "lr_end": lr_end, "keep_pre": keep_pre, "require_exposure": require_exposure,
            "reason": f"max_steps={total} < anneal.min_steps={min_steps} (tiny-run disable)",
        }
    if not 0 < frac <= 0.5:
        raise ValueError(f"train.anneal.frac={frac} must be in (0, 0.5]")
    start = int(total * (1.0 - frac))
    start = max(1, min(start, total - 1)) if total > 1 else total
    return {
        "active": start < total,
        "start_step": start,
        "frac": frac,
        "mix_tags": mix_tags,
        "lr_end": lr_end,
        "keep_pre": keep_pre,
        "require_exposure": require_exposure,
        "reason": f"final {frac:.0%} of {total} steps (start={start})",
    }


def anneal_lr(step: int, total: int, start: int, base_lr: float, lr_end: float) -> float:
    """Linear LR decay over the anneal window from base_lr → lr_end. Outside the window unused."""
    if step < start or total <= start:
        return base_lr
    t = (step - start) / max(1, total - start)
    t = min(1.0, max(0.0, t))
    return base_lr + (lr_end - base_lr) * t


def pre_anneal_exposure_check(cfg: dict, exposure: Counter | None, require: bool) -> dict:
    """Printable pre-anneal exposure check. Anneal is not a substitute for exposure (#760 floor).

    Returns a receipt dict the run contract / logs carry. Does not raise — under-exposure is a
    WARN so the operator sees it; the anneal still runs (the failure mode is polished weak install,
    which probes catch).
    """
    levers = cfg.get("levers") or {}
    lev4 = bool((levers.get("lever4_exposure_tracking") or {}).get("enabled"))
    if not lev4:
        return {
            "ok": not require,
            "lever4_enabled": False,
            "n_facts": 0,
            "message": (
                "pre-anneal exposure check SKIPPED — lever4_exposure_tracking is off; "
                "anneal will polish whatever is installed without an exposure floor proof"
            ),
        }
    n_facts = len(exposure) if exposure is not None else 0
    total_exp = int(sum(exposure.values())) if exposure else 0
    ok = n_facts > 0 and total_exp > 0
    return {
        "ok": ok,
        "lever4_enabled": True,
        "n_facts": n_facts,
        "total_exposures": total_exp,
        "message": (
            f"pre-anneal exposure check PASS — {n_facts} facts, {total_exp} cumulative exposures"
            if ok else
            "pre-anneal exposure check FAIL — lever4 on but exposure counter is empty "
            "(corpus missing fact_ids, or no facts seen yet); annealing an under-exposed "
            "corpus polishes a weak install"
        ),
    }


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
    from quartet2.linear import Quartet_II_linear
    from torchao.prototype.quantized_training import Int8QuantizedTrainingLinearWeight

    master_params = []
    for module in model.modules():
        for cn, child in list(module.named_children()):
            if type(child) is Quartet_II_linear and child.bias is None:
                child.__class__ = type("QuartetMasterLinear", (QuartetMasterMixin, Quartet_II_linear), {})
                child.weight = nn.Parameter(Int8QuantizedTrainingLinearWeight.from_float(child.weight.detach()))
                master_params.append(child.weight)
    log(f"int8-master MOAT: {len(master_params)} Quartet linears → torchao Int8 store (1 byte/param, SR on every "
        f"optimizer update; updated by the shared low-bit AdamW — FP4-native compute)")
    return master_params


class QuartetMasterMixin:
    """The int8-master Quartet forward: dequant_bridge gives the kernel a TRANSIENT bf16 view
    of the int8 master (grad routes to weight.grad); storage stays 1-byte. Module-level so the
    eval loader (pipeline.model_load.load_causal_ckpt_quartet) class-swaps identically."""

    def forward(self, x, disable_backward_quant=False, input_abs_max=None):
        from quartet2.linear import Quartet_II_fn, Quartet_II_pseudoquant_fn
        from quartet_bridge import dequant_bridge
        w = dequant_bridge(self.weight)
        major, _ = torch.cuda.get_device_capability(x.device)
        if major >= 10:
            return Quartet_II_fn.apply(x, w, self.had, self.mode,
                                       self.disable_backward_quant or disable_backward_quant,
                                       self.weight_abs_max, input_abs_max, self.scratch_amax)
        return Quartet_II_pseudoquant_fn.apply(x, w, self.had,
                                               self.disable_backward_quant or disable_backward_quant,
                                               self.mode == self.mode.FOUR_SIX)


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
    init = cfg.get("init_from")
    if init and (C.ROOT / init / "quant_keys.json").is_file():
        # int8-master checkpoint → NATIVE init: Quartet linears + int8 masters from the shard,
        # no bf16 dequant round-trip (the precision law: bf16 weight residency is forbidden).
        # swap_to_quartet / convert_quartet_to_master downstream no-op on the already-Quartet,
        # already-master linears (isinstance / type checks), so the tail below is shared.
        from pipeline.model_load import load_causal_ckpt_quartet
        model = load_causal_ckpt_quartet(C.ROOT / init, device="cpu", requires_grad=True)
        if cfg.get("freeze_embeddings"):
            frozen = 0
            for name, p in model.named_parameters():
                if "embed" in name.lower():
                    p.requires_grad_(False)
                    frozen += p.numel()
            log(f"froze {frozen/1e9:.2f}B embedding params (embed_tokens + PLE + tied lm_head)")
        model.config.use_cache = not bool(cfg.get("gradient_checkpointing"))
        if cfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
        log(f"init_from: int8-master checkpoint '{init}' loaded natively (Quartet + int8 masters)")
        return model
    if init:
        # Plain (non-int8) checkpoints are not loadable under the precision law (2026-07-26):
        # the bf16 round-trip (causal_bf16_state_dict) was deleted with the S4 legacy family.
        raise ConfigError(
            f"init_from '{init}' has no quant_keys.json — the precision law forbids the bf16 "
            f"round-trip. Convert the artifact once (scripts/convert_base_int8.py) and init from that.")
    else:
        sd = load_file(f"{cfg['model_path']}/model.safetensors")
        P = "model.language_model."
        remapped = {("model." + k[len(P):]): v for k, v in sd.items() if k.startswith(P)}
        if "lm_head.weight" in sd:
            remapped["lm_head.weight"] = sd["lm_head.weight"]
        n_dropped = len(sd) - len(remapped)              # multimodal vision/audio tower tensors dropped
    # Construct directly in the checkpoint's floating dtype. The default constructor otherwise
    # materializes ~20 GB of throw-away four-byte parameters while a resumed run also holds the
    # int8 checkpoint and its dequantized state dict. Inferring the dtype from the receipt keeps
    # buffers real (unlike meta construction) without changing checkpoint bytes or training policy.
    load_dtype = next(value.dtype for value in remapped.values() if value.is_floating_point())
    prior_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(load_dtype)
        model = Gemma4ForCausalLM(text_cfg)
    finally:
        torch.set_default_dtype(prior_default_dtype)
    # Guard the NaN-incident class (silent random-init) on BOTH sides: a wrong prefix yields a tiny
    # remap → fail loud before training on noise.
    if len(remapped) < 0.5 * len(model.state_dict()):
        raise RuntimeError(f"remap produced only {len(remapped)} tensors for {len(model.state_dict())} "
                           f"model params — prefix wrong for this checkpoint (init_from={bool(init)})")
    missing, unexpected = model.load_state_dict(remapped, strict=False, assign=True)
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


def anneal_composition(tags: list[str], mask: list[bool], mix_tags: set[str]) -> dict:
    """Replay vs domain block counts among the KEPT blocks, and the replay share they imply.

    Pure. Exists because the anneal's replay concentration was previously invisible: keeping every
    replay block while cutting domain by quality RAISES the replay share, so an educational-quality
    filter silently acts as a replay schedule. Reporting it is what makes that a decision.
    """
    kept_replay = sum(1 for t, k in zip(tags, mask, strict=True) if k and t == "replay")
    kept_domain = sum(1 for t, k in zip(tags, mask, strict=True)
                      if k and t != "replay" and t in mix_tags)
    total = kept_replay + kept_domain
    return {"kept_replay_blocks": kept_replay, "kept_domain_blocks": kept_domain,
            "kept_total_blocks": total,
            "replay_block_share": round(kept_replay / total, 6) if total else 0.0}


def _anneal_rank_keep_mask(tags: list[str], scores: list[float], mix_tags: set[str], *,
                           domain_quality_quantile: float = 0.5,
                           replay_fraction: float | None = None) -> list[bool]:
    """Keep replay; keep non-replay mix-tag blocks at/above a quality QUANTILE of the edu score.

    Pure helper (unit-tested). When scores are uniform the cut keeps everything in-tag.

    `domain_quality_quantile` defaults to 0.5 — the median, i.e. exactly the prior behaviour.
    `replay_fraction`, when set, caps kept replay so replay is that share of the kept set instead
    of whatever share falls out of the domain cut; None keeps every replay block (prior behaviour).
    Both default to the old semantics on purpose: this makes the composition EXPLICIT and
    reportable without silently changing any existing run.
    """
    ranked_scores = [
        float(score) for tag, score in zip(tags, scores, strict=True)
        if tag in mix_tags and tag != "replay"
    ]
    if not ranked_scores:
        return [tag in mix_tags for tag in tags]
    ordered = sorted(ranked_scores)
    # quantile 0.5 -> ordered[len//2], byte-identical to the median this replaced.
    cut = ordered[min(len(ordered) - 1, int(len(ordered) * domain_quality_quantile))]

    keep: list[bool] = []
    for tag, score in zip(tags, scores, strict=True):
        if tag not in mix_tags:
            keep.append(False)
        elif tag == "replay":
            keep.append(True)
        else:
            keep.append(float(score) >= cut)

    if replay_fraction is not None:
        if not 0 <= replay_fraction < 1:
            raise ValueError(f"anneal.replay_fraction must be in [0,1), got {replay_fraction}")
        kept_domain = sum(1 for t, k in zip(tags, keep, strict=True)
                          if k and t != "replay" and t in mix_tags)
        # r / (r + d) = f  ->  r = f*d/(1-f)
        want_replay = int(round(replay_fraction * kept_domain / (1.0 - replay_fraction))) \
            if replay_fraction > 0 else 0
        seen = 0
        for i, (tag, k) in enumerate(zip(tags, keep, strict=True)):
            if k and tag == "replay":
                seen += 1
                if seen > want_replay:
                    keep[i] = False                    # deterministic: keep the first `want` in order
    return keep


def make_loader(cfg, rank, world, want_mask=False, want_facts=False, tag_filter=None):
    # Dual-format: a directory is the Python HF-datasets save_to_disk output; a file is the Rust
    # festival-factory plain Arrow IPC (tokenize.rs). Both expose input_ids[/entity_mask/src/fact_ids].
    # tag_filter: optional set/list of block_tag values — the anneal mix (hydrated+replay). When the
    # corpus lacks block_tag (legacy shards), filter is skipped with a loud warning so anneal still
    # runs (LR decay only) rather than aborting a paid GPU run. When block_edu_score is present,
    # non-replay mix blocks are further ranked: keep score >= median (replay always kept).
    p = C.ROOT / cfg["tokenized"]
    ds = Dataset.from_file(str(p)) if p.is_file() else load_from_disk(str(p))
    has_mask = want_mask and "entity_mask" in ds.column_names
    has_facts = want_facts and "fact_ids" in ds.column_names
    has_labels = "labels" in ds.column_names           # SFT corpus (build_sft.py): assistant-only masked labels
    has_block_tag = "block_tag" in ds.column_names
    has_edu = "block_edu_score" in ds.column_names

    if tag_filter:
        wanted = set(tag_filter)
        if not has_block_tag:
            if rank0():
                log(f"[anneal] corpus has no block_tag column — cannot filter to {sorted(wanted)}; "
                    f"re-tokenize to emit block_tag. Continuing on the FULL mix (LR decay only).")
        else:
            before = len(ds)
            if has_edu:
                tags = list(ds["block_tag"])
                scores = [float(s) for s in ds["block_edu_score"]]
                _an = (cfg.get("anneal") or {}) if isinstance(cfg, dict) else {}
                _q = float(_an.get("domain_quality_quantile", 0.5))
                _rf = _an.get("replay_fraction")
                pre = anneal_composition(tags, [t in wanted for t in tags], wanted)
                mask = _anneal_rank_keep_mask(tags, scores, wanted,
                                              domain_quality_quantile=_q,
                                              replay_fraction=(None if _rf is None else float(_rf)))
                post = anneal_composition(tags, mask, wanted)
                keep_idx = [i for i, ok in enumerate(mask) if ok]
                ds = ds.select(keep_idx)
                if rank0():
                    log(f"[anneal] mix filter {sorted(wanted)} + edu-rank(q={_q}): "
                        f"{before} → {len(ds)} blocks")
                    # The distortion the review named: say the share BEFORE and AFTER, always.
                    log(f"[anneal] composition replay_block_share "
                        f"{pre['replay_block_share']:.4f} → {post['replay_block_share']:.4f} "
                        f"(replay {pre['kept_replay_blocks']}→{post['kept_replay_blocks']}, "
                        f"domain {pre['kept_domain_blocks']}→{post['kept_domain_blocks']}, "
                        f"target={'unset' if _rf is None else _rf})")
            else:
                ds = ds.filter(lambda row: row["block_tag"] in wanted)
                if rank0():
                    log(f"[anneal] mix filter {sorted(wanted)} (no block_edu_score — tag-only): "
                        f"{before} → {len(ds)} blocks")
            if len(ds) == 0:
                raise RuntimeError(
                    f"[anneal] mix filter {sorted(wanted)} matched ZERO blocks — refusing to train "
                    f"on an empty anneal loader. Re-tokenize so hydrated/replay blocks carry block_tag."
                )

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
    # Replay per optimizer window. Plain shuffling delivers the right replay share ON AVERAGE and
    # leaves a large fraction of updates with none at all — pure domain gradient, which is exactly
    # what replay exists to prevent, and invisible in the corpus stats.
    replay_report = None
    if sampler is None and has_block_tag:
        tags = list(ds["block_tag"])
        replay_idx = [i for i, t in enumerate(tags) if t == "replay"]
        domain_idx = [i for i, t in enumerate(tags) if t != "replay"]
        window = int(cfg["micro_batch_size"]) * int(cfg["grad_accum"])
        share = (len(replay_idx) / len(tags)) if tags else 0.0
        requested = cfg.get("replay_per_window")
        n_windows = max(1, int(cfg.get("max_steps") or 1))
        seed = int(cfg.get("seed", 42))
        # Per-window counts rather than one flat number: a flat quota either drops the realized
        # share (round 1.456 -> 1 turns 9.10% into 6.25%) or overshoots it. Spreading the
        # fractional part keeps the corpus share AND leaves no window empty.
        quotas = (window_sampler.window_quotas(n_windows, window, share, seed=seed)
                  if requested is None else [int(requested)] * n_windows)
        quota = max(quotas) if quotas else 0
        if any(q > 0 for q in quotas) and replay_idx and domain_idx:
            sampler = window_sampler.OptimizerWindowSampler(
                replay_idx, domain_idx, window=window, quota=quotas,
                n_windows=n_windows, seed=seed)
            replay_report = window_sampler.window_report(list(sampler), replay_idx, window=window)
            replay_report.update(
                corpus_replay_block_share=round(share, 6),
                replay_per_window_quota=quota,
                replay_per_window_target=round(window * share, 4),
                quota_source="derived" if requested is None else "configured",
                zero_replay_fraction_if_shuffled=round(
                    window_sampler.expected_zero_replay_fraction(window, share), 6),
            )
            if rank0():
                log(f"[replay] {replay_report['replay_per_window_mean']:.2f}/{window} blocks per optimizer window "
                    f"({replay_report['quota_source']}; corpus share {share:.4f}) — "
                    f"zero-replay updates {replay_report['zero_replay_fraction']:.4f} "
                    f"vs {replay_report['zero_replay_fraction_if_shuffled']:.4f} under plain shuffling")
        elif rank0() and requested != 0:
            log(f"[replay] per-window quota inactive (replay blocks {len(replay_idx)}, "
                f"domain {len(domain_idx)}, window {window}) — falling back to shuffling")
    dl = DataLoader(ds, batch_size=cfg["micro_batch_size"], sampler=sampler,
                    shuffle=(sampler is None), collate_fn=collate, drop_last=True)
    dl.replay_report = replay_report
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



# ── Save-time host-memory discipline ──────────────────────────────────────────
# 2026-07-25: five unshielded CPT runs finished 450 steps then died in save_checkpoint.
# whale-reaper journal (SIGTERM, unshielded python):
#   s2_reasoning rss=15018MB  s3_r005 rss=16257MB  s3_r010 rss=9193MB
#   s3_r020 rss=9071MB        s3_r030 rss=8219MB
# Root cause: the int8 path built the FULL CPU state dict (every int8 weight + scale +
# bf16 norm/embed) before safetensors.torch.save_file, so host RSS jumped from training
# steady-state to ~2× during the gather — the unprotected whale class under ambient
# pressure. Fix = (1) stream gather→write one tensor at a time so peak extra ≈ largest
# tensor, not the sum; (2) preflight MemAvailable against a need derived from that
# measured spike / the streaming peak, WAIT with backoff instead of dying mid-write.
# Layout stays single model.safetensors + quant_keys.json (eval loaders unchanged).


def _meminfo_bytes(field: str = "MemAvailable") -> int:
    """Read one /proc/meminfo field (kB) as bytes. MemAvailable is the kernel's own
    reclaimable-headroom estimate — same signal earlyoom/whale consult."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"/proc/meminfo missing field {field!r}")


def _tensor_nbytes(shape, dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()


def _tensor_tobytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian payload matching safetensors.torch._tobytes (no dtype tag)."""
    import ctypes
    import numpy as np
    if tensor.device.type != "cpu":
        tensor = tensor.detach().to("cpu", copy=True)
    else:
        tensor = tensor.detach()
    tensor = tensor.contiguous()
    total = tensor.numel() * tensor.element_size()
    if total == 0 or tensor.data_ptr() == 0:
        return b""
    ptr = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_ubyte))
    buf = np.ctypeslib.as_array(ptr, (total,))
    # copy out so the tensor can be freed immediately after
    return buf.tobytes()


_EMB_SCALE_SUFFIX = ".embed_scale"


def _fp8_folded_scale(scale):
    """Factory writing a scale as fp8 e4m3 with the ×128 range fold the loader unfolds.

    torch ships no fp8 arithmetic kernel — multiplying a Float8_e4m3fn raises
    NotImplementedError — so the fold is applied in bf16 and immediately narrowed. Transit only:
    nothing bf16 reaches the file. Mirrors convert_ckpt_v2.py and inverts the unfold in
    model_load.load_causal_ckpt_quartet.
    """
    def factory(t=scale):
        return (t.detach().to(torch.bfloat16) * 128.0).to(torch.float8_e4m3fn)  # bf16-sanctioned: S2 fp8 storage fold, transit only — the written tensor is e4m3
    return factory




def _int8_save_entries(sd, int8_keys):
    """Build (name, dtype_str, shape, nbytes, factory) entries WITHOUT gathering CPU data.
    Factories run later, one at a time, during the streaming write.

    Returns (entries, embedding_keys). embedding_keys is non-empty exactly when the model carries
    v2 Int8ScaledEmbedding wrappers, and the caller writes the v2 quant manifest in that case.

    THE BUG THIS FIXES (measured 2026-07-28): Int8ScaledEmbedding registers `weight` and
    `weight_scale` as BUFFERS, not as a tensor subclass carrying .int_data/.scale — so the
    embeddings never entered int8_keys and fell through to the bf16 branch below, which wrote their
    raw int8 VALUES reinterpreted as bf16 floats (±127 instead of ~±0.05) and recorded no
    embedding_keys. On reload the loader computed is_v2=False, never rebuilt the wrappers, and the
    model came back with garbage embeddings: a run at loss 0.4119 reloaded at 36.12. Every
    continuation — booster, SFT-from-CPT, anneal, crash resume — was silently starting from that.
    The eval path was unaffected, which is why it survived.
    """
    int8_set = set(int8_keys)
    # v2 embedding wrappers announce themselves with an embed_scale buffer. lm_head is excluded:
    # the loader rebuilds it as Int8TiedLmHead over embed_tokens, so storing it duplicates the
    # whole vocab table and is never read.
    emb_prefixes = sorted(
        k[: -len(_EMB_SCALE_SUFFIX)] for k in sd
        if k.endswith(_EMB_SCALE_SUFFIX) and not k.startswith("lm_head")
    )
    handled = {f"{p}{suffix}" for p in emb_prefixes
               for suffix in (".weight", ".weight_scale", _EMB_SCALE_SUFFIX)}
    handled |= {k for k in sd if k.startswith("lm_head._emb")}
    v2 = bool(emb_prefixes)

    def scale_entry(name, scale):
        """v2 stores scales as fp8 e4m3 with the ×128 range fold the loader unfolds at load."""
        shape = tuple(scale.shape)
        if not v2:
            return {"name": name, "dtype_str": "BF16", "shape": shape,
                    "nbytes": _tensor_nbytes(shape, torch.bfloat16),
                    "factory": (lambda t=scale: t.detach().to(dtype=torch.bfloat16))}
        # bf16-sanctioned: S2 fp8 STORAGE fold. torch ships no fp8 arithmetic kernel (mul on
        # Float8_e4m3fn raises NotImplementedError), so the ×128 range fold is applied in bf16 and
        # immediately narrowed to e4m3. Transit only — nothing bf16 is written. Mirrors
        # convert_ckpt_v2.py and inverts the unfold in model_load.load_causal_ckpt_quartet.
        return {"name": name, "dtype_str": "F8_E4M3", "shape": shape,
                "nbytes": shape[0] if shape else 1,
                "factory": _fp8_folded_scale(scale)}

    entries = []
    for k, v in sd.items():
        if k in handled:
            continue
        if k in int8_set:
            idata, scale = v.int_data, v.scale
            shape_i = tuple(idata.shape)
            entries.append({
                "name": k, "dtype_str": "I8", "shape": shape_i,
                "nbytes": _tensor_nbytes(shape_i, torch.int8),
                "factory": (lambda t=idata: t.detach()),
            })
            entries.append(scale_entry(k + ".weight_scale", scale))
        else:
            # norms — stored bf16, may already be on CPU (PLE offload)
            shape = tuple(v.shape)
            entries.append({
                "name": k, "dtype_str": "BF16", "shape": shape,
                "nbytes": _tensor_nbytes(shape, torch.bfloat16),
                "factory": (lambda t=v: t.detach().to(dtype=torch.bfloat16)),
            })

    # v2 embeddings: int8 data + fp8 scale, under the names load_causal_ckpt_quartet reads
    # (`<module>.weight` and `<module>.weight.weight_scale`). embed_scale is NOT stored — the
    # loader derives it from the config.
    embedding_keys = []
    for prefix in emb_prefixes:
        name = f"{prefix}.weight"
        weight, scale = sd[name], sd[f"{prefix}.weight_scale"]
        shape_w = tuple(weight.shape)
        embedding_keys.append(name)
        entries.append({
            "name": name, "dtype_str": "I8", "shape": shape_w,
            "nbytes": _tensor_nbytes(shape_w, torch.int8),
            "factory": (lambda t=weight: t.detach().to(torch.int8)),
        })
        entries.append(scale_entry(name + ".weight_scale", scale))
    return entries, embedding_keys


def estimate_streaming_save_need_bytes(entries, headroom_bytes: int = 512 * 1024 ** 2) -> int:
    """Watermark for the STREAMING path: peak extra host RAM ≈ largest single tensor
    (gathered, written, freed) + small headroom. Derived from the entry sizes themselves
    (the measured full-gather RSS spike of 9–16GB was sum(entries); streaming bound is max)."""
    if not entries:
        return headroom_bytes
    largest = max(e["nbytes"] for e in entries)
    # Also hold a small fraction of total as FS/page-cache slack so a quiet box isn't
    # required to be emptier than the checkpoint itself.
    total = sum(e["nbytes"] for e in entries)
    return largest + min(total // 16, 1024 * 1024 ** 2) + headroom_bytes


def save_preflight(need_bytes: int, *, timeout_s: float = 600.0, log_fn=None,
                   sleep_fn=None, mem_fn=None) -> int:
    """Block until MemAvailable >= need_bytes, or raise. Loud log on every wait.
    sleep_fn/mem_fn are seams for the red-green synthetic-pressure test."""
    log_fn = log_fn or log
    sleep_fn = sleep_fn or time.sleep
    mem_fn = mem_fn or _meminfo_bytes
    attempt = 0
    backoff = 5.0
    deadline = time.time() + timeout_s
    while True:
        avail = int(mem_fn("MemAvailable"))
        if avail >= need_bytes:
            if attempt:
                log_fn(f"save preflight: MemAvailable={avail / 1e9:.2f}GB >= need="
                       f"{need_bytes / 1e9:.2f}GB — proceeding after {attempt} wait(s)")
            else:
                log_fn(f"save preflight: MemAvailable={avail / 1e9:.2f}GB >= need="
                       f"{need_bytes / 1e9:.2f}GB — clear")
            return avail
        if time.time() >= deadline:
            raise RuntimeError(
                f"save preflight: timed out after {timeout_s:.0f}s waiting for "
                f"{need_bytes / 1e9:.2f}GB free (MemAvailable={avail / 1e9:.2f}GB). "
                f"Refusing to start a mid-write death — free host memory and retry."
            )
        attempt += 1
        log_fn(
            f"save preflight: WAITING — MemAvailable={avail / 1e9:.2f}GB < need="
            f"{need_bytes / 1e9:.2f}GB (attempt {attempt}, sleep {backoff:.0f}s). "
            f"Save will not start until headroom exists."
        )
        sleep_fn(backoff)
        backoff = min(backoff * 1.5, 60.0)


def _safetensors_key_order(entries):
    """Match safetensors.torch.save_file header/data order for byte-identical files.

    Observed rule for the int8-native path (I8 + 2-byte scales/norms only): wider
    element size first, then tensor name ascending. (Full safetensors also subtypes
    same-width dtypes; we only emit one 2-byte and one 1-byte kind.)"""
    def sort_key(e):
        width = 1 if e["dtype_str"] == "I8" else 2
        return (-width, e["name"])
    return [e["name"] for e in sorted(entries, key=sort_key)]


def save_file_streaming(filename, entries, metadata=None):
    """Write a safetensors file gathering ONE tensor at a time.

    Byte layout matches safetensors.torch.save_file: size/name key order, 8-byte-aligned
    header padded with spaces, little-endian u64 header size, contiguous tensor payloads.
    Peak host RSS during the write ≈ max(entry.nbytes), not sum(entry.nbytes).
    """
    import struct
    from pathlib import Path
    by_name = {e["name"]: e for e in entries}
    if len(by_name) != len(entries):
        dupes = [e["name"] for e in entries if sum(1 for x in entries if x["name"] == e["name"]) > 1]
        raise RuntimeError(f"duplicate tensor names in save entries: {sorted(set(dupes))[:8]}")
    order = _safetensors_key_order(entries)
    header = {}
    if metadata:
        # safetensors Rust serializes metadata keys in reverse-alphabetical order
        # (observed: quant before format). Match for byte-identical files.
        header["__metadata__"] = {
            str(k): str(v) for k, v in sorted(metadata.items(), key=lambda kv: str(kv[0]), reverse=True)
        }
    offset = 0
    for name in order:
        e = by_name[name]
        header[name] = {
            "dtype": e["dtype_str"],
            "shape": list(e["shape"]),
            "data_offsets": [offset, offset + e["nbytes"]],
        }
        offset += e["nbytes"]
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_json) % 8)) % 8
    header_json = header_json + (b" " * pad)

    path = Path(filename)
    tmp = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for name in order:
            e = by_name[name]
            tensor = e["factory"]()
            if tuple(tensor.shape) != tuple(e["shape"]):
                raise RuntimeError(
                    f"save streaming shape drift on {name}: factory {tuple(tensor.shape)} "
                    f"!= planned {tuple(e['shape'])}"
                )
            payload = _tensor_tobytes(tensor)
            if len(payload) != e["nbytes"]:
                raise RuntimeError(
                    f"save streaming nbytes drift on {name}: got {len(payload)} planned {e['nbytes']}"
                )
            f.write(payload)
            written += len(payload)
            del payload, tensor
    os.replace(tmp, path)
    return written



def _load_probe_slices(cfg) -> dict:
    """Frozen forced-choice slices for mid-run probing, or {} when not configured.

    Deliberately a path rather than a guess at the domain's sentinel: a probe drawn from a
    different set than the post-run verdict would not be comparable to it.
    """
    ref = str(cfg.get("probe_slices") or "")
    if not ref:
        return {}
    import json as _json
    from pathlib import Path as _Path
    path = _Path(ref) if str(ref).startswith("/") else C.ROOT / ref
    try:
        loaded = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        log(f"[probe] WARNING cannot read probe_slices {path}: {type(error).__name__}: {error}")
        return {}
    return {k: v for k, v in loaded.items() if k in training_probe.FAMILIES and v}


def _write_recovery(out_dir, cfg, rank, step, micro, opt, exposure, anneal_started,
                    scored_checkpoint, es_best, es_bad, sampler, contract):
    """Write the recovery checkpoint beside a deploy checkpoint. Never fails the run.

    A recovery file is an insurance policy; losing the policy must not destroy the thing it
    insures. A failure here is logged loudly and the trained weights, already written, stand.
    """
    if rank != 0 or not cfg.get("write_recovery", True):
        return
    try:
        state = recovery.capture(
            step=step, micro=micro, optimizer=opt, exposure=exposure,
            anneal_started=anneal_started, scored_checkpoint=scored_checkpoint,
            early_stop_best=es_best, early_stop_bad=es_bad,
            sampler_epoch=step if sampler is not None else None,
            recipe_fingerprint=cfg.get("continuation_fingerprint"),
        )
        path = recovery.save(out_dir, state)
        log(f"[recovery] wrote {path} ({path.stat().st_size / 1e9:.2f} GB) at step {step}")
    except Exception as error:  # noqa: BLE001 — never lose a saved checkpoint over its sidecar
        log(f"[recovery] WARNING could not write recovery state at step {step}: "
            f"{type(error).__name__}: {error}")


def save_checkpoint(model, cfg, out_dir, rank, world):
    """Save a Gemma4ForCausalLM checkpoint on rank 0. For an int8-master run, save the weights in their NATIVE
    int8 form (int8 `_data` + per-row `scale`) — half the size, and crucially NO full-bf16 materialization: the
    old path built a 9GB bf16 state dict AND a second 9GB fresh model (~18GB), the spike that got the run
    killed at the first save. The quartet loader (pipeline.model_load.load_causal_ckpt_quartet) runs int8
    checkpoints on load. bf16/snap runs (no int8 masters) keep the standard from_pretrained save.

    int8 path (2026-07-25+): STREAM gather→write (one tensor at a time) + host-memory SAVE PREFLIGHT
    so save-time RSS stays near training steady-state instead of 2× it. Layout unchanged
    (model.safetensors + quant_keys.json) — eval loaders do not change."""
    import json as _json
    import copy
    from pathlib import Path
    out_dir = Path(out_dir)
    if world > 1:                                       # FSDP consolidation → bf16 HF save (int8 master is single-GPU)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        full_sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            raise ConfigError(
                "world>1 FSDP saves write bf16 — unsupported under the precision law (2026-07-26). "
                "int8-master storage is single-GPU; run world=1 with int8_master:storage.")
        return
    if rank != 0:
        return
    sd = model.state_dict()
    int8_keys = [k for k, v in sd.items() if hasattr(v, "int_data") and hasattr(v, "scale")]
    if not int8_keys:
        raise ConfigError(
            "no int8 masters in this run (int8_master:snap/off) — unsupported under the precision law "
            "(2026-07-26): bf16 saves are gone. Run int8_master:storage.")
    # int8-master run → STREAMING native-int8 save (no full CPU state dict, no fresh model)
    entries, embedding_keys = _int8_save_entries(sd, int8_keys)
    need = estimate_streaming_save_need_bytes(entries)
    # Full-gather spike measured 9–16GB RSS (journal 2026-07-25). Streaming need is max-tensor
    # sized; still refuse to start when the box is below that watermark.
    save_preflight(need)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = save_file_streaming(
        out_dir / "model.safetensors",
        entries,
        metadata={"format": "pt", "quant": "int8_rowwise_v2" if embedding_keys else "int8_rowwise"},
    )
    tcfg = copy.deepcopy(model.config)          # in-memory config — never re-read model_path/config.json at
    tcfg._attn_implementation = "sdpa"          # save time (that disk read, under FS pressure from the 8.6GB
    tcfg.architectures = ["Gemma4ForCausalLM"]  # weight write, is what crashed the step-3100 save of the v2 run).
    tcfg.save_pretrained(out_dir)
    # The manifest is what makes the artifact reloadable: is_v2 in load_causal_ckpt_quartet is
    # (method == int8_rowwise_v2 or embedding_keys present), and it gates both the fp8 scale unfold
    # and rebuilding the Int8ScaledEmbedding wrappers.
    (out_dir / "quant_keys.json").write_text(_json.dumps(
        {"method": "int8_rowwise_v2", "keys": int8_keys, "embedding_keys": embedding_keys,
         "scale_dtype": "fp8_e4m3"} if embedding_keys
        else {"method": "int8_rowwise", "keys": int8_keys}))
    _cached_tokenizer(cfg["model_path"]).save_pretrained(out_dir)
    gb = written / 1e9
    largest = max(e["nbytes"] for e in entries) / 1e9
    log(f"saved int8 checkpoint -> {out_dir}  ({len(int8_keys)} int8 weights + scales, "
        f"{gb:.1f}GB on disk, streaming peak_tensor={largest:.2f}GB, no full-gather)")


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
    max_steps = int(cfg.get("max_steps") or 0)
    anneal_plan = resolve_anneal(cfg, max_steps if not smoke_steps else 0)
    contract = {
        "run_name": cfg.get("run_name"), "precision": cfg.get("precision"), "int8_master": int8,
        "optimizer_lowbit": lowbit or "bf16", "save_dir": save_dir, "init_from": init_from or "(base model)",
        "tokenized": tokenized, "tokenized_total_tokens": tok_count, "seq_len": cfg.get("seq_len"),
        "effective_batch": eff, "max_steps": cfg.get("max_steps"), "decay_frac": cfg.get("decay_frac"),
        "anneal": anneal_plan,
        "scored_checkpoint": "anneal-end" if anneal_plan.get("active") else "final",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "levers_on": [k for k, v in lev_on.items() if v], "smoke": bool(smoke_steps),
        "mode": typed.mode, "fingerprint": typed.fingerprint(),
    }
    if rank0():
        bar = "─" * 70
        anneal_row = (
            f"ON start={anneal_plan['start_step']} mix={anneal_plan['mix_tags']}"
            if anneal_plan.get("active") else f"off ({anneal_plan.get('reason', '?')})"
        )
        rows = [("mode", f"{typed.mode}   fingerprint={contract['fingerprint']}"),
                ("precision", contract["precision"]), ("int8_master", int8),
                ("optimizer_lowbit", contract["optimizer_lowbit"]),
                ("save_dir", save_dir or "‼ UNSET — NOTHING WILL BE SAVED"),
                ("init_from", contract["init_from"]),
                ("tokenized", f"{tokenized}  [{tok_count:,} tok]" if tok_count else (tokenized or "‼ UNSET")),
                ("seq_len", contract["seq_len"]),
                ("batch", f"{cfg.get('micro_batch_size')}×{cfg.get('grad_accum')}×{max(1,world)}gpu = {eff} eff"),
                ("max_steps", contract["max_steps"]), ("decay_frac", contract["decay_frac"]),
                ("anneal", anneal_row),
                ("scored_ckpt", contract["scored_checkpoint"]),
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
    # Point CUDA_HOME at a toolkit that can actually compile BEFORE anything imports a JIT. The
    # shell default here is 12.8, a debug-tools-only install with no compiler, so flashinfer's fp4
    # GEMM dies with "cuda-12.8/bin/nvcc: not found" — but only after the model has loaded, minutes
    # in. train_fp4.sh resolved its own toolkit; a direct `python scripts/train_cpt.py` inherited
    # the broken default. Fixing the DEFAULT rather than each call site is the point.
    _cuda_root = cuda_home.apply()
    if _cuda_root is None:
        log("[cuda] WARNING no toolkit with a compiler found; a JIT-compiling kernel will fail. "
            "Set QGRE_CUDA_HOME.")
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
        anneal_plan = {"active": False, "start_step": total, "reason": "smoke run — anneal disabled"}
    else:
        dl, sampler = make_loader(cfg, rank, world, want_mask=lever1_on, want_facts=lever4_on)
        total = cfg["max_steps"]
        anneal_plan = resolve_anneal(cfg, total)
        if rank0() and lever1_on:
            log(f"[lever1] entity-targeted loss ON — entity_loss_weight={entity_w} (weighted-mean CE)")
        if rank0() and lever4_on:
            log("[lever4] exposure tracking ON — fact_id exposures snapshotted at each checkpoint")
        if rank0():
            if anneal_plan["active"]:
                log(f"[anneal] ACTIVE — {anneal_plan['reason']}; mix={anneal_plan['mix_tags']}; "
                    f"lr → {anneal_plan['lr_end']} over final window; pre-anneal ckpt kept="
                    f"{anneal_plan['keep_pre']}")
            else:
                log(f"[anneal] inactive — {anneal_plan['reason']}")
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
    es_pat = int(cfg.get("early_stop_patience", 0) or 0)   # 0 = OFF (default): the block below never runs
    es_best, es_bad = float("inf"), 0
    anneal_started = False
    scored_checkpoint = "final"  # overwritten to anneal-end when anneal runs to completion
    probe_history: list = []
    last_window_loss = float("nan")   # the most recent logged window mean, for the probe record
    probe_verdict: dict | None = None
    probe_slices = _load_probe_slices(cfg)

    # RECOVERY (not init_from). init_from reloads weights and starts a fresh optimizer at step 0;
    # this restores optimizer moments, RNG, sampler epoch, exposure and the step itself, so a run
    # killed mid-flight continues instead of silently becoming a different experiment.
    resume_dir = str(cfg.get("resume_from") or "")
    if resume_dir and not smoke_steps:
        saved = recovery.load(resume_dir)
        if saved is None:
            raise RuntimeError(f"run.resume_from={resume_dir!r} holds no {recovery.FILENAME}")
        loop = recovery.restore(saved, optimizer=opt, exposure=exposure, sampler=sampler,
                                recipe_fingerprint=cfg.get("continuation_fingerprint"))
        step, micro = loop["step"], loop["micro"]
        anneal_started, scored_checkpoint = loop["anneal_started"], loop["scored_checkpoint"]
        es_best, es_bad = loop["early_stop_best"], loop["early_stop_bad"]
        if rank0():
            log(f"[recovery] resumed at step {step}/{total} from {resume_dir} "
                f"(rng restored: {', '.join(loop['restored_rng']) or 'none'}; "
                f"exposure facts {len(exposure)})")
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
                # Receipt from INSIDE the branch: make_loader only LOGS when fact_ids is absent, so
                # without this the lever can be on for a whole run, never reach here, and still be
                # reported as an active intervention.
                lever_receipts.fire("lever4_exposure_tracking")
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
            lever_receipts.fire("lever1_entity_loss")   # the weighting actually reached the loss
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
            # Anneal window: linear LR → lr_end on the anneal mix. Pre-anneal uses the WSD schedule
            # as before. Precision floor untouched — schedule is not dtype.
            if anneal_started:
                lr = anneal_lr(
                    step, total, anneal_plan["start_step"], cfg["lr"], anneal_plan["lr_end"],
                )
            else:
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
                raise ConfigError("int8_master:snap (bf16-grid SR) is retired under the precision law — use int8_master:storage")
            opt.zero_grad(set_to_none=True)
            if gctl is not None:
                gctl.reset()                # empty the accumulator for the next window
            step += 1
            # ── anneal switch at the 90% mark (or configured frac) ─────────────────
            if (
                anneal_plan.get("active")
                and not anneal_started
                and step >= anneal_plan["start_step"]
                and not smoke_steps
            ):
                exp_check = pre_anneal_exposure_check(
                    cfg, exposure if lever4_on else None, anneal_plan["require_exposure"],
                )
                if rank0():
                    log(f"[anneal] {exp_check['message']}")
                    if not exp_check["ok"]:
                        log("[anneal] WARNING — exposure floor not proven; anneal will still run "
                            "(probes decide install; do not read a pretty loss curve as install)")
                if anneal_plan["keep_pre"] and save_dir:
                    pre_dir = f"{save_dir}/pre_anneal"
                    if rank0():
                        log(f"[anneal] saving pre-anneal fallback checkpoint → {pre_dir}")
                    save_checkpoint(model, cfg, pre_dir, rank, world)
                    if lever4_on and rank0():
                        _save_exposure(pre_dir, step, exposure)
                    model.train()
                # Switch the loader to the anneal mix (hydrated + replay by default).
                dl, sampler = make_loader(
                    cfg, rank, world,
                    want_mask=lever1_on, want_facts=lever4_on,
                    tag_filter=anneal_plan["mix_tags"],
                )
                data_iter = iter(dl)
                anneal_started = True
                scored_checkpoint = "anneal-end"
                if rank0():
                    log(f"[anneal] MIX SWITCH + LR DECAY at step {step}/{total} "
                        f"(mix={anneal_plan['mix_tags']}, lr_end={anneal_plan['lr_end']})")
                if world > 1:
                    dist.barrier()
            if step % cfg["log_every"] == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                cur = torch.cuda.memory_allocated() / 1e9    # post-step+zero_grad: weights + optimizer state (grads freed)
                wmean = window_loss / (accum * cfg['log_every'])
                log(f"step {step}/{total} loss {wmean:.4f} "
                    f"lr {lr:.2e} peakmem {mem:.1f}GB (resident {cur:.1f}GB; transient {mem-cur:.1f}GB) {(time.time()-t0)/step:.2f}s/step")
                last_window_loss = wmean
                window_loss = 0.0
                if es_pat:                       # loss-plateau early stop — inert unless run.early_stop_patience>0
                    if wmean < es_best - 1e-4:
                        es_best, es_bad = wmean, 0
                    else:
                        es_bad += 1
                        if es_bad >= es_pat:
                            log(f"early stop: loss plateaued for {es_pat} log-windows (best {es_best:.4f}) — stopping at step {step}/{total}")
                            break
            # Honor save_every (was silently ignored): periodic checkpoint → crash-recovery for long
            # fleet runs + the per-step checkpoints the geometry probe sweeps to watch geometry form.
            se = cfg.get("save_every", 0)
            if save_dir and not smoke_steps and se and step % se == 0 and step < total:
                save_checkpoint(model, cfg, f"{save_dir}/step_{step:06d}", rank, world)
                if lever4_on and rank0():    # snapshot cumulative fact exposures alongside the checkpoint
                    _save_exposure(f"{save_dir}/step_{step:06d}", step, exposure)
                _write_recovery(f"{save_dir}/step_{step:06d}", cfg, rank, step, micro, opt, exposure,
                                anneal_started, scored_checkpoint, es_best, es_bad, sampler, contract)
                # Measure LEARNING, not loss. Forced choice is a logprob pass on the model already
                # in memory, so this costs a fraction of a step and needs no checkpoint reload.
                if probe_slices and rank0():
                    try:
                        model.eval()
                        with torch.no_grad():
                            scored = training_probe.probe_loaded_model(
                                _cached_tokenizer(cfg["model_path"]), model, str(device), probe_slices)
                        probe_history.append(training_probe.record(
                            step=step, loss=last_window_loss, scores=scored))
                        probe_verdict = training_probe.decide(
                            probe_history, general_base_acc=cfg.get("probe_general_base_acc"))
                        log(f"[probe] step {step}: "
                            + ", ".join(f"{k}={v['acc']:.4f}" for k, v in scored.items())
                            + (f" — STOP: {probe_verdict['reasons'][0]}" if probe_verdict["stop"] else ""))
                        if probe_verdict["stop"] and cfg.get("probe_stop"):
                            log(f"[probe] stopping at step {step}/{total} on measured learning: "
                                + "; ".join(probe_verdict["reasons"]))
                            model.train()
                            break
                    except Exception as error:  # noqa: BLE001 — a probe must never kill a paid run
                        log(f"[probe] WARNING probe failed at step {step}: "
                            f"{type(error).__name__}: {error}")
                model.train()

    log(f"done {step} steps in {time.time()-t0:.0f}s")
    # Reconcile ENABLED levers against receipts fired from inside their effect branches. Smoke runs
    # force lever1_on/lever4_on off against a config that may still enable them, so they are exempt
    # by construction rather than through a status the audit would have to special-case.
    lever_audit = ({"status": "SKIPPED_SMOKE"} if smoke_steps
                   else lever_receipts.audit(_levers, optimizer_steps=step))
    if lever_audit.get("status") == "FAIL":
        log(f"[levers] RECEIPT FAILURE — enabled but never executed: {lever_audit['silent']}")
    if save_dir and not smoke_steps:
        save_checkpoint(model, cfg, save_dir, rank, world)
        if rank0():                          # recipe provenance: what RUN CONTRACT produced this checkpoint
            import json as _json
            from pathlib import Path
            # Name which checkpoint is the eval candidate (anneal-end vs final / pre-anneal fallback).
            contract = dict(contract)
            contract["scored_checkpoint"] = scored_checkpoint
            contract["anneal_started"] = anneal_started
            contract["final_step"] = step
            contract["lever_receipts"] = lever_audit   # what RAN, beside what was configured
            # Per-window replay: the distribution and the zero-replay count, not just the
            # realized share — a share cannot distinguish a healthy schedule from a clumped one.
            contract["replay_schedule"] = getattr(dl, "replay_report", None)
            # The learning trajectory, recorded whether or not the stop rules are armed.
            contract["training_probe"] = {"history": probe_history, "verdict": probe_verdict,
                                          "armed": bool(cfg.get("probe_stop"))}
            if anneal_started and anneal_plan.get("keep_pre"):
                contract["pre_anneal_checkpoint"] = f"{save_dir}/pre_anneal"
            (Path(save_dir) / "run_contract.json").write_text(_json.dumps(contract, indent=2, default=str))
            log(f"scored checkpoint = {scored_checkpoint}"
                + (f" (pre-anneal fallback at {save_dir}/pre_anneal)" if anneal_started else ""))
        if lever4_on and rank0():
            _save_exposure(save_dir, step, exposure)
        _write_recovery(save_dir, cfg, rank, step, micro, opt, exposure, anneal_started,
                        scored_checkpoint, es_best, es_bad, sampler, contract)
    # Enforced AFTER the checkpoint and run_contract are on disk: the weights are already trained and
    # a silent lever is a reporting failure, not a reason to discard them. The artifact carries the
    # FAIL record, so what died and why is readable from the checkpoint itself.
    lever_receipts.enforce(lever_audit)
    if rank0():
        _finish_status("done")               # flip the console heartbeat to done; close the run log
        # The canonical launcher starts the config-bound trust chain after this process exits. That
        # boundary releases the training model's GPU allocation before live reference/candidate scoring.
    # NOTE: NVFP4 QAT→convert is intentionally NOT done here. Running quantize_(step="convert") on the
    # live FSDP2-sharded model on rank 0 only would (a) risk a collective deadlock and (b) discard the
    # result (it's never re-saved). The convert is a separate OFFLINE step: load the saved bf16
    # checkpoint single-process, quantize_, save. (TODO: scripts/convert_nvfp4.py)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
