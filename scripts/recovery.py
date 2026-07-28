#!/usr/bin/env python3
"""Recovery checkpoints — the state a run needs to CONTINUE, not the state it needs to DEPLOY.

`data.init_from` reloads model weights and starts a fresh optimizer at step 0 with manually chosen
continuation parameters. That is a warm restart, not recovery: the Adam moments are gone, the LR
schedule restarts, the data order restarts (the sampler is seeded from `step`), the RNG that drives
stochastic rounding restarts, and the exposure counters reset. A run killed at step 400 of 450 and
"resumed" produces a different model than one that never died — which makes a crash silently change
the experiment rather than pause it.

Two artifact types, deliberately separate:

  deploy checkpoint    model.safetensors + quant_keys.json + config + tokenizer
                       what inference and evaluation load; written by save_checkpoint
  recovery checkpoint  THIS module — optimizer state, LR/anneal position, global step,
                       RNG states, sampler epoch, grad-accumulation position, exposure
                       counters, early-stop state, and the recipe fingerprint that says
                       which run it belongs to

Keeping them separate matters because they have different lifetimes and different sizes: a deploy
checkpoint is the thing you keep, a recovery checkpoint is dead weight the moment its run finishes.

Precision note: this module stores optimizer state EXACTLY as the optimizer holds it. The low-bit
AdamW moments are saved at their own width and restored at their own width; nothing here widens,
casts, or "normalises" a dtype. Recovery must reproduce the run, not improve it.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch

SCHEMA = "recovery/1"
FILENAME = "recovery.pt"


def _rng_state() -> dict[str, Any]:
    """Every generator that can change a training step, captured together.

    Stochastic rounding draws from the torch CUDA generator, so omitting it would make a resumed
    fp4 run diverge from an uninterrupted one even with identical weights and optimizer state.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy
        state["numpy"] = numpy.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state: dict[str, Any]) -> list[str]:
    """Restore what is present; report what was restored so a caller can log it rather than assume."""
    restored = []
    if "python" in state:
        random.setstate(state["python"])
        restored.append("python")
    if "torch_cpu" in state:
        # Forced to a CPU uint8 tensor: a recovery file loaded with map_location="cuda" hands back
        # a CUDA tensor here and set_rng_state rejects it ("RNG state must be a torch.ByteTensor").
        torch.set_rng_state(state["torch_cpu"].to(device="cpu", dtype=torch.uint8))
        restored.append("torch_cpu")
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_state = state["torch_cuda"]
        # Refuse a partial restore rather than seeding some devices and not others.
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([t.to(device="cpu", dtype=torch.uint8) for t in cuda_state])
            restored.append("torch_cuda")
    if "numpy" in state:
        try:
            import numpy
            numpy.random.set_state(state["numpy"])
            restored.append("numpy")
        except ImportError:
            pass
    return restored


def capture(
    *,
    step: int,
    micro: int,
    optimizer: Any,
    exposure: Any,
    anneal_started: bool,
    scored_checkpoint: str,
    early_stop_best: float,
    early_stop_bad: int,
    sampler_epoch: int | None,
    recipe_fingerprint: str | None,
) -> dict[str, Any]:
    """Everything needed to resume mid-run. Pure with respect to the objects it reads."""
    return {
        "schema": SCHEMA,
        "step": int(step),
        "micro": int(micro),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "exposure": dict(exposure) if exposure else {},
        "anneal_started": bool(anneal_started),
        "scored_checkpoint": str(scored_checkpoint),
        "early_stop_best": float(early_stop_best),
        "early_stop_bad": int(early_stop_bad),
        "sampler_epoch": sampler_epoch,
        "recipe_fingerprint": recipe_fingerprint,
        "rng": _rng_state(),
    }


def save(directory: str | Path, state: dict[str, Any]) -> Path:
    """Write recovery.pt atomically — a half-written recovery file is worse than none.

    Torn state would be loaded by a later resume and either explode mid-run or, worse, restore
    partial optimizer moments. Written to a sibling temp name and renamed, so the final path
    either holds a complete checkpoint or does not exist.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / FILENAME
    staging = final.with_suffix(final.suffix + ".partial")
    torch.save(state, staging)
    os.replace(staging, final)
    return final


def load(directory: str | Path) -> dict[str, Any] | None:
    """Read recovery.pt from a run directory, or None when there is none to read.

    Always onto CPU. optimizer.load_state_dict moves its own tensors to each parameter's device,
    while the RNG states must stay CPU ByteTensors — loading straight onto CUDA breaks the RNG
    restore, which is the one piece a resumed fp4 run cannot do without.
    """
    path = Path(directory) / FILENAME
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


class RecoveryMismatch(RuntimeError):
    """A recovery checkpoint that does not belong to this run."""


def restore(
    state: dict[str, Any],
    *,
    optimizer: Any,
    exposure: Any,
    sampler: Any,
    recipe_fingerprint: str | None,
) -> dict[str, Any]:
    """Rehydrate optimizer, RNG, sampler and counters; return the scalar loop state.

    Fails closed on a fingerprint mismatch. Resuming one run from another run's optimizer moments
    silently produces a model neither recipe describes, and the resulting checkpoint would carry a
    contract claiming a lineage it does not have.
    """
    if str(state.get("schema")) != SCHEMA:
        raise RecoveryMismatch(f"recovery schema {state.get('schema')!r}, expected {SCHEMA!r}")
    stored = state.get("recipe_fingerprint")
    if stored is not None and recipe_fingerprint is not None and stored != recipe_fingerprint:
        raise RecoveryMismatch(
            f"recovery checkpoint belongs to recipe {stored} but this run is {recipe_fingerprint}. "
            "Resuming across recipes yields a model neither recipe describes."
        )
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if exposure is not None and state.get("exposure"):
        exposure.update(state["exposure"])
    if sampler is not None and state.get("sampler_epoch") is not None:
        sampler.set_epoch(int(state["sampler_epoch"]))
    restored_rng = _restore_rng(state.get("rng") or {})
    return {
        "step": int(state["step"]),
        "micro": int(state["micro"]),
        "anneal_started": bool(state["anneal_started"]),
        "scored_checkpoint": str(state["scored_checkpoint"]),
        "early_stop_best": float(state["early_stop_best"]),
        "early_stop_bad": int(state["early_stop_bad"]),
        "restored_rng": restored_rng,
    }
