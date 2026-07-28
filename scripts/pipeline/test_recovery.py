#!/usr/bin/env python3
"""Recovery checkpoints carry the state a run needs to CONTINUE.

`data.init_from` reloads weights and starts a fresh optimizer at step 0 — a warm restart, so a run
killed at step 400 of 450 and "resumed" produces a different model than one that never died. These
tests pin the pieces that distinguish continuation from restart, and the refusals that stop a
recovery file being applied to a run it does not belong to.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_recovery.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import recovery


class _Sampler:
    def __init__(self) -> None:
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.full((4,), 0.5)
    opt.step()                                    # give the moments something to lose
    return opt


def _capture(opt, exposure, *, step=400, fingerprint="recipe-a") -> dict:
    return recovery.capture(
        step=step, micro=7, optimizer=opt, exposure=exposure, anneal_started=True,
        scored_checkpoint="anneal-end", early_stop_best=1.25, early_stop_bad=2,
        sampler_epoch=step, recipe_fingerprint=fingerprint,
    )


def test_round_trip_restores_optimizer_rng_and_counters() -> None:
    opt = _optimizer()
    exposure = Counter({"fact:1": 3, "fact:2": 5})
    state = _capture(opt, exposure)

    with tempfile.TemporaryDirectory() as tmp:
        path = recovery.save(tmp, state)
        assert path.name == recovery.FILENAME and path.is_file()
        assert not list(Path(tmp).glob("*.partial")), "atomic write left staging debris"
        reloaded = recovery.load(tmp)
    assert reloaded is not None

    # A DIFFERENT optimizer and empty counters — restoring must repopulate them.
    fresh, fresh_exposure, sampler = _optimizer(), Counter(), _Sampler()
    random.seed(999)
    loop = recovery.restore(reloaded, optimizer=fresh, exposure=fresh_exposure,
                            sampler=sampler, recipe_fingerprint="recipe-a")

    assert loop["step"] == 400 and loop["micro"] == 7
    assert loop["anneal_started"] is True and loop["scored_checkpoint"] == "anneal-end"
    assert loop["early_stop_best"] == 1.25 and loop["early_stop_bad"] == 2
    assert sampler.epoch == 400, "sampler epoch drives data order; a reset restarts the corpus"
    assert dict(fresh_exposure) == {"fact:1": 3, "fact:2": 5}
    assert "python" in loop["restored_rng"] and "torch_cpu" in loop["restored_rng"]

    # The Adam moments are the thing a warm restart throws away.
    original = opt.state_dict()["state"]
    restored = fresh.state_dict()["state"]
    assert original.keys() == restored.keys() and original, "no optimizer state was captured"
    for key, value in original.items():
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(value[name], restored[key][name]), f"{name} did not survive round trip"


def test_rng_restore_makes_the_next_draw_reproducible() -> None:
    """Stochastic rounding draws from these generators; a reset RNG diverges a resumed fp4 run."""
    opt = _optimizer()
    state = _capture(opt, Counter())
    expected_python = random.random()
    expected_torch = torch.rand(3)

    random.seed(12345)
    torch.manual_seed(12345)
    recovery.restore(state, optimizer=opt, exposure=Counter(), sampler=None,
                     recipe_fingerprint="recipe-a")
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)


def test_restore_refuses_a_foreign_or_unknown_recovery_file() -> None:
    opt = _optimizer()
    state = _capture(opt, Counter(), fingerprint="recipe-a")

    for label, kwargs, mutate in (
        ("a different recipe", {"recipe_fingerprint": "recipe-b"}, None),
        ("an unknown schema", {"recipe_fingerprint": "recipe-a"}, lambda d: d.update(schema="recovery/999")),
    ):
        candidate = dict(state)
        if mutate:
            mutate(candidate)
        try:
            recovery.restore(candidate, optimizer=opt, exposure=Counter(), sampler=None, **kwargs)
        except recovery.RecoveryMismatch:
            continue
        raise AssertionError(f"restore accepted {label}")

    # No fingerprint recorded on either side is not a mismatch — it is an older or anonymous run.
    anonymous = _capture(opt, Counter(), fingerprint=None)
    recovery.restore(anonymous, optimizer=opt, exposure=Counter(), sampler=None,
                     recipe_fingerprint=None)


def test_rng_restore_survives_a_non_cpu_uint8_state() -> None:
    """torch.set_rng_state demands a CPU ByteTensor.

    A recovery file loaded with map_location pointing at the training device hands back a CUDA
    tensor here and the restore dies with "RNG state must be a torch.ByteTensor" — after the model
    has loaded, minutes into the resume. Found by running the resume, not by reading it. load() now
    always maps to CPU; this pins the coercion so a future map_location change cannot re-break it.
    """
    opt = _optimizer()
    state = _capture(opt, Counter())
    state["rng"]["torch_cpu"] = state["rng"]["torch_cpu"].to(torch.int64)   # wrong dtype, as a proxy
    loop = recovery.restore(state, optimizer=opt, exposure=Counter(), sampler=None,
                            recipe_fingerprint="recipe-a")
    assert "torch_cpu" in loop["restored_rng"]


def test_load_returns_none_when_there_is_nothing_to_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert recovery.load(tmp) is None


if __name__ == "__main__":
    test_round_trip_restores_optimizer_rng_and_counters()
    test_rng_restore_makes_the_next_draw_reproducible()
    test_restore_refuses_a_foreign_or_unknown_recovery_file()
    test_rng_restore_survives_a_non_cpu_uint8_state()
    test_load_returns_none_when_there_is_nothing_to_resume()
    print("OK — optimizer moments, RNG, sampler epoch and counters survive; foreign files refused")
