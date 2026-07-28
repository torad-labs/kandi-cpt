#!/usr/bin/env python3
"""Guarantee replay inside every optimizer window, instead of only in expectation.

Replay is mixed at corpus construction and delivered through a shuffled loader, so its frequency is
correct on average and unreliable per update. With a 9.09% replay share and a 16-block optimizer
window, ~22% of optimizer updates see NO replay block at all; at the 4.76% arm it is ~46%. Those
updates are pure domain gradient — exactly the updates replay exists to prevent. Averaged over a
run the share looks right, which is why this is invisible in the stats sidecar.

The fix is a schedule, not a bigger corpus: draw each window of `micro_batch_size × grad_accum`
blocks with an explicit replay quota. Both pools are consumed as reshuffled cycles, so no block is
starved and replay (a much smaller pool) simply repeats — which is what a replay set is for.

This is a DATA ORDER change, not a precision or objective change. It is part of run identity
(train.replay_per_window is fingerprinted) so runs before and after are never silently compared.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence


def window_quotas(n_windows: int, window: int, replay_share: float, *, seed: int = 0) -> list[int]:
    """Per-window replay counts: mean tracks the corpus share, minimum is 1, maximum is window-1.

    A single flat quota cannot do both. Rounding 16 x 0.0910 = 1.456 down to 1 removes every
    zero-replay update but also cuts the realized share from 9.10% to 6.25% — trading one
    distortion for another. Spreading the fractional part across windows (some get 1, some get 2)
    holds the share AND keeps every window non-empty.

    Floored at 1 because eliminating zero-replay updates is the point; capped at window-1 because a
    replay-only update is the opposite failure.
    """
    if n_windows <= 0 or window < 2 or replay_share <= 0:
        return [0] * max(0, n_windows)
    total = round(n_windows * window * replay_share)
    total = max(total, n_windows * 1)
    total = min(total, n_windows * (window - 1))
    base, remainder = divmod(total, n_windows)
    counts = [base + 1] * remainder + [base] * (n_windows - remainder)
    # Deterministically spread the +1 windows rather than front-loading them, so replay density is
    # not systematically higher early in training.
    random.Random(seed).shuffle(counts)
    return counts


def derive_quota(window: int, replay_share: float) -> int:
    """Flat per-window quota — the single-number view, kept for reporting and explicit config."""
    if window < 2 or replay_share <= 0:
        return 0
    return max(1, min(window - 1, round(window * replay_share)))


class _Cycle:
    """A reshuffled cycle over a pool — every element is used once before any repeats."""

    def __init__(self, items: Sequence[int], rng: random.Random) -> None:
        self._items = list(items)
        self._rng = rng
        self._order: list[int] = []

    def take(self, count: int) -> list[int]:
        drawn = []
        while len(drawn) < count:
            if not self._order:
                self._order = list(self._items)
                self._rng.shuffle(self._order)
            drawn.append(self._order.pop())
        return drawn


def plan_windows(
    replay_indices: Sequence[int],
    domain_indices: Sequence[int],
    *,
    window: int,
    quota: int | Sequence[int],
    n_windows: int,
    seed: int,
) -> list[int]:
    """Flat index order whose every `window`-sized slice holds exactly `quota` replay indices.

    Positions are shuffled WITHIN each window so replay does not land at a fixed offset — a fixed
    offset would correlate replay with position in the accumulation, which is its own artifact.
    """
    quotas = [quota] * n_windows if isinstance(quota, int) else list(quota)
    if not any(q > 0 for q in quotas) or not replay_indices:
        rng = random.Random(seed)
        pool = _Cycle(list(domain_indices) or list(replay_indices), rng)
        return pool.take(window * n_windows)

    rng = random.Random(seed)
    replay_pool = _Cycle(replay_indices, rng)
    domain_pool = _Cycle(domain_indices or replay_indices, rng)
    order: list[int] = []
    for per_window in quotas[:n_windows]:
        per_window = max(0, min(window, int(per_window)))
        chunk = replay_pool.take(per_window) + domain_pool.take(window - per_window)
        rng.shuffle(chunk)
        order.extend(chunk)
    return order


def window_report(
    order: Sequence[int],
    replay_indices: Sequence[int],
    *,
    window: int,
) -> dict:
    """Per-window replay distribution and the zero-replay count, for the run record.

    "Realized share" alone cannot distinguish a healthy schedule from one that clumps all its
    replay into a few windows, so the distribution and the zero-replay count are reported too.
    """
    replay = set(replay_indices)
    counts = [
        sum(1 for i in order[start:start + window] if i in replay)
        for start in range(0, len(order) - window + 1, window)
    ]
    total = len(counts) * window
    return {
        "windows": len(counts),
        "window_size": window,
        "replay_per_window_min": min(counts) if counts else 0,
        "replay_per_window_max": max(counts) if counts else 0,
        "replay_per_window_mean": (sum(counts) / len(counts)) if counts else 0.0,
        "zero_replay_windows": sum(1 for c in counts if c == 0),
        "zero_replay_fraction": (sum(1 for c in counts if c == 0) / len(counts)) if counts else 0.0,
        "realized_replay_block_share": (sum(counts) / total) if total else 0.0,
    }


def expected_zero_replay_fraction(window: int, replay_share: float) -> float:
    """P(a shuffled window contains no replay block) — the number this scheduler drives to zero.

    Reported beside the realized figure so the run record states what the OLD behaviour would have
    produced on this exact corpus, rather than quoting the review's example numbers.
    """
    if replay_share <= 0:
        return 1.0
    return float(math.pow(1.0 - min(1.0, replay_share), window))


class OptimizerWindowSampler:
    """torch Sampler yielding `plan_windows` order. Deterministic given `seed` and `epoch`.

    Deliberately not a DistributedSampler subclass: int8-master storage is single-GPU (world>1 is
    refused at save time), so a multi-rank variant would be untested code guarding a path that
    cannot run.
    """

    def __init__(
        self,
        replay_indices: Sequence[int],
        domain_indices: Sequence[int],
        *,
        window: int,
        quota: int | Sequence[int],
        n_windows: int,
        seed: int = 42,
    ) -> None:
        self.replay_indices = list(replay_indices)
        self.domain_indices = list(domain_indices)
        self.window = int(window)
        self.quota = int(quota) if isinstance(quota, int) else list(quota)
        self.n_windows = int(n_windows)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.window * self.n_windows

    def __iter__(self) -> Iterator[int]:
        return iter(plan_windows(
            self.replay_indices, self.domain_indices,
            window=self.window, quota=self.quota, n_windows=self.n_windows,
            seed=self.seed + self.epoch,
        ))
