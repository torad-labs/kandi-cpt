#!/usr/bin/env python3
"""Replay must land in EVERY optimizer window, not just on average.

At a 9.09% replay share with a 16-block window, ~22% of optimizer updates see no replay block; at
4.76% it is ~46%. Those updates are pure domain gradient — the exact thing replay exists to
prevent — and the run's average share hides it completely.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_window_sampler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import window_sampler as W


def test_the_defect_is_real_before_asserting_the_fix() -> None:
    """Quantify the old behaviour on this corpus shape, so the fix is measured against a number."""
    assert round(W.expected_zero_replay_fraction(16, 0.0909), 2) == 0.22
    assert round(W.expected_zero_replay_fraction(16, 0.0476), 2) == 0.46
    assert W.expected_zero_replay_fraction(16, 0.0) == 1.0


def test_every_window_carries_the_quota() -> None:
    replay, domain = list(range(0, 20)), list(range(100, 900))
    order = W.plan_windows(replay, domain, window=16, quota=2, n_windows=50, seed=7)
    report = W.window_report(order, replay, window=16)
    assert report["windows"] == 50
    assert report["zero_replay_windows"] == 0
    assert report["replay_per_window_min"] == report["replay_per_window_max"] == 2


def test_a_tiny_replay_pool_repeats_rather_than_starving() -> None:
    """Replay sets are small by construction; repeating them is the point, not a defect."""
    replay, domain = [1, 2], list(range(100, 500))
    order = W.plan_windows(replay, domain, window=16, quota=2, n_windows=40, seed=3)
    report = W.window_report(order, replay, window=16)
    assert report["zero_replay_windows"] == 0
    assert order.count(1) + order.count(2) == 80        # 2 per window x 40

    # Every element is used before any repeats, so no replay block is starved.
    assert abs(order.count(1) - order.count(2)) <= 1


def test_domain_blocks_are_never_fully_displaced() -> None:
    replay, domain = list(range(0, 500)), list(range(1000, 1010))
    order = W.plan_windows(replay, domain, window=8, quota=7, n_windows=20, seed=11)
    report = W.window_report(order, replay, window=8)
    assert report["replay_per_window_max"] == 7, "a replay-only window is the opposite failure"
    assert all(any(i >= 1000 for i in order[s:s + 8]) for s in range(0, len(order), 8))


def test_derive_quota_floors_at_one_and_leaves_room_for_domain() -> None:
    assert W.derive_quota(16, 0.0909) == 1        # round(1.45)=1
    assert W.derive_quota(16, 0.0476) == 1        # round(0.76)=1 -> floored at 1 anyway
    assert W.derive_quota(16, 0.25) == 4
    assert W.derive_quota(16, 0.99) == 15, "never a replay-only window"
    assert W.derive_quota(16, 0.0) == 0, "no replay in the corpus -> no quota to meet"
    assert W.derive_quota(1, 0.5) == 0, "a 1-block window cannot hold both"


def test_window_quotas_hold_the_share_and_never_empty_a_window() -> None:
    """A flat quota can do one or the other, not both — this is why quotas vary per window.

    16 x 0.0910 = 1.456. Rounding to a flat 1 removes every zero-replay update but cuts the
    realized share from 9.10% to 6.25%, trading one distortion for another.
    """
    assert W.derive_quota(16, 0.09097) == 1                       # the flat view, and its problem
    assert 1 / 16 < 0.09097, "a flat quota of 1 under-delivers replay at this share"

    quotas = W.window_quotas(277, 16, 0.09097, seed=42)
    assert len(quotas) == 277
    assert min(quotas) >= 1, "no window may be empty of replay"
    assert max(quotas) <= 15, "no window may be replay-only"
    realized = sum(quotas) / (277 * 16)
    assert abs(realized - 0.09097) < 0.001, f"share drifted: {realized}"

    # Applied end to end, the order carries both properties.
    replay, domain = list(range(0, 135)), list(range(1000, 2349))
    order = W.plan_windows(replay, domain, window=16, quota=quotas, n_windows=277, seed=42)
    report = W.window_report(order, replay, window=16)
    assert report["zero_replay_windows"] == 0
    assert abs(report["realized_replay_block_share"] - 0.09097) < 0.001


def test_window_quotas_degenerate_cases() -> None:
    assert W.window_quotas(0, 16, 0.1) == []
    assert W.window_quotas(5, 16, 0.0) == [0, 0, 0, 0, 0], "no replay in the corpus -> no quota"
    assert W.window_quotas(5, 1, 0.5) == [0] * 5, "a 1-block window cannot hold both"
    # A share so high it would empty the window of domain blocks is capped, not honoured.
    assert max(W.window_quotas(10, 4, 0.99)) <= 3


def test_replay_is_not_pinned_to_a_fixed_offset() -> None:
    """A fixed position would correlate replay with accumulation order — its own artifact."""
    replay, domain = list(range(0, 50)), list(range(100, 900))
    order = W.plan_windows(replay, domain, window=8, quota=2, n_windows=60, seed=5)
    positions = {
        offset
        for start in range(0, len(order), 8)
        for offset, i in enumerate(order[start:start + 8]) if i in set(replay)
    }
    assert len(positions) > 2, f"replay only ever appeared at offsets {sorted(positions)}"


def test_deterministic_for_a_seed_and_varies_by_epoch() -> None:
    replay, domain = list(range(0, 20)), list(range(100, 500))
    kwargs = dict(window=8, quota=2, n_windows=10)
    assert W.plan_windows(replay, domain, seed=1, **kwargs) == W.plan_windows(replay, domain, seed=1, **kwargs)
    assert W.plan_windows(replay, domain, seed=1, **kwargs) != W.plan_windows(replay, domain, seed=2, **kwargs)

    sampler = W.OptimizerWindowSampler(replay, domain, window=8, quota=2, n_windows=10, seed=1)
    first = list(sampler)
    assert list(sampler) == first, "same epoch must reproduce"
    sampler.set_epoch(1)
    assert list(sampler) != first, "a new epoch must reshuffle"
    assert len(sampler) == 80


def test_no_replay_in_the_corpus_degrades_to_plain_cycling() -> None:
    order = W.plan_windows([], list(range(100, 200)), window=8, quota=0, n_windows=5, seed=1)
    assert len(order) == 40 and all(100 <= i < 200 for i in order)
    report = W.window_report(order, [], window=8)
    assert report["zero_replay_windows"] == 5, "honest: no replay means every window is zero-replay"


def test_report_distinguishes_a_clumped_schedule_from_a_healthy_one() -> None:
    """Realized share alone cannot tell these apart — which is why the distribution is reported."""
    replay = [0, 1]
    clumped = [0, 1, 2, 3] + [4, 5, 6, 7]          # both replay blocks in window 0, none in window 1
    healthy = [0, 2, 3, 4] + [1, 5, 6, 7]
    clumped_report = W.window_report(clumped, replay, window=4)
    healthy_report = W.window_report(healthy, replay, window=4)
    assert clumped_report["realized_replay_block_share"] == healthy_report["realized_replay_block_share"]
    assert clumped_report["zero_replay_windows"] == 1
    assert healthy_report["zero_replay_windows"] == 0


if __name__ == "__main__":
    test_the_defect_is_real_before_asserting_the_fix()
    test_every_window_carries_the_quota()
    test_a_tiny_replay_pool_repeats_rather_than_starving()
    test_domain_blocks_are_never_fully_displaced()
    test_derive_quota_floors_at_one_and_leaves_room_for_domain()
    test_window_quotas_hold_the_share_and_never_empty_a_window()
    test_window_quotas_degenerate_cases()
    test_replay_is_not_pinned_to_a_fixed_offset()
    test_deterministic_for_a_seed_and_varies_by_epoch()
    test_no_replay_in_the_corpus_degrades_to_plain_cycling()
    test_report_distinguishes_a_clumped_schedule_from_a_healthy_one()
    print("OK — every window carries its quota, zero-replay updates eliminated, clumping detectable")
