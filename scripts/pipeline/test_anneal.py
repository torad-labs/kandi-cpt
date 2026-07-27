#!/usr/bin/env python3
"""Red-green proofs for the CPT anneal phase (no GPU)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_cpt as T  # noqa: E402
from cpt_config import AnnealConfig, ConfigError, CptConfig, TrainConfig  # noqa: E402
from tokenize_corpus import _block_majority_tags  # noqa: E402


def test_resolve_anneal_default_on_at_10pct():
    plan = T.resolve_anneal({"anneal": {"enabled": True, "frac": 0.10, "min_steps": 500}}, 1000)
    assert plan["active"] is True
    assert plan["start_step"] == 900
    assert plan["mix_tags"] == ["hydrated", "replay"]
    assert plan["lr_end"] == 0.0
    assert plan["keep_pre"] is True


def test_resolve_anneal_disabled_on_tiny_run():
    plan = T.resolve_anneal({"anneal": {"enabled": True, "frac": 0.10, "min_steps": 500}}, 450)
    assert plan["active"] is False
    assert "tiny-run" in plan["reason"]


def test_resolve_anneal_explicit_off():
    plan = T.resolve_anneal({"anneal": {"enabled": False}}, 5000)
    assert plan["active"] is False
    assert "enabled=false" in plan["reason"]


def test_anneal_lr_linear_decay():
    # step at start → base; step at end → lr_end; midpoint halfway
    assert abs(T.anneal_lr(900, 1000, 900, 5e-5, 0.0) - 5e-5) < 1e-12
    assert abs(T.anneal_lr(1000, 1000, 900, 5e-5, 0.0) - 0.0) < 1e-12
    mid = T.anneal_lr(950, 1000, 900, 5e-5, 0.0)
    assert abs(mid - 2.5e-5) < 1e-12


def test_pre_anneal_exposure_check():
    cfg_off = {"levers": {"lever4_exposure_tracking": {"enabled": False}}}
    skip = T.pre_anneal_exposure_check(cfg_off, None, require=True)
    assert skip["ok"] is False
    assert skip["lever4_enabled"] is False
    assert "SKIPPED" in skip["message"]

    cfg_on = {"levers": {"lever4_exposure_tracking": {"enabled": True}}}
    empty = T.pre_anneal_exposure_check(cfg_on, Counter(), require=True)
    assert empty["ok"] is False
    assert "FAIL" in empty["message"]

    ok = T.pre_anneal_exposure_check(cfg_on, Counter({"fact-a": 3, "fact-b": 7}), require=True)
    assert ok["ok"] is True
    assert ok["n_facts"] == 2
    assert ok["total_exposures"] == 10
    assert "PASS" in ok["message"]


def test_block_majority_tags():
    # 4 tokens/block, 3 blocks
    tags = (
        ["hydrated"] * 4
        + ["clean"] * 3 + ["hydrated"]  # clean wins 3-1
        + ["replay"] * 2 + ["hydrated"] * 2  # tie → first-seen = replay
    )
    out = _block_majority_tags(tags, seq_len=4, n_blocks=3)
    assert out == ["hydrated", "clean", "replay"]


def test_anneal_config_defaults_and_validation():
    a = AnnealConfig()
    assert a.enabled is True and a.frac == 0.10 and a.min_steps == 500
    assert a.mix_tags == ["hydrated", "replay"]
    try:
        AnnealConfig(frac=0.9)
    except ConfigError as error:
        assert "frac" in str(error)
    else:
        raise AssertionError("frac=0.9 should hard-error")


def test_train_config_carries_anneal_into_trainer_dict():
    c = CptConfig()
    assert c.train.anneal.enabled is True
    td = c.to_trainer_dict()
    assert "anneal" in td
    assert td["anneal"]["frac"] == 0.10
    assert td["anneal"]["mix_tags"] == ["hydrated", "replay"]
    # nested override
    c2 = CptConfig._from_dict({"train": {"anneal": {"frac": 0.15, "enabled": True}}})
    assert c2.train.anneal.frac == 0.15
    plan = T.resolve_anneal(c2.to_trainer_dict(), 2000)
    assert plan["start_step"] == 1700


def test_fingerprint_tracks_anneal():
    f1 = CptConfig().fingerprint()
    f2 = CptConfig._from_dict({"train": {"anneal": {"enabled": False}}}).fingerprint()
    f3 = CptConfig._from_dict({"train": {"anneal": {"frac": 0.15}}}).fingerprint()
    assert f1 != f2
    assert f1 != f3


def test_anneal_rank_keep_mask_median_and_replay():
    tags = ["hydrated", "hydrated", "hydrated", "hydrated", "replay", "clean"]
    scores = [1.0, 2.0, 4.0, 5.0, 0.0, 9.0]
    keep = T._anneal_rank_keep_mask(tags, scores, {"hydrated", "replay"})
    # non-replay hydrated scores 1,2,4,5 → median = 4; keep >= 4 plus all replay; drop clean
    assert keep == [False, False, True, True, True, False]


def test_anneal_rank_keep_mask_uniform_keeps_all_in_tag():
    tags = ["hydrated", "hydrated", "replay"]
    scores = [3.0, 3.0, 3.0]
    keep = T._anneal_rank_keep_mask(tags, scores, {"hydrated", "replay"})
    assert keep == [True, True, True]


def test_anneal_defaults_are_byte_identical_to_the_median_cut():
    """The new quantile/replay knobs must not change any existing run: q=0.5 + replay_fraction=None
    has to reproduce the exact mask the median implementation produced."""
    import random as _r
    from train_cpt import _anneal_rank_keep_mask

    rng = _r.Random(0)
    for trial in range(40):
        n = rng.randint(1, 60)
        tags = [rng.choice(["hydrated", "replay", "clean"]) for _ in range(n)]
        scores = [rng.choice([1.0, 2.0, 3.0, 3.0, 5.0]) for _ in range(n)]
        wanted = {"hydrated", "replay"}
        new = _anneal_rank_keep_mask(tags, scores, wanted)

        ranked = [s for t, s in zip(tags, scores) if t in wanted and t != "replay"]
        if not ranked:
            old = [t in wanted for t in tags]
        else:
            med = sorted(ranked)[len(ranked) // 2]
            old = [False if t not in wanted else (True if t == "replay" else s >= med)
                   for t, s in zip(tags, scores)]
        assert new == old, f"default behaviour drifted on trial {trial}"


def test_anneal_replay_fraction_is_an_explicit_target():
    """Setting replay_fraction turns the emergent share into a declared one."""
    from train_cpt import _anneal_rank_keep_mask, anneal_composition

    tags = ["replay"] * 50 + ["hydrated"] * 50
    scores = [5.0] * 50 + [3.0] * 50               # uniform in-tag -> domain cut keeps all 50
    wanted = {"hydrated", "replay"}

    unset = _anneal_rank_keep_mask(tags, scores, wanted)
    comp_unset = anneal_composition(tags, unset, wanted)
    assert comp_unset["replay_block_share"] == 0.5, comp_unset      # emergent 50/50

    targeted = _anneal_rank_keep_mask(tags, scores, wanted, replay_fraction=0.20)
    comp = anneal_composition(tags, targeted, wanted)
    assert comp["kept_domain_blocks"] == 50
    assert comp["kept_replay_blocks"] == 12        # 0.2*50/0.8 = 12.5 -> 12
    assert abs(comp["replay_block_share"] - 12 / 62) < 1e-6

    # Deterministic and idempotent.
    assert _anneal_rank_keep_mask(tags, scores, wanted, replay_fraction=0.20) == targeted
    # Zero is reachable and means zero.
    z = anneal_composition(tags, _anneal_rank_keep_mask(tags, scores, wanted, replay_fraction=0.0), wanted)
    assert z["kept_replay_blocks"] == 0


def test_anneal_composition_reports_the_hidden_concentration():
    """The distortion the review named: all-replay-kept + half-domain-cut raises the share."""
    from train_cpt import _anneal_rank_keep_mask, anneal_composition

    tags = ["replay"] * 10 + ["hydrated"] * 90
    scores = [5.0] * 10 + [float(i % 10) for i in range(90)]        # spread -> real median cut
    wanted = {"hydrated", "replay"}
    before = anneal_composition(tags, [t in wanted for t in tags], wanted)
    after = anneal_composition(tags, _anneal_rank_keep_mask(tags, scores, wanted), wanted)
    assert after["replay_block_share"] > before["replay_block_share"], (before, after)


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} ANNEAL TESTS PASS")
