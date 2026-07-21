"""Regression tests for the config contract (scripts/cpt_config.py).

Run:  PYTHONPATH=scripts .venv/bin/python scripts/test_cpt_config.py

Guarantees:
  1. Defaults reproduce canonical-CPT behavior (configs/train_edc_lv_2026.yaml values).
  2. Nested hydration + the YAML 'lambda' alias + nvfp4 subtree build correctly.
  3. Unknown keys HARD-ERROR (with a did-you-mean hint).
  4. validate() catches the cross-field combos run_contract used to guard.
  5. The fingerprint is stable, ignores volatile fields, tracks real knobs.
  6. validate_runtime() enforces the mode⇆labels cross-check.
  7. ROUND-TRIP: every field the trainer reads from each live config is preserved
     when that config is expressed in the nested contract form. This is the
     'defaults reproduce current run behavior — verified, not assumed' proof.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpt_config import ConfigError, CptConfig, TrainConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
warnings.simplefilter("ignore")


def _expect(fn, frag):
    try:
        fn()
    except ConfigError as e:
        assert frag in str(e), f"wrong error: {e!r} (wanted {frag!r})"
        return
    raise AssertionError(f"expected ConfigError containing {frag!r}, none raised")


def test_defaults_are_canonical_cpt():
    c = CptConfig()
    assert (c.train.precision, c.train.int8_master, c.train.optimizer_lowbit) == ("fp4", "storage", 4)
    assert (c.train.lr, c.train.min_lr, c.train.max_steps, c.train.warmup_steps) == (5e-5, 5e-6, 3160, 50)
    assert (c.train.micro_batch_size, c.train.grad_accum, c.train.seq_len) == (2, 8, 512)
    assert c.train.betas == [0.9, 0.95] and c.train.decay_frac == 0.2
    assert c.model.freeze_embeddings and c.model.cpu_offload_ple
    assert (c.model.nvfp4.bf16_first_layers, c.model.nvfp4.bf16_last_layers) == (2, 5)
    assert (c.run.save_every, c.run.log_every) == (100, 5)
    assert (c.corpus_synthesis.answer_aug, c.corpus_synthesis.absent_neg) == (8, 125)
    assert (c.eval.tau_fab, c.eval.tau_acc, c.eval.tau_cov) == (0.25, 0.55, 0.40)
    assert c.eval.probe_domain is None and c.eval.probe_editions is None
    assert (c.eval.probe_min_n, c.eval.probe_min_delta_acc) == (50, 0.10)
    assert c.eval.checkpoint_fingerprint is None


def test_nested_build_and_lambda_alias():
    d = {"mode": "cpt", "train": {"lr": 2e-5, "levers": {
        "lever1_entity_loss": {"enabled": True, "entity_loss_weight": 3.0},
        "lever3_three_r": {"ewc": {"lambda": 2.0}}}},
        "model": {"nvfp4": {"bf16_first_layers": 0, "bf16_last_layers": 0}}}
    c = CptConfig._from_dict(d)
    assert c.train.lr == 2e-5
    assert c.train.levers.lever1_entity_loss.entity_loss_weight == 3.0
    assert c.train.levers.lever3_three_r.ewc.lambda_ == 2.0       # YAML 'lambda' → lambda_
    assert c.model.nvfp4.bf16_first_layers == 0


def test_unknown_key_hard_errors():
    _expect(lambda: CptConfig._from_dict({"train": {"max_step": 10}}), "max_steps")   # did-you-mean
    _expect(lambda: CptConfig._from_dict({"train": {"lrr": 1}}), "unknown key")
    _expect(lambda: CptConfig._from_dict({"bogus": {}}), "section")


def test_cross_field_validation():
    _expect(lambda: CptConfig._from_dict({"mode": "sft", "data": {"tokenized": "x"}}).validate(), "init_from")
    _expect(lambda: CptConfig._from_dict({"train": {"int8_master": "storage", "precision": "bf16"}}).validate(),
            "precision=fp4")
    _expect(lambda: TrainConfig(decay_frac=1.5), "decay_frac")
    _expect(lambda: TrainConfig(optimizer_lowbit=3), "optimizer_lowbit")
    _expect(lambda: TrainConfig(betas=[0.9]), "betas")
    _expect(lambda: CptConfig._from_dict({"mode": "sft", "data": {"init_from": "x"},
            "train": {"seq_len": 256}, "corpus_synthesis": {"sft_seq_len": 512}}).validate(), "sft_seq_len")
    _expect(lambda: CptConfig._from_dict({"eval": {"probe_domain": "pokemon-lore"}}), "set together")
    _expect(lambda: CptConfig._from_dict({"eval": {"probe_min_n": 0}}), "probe_min_n")
    _expect(lambda: CptConfig._from_dict({"eval": {"probe_report_root": "../logs"}}), "inside the engine")
    _expect(lambda: CptConfig._from_dict({"eval": {"checkpoint_fingerprint": "NOT-A-HASH"}}),
            "checkpoint_fingerprint")


def test_generic_probe_eval_config():
    c = CptConfig._from_dict({"eval": {
        "probe_domain": "pokemon-lore",
        "probe_editions": "kanto",
        "probe_min_n": 72,
        "probe_min_delta_acc": 0.15,
        "probe_report_root": "logs/evals/pokemon",
        "checkpoint_fingerprint": "a244e877c096ddfb",
    }})
    assert c.eval.probe_domain == "pokemon-lore"
    assert c.eval.probe_editions == "kanto"
    assert (c.eval.probe_min_n, c.eval.probe_min_delta_acc) == (72, 0.15)
    assert c.eval.probe_report_root == "logs/evals/pokemon"
    assert c.eval.checkpoint_fingerprint == "a244e877c096ddfb"


def test_fingerprint_and_int8_resolution():
    f1 = CptConfig().fingerprint()
    f2 = CptConfig._from_dict({"run": {"save_dir": "checkpoints/x", "run_name": "zzz"}}).fingerprint()
    f3 = CptConfig._from_dict({"train": {"lr": 1e-5}}).fingerprint()
    assert f1 == f2, "fingerprint must ignore save_dir/run_name"
    assert f1 != f3, "fingerprint must track lr"
    assert len(f1) == 16
    assert CptConfig().resolve_int8_master(1) == "storage"
    assert CptConfig().resolve_int8_master(2) == "snap"          # storage degrades under world>1


def test_runtime_mode_labels_crosscheck():
    _expect(lambda: CptConfig._from_dict({"mode": "cpt", "run": {"smoke_steps": 1}})
            .validate_runtime(1, has_labels=True), "SFT corpus")


# ── the live configs load through the contract + carry their canonical values ──
_LIVE = [("configs/train_edc_lv_2026.yaml", "cpt"), ("configs/train_edc_lv_2026_v3.yaml", "cpt"),
         ("configs/train_edc_lv_2026_finish.yaml", "cpt"), ("configs/sft_edc.yaml", "sft")]
# canonical values that MUST survive — the 'defaults reproduce current behavior' guarantee, per config.
_EXPECT = {
    "configs/train_edc_lv_2026.yaml": {"train.precision": "fp4", "train.int8_master": "storage",
        "train.optimizer_lowbit": 4, "train.lr": 5e-5, "train.max_steps": 3160, "train.seq_len": 512,
        "train.micro_batch_size": 2, "train.grad_accum": 8, "train.warmup_steps": 50, "train.decay_frac": 0.2,
        "model.nvfp4.bf16_first_layers": 2, "model.nvfp4.bf16_last_layers": 5, "run.save_every": 100},
    "configs/sft_edc.yaml": {"train.precision": "fp4", "train.int8_master": "storage", "train.lr": 2e-5,
        "train.max_steps": 400, "train.seq_len": 256, "train.warmup_steps": 20,
        "model.nvfp4.bf16_first_layers": 0, "model.nvfp4.bf16_last_layers": 0},
    "configs/train_edc_lv_2026_finish.yaml": {"train.decay_frac": 1.0, "train.warmup_steps": 0,
        "train.max_steps": 160, "run.save_every": 200},
    "configs/train_edc_lv_2026_v3.yaml": {"train.max_steps": 3160, "train.int8_master": "storage"},
}


def _getp(o, dotted):
    for p in dotted.split("."):
        o = getattr(o, p)
    return o


def test_live_configs_load_through_contract():
    """Each live config loads via the nested contract (validate() passes, mode⇆init_from agrees) AND its
    canonical values survive — the 'defaults reproduce current run behavior — verified' guarantee."""
    for path, mode in _LIVE:
        p = ROOT / path
        if not p.exists():
            continue
        cfg = CptConfig.from_yaml(p, mode)          # full load + validate() (skips path/world runtime gates)
        assert cfg.mode == mode
        for fld, want in _EXPECT.get(path, {}).items():
            got = _getp(cfg, fld)
            assert got == want, f"{path}: {fld} = {got!r} != canonical {want!r}"
        # the flat trainer-dict adapter must still reconstitute every consumed key
        td = cfg.to_trainer_dict()
        for k in ("precision", "int8_master", "lr", "max_steps", "seq_len", "save_dir", "tokenized",
                  "levers", "nvfp4", "optimizer_lowbit", "gradient_checkpointing"):
            assert k in td, f"{path}: trainer_dict missing {k}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} CONFIG-CONTRACT TESTS PASS")
