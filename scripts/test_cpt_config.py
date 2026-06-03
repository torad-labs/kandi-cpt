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

import yaml

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


# ── the round-trip proof: live flat configs ⇄ nested contract, fields preserved ──
_MODEL = {"model_path", "freeze_embeddings", "cpu_offload_ple", "nvfp4"}
_DATA = {"tokenized", "init_from"}
_RUN = {"run_name", "save_dir", "log_every", "save_every"}
_LIVE = [("configs/train_edc_lv_2026.yaml", "cpt"), ("configs/train_edc_lv_2026_v3.yaml", "cpt"),
         ("configs/train_edc_lv_2026_finish.yaml", "cpt"), ("configs/sft_edc.yaml", "sft")]
_CHECK = ["train.precision", "train.int8_master", "train.optimizer_lowbit", "train.lr", "train.min_lr",
          "train.warmup_steps", "train.max_steps", "train.decay_frac", "train.micro_batch_size",
          "train.grad_accum", "train.seq_len", "train.weight_decay", "train.betas", "train.adam_eps",
          "train.grad_clip", "train.gradient_checkpointing", "model.freeze_embeddings",
          "model.nvfp4.bf16_first_layers", "model.nvfp4.bf16_last_layers", "run.save_every", "run.log_every"]


def _nest(flat: dict, mode: str) -> dict:
    out: dict = {"mode": mode, "model": {}, "data": {}, "train": {}, "run": {}}
    for k, v in flat.items():
        sec = "model" if k in _MODEL else "data" if k in _DATA else "run" if k in _RUN else "train"
        out[sec][k] = v
    return {s: vv for s, vv in out.items() if s == "mode" or vv}


def _getp(o, dotted):
    for p in dotted.split("."):
        o = getattr(o, p)
    return o


def test_live_configs_round_trip_identical():
    for path, mode in _LIVE:
        p = ROOT / path
        if not p.exists():
            continue
        flat = yaml.safe_load(p.read_text())
        cfg = CptConfig._from_dict(_nest(flat, mode))
        for fld in _CHECK:
            if fld.startswith("model.nvfp4"):
                want = (flat.get("nvfp4") or {}).get(fld.split(".")[-1])
            else:
                want = flat.get(fld.split(".")[-1])
            if want is None:
                continue
            got = _getp(cfg, fld)
            assert got == want, f"{path}: {fld} flat={want!r} != nested={got!r}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} CONFIG-CONTRACT TESTS PASS")
