"""The ONE config contract for kandi-cpt — CPT + SFT, every knob, one typed object.

THE RULE (also in AGENTS.md and at the top of train_cpt.py / eval.py):
  All run configuration lives in this object and one YAML file. There is exactly ONE training
  entrypoint (train_cpt.py) and ONE eval entrypoint (eval.py), each taking only --config (+ --mode).
  Do NOT add a behavior-affecting flag. Do NOT create a second entrypoint or a parallel trainer.
  New knobs go in a sub-dataclass here + the emitted schema. Sweeps generate config files and call
  the entrypoint N times — they never pass behavior flags.

Pattern (matches qgre/config.py): stdlib @dataclass, nested sub-configs via field(default_factory=...),
Literal enums, a from_yaml(path, mode) staticmethod → yaml.safe_load → explicit _from_dict hydrator
→ validate(). Unknown keys HARD-ERROR (project decision 2026-06-03 — a typo'd key must never silently
fall back to a default). World/path-dependent gates run in validate_runtime() AFTER env-read, mirroring
the old run_contract() timing exactly so fail-loud behavior is preserved, not weakened.

Defaults reproduce CURRENT canonical-CPT behavior (verified against configs/train_edc_lv_2026.yaml +
train_cpt.py). SFT configs override the mode-divergent fields (lr, seq_len, max_steps, …) explicitly.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

# Single-sourced canonical paths (the machine-specific seam stays in scripts/config.py).
try:
    from config import MODEL_PATH as _MODEL_PATH, ROOT as _ROOT
    _MODEL_PATH = str(_MODEL_PATH)
    _ROOT = Path(_ROOT)
except Exception:                                   # pragma: no cover - import-context fallback
    _ROOT = Path(__file__).resolve().parent.parent
    _MODEL_PATH = "/home/marcos/Documents/dev/infra/models/gemma-4-E2B-it"

Precision = Literal["bf16", "fp4", "nvfp4"]
Int8Master = Literal["storage", "snap", "off"]
Mode = Literal["cpt", "sft"]


class ConfigError(ValueError):
    """Raised for any invalid config — typo'd key, out-of-range value, or bad cross-field combo."""


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Nvfp4Config:
    """Selective-precision knobs for the legacy NVFP4-QAT path (INERT unless train.precision==nvfp4)."""
    skip_lm_head: bool = True
    skip_embeddings: bool = True
    bf16_first_layers: int = 2          # CPT canonical; SFT sets 0
    bf16_last_layers: int = 5           # CPT canonical; SFT sets 0  (~15% of 35 text layers kept bf16)

    def __post_init__(self) -> None:
        if self.bf16_first_layers < 0 or self.bf16_last_layers < 0:
            raise ConfigError("model.nvfp4.bf16_first_layers/bf16_last_layers must be >= 0")
        if self.bf16_first_layers + self.bf16_last_layers > 35:
            raise ConfigError(
                f"model.nvfp4: bf16_first_layers({self.bf16_first_layers})+bf16_last_layers"
                f"({self.bf16_last_layers}) > 35 (Gemma 4 E2B has 35 text decoder layers)")


@dataclass
class ModelConfig:
    model_path: str = _MODEL_PATH
    freeze_embeddings: bool = True      # freezes ALL 'embed'-named params (embed_tokens + 2.35B PLE + tied lm_head)
    cpu_offload_ple: bool = True        # stream the 4.7GB frozen PLE from CPU (read at train_cpt.py; absent from YAMLs → code default)
    nvfp4: Nvfp4Config = field(default_factory=Nvfp4Config)

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ConfigError("model.model_path must not be empty")


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    tokenized: str = ""                 # REQUIRED for non-smoke runs (.arrow); validated in validate_runtime
    init_from: str | None = None        # unset → fresh CPT from base safetensors; set → resume/SFT via causal_bf16_state_dict (THE mode source signal)
    dataset_hash: str | None = None     # freeze-time fingerprint input (from <name>_stats.json sidecar); never authored, never gates a run


# ─────────────────────────────────────────────────────────────────────────────
# train (+ levers)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Lever1EntityLoss:
    enabled: bool = False
    entity_loss_weight: float = 2.5     # 1.0 == baseline mean CE

    def __post_init__(self) -> None:
        if self.entity_loss_weight <= 0:
            raise ConfigError("levers.lever1_entity_loss.entity_loss_weight must be > 0")


@dataclass
class Lever2QaMix:
    enabled: bool = False
    qa_token_fraction: float = 0.15
    use_chat_template: bool = False
    system_prefix: str | None = None


@dataclass
class Lever3Ewc:
    enabled: bool = False
    lambda_: float = 1.0                # YAML key is 'lambda' (reserved word) → aliased in _from_dict


@dataclass
class Lever3ThreeR:
    enabled: bool = False
    replay_token_fraction: float = 0.02
    rewarm_peak_lr: float = 5.0e-5
    ewc: Lever3Ewc = field(default_factory=Lever3Ewc)


@dataclass
class Lever4ExposureTracking:
    enabled: bool = False
    threshold: float = 0.8              # DEAD: zero reads in train_cpt.py (only `enabled` is wired); kept for back-compat


@dataclass
class Lever5CausalTrace:
    enabled: bool = False
    n_facts: int = 200
    noise_scale: float = 0.1
    checkpoint: str | None = None


@dataclass
class LeversConfig:
    """CPT training levers. Only lever1 (entity-weighted CE) + lever4.enabled (exposure log) are WIRED in
    train_cpt.py; lever2/lever3/lever5 are declared-but-inert (documented future). Kept here so the 4 YAMLs
    that carry the full subtree load under the hard-error-on-unknown-key policy."""
    lever1_entity_loss: Lever1EntityLoss = field(default_factory=Lever1EntityLoss)
    lever2_qa_mix: Lever2QaMix = field(default_factory=Lever2QaMix)
    lever3_three_r: Lever3ThreeR = field(default_factory=Lever3ThreeR)
    lever4_exposure_tracking: Lever4ExposureTracking = field(default_factory=Lever4ExposureTracking)
    lever5_causal_trace: Lever5CausalTrace = field(default_factory=Lever5CausalTrace)


@dataclass
class AnnealConfig:
    """Final-phase anneal on top-quality surfaces + replay (docs/cpt-overhaul/04).

    Default ON and derived: final `frac` of max_steps (default 0.10) switches the loader to the
    anneal mix (hydrated + replay by block_tag; educational-score ranking plugs in when workstream
    01 lands) and decays LR to ~0. Disabled automatically on tiny runs (max_steps < min_steps)
    where 10% is noise — never asked. Precision floor untouched — schedule is not dtype.
    """
    enabled: bool = True
    frac: float = 0.10                  # final fraction of max_steps
    min_steps: int = 500                # disable when max_steps < this (derived, not asked)
    mix_tags: list[str] = field(default_factory=lambda: ["hydrated", "replay"])
    lr_end: float = 0.0                 # LR target at anneal end (linear decay from base lr)
    require_exposure: bool = True       # print pre-anneal exposure check; warn if lever4 off
    keep_pre_anneal_checkpoint: bool = True  # save pre-anneal/ as fallback; anneal-end is eval candidate

    def __post_init__(self) -> None:
        if not 0 < self.frac <= 0.5:
            raise ConfigError(f"train.anneal.frac({self.frac}) must be in (0, 0.5]")
        if self.min_steps < 1:
            raise ConfigError("train.anneal.min_steps must be >= 1")
        if self.lr_end < 0:
            raise ConfigError("train.anneal.lr_end must be >= 0")
        if not self.mix_tags:
            raise ConfigError("train.anneal.mix_tags must be non-empty")


@dataclass
class TrainConfig:
    precision: Precision = "fp4"        # bf16 | fp4 (Quartet-II native NVFP4) | nvfp4 (legacy QAT)
    int8_master: Int8Master = "storage"  # THE MOAT: storage=true int8-stored master+SR (needs fp4+world1) | snap=legacy bf16-grid SR | off
    optimizer_lowbit: int = 4           # torchao AdamW{4,8}bit states (single-GPU); 0 = bf16 AdamW. CANONICAL 4 (NOT argparse's old 0).
    optimizer_8bit: bool = False        # DEAD: never read by train_cpt.py. Vestigial; emits DeprecationWarning. Real switch is optimizer_lowbit.
    int8_grad_accum: bool = False       # SFT-only: fold bf16 grads into a persistent int8 SR accumulator (world==1)
    lr: float = 5.0e-5                  # CPT canonical (SFT 2.0e-5, finish 1.64e-5)
    min_lr: float = 5.0e-6
    warmup_steps: int = 50              # CPT canonical (SFT 20, finish 0)
    max_steps: int = 3160               # CPT canonical (SFT 400, finish 160)
    decay_frac: float = 0.2             # WSD: last frac decays to min_lr (finish 1.0)
    micro_batch_size: int = 2
    grad_accum: int = 8                 # mb2×accum8×world1 = eff batch 16
    seq_len: int = 512                  # CPT corpus blocks (SFT chat 256)
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    adam_eps: float = 1.0e-8
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True  # forces use_cache=False
    levers: LeversConfig = field(default_factory=LeversConfig)
    anneal: AnnealConfig = field(default_factory=AnnealConfig)

    def __post_init__(self) -> None:
        if self.precision not in ("bf16", "fp4", "nvfp4"):
            raise ConfigError(f"train.precision='{self.precision}' invalid — must be bf16|fp4|nvfp4")
        if self.int8_master not in ("storage", "snap", "off"):
            raise ConfigError(f"train.int8_master='{self.int8_master}' invalid — must be storage|snap|off")
        if self.optimizer_lowbit not in (0, 4, 8):
            raise ConfigError(f"train.optimizer_lowbit={self.optimizer_lowbit} invalid — must be 0|4|8")
        if self.lr <= 0:
            raise ConfigError("train.lr must be > 0")
        if self.min_lr < 0 or self.min_lr > self.lr:
            raise ConfigError(f"train.min_lr({self.min_lr}) must satisfy 0 <= min_lr <= lr({self.lr})")
        if self.warmup_steps < 0:
            raise ConfigError("train.warmup_steps must be >= 0")
        if self.max_steps <= 0:
            raise ConfigError("train.max_steps must be > 0")
        if not 0 < self.decay_frac <= 1:
            raise ConfigError(f"train.decay_frac({self.decay_frac}) must be in (0, 1]")
        if self.micro_batch_size < 1 or self.grad_accum < 1:
            raise ConfigError("train.micro_batch_size and train.grad_accum must be >= 1")
        if self.seq_len <= 0:
            raise ConfigError("train.seq_len must be > 0")
        if len(self.betas) != 2 or not all(0 < b < 1 for b in self.betas):
            raise ConfigError(f"train.betas must be two values in (0,1), got {self.betas}")
        if self.grad_clip < 0:
            raise ConfigError("train.grad_clip must be >= 0")
        if self.optimizer_8bit:
            warnings.warn("train.optimizer_8bit is DEAD config — never read by the trainer; "
                          "the real switch is train.optimizer_lowbit. Remove it.", DeprecationWarning, stacklevel=3)


# ─────────────────────────────────────────────────────────────────────────────
# corpus_synthesis (the data-generation recipe; single-sources what run_v9.sh/rehydrate baked as flags)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorpusSynthesisConfig:
    gen_model: str = "google/gemma-4-31b-it"   # OpenRouter data-gen LLM — DISTINCT from model.model_path on purpose
    hydrate_base_n: int = 8
    hydrate_concurrency: int = 10
    hydrate_parallelism: int = 8
    replay_frac: float = 0.10
    answer_aug: int = 8                 # LIVE V9 recipe (build_sft.py argparse default 2 is the naive floor)
    abstention_frac: float = 0.0        # V9 dropped global abstention
    absent_neg: int = 125               # fabrication lever
    fake_city_neg: int = 0
    fine_grained_neg: int = 0
    max_per_relation: int = 6
    sft_seq_len: int = 256              # build_sft producer side — MUST equal train.seq_len when mode==sft
    score_samples: int = 4
    score_batch: int = 16
    knowledge_filter: bool = True
    self_gen: bool = True
    drop_generic: bool = True
    score_model: str | None = "checkpoints/cpt/edc_all97_v1"   # knownness scorer; required if knowledge_filter or self_gen

    def __post_init__(self) -> None:
        if not 0 <= self.replay_frac <= 1:
            raise ConfigError("corpus_synthesis.replay_frac must be in [0,1]")
        if not 0 <= self.abstention_frac <= 1:
            raise ConfigError("corpus_synthesis.abstention_frac must be in [0,1]")
        if not 1 <= self.answer_aug <= 8:
            raise ConfigError("corpus_synthesis.answer_aug must be in [1,8]")
        for k in ("absent_neg", "fake_city_neg", "fine_grained_neg", "max_per_relation"):
            if getattr(self, k) < 0:
                raise ConfigError(f"corpus_synthesis.{k} must be >= 0")
        if self.sft_seq_len <= 0:
            raise ConfigError("corpus_synthesis.sft_seq_len must be > 0")
        if min(self.hydrate_base_n, self.hydrate_concurrency, self.hydrate_parallelism) <= 0:
            raise ConfigError("corpus_synthesis.hydrate_* must be > 0")


# ─────────────────────────────────────────────────────────────────────────────
# eval (the gate; generation temps stay hardcoded per-script — determinism contrast is load-bearing)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    tau_fab: float = 0.25               # THE GATE: PASS iff fab<=tau_fab AND acc>=tau_acc AND cov>=tau_cov
    tau_acc: float = 0.55
    tau_cov: float = 0.40
    n_absent: int = 90
    seed: int = 42
    scale: float = 1.0
    batch: int = 16
    device: str = "cuda:0"              # single-GPU PSU rule
    probe_domain: str | None = None      # generic CPT base-vs-tuned path; disabled when unset
    probe_editions: str | None = None    # exact comma-separated unit scope; required with probe_domain
    probe_min_n: int = 50                # no PASS/FAIL number below the handbook's minimum denominator
    probe_min_delta_acc: float = 0.10
    probe_report_root: str = "logs/probe_evals"
    # Generative axis (docs/cpt-overhaul/04): open-answer judged by different-family κ-calibrated model.
    # Bar starts lenient (harder measurement); recorded per report, never mid-iteration.
    generative_bar: float = 0.40
    generative_min_delta_acc: float = 0.05
    generative_enabled: bool = True      # dual-axis install-strength when probe_domain is set

    def __post_init__(self) -> None:
        for k in ("tau_fab", "tau_acc", "tau_cov", "probe_min_delta_acc",
                  "generative_bar", "generative_min_delta_acc"):
            v = getattr(self, k)
            if not 0 <= v <= 1:
                raise ConfigError(f"eval.{k}={v} must be in [0,1]")
        if self.n_absent < 0 or self.seed < 0 or self.scale <= 0 or self.batch < 1:
            raise ConfigError("eval.n_absent>=0, seed>=0, scale>0, batch>=1 required")
        if self.probe_min_n < 1:
            raise ConfigError("eval.probe_min_n must be >= 1")
        if bool(self.probe_domain) != bool(self.probe_editions):
            raise ConfigError("eval.probe_domain and eval.probe_editions must be set together")
        if not self.probe_report_root or Path(self.probe_report_root).is_absolute() or ".." in Path(self.probe_report_root).parts:
            raise ConfigError("eval.probe_report_root must be a non-empty path inside the engine repo")


# THE DEPLOY GATE — single-sourced from EvalConfig defaults so eval_sft_calibration.py + serve.py can't
# drift (was triple-located: eval_sft_calibration TAU_* + serve.py GATE + serve.py GATE_T). Canonical metric
# names; each consumer maps them to its own local labels. PASS iff fab<=fabrication ∧ acc>=accuracy ∧ cov>=coverage.
GATE_THRESHOLDS: dict[str, float] = {
    "fabrication": EvalConfig().tau_fab,
    "accuracy": EvalConfig().tau_acc,
    "coverage": EvalConfig().tau_cov,
}


# ─────────────────────────────────────────────────────────────────────────────
# run (output + provenance; toolchain env + CUDA_VISIBLE_DEVICES are intentionally NOT config)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    run_name: str = "edc_cpt"
    save_dir: str = ""                  # REQUIRED non-smoke (no save_dir ⇒ trains & saves NOTHING)
    log_every: int = 5
    save_every: int = 100
    seed: int = 42
    smoke_steps: int = 0                # >0 → synthetic-data sanity loop (folds the old --smoke flag)
    dry_run: bool = False               # True → validate + print contract + exit (folds the old --dry-run flag)
    early_stop_patience: int = 0        # >0 → stop if smoothed loss hasn't improved for N log-windows. 0 = OFF (default reproduces current behavior). Saves scarce single-GPU compute past the optimum.

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ConfigError("run.run_name must not be empty")
        if self.log_every < 1 or self.save_every < 1:
            raise ConfigError("run.log_every and run.save_every must be >= 1")
        if self.seed < 0 or self.smoke_steps < 0 or self.early_stop_patience < 0:
            raise ConfigError("run.seed, run.smoke_steps, run.early_stop_patience must be >= 0")


# ─────────────────────────────────────────────────────────────────────────────
# top-level
# ─────────────────────────────────────────────────────────────────────────────
# Fields the run-identity fingerprint is computed over (stable subset; excludes volatile save_dir/run_name/etc).
_FINGERPRINT_FIELDS = (
    "train.precision", "train.int8_master", "train.optimizer_lowbit", "train.lr", "train.min_lr",
    "train.warmup_steps", "train.max_steps", "train.decay_frac", "train.micro_batch_size",
    "train.grad_accum", "train.seq_len", "train.weight_decay", "train.betas",
    "train.anneal.enabled", "train.anneal.frac", "train.anneal.min_steps", "train.anneal.lr_end",
    "model.model_path", "data.tokenized", "data.init_from", "mode",
)


@dataclass
class CptConfig:
    """Top-level config for kandi-cpt training + eval. mode ∈ {cpt,sft} is a DECLARED-INTENT cross-check —
    the runtime truth stays the dual implicit detection (data.init_from presence + the corpus `labels`
    column). validate() errors if mode disagrees with init_from; it never replaces the detection."""
    mode: Mode = "cpt"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    corpus_synthesis: CorpusSynthesisConfig = field(default_factory=CorpusSynthesisConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ── loading ──────────────────────────────────────────────────────────────
    @staticmethod
    def from_yaml(path: str | Path, mode: Mode | None = None) -> "CptConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
        cfg = CptConfig._from_dict(raw, mode)
        cfg.validate()
        return cfg

    @staticmethod
    def _from_dict(d: dict, mode: Mode | None = None) -> "CptConfig":
        d = dict(d)
        m = mode or d.pop("mode", "cpt")
        if m not in ("cpt", "sft"):
            raise ConfigError(f"mode='{m}' invalid — must be cpt|sft")
        d.pop("mode", None)
        sections = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig,
                    "corpus_synthesis": CorpusSynthesisConfig, "eval": EvalConfig, "run": RunConfig}
        unknown_sections = set(d) - set(sections)
        if unknown_sections:
            raise ConfigError(
                f"unknown top-level config section(s): {sorted(unknown_sections)} — "
                f"valid sections are {sorted(sections)}. (Is this a flat pre-2026-06 config? "
                f"Nest knobs under model/data/train/corpus_synthesis/eval/run.)")
        cfg = CptConfig(mode=cast(Mode, m))
        for name, cls in sections.items():
            if name in d and d[name] is not None:
                setattr(cfg, name, _build(cls, d[name], name))
        return cfg

    # ── validation (world/path-independent — safe at load time) ───────────────
    def validate(self) -> None:
        # mode ⇆ init_from cross-check (declared intent must agree with the runtime source signal).
        if self.mode == "sft" and not self.data.init_from:
            raise ConfigError(
                "mode:sft requires data.init_from — SFT resumes a CPT checkpoint via causal_bf16_state_dict. "
                "Set data.init_from: checkpoints/cpt/edc_all97_v2")
        # int8_master=storage needs the Quartet FP4 linears (world gate is deferred to validate_runtime).
        if self.train.int8_master == "storage" and self.train.precision != "fp4":
            raise ConfigError(
                "train.int8_master=storage needs train.precision=fp4 (it stores int8 + SR through Quartet "
                "linears). Set precision:fp4, or int8_master:snap/off.")
        # corpus knownness scorer prerequisite (mirrors build_sft.py hard-exit).
        cs = self.corpus_synthesis
        if (cs.knowledge_filter or cs.self_gen) and not cs.score_model:
            raise ConfigError(
                "corpus_synthesis.score_model is required when knowledge_filter or self_gen is true "
                "(the knownness scorer checkpoint).")
        # SFT producer/consumer seq_len must match or blocks are mis-sized.
        if self.mode == "sft" and cs.sft_seq_len != self.train.seq_len:
            raise ConfigError(
                f"corpus_synthesis.sft_seq_len({cs.sft_seq_len}) must equal train.seq_len({self.train.seq_len}) "
                f"for SFT, or the SFT blocks are mis-tokenized.")

    def validate_runtime(self, world: int = 1, *, has_labels: bool | None = None) -> None:
        """Path/world/corpus-dependent gates — call AFTER env-read + corpus open, mirroring run_contract timing."""
        smoke = self.run.smoke_steps > 0
        if not smoke:
            if not self.run.save_dir:
                raise ConfigError(
                    "run.save_dir UNSET — the run would train and save NOTHING (and the status heartbeat "
                    "never fires). Set run.save_dir in the config.")
            if not self.data.tokenized:
                raise ConfigError("data.tokenized UNSET — no training corpus configured.")
            if not (_ROOT / self.data.tokenized).exists():
                raise ConfigError(f"data.tokenized not found: {_ROOT / self.data.tokenized}")
        if self.data.init_from and not (_ROOT / self.data.init_from).exists():
            raise ConfigError(f"data.init_from checkpoint not found: {_ROOT / self.data.init_from}")
        # int8_master=storage is single-GPU only (FSDP/DTensor-incompatible). Graceful runtime fallback → WARN.
        if self.train.int8_master == "storage" and world > 1:
            warnings.warn("train.int8_master=storage is single-GPU only — falls back to snap under world>1.",
                          RuntimeWarning, stacklevel=2)
        if self.train.optimizer_lowbit in (4, 8) and world > 1:
            warnings.warn("train.optimizer_lowbit is DTensor-incompatible under world>1 — falls back to bf16 AdamW.",
                          RuntimeWarning, stacklevel=2)
        # mode ⇆ labels-column cross-check (the second runtime detection).
        if has_labels is not None:
            if self.mode == "sft" and not has_labels:
                raise ConfigError("mode:sft but the corpus has no `labels` column — this is a CPT corpus.")
            if self.mode == "cpt" and has_labels:
                raise ConfigError("mode:cpt but the corpus HAS a `labels` column — this is an SFT corpus "
                                  "(set mode:sft + data.init_from).")

    # ── helpers ───────────────────────────────────────────────────────────────
    def resolve_int8_master(self, world: int) -> Int8Master:
        """The single int8_master resolution (was duplicated in run_contract + main). storage degrades to
        snap under world>1, and is only honored with precision=fp4 (validate() already gated precision)."""
        mode = self.train.int8_master
        if mode == "storage" and world > 1:
            return "snap"
        return mode

    def fingerprint(self) -> str:
        flat = _flatten(asdict(self))
        sub = {k: flat.get(k) for k in _FINGERPRINT_FIELDS}
        return hashlib.sha256(json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def to_resolved_dict(self) -> dict:
        d = asdict(self)
        d["_fingerprint"] = self.fingerprint()
        return d

    def to_trainer_dict(self) -> dict:
        """Flatten to the flat-key dict the train_cpt.py INTERNALS already consume (model/data/train/run
        fields hoisted to the top level; `nvfp4` + `levers` + `anneal` stay nested dicts exactly as the old
        flat YAML produced them). This is an internal adapter so the ~900 lines of trainer plumbing stay
        untouched and proven-working — it is NOT a config surface. corpus_synthesis/eval are not read by
        the trainer."""
        d: dict = {}
        for section in (self.model, self.data, self.train, self.run):
            for f in fields(section):
                d[f.name] = getattr(section, f.name)
        d["nvfp4"] = asdict(self.model.nvfp4)
        d["levers"] = _unalias(asdict(self.train.levers))   # lambda_ → lambda (match original YAML shape)
        d["anneal"] = asdict(self.train.anneal)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# hydrator internals
# ─────────────────────────────────────────────────────────────────────────────
_ALIASES: dict[str, str] = {"lambda": "lambda_"}   # YAML 'lambda' (reserved word) → dataclass field lambda_


def _build(cls, raw, path: str):
    """Recursively build a dataclass from a raw dict, HARD-ERRORING on any unknown key (with a 'did you
    mean' hint), and recursing into nested dataclass fields."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping for {cls.__name__}, got {type(raw).__name__}")
    aliased: dict[str, object] = {_ALIASES.get(str(k), str(k)): v for k, v in raw.items()}
    known = {f.name: f for f in fields(cls)}
    for k in aliased:
        if k not in known:
            hint = _did_you_mean(k, known)
            raise ConfigError(f"unknown key '{path}.{k}'{hint} — not a field of {cls.__name__}")
    kwargs = {}
    for k, v in aliased.items():
        f = known[k]
        nested = _dataclass_type(f.type)
        kwargs[k] = _build(nested, v, f"{path}.{k}") if (nested and isinstance(v, dict)) else v
    return cls(**kwargs)


def _dataclass_type(tp):
    """Return the dataclass class if a field's type is (a possibly-string-annotated) dataclass, else None."""
    if is_dataclass(tp):
        return tp
    if isinstance(tp, str):                          # from __future__ annotations → types are strings
        return _DATACLASS_BY_NAME.get(tp)
    return None


def _did_you_mean(key: str, known: dict) -> str:
    import difflib
    m = difflib.get_close_matches(key, list(known), n=1, cutoff=0.7)
    return f" — did you mean '{m[0]}'?" if m else ""


_REV_ALIASES = {v: k for k, v in _ALIASES.items()}


def _unalias(d):
    """Reverse the field-name aliases (lambda_ → lambda) recursively, so the flattened trainer dict matches
    the original YAML key spelling exactly."""
    if isinstance(d, dict):
        return {_REV_ALIASES.get(k, k): _unalias(v) for k, v in d.items()}
    return d


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, p + "."))
        else:
            out[p] = v
    return out


_DATACLASS_BY_NAME = {
    c.__name__: c for c in (
        Nvfp4Config, ModelConfig, DataConfig, Lever1EntityLoss, Lever2QaMix, Lever3Ewc, Lever3ThreeR,
        Lever4ExposureTracking, Lever5CausalTrace, LeversConfig, AnnealConfig, TrainConfig,
        CorpusSynthesisConfig, EvalConfig, RunConfig,
    )
}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate/inspect a kandi-cpt config (the contract).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cpt", "sft"], default=None)
    ap.add_argument("--get", help="print one resolved dotted field (e.g. run.save_dir) and exit")
    args = ap.parse_args()
    cfg = CptConfig.from_yaml(args.config, args.mode)
    if args.get:
        print(_flatten(asdict(cfg)).get(args.get, ""))
    else:
        print(f"OK  {args.config}  mode={cfg.mode}  fingerprint={cfg.fingerprint()}")
