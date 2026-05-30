#!/usr/bin/env python3
"""Checkpoint 1 — bf16 full-FT CPT pass on Gemma 4 E2B.

Pipeline integration test. ~25K tokens of corpus, 1 epoch, low LR.
Validates the trainer setup, loss stability, throughput, and that the
resulting checkpoint still produces coherent text at temp=1.2.

NOT a knowledge install — corpus is too small for probes to move.
Gate to scale to checkpoint 2: pipeline ran clean + loss descended +
voice survived + should-NOT-know probes still pass.

Run:
    python scripts/train_checkpoint_1.py

Output:
    checkpoints/ckpt_1/ — HF format checkpoint, reloadable for inference
"""
import os
# Default to GPU 1 (RTX 5070 Ti) because GPU 0 (RTX 5080) hosts QGRE-runtime
# training (`qwen3_tq4_synthetic`). Both are Blackwell sm_120 / 16 GB; the
# 5070 Ti has marginally less compute but is equivalent for QLoRA CPT.
# Override externally with CUDA_VISIBLE_DEVICES=N if the 5080 frees up.
# Also single-GPU pinning prevents HF Trainer from auto-enabling DataParallel.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from pathlib import Path
import json
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

ROOT = Path(__file__).resolve().parents[1]
from config import MODEL_PATH, TOKENIZED_ARROW as TOKENIZED
CKPT_DIR = ROOT / "checkpoints" / "ckpt_1"
LOGS_DIR = ROOT / "logs" / "ckpt_1"

# CPT config — calibrated for 5B-param E2B on 16 GB RTX 5080.
# Gemma 4 E2B name is "effective 2B" but actual param count is ~5.1B
# (10 GB bf16 safetensors confirms it). Full-FT OOMs on 16 GB; switching to
# LoRA on attention + MLP, plus full-FT on embed_tokens (tied with lm_head).
LR = 2e-5
NUM_EPOCHS = 1
SEQ_LEN = 1024   # batch=1/seq=1024/vocab=262144/bf16 logits = 0.5 GB; CE backward ~1 GB
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8   # effective batch 8
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

LORA_R = 16
LORA_ALPHA = 32
# Regex restricts adapter injection to the text decoder path. The vision and
# audio sub-encoders also have q_proj/k_proj/v_proj/o_proj, but they use the
# custom Gemma4ClippableLinear class which PEFT can't wrap — and we don't want
# to train them anyway for text-only CPT.
LORA_TARGET_MODULES = r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
# CKPT 1 NOTE: embed_tokens training is dropped here to fit in 14 GB on the
# 5070 Ti with vocab=262144. For ckpt 2+ when we have more corpus and want
# real knowledge install, re-enable this and switch to Liger Kernel fused
# linear+CE (which eliminates the logits.float() upcast that's eating ~1 GB
# in the CE backward).
LORA_MODULES_TO_SAVE: list[str] = []

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — checkpoint 1 expects RTX 5080")
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}  (sm_{cap[0]}{cap[1]})")

    ds = load_from_disk(str(TOKENIZED))
    print(f"Dataset: {len(ds)} chunks of {SEQ_LEN} tokens")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True)

    # Gemma 4 "E2B" is effective-2B but actually 5.1B params total. Full-FT
    # doesn't fit on 16 GB — even LoRA full-bf16 OOMs at the vocab=262144 CE
    # loss step. QLoRA (NF4 4-bit base) drops the frozen base from ~10 GB to
    # ~2.5 GB while LoRA adapters and modules_to_save embed_tokens train in
    # bf16. This is the Unsloth CPT recipe for consumer GPUs.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # sm_120 supports SDPA; fallback if FA2 absent
    )
    # Manual k-bit prep: skip PEFT's prepare_model_for_kbit_training because
    # its blanket fp32 upcast of non-quantized layers (RMSNorm, LayerNorm)
    # costs ~8 GB on a 5B model — bigger than the 4-bit base itself. The two
    # things prepare_model_for_kbit_training does that we actually need:
    model.enable_input_require_grads()   # for grad-ckpt through frozen base
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_kwargs = dict(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=LORA_MODULES_TO_SAVE,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        lora_config = LoraConfig(**lora_kwargs, ensure_weight_tying=True)
    except TypeError:
        # PEFT version doesn't support ensure_weight_tying — fall back.
        lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    n_train = len(ds)
    steps_per_epoch = max(1, n_train // (PER_DEVICE_BATCH * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS
    save_steps = max(50, total_steps // 4)
    print(f"Estimated total optimizer steps: {total_steps}; save every {save_steps}")

    args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        logging_dir=str(LOGS_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=3,
        report_to=[],   # disable wandb/tensorboard auto-init for ckpt 1
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(str(CKPT_DIR / "final"))
    tokenizer.save_pretrained(str(CKPT_DIR / "final"))

    tokens_seen = total_steps * PER_DEVICE_BATCH * GRAD_ACCUM * SEQ_LEN
    metrics = {
        "elapsed_seconds": round(elapsed, 1),
        "total_optimizer_steps": total_steps,
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / elapsed, 1) if elapsed > 0 else None,
        "tokens_per_hour": round(tokens_seen / elapsed * 3600, 1) if elapsed > 0 else None,
        "final_train_loss": train_result.metrics.get("train_loss"),
        "gpu": device_name,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
    }
    (CKPT_DIR / "ckpt_1_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
