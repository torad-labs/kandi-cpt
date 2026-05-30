#!/usr/bin/env python3
"""Parameterized CPT trainer driven by env vars — for use by autonomous_loop.py.

Env vars:
  KANDI_CKPT_NAME    — checkpoint dir name under checkpoints/ (required)
  KANDI_LR           — learning rate (default 5e-6)
  KANDI_EPOCHS       — num training epochs (default 1)
  KANDI_RANK         — LoRA rank (default 16)
  KANDI_EMBED_IN_LORA— "1" to include embed_tokens in target_modules (default "1")
  KANDI_EMBED_SAVE   — "1" to add embed_tokens to modules_to_save (default "0")
  KANDI_OPTIM        — optimizer name (default "adamw_8bit")
  KANDI_SEQ_LEN      — sequence length (default 512)

Single 5070 Ti (sm_120) target. QLoRA NF4 base. Writes ckpt to
checkpoints/<name>/final + metrics to checkpoints/<name>/metrics.json.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import json
import time
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
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

CKPT_NAME = os.environ.get("KANDI_CKPT_NAME")
if not CKPT_NAME:
    sys.exit("KANDI_CKPT_NAME env var required")
CKPT_DIR = ROOT / "checkpoints" / CKPT_NAME
LOGS_DIR = ROOT / "logs" / CKPT_NAME

LR             = float(os.environ.get("KANDI_LR", "5e-6"))
NUM_EPOCHS     = int(os.environ.get("KANDI_EPOCHS", "1"))
LORA_R         = int(os.environ.get("KANDI_RANK", "16"))
EMBED_IN_LORA  = os.environ.get("KANDI_EMBED_IN_LORA", "1") == "1"
EMBED_SAVE     = os.environ.get("KANDI_EMBED_SAVE", "0") == "1"
OPTIM          = os.environ.get("KANDI_OPTIM", "adamw_8bit")
SEQ_LEN        = int(os.environ.get("KANDI_SEQ_LEN", "512"))

PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LORA_ALPHA = 2 * LORA_R   # idiomatic 2× rank
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# Regex restricts adapter injection to the text decoder path
_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
if EMBED_IN_LORA:
    _modules.append("embed_tokens")
LORA_TARGET_MODULES = r"^model\.language_model\..*\.(" + "|".join(_modules) + r")$"
LORA_MODULES_TO_SAVE = ["embed_tokens"] if EMBED_SAVE else []

def main():
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"=== {CKPT_NAME} ===")
    print(f"GPU: {device_name} sm_{cap[0]}{cap[1]}")
    print(f"LR={LR}  epochs={NUM_EPOCHS}  rank={LORA_R}  embed_in_lora={EMBED_IN_LORA}  embed_save={EMBED_SAVE}  optim={OPTIM}  seq={SEQ_LEN}")

    ds = load_from_disk(str(TOKENIZED))
    print(f"Dataset: {len(ds)} chunks")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True)
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
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()
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
        lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    n_train = len(ds)
    steps_per_epoch = max(1, n_train // (PER_DEVICE_BATCH * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS
    save_steps = max(10, total_steps // 2)
    print(f"Total optimizer steps: {total_steps}")

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
        optim=OPTIM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=2,
        save_steps=save_steps,
        save_total_limit=1,
        report_to=[],
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(str(CKPT_DIR / "final"))
    tokenizer.save_pretrained(str(CKPT_DIR / "final"))

    tokens_seen = total_steps * PER_DEVICE_BATCH * GRAD_ACCUM * SEQ_LEN
    metrics = {
        "name": CKPT_NAME,
        "lr": LR,
        "epochs": NUM_EPOCHS,
        "rank": LORA_R,
        "embed_in_lora": EMBED_IN_LORA,
        "embed_save": EMBED_SAVE,
        "optim": OPTIM,
        "seq_len": SEQ_LEN,
        "elapsed_seconds": round(elapsed, 1),
        "total_optimizer_steps": total_steps,
        "tokens_seen": tokens_seen,
        "final_train_loss": train_result.metrics.get("train_loss"),
    }
    (CKPT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
