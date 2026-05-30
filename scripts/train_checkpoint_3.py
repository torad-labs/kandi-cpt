#!/usr/bin/env python3
"""Checkpoint 3 — QLoRA CPT with corrected hyperparameters.

Ckpt 2 showed classic catastrophic forgetting: loss descended cleanly but
the model became MORE confident with WRONG answers (Quantum Valley said
"techno and house" instead of trance). Root cause per Unsloth + QLoRA
literature: LR too high for QLoRA-CPT + too many epochs.

Ckpt 3 fixes:
1. LR 2e-5 → 5e-6 (4× lower; matches Unsloth CPT canonical recipe)
2. Epochs 3 → 1 (catastrophic-forgetting Fix #2 across multiple sources)
3. embed_tokens in LoRA target_modules (per HF PEFT docs — knowledge install
   path that's memory-friendly: ~4M extra params vs full-FT's ~400M)
4. seq_len 512 maintained (apples-to-apples comparison vs ckpt 2)

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_checkpoint_3.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import json
import time

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
MODEL_PATH = Path("/home/marcos/Documents/dev/models/gemma-4-E2B-it")
TOKENIZED = ROOT / "tokenized" / "corpus_v1.arrow"
CKPT_DIR = ROOT / "checkpoints" / "ckpt_3"
LOGS_DIR = ROOT / "logs" / "ckpt_3"

# Corrected for QLoRA-CPT — see ckpt 2 forgetting analysis.
LR = 5e-6
NUM_EPOCHS = 1
SEQ_LEN = 512
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

LORA_R = 16
LORA_ALPHA = 32
# embed_tokens added as a LoRA target — installs vocabulary updates with
# memory cost of ~4M params instead of ~400M for full-FT. HF PEFT docs
# explicitly support `target_modules=[..., "embed_tokens"]` as the memory-
# friendly alternative to modules_to_save when training new domain vocab.
LORA_TARGET_MODULES = r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|embed_tokens)$"
LORA_MODULES_TO_SAVE: list[str] = []

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}  (sm_{cap[0]}{cap[1]})")

    ds = load_from_disk(str(TOKENIZED))
    print(f"Dataset: {len(ds)} chunks of {SEQ_LEN} tokens")

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
    save_steps = max(10, total_steps // 4)
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
        report_to=[],
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
        "num_epochs": NUM_EPOCHS,
        "total_optimizer_steps": total_steps,
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / elapsed, 1) if elapsed > 0 else None,
        "learning_rate": LR,
        "final_train_loss": train_result.metrics.get("train_loss"),
        "gpu": device_name,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "lora_targets_include_embed": True,
        "modules_to_save": LORA_MODULES_TO_SAVE,
    }
    (CKPT_DIR / "ckpt_3_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
