#!/usr/bin/env python3
"""Checkpoint 2 — QLoRA + Liger Kernel + embed_tokens full-FT on Gemma 4 E2B.

Builds on ckpt 1:
  - Liger Kernel fused linear+CE eliminates the logits.float() upcast that
    cost ~1 GB in the CE backward at ckpt 1. With Liger we can re-enable
    embed_tokens full-FT (the proper Unsloth CPT pattern for knowledge install).
  - Corpus expanded from 15K → ~78K tokens (Wikipedia history + 4 YouTube
    vlog transcripts + more Insomniac magazine articles).
  - 3 epochs (more optimizer steps for facts to land).

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_checkpoint_2.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import json
import time

import torch
from datasets import load_from_disk
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma4_text
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
CKPT_DIR = ROOT / "checkpoints" / "ckpt_2"
LOGS_DIR = ROOT / "logs" / "ckpt_2"

LR = 2e-5
NUM_EPOCHS = 3
SEQ_LEN = 512   # halved from 1024 — Liger FLCE patch didn't intercept the loss path, so memory headroom for embed_tokens FT comes from shorter sequences instead
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
# Aspirational: full-FT on embed_tokens. Reality on 16 GB / 5B / vocab=262144:
# even at seq=512 the embed gradient + optimizer state pushes us 22 MB over
# at backward-pass time (HF's logits.float() upcast in ForCausalLMLoss is
# the irreducible cost, and Liger's gemma4_text patch targets
# Gemma4ForCausalLM, not Gemma4ForConditionalGeneration which is what we load).
# For ckpt 2 we drop back to LoRA-only and rely on 5× corpus + 54 optimizer
# steps (vs ckpt 1's 1 step) to show real signal. Re-enable embed_tokens
# once we have either (a) a working Liger integration for the multimodal
# forward path, or (b) Adam8bit-on-CPU offload via bitsandbytes paged optim.
LORA_MODULES_TO_SAVE: list[str] = []

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}  (sm_{cap[0]}{cap[1]})")

    # Apply Liger Kernel BEFORE loading the model — patches Gemma 4's text
    # module so it uses fused linear+CE (eliminates the 1 GB logits tensor at
    # vocab=262144 / seq=1024). RMSNorm and GEGLU patches are skipped because
    # Liger's gemma4_text patch was designed for 31B; E2B's PLE + KV-sharing
    # + double-wide-MLP layer schedule may break those patches. Fused
    # linear+CE is at the LM-head boundary and is architecture-independent.
    apply_liger_kernel_to_gemma4_text(
        rms_norm=False,
        geglu=False,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )
    print("Liger Kernel: fused_linear_cross_entropy applied to gemma4_text")

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
        "tokens_per_hour": round(tokens_seen / elapsed * 3600, 1) if elapsed > 0 else None,
        "final_train_loss": train_result.metrics.get("train_loss"),
        "gpu": device_name,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "liger_kernel": True,
        "modules_to_save": LORA_MODULES_TO_SAVE,
    }
    (CKPT_DIR / "ckpt_2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
