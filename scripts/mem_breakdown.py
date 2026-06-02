#!/usr/bin/env python3
"""Where does the SFT footprint actually go? Counts params by category so we can see what fraction the
int8 master weights are (and therefore the ceiling on what int8-master alone can save)."""
import re, yaml, torch
from transformers import Gemma4ForCausalLM

cfg = yaml.safe_load(open("configs/sft_edc.yaml"))
m = Gemma4ForCausalLM.from_pretrained(cfg["init_from"], dtype=torch.bfloat16)
nl = m.config.num_hidden_layers
n = cfg["nvfp4"]
bf16_layers = set(range(n["bf16_first_layers"])) | set(range(nl - n["bf16_last_layers"], nl))


def lidx(name):
    mm = re.search(r"\.layers\.(\d+)\.", name)
    return int(mm.group(1)) if mm else None


cat = {}
for name, p in m.named_parameters():
    li = lidx(name)
    if "embed" in name or "lm_head" in name:
        k = "frozen_embeds/lm_head"
    elif p.dim() == 2 and li is not None and li not in bf16_layers and p.shape[0] % 16 == 0 and p.shape[1] % 16 == 0:
        k = "int8_master (Quartet linears)"
    elif p.dim() == 2 and li in bf16_layers:
        k = "bf16-kept layers (first2/last5)"
    elif p.dim() == 2:
        k = "other 2-D (bf16)"
    else:
        k = "norms/1-D (bf16)"
    cat[k] = cat.get(k, 0) + p.numel()

GB = 1e9
print(f"\n{'category':34s} {'params':>10s}   {'bf16 GB':>8s} {'int8 GB':>8s}")
print("-" * 66)
tot = 0
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    tot += v
    print(f"{k:34s} {v/GB:9.3f}B   {v*2/GB:7.2f}  {v/GB:7.2f}")
print("-" * 66)
print(f"{'TOTAL':34s} {tot/GB:9.3f}B")

mast = cat.get("int8_master (Quartet linears)", 0)
train = sum(v for k, v in cat.items() if "frozen" not in k)
print(f"\nint8 master weights:      {mast/GB:.2f}B params  → bf16 {mast*2/GB:.2f}GB  vs  int8 {mast/GB:.2f}GB  "
      f"(SAVING {mast/GB:.2f}GB)")
print(f"trainable (non-frozen):   {train/GB:.2f}B params  → 4-bit AdamW m+v ≈ {train*1.0/GB:.2f}GB optimizer state")
print(f"frozen embeds/lm_head:    {cat.get('frozen_embeds/lm_head',0)/GB:.2f}B params  "
      f"(~2.35B CPU-offloaded; ~{(cat.get('frozen_embeds/lm_head',0)/GB-2.35):.2f}B + tied head resident bf16)")
