
# Autonomous loop started at 2026-05-12 05:18:29.485655
  configs to try: 8
  success criterion: era=2026 ≥ 3 AND voice == 20
  best ckpt kept at: checkpoints/<best_name>; others auto-deleted


## [1/8] iter_lr1e-5_e1
  config: {"lr": 1e-05, "epochs": 1, "rank": 16, "embed_in_lora": true, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 2/13  loss: 7.989  score: 2  (110s)
  *** NEW BEST (score 2) ***

## [2/8] iter_lr2e-5_e1
  config: {"lr": 2e-05, "epochs": 1, "rank": 16, "embed_in_lora": true, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 2/13  loss: 7.550  score: 2  (108s)

## [3/8] iter_lr5e-5_e1
  config: {"lr": 5e-05, "epochs": 1, "rank": 16, "embed_in_lora": true, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 3/13  loss: 6.071  score: 3  (107s)
  *** NEW BEST (score 3) ***

## [4/8] iter_lr1e-5_e2
  config: {"lr": 1e-05, "epochs": 2, "rank": 16, "embed_in_lora": true, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 2/13  loss: 7.493  score: 2  (144s)

## [5/8] iter_lr5e-6_e3
  config: {"lr": 5e-06, "epochs": 3, "rank": 16, "embed_in_lora": true, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 2/13  loss: 7.741  score: 2  (186s)

## [6/8] iter_lr1e-5_e1_r32
  config: {"lr": 1e-05, "epochs": 1, "rank": 32, "embed_in_lora": true, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 2/13  loss: 7.631  score: 2  (112s)

## [7/8] iter_lr1e-5_e1_no_embed
  config: {"lr": 1e-05, "epochs": 1, "rank": 16, "embed_in_lora": false, "embed_save": false}
  voice: 20/20  2026: 0/17  stable: 2/13  loss: 7.985  score: 2  (109s)

## [8/8] iter_lr1e-5_e1_embed_ft
  config: {"lr": 1e-05, "epochs": 1, "rank": 16, "embed_in_lora": false, "embed_save": true, "optim": "paged_adamw_8bit"}
  eval[iter_lr1e-5_e1_embed_ft] FAILED rc=1
  stderr tail:
```
ires_grad=False)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.21 GiB. GPU 0 has a total capacity of 15.47 GiB of which 4.92 GiB is free. Process 4146626 has 596.58 MiB memory in use. Process 40136 has 48.53 MiB memory in use. Process 183000 has 57.03 MiB memory in use. Process 554735 has 254.69 MiB memory in use. Process 2708588 has 291.41 MiB memory in use. Including non-PyTorch memory, this process has 7.34 GiB memory in use. Of the allocated memory 7.07 GiB is allocated by PyTorch, and 9.70 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```
  failed after 86s (consec fails: 1/3)

# Loop done: completed all configs
# Best: iter_lr5e-5_e1 score=3
# Checkpoint at: checkpoints/iter_lr5e-5_e1/final
