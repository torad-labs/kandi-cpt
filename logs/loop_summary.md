# Autonomous loop summary
Finished: 2026-05-12 05:34:30.928596
Exit: completed all configs

## Best config
```json
{
  "name": "iter_lr5e-5_e1",
  "lr": 5e-05,
  "epochs": 1,
  "rank": 16,
  "embed_in_lora": true,
  "embed_save": false,
  "optim": "adamw_8bit",
  "seq_len": 512,
  "elapsed_seconds": 37.8,
  "total_optimizer_steps": 18,
  "tokens_seen": 73728,
  "final_train_loss": 6.071213772422389,
  "voice_pass": 20,
  "voice_total": 20,
  "probes_pass": 3,
  "probes_total": 30,
  "era_2026_pass": 0,
  "era_2026_total": 17,
  "era_stable_pass": 3,
  "era_stable_total": 13,
  "eval_dir": "/home/marcos/Documents/dev/training/kandi-cpt/logs/eval_iter_lr5e-5_e1"
}
```

## All iterations (7)

| name | lr | epochs | rank | embed_in | embed_ft | voice | 2026 | stable | score |
|---|---|---|---|---|---|---|---|---|---|
| iter_lr1e-5_e1 | 1e-05 | 1 | 16 | True | False | 20/20 | 0/17 | 2/13 | 2 |
| iter_lr2e-5_e1 | 2e-05 | 1 | 16 | True | False | 20/20 | 0/17 | 2/13 | 2 |
| iter_lr5e-5_e1 | 5e-05 | 1 | 16 | True | False | 20/20 | 0/17 | 3/13 | 3 |
| iter_lr1e-5_e2 | 1e-05 | 2 | 16 | True | False | 20/20 | 0/17 | 2/13 | 2 |
| iter_lr5e-6_e3 | 5e-06 | 3 | 16 | True | False | 20/20 | 0/17 | 2/13 | 2 |
| iter_lr1e-5_e1_r32 | 1e-05 | 1 | 32 | True | False | 20/20 | 0/17 | 2/13 | 2 |
| iter_lr1e-5_e1_no_embed | 1e-05 | 1 | 16 | False | False | 20/20 | 0/17 | 2/13 | 2 |
