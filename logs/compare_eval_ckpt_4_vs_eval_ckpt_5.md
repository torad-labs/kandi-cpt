# Eval comparison: eval_ckpt_4 → eval_ckpt_5

## Headline

| Metric            |    eval_ckpt_4     |    eval_ckpt_5     | Delta |
|-------------------|--------------------|--------------------|-------|
| Voice survival    |    20/20 (100%)    |    20/20 (100%)    | +0    |
| Knowledge probes  |     3/30 (10%)     |     1/30 (3%)      | -2    |

### By era

| Era       |    eval_ckpt_4     |    eval_ckpt_5     | Delta |
|-----------|--------------------|--------------------|-------|
| 2026      |     0/17 (0%)      |     0/17 (0%)      | +0    |
| stable    |     3/13 (23%)     |     1/13 (8%)      | -2    |

## Voice — per-feature counts

| Feature              | eval_ckpt_4 | eval_ckpt_5 | Delta |
|----------------------|------------|------------|-------|
| brief                |     20     |     20     | +0    |
| no_emoji             |     20     |     20     | +0    |
| on_topic_or_redirect |     13     |     18     | +5    |

## Probe flips

- Gained (newly passing): **0**
- Lost (newly failing):   **2**
- Stable pass:            1
- Stable fail:            27

### Newly failing (CPT broke or perturbed)
- [direct] What kind of music plays at Quantum Valley?
- [direct] Is EDC cash or card?

