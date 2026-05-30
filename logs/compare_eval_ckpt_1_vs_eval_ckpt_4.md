# Eval comparison: eval_ckpt_1 → eval_ckpt_4

## Headline

| Metric            |    eval_ckpt_1     |    eval_ckpt_4     | Delta |
|-------------------|--------------------|--------------------|-------|
| Voice survival    |    20/20 (100%)    |    20/20 (100%)    | +0    |
| Knowledge probes  |     3/30 (10%)     |     3/30 (10%)     | +0    |

### By era

| Era       |    eval_ckpt_1     |    eval_ckpt_4     | Delta |
|-----------|--------------------|--------------------|-------|
| 2026      |     0/17 (0%)      |     0/17 (0%)      | +0    |
| stable    |     3/13 (23%)     |     3/13 (23%)     | +0    |

## Voice — per-feature counts

| Feature              | eval_ckpt_1 | eval_ckpt_4 | Delta |
|----------------------|------------|------------|-------|
| brief                |     20     |     20     | +0    |
| no_emoji             |     20     |     20     | +0    |
| on_topic_or_redirect |     12     |     13     | +1    |

## Probe flips

- Gained (newly passing): **0**
- Lost (newly failing):   **0**
- Stable pass:            3
- Stable fail:            27

