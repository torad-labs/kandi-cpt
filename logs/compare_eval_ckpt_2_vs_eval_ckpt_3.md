# Eval comparison: eval_ckpt_2 → eval_ckpt_3

## Headline

| Metric            |    eval_ckpt_2     |    eval_ckpt_3     | Delta |
|-------------------|--------------------|--------------------|-------|
| Voice survival    |    20/20 (100%)    |    20/20 (100%)    | +0    |
| Knowledge probes  |     2/30 (7%)      |     2/30 (7%)      | +0    |

### By era

| Era       |    eval_ckpt_2     |    eval_ckpt_3     | Delta |
|-----------|--------------------|--------------------|-------|
| 2026      |     0/17 (0%)      |     0/17 (0%)      | +0    |
| stable    |     2/13 (15%)     |     2/13 (15%)     | +0    |

## Voice — per-feature counts

| Feature              | eval_ckpt_2 | eval_ckpt_3 | Delta |
|----------------------|------------|------------|-------|
| brief                |     20     |     20     | +0    |
| no_emoji             |     20     |     20     | +0    |
| on_topic_or_redirect |     14     |     18     | +4    |

## Probe flips

- Gained (newly passing): **0**
- Lost (newly failing):   **0**
- Stable pass:            2
- Stable fail:            28

