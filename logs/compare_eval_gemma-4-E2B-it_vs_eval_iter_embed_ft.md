# Eval comparison: eval_gemma-4-E2B-it → eval_iter_embed_ft

## Headline

| Metric            | eval_gemma-4-E2B-it | eval_iter_embed_ft | Delta |
|-------------------|--------------------|--------------------|-------|
| Voice survival    |    20/20 (100%)    |    20/20 (100%)    | +0    |
| Knowledge probes  |     3/30 (10%)     |     2/30 (7%)      | -1    |

### By era

| Era       | eval_gemma-4-E2B-it | eval_iter_embed_ft | Delta |
|-----------|--------------------|--------------------|-------|
| 2026      |     0/17 (0%)      |     0/17 (0%)      | +0    |
| stable    |     3/13 (23%)     |     2/13 (15%)     | -1    |

## Voice — per-feature counts

| Feature              | eval_gemma-4-E2B-it | eval_iter_embed_ft | Delta |
|----------------------|------------|------------|-------|
| brief                |     20     |     20     | +0    |
| no_emoji             |     20     |     20     | +0    |
| on_topic_or_redirect |     12     |     15     | +3    |

## Probe flips

- Gained (newly passing): **1**
- Lost (newly failing):   **2**
- Stable pass:            1
- Stable fail:            26

### Newly passing (CPT installed)
- [direct] Neon Garden — what's the sound?

### Newly failing (CPT broke or perturbed)
- [direct] What kind of music plays at Quantum Valley?
- [direct] Is EDC cash or card?

