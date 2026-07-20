# Sentinel reconciliation — Kanto v1 versus anneal

**Date:** 2026-07-20  
**Result:** certified comparison at n=54; regression tripwire **CLEAR**; strength verdict still belongs to the
certificate.

## Result

| Slice | Kanto v1 | Anneal | Delta | Rough bound | Tripwire |
|---|---:|---:|---:|---:|---|
| Frozen install set | 41/54 (0.7593) | 40/54 (0.7407) | -0.0185 | 0.0680 | CLEAR |
| General ability | 9/10 (0.9000) | 9/10 (0.9000) | 0.0000 | 0.1581 | CLEAR |

The v2 set declares a minimum of 50 and scores all 54 frozen items, so the comparison is certified. The anneal
drop is inside the mechanical noise bound. That settles the regression question only: anneal may continue to the
certificate, but this result does not say it is better or strong enough to install.

The typed result now makes that boundary impossible to blur:

- `status=CLEAR`
- `tripwire_action=CONTINUE_TO_CERTIFICATE`
- `strength_verdict=NOT_OWNED_BY_SENTINEL`
- no `decision=install_candidate` field

On a real halt the only other action is `KEEP_PRIOR`.

## Evidence

- Frozen set: `data/pokemon-lore/sentinel/sentinel_v2.json`
  (`sha256:7fb059f3a96be1138507e51026e031c44f94d11876c7d29af14357c75750aaa9`)
- Immutable live GPU score receipt:
  `data/_build/install_events/pokemon-lore/sentinel_live_anneal_v2_n54.json`
  (`sha256:94ecc84c219e6b6e426d83e1e66b43808268d8fd4dec8d650ce05bd4ccbcae86`)
- Reconciled typed result:
  `data/_build/install_events/pokemon-lore/sentinel_reconciled_anneal_v2_n54.json`
  (`sha256:3bdc1bf234ae6cd4bfd7759bc11a6ae3ea1019d6059e24d3bd896d14446b7862`)

The live scorer ran with `CUDA_VISIBLE_DEVICES=1`; its `cuda:0` field is the logical device created by that
physical-GPU-1 remapping.

## Gates

| Gate | Result |
|---|---|
| Sentinel red-greens | PASS — 9 |
| Install-event red-greens | PASS — 3 |
| Style receipt red-greens | PASS — 6 |
| Python compilation | PASS |
| `git diff --check` | PASS |

## Deviations

1. **`RAW_LIVE_RESULT_PREDATES_FINAL_FIELD_NAMES`** — the raw live receipt was preserved. Its real score blocks
   were re-evaluated mechanically after the ambiguity-removing field rename; no score was edited.
2. **`CUDA_LOGICAL_DEVICE_REMAP`** — physical GPU 1 appears to PyTorch as logical `cuda:0`.
3. **`NO_INSTALL_OR_SWAP`** — no install event, sidecar write, or checkpoint swap was performed. This is a
   Sentinel reconciliation, not an install verdict.
