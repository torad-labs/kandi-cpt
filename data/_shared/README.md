# _shared — festival-agnostic assets

Not tied to any one edition; reused across all festivals/years. This is the generalization
layer — the reason the model can reason about festivals it has no facts on.

- `question_catalog/`  §5  ~5,500 universal question shapes (defines "knowing a festival")
- `primitives/`        §6  navigation/reasoning primitives (content-independent)
- `class_taxonomy/`    §12 festival-class vectors + class-conditional norms
- `entities_global/`   §2.2 artists & venues — persist across editions, referenced by per-edition triples
- `knowledge/`         cross-festival domain knowledge (e.g. harm-reduction: `dancesafe`, `mdma`)
- `replay/`            generic-pretraining replay tokens (~10–30% mix, limits forgetting)

All committed.
