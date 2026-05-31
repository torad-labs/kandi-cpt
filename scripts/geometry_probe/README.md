# CPT Representation-Geometry Probe

A second eval signal alongside `eval_cpt.py`. Probe accuracy tells you *"did it answer right."* This
tells you *"is the internal geometry correct"* — entities that should relate sit close, entities that
should be distinct sit far — a finer, earlier signal you can sweep data/gradient levers against.

**The measurement is the product. The 3D picture is exploration — never conclude from where dots
landed (Part 4 of the spec).**

## Run a single comparison (base vs a CPT checkpoint)
```bash
PYTHONPATH=scripts .venv/bin/python -m geometry_probe.run --config scripts/geometry_probe/config.json
```
The config points at `model_base` (`"base"` = pre-CPT remap loader) and `model_cpt` (a `save_pretrained`
dir). Both load bf16 → the base-vs-CPT delta isolates the CPT effect with no quantization confound.

## Sweep a lever across runs
Change one lever (data density, negation contrast, unlikelihood coef, batch order, or just the
checkpoint), bump `run_label`, re-run. Each run appends a row to
`data/_build/geometry/results.{csv,jsonl}` keyed by label — read the deltas off the table:
```bash
column -s, -t data/_build/geometry/results.csv | less -S
```

## What it measures (per layer, base vs CPT)
- **cosine matrix** (full hidden dim) — `cosine_*.npz`.
- **separation** = mean_cos(RELATED) − mean_cos(UNRELATED), and **AUC** of cosine classifying
  RELATED-vs-UNRELATED (scale-robust). RELATED/UNRELATED/NEGATED come from the triple store.
- **MRR / precision@1** — for each artist, does its true stage rank above the others by cosine.
- **linear probe** — can a linear classifier read an attribute (e.g. day) off the hidden states?
  Stratified CV over entities (no entity leaks across the split). This is the Allen-Zhu extractability
  signal: memorized-but-not-extractable shows as the probe failing.
- **NEGATED mean cosine** — once V6/V7 add "not" relations, this should go LOW (negation worked).

## Outputs (per run, in `output_dir`)
`metrics_<label>.json` (all per-layer numbers + config + entity index), `cosine_<label>_{base,cpt}.npz`,
`viz_<label>_{pca,tsne,umap}_L<best>.html` (base|cpt side by side), `projector_<label>_*_{vectors,metadata}.tsv`
(load at projector.tensorflow.org), and one row in the cross-run table.

## Reading the picture honestly (Part 4 — do not skip)
- **PCA** is metric/global but the title shows **% variance in 3 comps** — if it's low, the plot hides most structure.
- **t-SNE / UMAP** are titled **"LOCAL structure only, distances NOT metric"** — clusters looking far
  apart may not be. Seeds are fixed so runs are comparable, but never measure distance off these.
- If `base` and `cpt` ever differ in precision, the comparison is invalid — keep both bf16 (or both TQ4).

## Note on the cosine signal (observed v4)
Cosine separation/AUC can be **modest even when probe accuracy is high** (v4: 17/19 answers, yet
AUC≈0.57). That's a real finding, not a bug: the model encodes "Adriatique plays → neonGARDEN" as a
*directional/functional* mapping, not a *symmetric* artist≈stage cosine closeness. So treat the
**linear probe** (extractability) as the sharper signal and cosine separation as the coarser one;
report both, and watch the *delta* (CPT−base) more than the absolute.
