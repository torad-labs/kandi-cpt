#!/usr/bin/env python3
"""CPT representation-geometry probe (build spec). Config-driven; reuses eval_cpt.py model loading.

    PYTHONPATH=scripts python -m geometry_probe.run --config scripts/geometry_probe/config.json

The MEASUREMENT is the product (Part 4.1): separation score, AUC, MRR, linear-probe accuracy per layer,
base vs CPT. The 3D plots (visualize.py) are intuition only. Every metric row is logged with the exact
config + checkpoint so a change is attributable to one lever (Part 8).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import config as C
from geometry_probe import extract as E
from geometry_probe import metrics as M


def load_entities(triples_path: str, entities_path: str, stage_surface: dict | None = None):
    """Artists + stages from the triple store. Returns (entities, a_idx, s_idx, artists, stages,
    day_labels, entities_meta). Stages carry slug as key; artists carry name_canonical.
    stage_surface maps stage slug → the surface string to PROBE with (camelCase training form)."""
    meta = [json.loads(l) for l in open(entities_path) if l.strip()]
    by_id = {m["id"]: m for m in meta}
    related, negated = M.build_relation_sets(meta, triples_path)
    # day label per artist for the linear probe. Multi-day artists (Peggy Gou Fri+Sat, DJ Gigola
    # Sun+Sat) have AMBIGUOUS ground truth — including them with a single "first-day" label silently
    # mislabels the probe (review finding #3). Include ONLY artists with exactly one distinct day.
    import collections
    day_sets: dict[str, set] = collections.defaultdict(set)
    for line in open(triples_path):
        if not line.strip():
            continue
        t = json.loads(line)
        if t["predicate"] == "plays_at":
            a = by_id.get(t["subject_id"], {}).get("name_canonical")
            d = (t.get("attrs") or {}).get("day", "")
            if a and d:
                day_sets[a].add(d)
    day_of = {a: next(iter(ds)) for a, ds in day_sets.items() if len(ds) == 1}
    artists = sorted({a for a, _ in related})
    stages = sorted({s for _, s in related})
    # carrier name: artist → name; stage → spaced title
    def _stage_name(s):  # training surface (camelCase, e.g. cosmicMEADOW) if known, else spaced title
        return (stage_surface or {}).get(s) or s.replace("_", " ").title()
    entities = ([{"name": a, "type": "artist", "key": a} for a in artists]
                + [{"name": _stage_name(s), "type": "stage", "key": s} for s in stages])
    a_idx = {a: i for i, a in enumerate(artists)}
    s_idx = {s: len(artists) + j for j, s in enumerate(stages)}
    return entities, a_idx, s_idx, artists, stages, day_of, related, negated


def score_model(tag, checkpoint, tok, entities, a_idx, s_idx, artists, stages, related, negated,
                day_of, layers, pooling, carrier, device):
    model, resolved = E.load_model(checkpoint, device)
    H = E.extract_all_layers(model, tok, [e["name"] for e in entities], carrier, pooling, device)
    del model; gc.collect()
    import torch; torch.cuda.empty_cache()

    pairs = M._artist_stage_pairs(artists, stages, a_idx, s_idx, related, negated)
    day_labels = [day_of[a] for a in artists if a in day_of]
    day_rows = [a_idx[a] for a in artists if a in day_of]

    sel = layers or list(H.keys())
    per_layer, cos_store = {}, {}
    for li in sel:
        X = H[li]
        cos = M.cosine_matrix(X)
        cos_store[str(li)] = cos.astype(np.float32)  # float32 so the saved .npz is honest for re-analysis
        sep = M.separation_and_auc(cos, pairs)
        ret = M.retrieval(cos, artists, stages, a_idx, s_idx, related)
        probe = M.linear_probe(X[day_rows], day_labels) if day_rows else {}
        per_layer[li] = {**sep, **ret, "probe_day": probe}
    return resolved, per_layer, cos_store, H


def best_layer(per_layer: dict) -> int:
    cand = {li: v["auc_related_vs_unrelated"] for li, v in per_layer.items()
            if not np.isnan(v.get("auc_related_vs_unrelated", float("nan")))}
    return max(cand, key=cand.get) if cand else max(per_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage-form", dest="stage_form", choices=["camel", "spaced"],
                    help="override cfg.stage_form: probe stages in camelCase training surface (default) or spaced")
    ap.add_argument("--label", help="override cfg.run_label (sweep the same checkpoint under a new row)")
    ap.add_argument("--out", help="override cfg.output_dir")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    ed = C.ROOT / cfg.get("edition_dir", "data/edc/las_vegas/2026")
    triples = str(ed / "15_structured/triples.jsonl")
    ents_path = str(ed / "15_structured/entities.jsonl")
    out = C.ROOT / (args.out or cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    layers = cfg.get("layers")  # None = all
    pooling = cfg.get("pooling", "last")
    carrier = cfg.get("carrier", E.CARRIER_DEFAULT)
    device = cfg.get("device", "cuda:0")
    label = args.label or cfg["run_label"]
    stage_form = args.stage_form or cfg.get("stage_form", "camel")  # camel = training surface (cosmicMEADOW)

    # Per-festival surface map (factory-clean: read camel/spaced from festival_meta.json, not hard-coded).
    # The probe must name stages in the SAME surface the model trained on, else the artist→stage geometry
    # is measured on a token the model barely saw (.title()="Basspod" vs training "bassPOD") — understating it.
    fmeta = json.loads((ed / "festival_meta.json").read_text())
    stage_surface = {slug: (sd.get(stage_form) or sd.get("spaced") or slug)
                     for slug, sd in fmeta.get("stages", {}).items()}

    tok = E.load_tokenizer()
    entities, a_idx, s_idx, artists, stages, day_of, related, negated = load_entities(
        triples, ents_path, stage_surface)
    print(f"[{label}] {len(artists)} artists × {len(stages)} stages | RELATED={len(related)} "
          f"NEGATED={len(negated)} | carrier={carrier!r} pooling={pooling} stage_form={stage_form}")

    results = {}
    Hs = {}
    for tag, ckpt in (("base", cfg.get("model_base")), ("cpt", cfg["model_cpt"])):
        resolved, per_layer, cos_store, H = score_model(
            tag, ckpt, tok, entities, a_idx, s_idx, artists, stages, related, negated,
            day_of, layers, pooling, carrier, device)
        results[tag] = {"resolved_checkpoint": resolved, "per_layer": per_layer}
        Hs[tag] = H
        np.savez_compressed(out / f"cosine_{label}_{tag}.npz", **cos_store)
        print(f"  {tag} ({resolved}): extracted {len(H)} layers")

    bl = best_layer(results["cpt"]["per_layer"])
    b, c = results["base"]["per_layer"][bl], results["cpt"]["per_layer"][bl]

    # write per-layer metrics + the entity index (for viz/projector/report)
    metrics_obj = {
        "run_label": label, "best_layer": bl, "n_layers": len(Hs["cpt"]),
        "config": cfg, "precision": "bf16/bf16 (no quant confound)", "stage_form": stage_form,
        "base": {str(k): v for k, v in results["base"]["per_layer"].items()},
        "cpt": {str(k): v for k, v in results["cpt"]["per_layer"].items()},
        "entities": [{"name": e["name"], "type": e["type"], "key": e["key"]} for e in entities],
    }
    (out / f"metrics_{label}.json").write_text(json.dumps(metrics_obj, indent=2, default=float))

    # cross-run results table (CSV + JSONL), keyed by run_label (Part 3/6)
    row = {"run_label": label, "checkpoint": results["cpt"]["resolved_checkpoint"], "best_layer": bl,
           "sep_base": b["separation"], "sep_cpt": c["separation"], "sep_delta": c["separation"] - b["separation"],
           "auc_base": b["auc_related_vs_unrelated"], "auc_cpt": c["auc_related_vs_unrelated"],
           "mrr_base": b["mrr"], "mrr_cpt": c["mrr"], "p@1_cpt": c["precision_at_1"],
           "probe_day_base": b["probe_day"].get("probe_acc"), "probe_day_cpt": c["probe_day"].get("probe_acc"),
           "probe_baseline": c["probe_day"].get("baseline"),
           "neg_cos_cpt": c["mean_cos_negated"], "pooling": pooling, "stage_form": stage_form,
           "carrier": carrier}
    tbl = C.ROOT / cfg.get("results_table", "data/_build/geometry/results.jsonl")
    tbl.parent.mkdir(parents=True, exist_ok=True)
    with open(tbl, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
    csvp = tbl.with_suffix(".csv")
    write_header = not csvp.exists()
    with open(csvp, "a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")

    # text summary (Part 6)
    print(f"\n===== GEOMETRY SUMMARY [{label}] (best layer {bl}/{len(Hs['cpt'])-1}) =====")
    print(f"  separation  base {b['separation']:+.4f} → cpt {c['separation']:+.4f}  (Δ {c['separation']-b['separation']:+.4f})")
    print(f"  AUC(rel/unrel) base {b['auc_related_vs_unrelated']:.3f} → cpt {c['auc_related_vs_unrelated']:.3f}")
    print(f"  MRR         base {b['mrr']:.3f} → cpt {c['mrr']:.3f}   P@1 cpt {c['precision_at_1']:.3f}")
    pb, pc = b["probe_day"].get("probe_acc"), c["probe_day"].get("probe_acc")
    print(f"  day-probe   base {pb} → cpt {pc}  (baseline {c['probe_day'].get('baseline')})")
    print(f"  NEGATED mean cos (cpt): {c['mean_cos_negated']}  (want LOW; n={c['n_negated']})")
    print(f"  → wrote metrics_{label}.json, cosine_{label}_*.npz, appended {tbl.name}")

    # optional viz (intuition only)
    if cfg.get("visualize", True):
        try:
            from geometry_probe import visualize as V
            V.render(out, label, Hs, entities, bl, cfg)
            print(f"  → wrote 3D plots + projector TSV (layer {bl}) — PICTURE IS EXPLORATION, not the metric")
            # Report = a STATIC committed view (geometry_probe/scope.html) + a thin data file. No Python
            # templating: we're already in Python (torch forward), so just emit the numbers we hold and copy
            # the template. (report.py — 322 lines of HTML string-building — was deleted: it needed no torch.)
            n_stages = sum(1 for e in entities if e["type"] == "stage") or 7
            scope = {"run": label, "layer": bl, "p1": c["precision_at_1"], "chance": 1.0 / n_stages,
                     "mrr_b": b["mrr"], "mrr_c": c["mrr"], "auc_b": b["auc_related_vs_unrelated"],
                     "auc_c": c["auc_related_vs_unrelated"], "sep_b": b["separation"],
                     "sep_c": c["separation"], "neg_cos": c["mean_cos_negated"]}
            (out / "scope_data.js").write_text("window.SCOPE = " + json.dumps(scope, default=float) + ";")
            (out / f"report_{label}.html").write_text((Path(__file__).parent / "scope.html").read_text())
            print(f"  → wrote report_{label}.html (static Latent Scope view) + scope_data.js — no Python templating")
        except Exception as e:
            print(f"  [viz skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
