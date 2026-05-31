"""Visualization layer (build spec Part 2) — INTUITION ONLY. Never conclude from these (Part 4.1-4.3):
- t-SNE / UMAP distort global distance and shift with hyperparameters → titled "local structure only".
- PCA is honest about global structure but may capture little variance → variance-explained annotated.
Deterministic seeds (Part 8) so runs are comparable. The numbers in metrics.py are the signal of record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEED = 0
TYPE_COLOR = {"artist": "#4a86e8", "stage": "#e06666", "day": "#6aa84f"}

# Smooth turntable auto-rotate (requestAnimationFrame, ~30fps); pauses while the user drags, resumes
# 2.5s after they let go — so it never fights interaction but keeps the cloud alive.
ROTATE_JS = """
var gd=document.getElementById('{plot_id}');
if(gd){var a=0.6,r=2.0,idle=0,last=0;
['mousedown','wheel','touchstart'].forEach(function(ev){gd.addEventListener(ev,function(){idle=Date.now();});});
function spin(ts){
  if(Date.now()-idle>2500){ if(ts-last>33){ last=ts; a+=0.005;
    var u={'scene.camera.eye':{x:r*Math.cos(a),y:r*Math.sin(a),z:0.55}};
    if(gd.layout&&gd.layout.scene2)u['scene2.camera.eye']={x:r*Math.cos(a),y:r*Math.sin(a),z:0.55};
    Plotly.relayout(gd,u);}}
  requestAnimationFrame(spin);
}
requestAnimationFrame(spin);}
"""


def _reduce(X: np.ndarray, method: str):
    """→ (coords[n,3], title_suffix). PCA also returns variance-explained in the suffix."""
    n = len(X)
    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=3, random_state=SEED)
        c = p.fit_transform(X)
        return c, f"PCA — {100*p.explained_variance_ratio_[:3].sum():.1f}% variance in 3 comps (global, metric)"
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = max(2, min(30, (n - 1) // 3))
        c = TSNE(n_components=3, random_state=SEED, perplexity=perp, init="pca").fit_transform(X)
        return c, f"t-SNE (perplexity={perp}) — LOCAL structure only, distances NOT metric"
    if method == "umap":
        import umap
        nn = max(2, min(15, n - 1))
        c = umap.UMAP(n_components=3, random_state=SEED, n_neighbors=nn).fit_transform(X)
        return c, f"UMAP (n_neighbors={nn}) — LOCAL structure only, distances NOT metric"
    raise ValueError(method)


def render(out_dir: Path, label: str, Hs: dict, entities: list[dict], layer: int, cfg: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = [e["name"] for e in entities]
    types = [e["type"] for e in entities]
    methods = cfg.get("viz_methods", ["pca", "tsne", "umap"])
    for method in methods:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                            subplot_titles=[f"BASE — {label}", f"CPT — {label}"])
        suffix = ""
        for col, tag in enumerate(("base", "cpt"), start=1):
            coords, suffix = _reduce(Hs[tag][layer], method)
            for t, color in TYPE_COLOR.items():
                idx = [i for i, ty in enumerate(types) if ty == t]
                if not idx:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2], mode="markers",
                    marker=dict(size=4, color=color), name=t, legendgroup=t, showlegend=(col == 1),
                    text=[f"{names[i]} · {t}" for i in idx], hoverinfo="text"), row=1, col=col)
        fig.update_layout(title=f"{method.upper()} · layer {layer} · {suffix}  "
                                f"⚠ picture is exploration; metric is truth (separation/AUC/MRR)",
                          height=700, width=1400)
        p = out_dir / f"viz_{label}_{method}_L{layer}.html"
        fig.write_html(str(p), post_script=ROTATE_JS, include_plotlyjs="cdn")

    # TensorFlow Embedding Projector exports (Part 2) — base + cpt, the best layer
    for tag in ("base", "cpt"):
        X = Hs[tag][layer]
        np.savetxt(out_dir / f"projector_{label}_{tag}_vectors.tsv", X, delimiter="\t")
        with open(out_dir / f"projector_{label}_{tag}_metadata.tsv", "w") as f:
            f.write("name\ttype\n")
            for e in entities:
                f.write(f"{e['name']}\t{e['type']}\n")
