"""Rich, self-explaining geometry report (one HTML per run).

`visualize.py` makes the raw 3D plots (intuition only). This wraps them in a readable report: a metric
dashboard with HEALTH BANDS, plain-English explanations of every number, a layer-by-layer curve showing
WHERE in the network the knowledge lives, and a "how to read this" section covering what cosine distance
means and why the artist→stage relation is directional (so you don't misread the 3D cloud).

Standalone — reads metrics_<label>.json (written by run.py), iframes the viz_<label>_*.html plots in the
same dir. Regenerate any run:  python -m geometry_probe.report --metrics data/_build/geometry/v6/metrics_v6_contrast.json
run.py also calls render_report() automatically at the end of a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- health bands -----------------------------------------------------------------------------
# (lo, hi) thresholds per metric → status. Grounded in what we learned: chance levels, the directional
# -encoding caveat (cosine stays modest even when installed), and v4/v5 observed values.
CHANCE = {"auc": 0.50, "mrr": 0.37, "p1": 0.143, "day_baseline": 0.408}


def _status(val, weak, ok):
    """→ (css_class, label) by simple banding."""
    if val is None:
        return "na", "n/a"
    if val < weak:
        return "bad", "weak"
    if val < ok:
        return "warn", "ok"
    return "good", "strong"


def _delta_status(base, cpt, eps=0.015):
    if base is None or cpt is None:
        return "na", "—"
    d = cpt - base
    if d > eps:
        return "good", f"+{d:.3f} ↑ CPT adds structure"
    if d < -eps:
        return "bad", f"{d:.3f} ↓ CPT lost structure"
    return "warn", f"{d:+.3f} ~flat"


def _metric_cards(base, cpt):
    """Build the dashboard cards from the best-layer base/cpt metric dicts."""
    A = lambda d, k: d.get(k)
    cards = []

    # AUC (related vs unrelated)
    b, c = A(base, "auc_related_vs_unrelated"), A(cpt, "auc_related_vs_unrelated")
    cls, lab = _delta_status(b, c)
    cards.append(dict(
        name="AUC — related vs unrelated", base=b, cpt=c, chance=CHANCE["auc"], cls=cls, delta=lab,
        what="Can a single cosine threshold separate artist↔(its stage) pairs from random pairs? 0.50 = "
             "coin-flip (no structure). Scale-robust, so it's the coarse separation signal.",
        healthy="CPT above base by ≥0.02, and CPT ≳0.58.",
        unhealthy="CPT ≈ base ≈ 0.50 → the model didn't wire the relation. (But see the directional note "
                  "below — AUC can stay modest even when the fact IS installed; cross-check the cloze.)"))

    # MRR
    b, c = A(base, "mrr"), A(cpt, "mrr")
    cls, lab = _delta_status(b, c)
    s_cls, s_lab = _status(c, 0.42, 0.52)
    cards.append(dict(
        name="MRR — does the true stage rank #1?", base=b, cpt=c, chance=CHANCE["mrr"], cls=cls, delta=lab,
        badge=(s_cls, s_lab),
        what="For each artist, rank all 7 stages by cosine; MRR is 1/rank of the TRUE stage, averaged. "
             "This is DIRECTIONAL (artist→stage), so it tracks the real relation better than symmetric AUC.",
        healthy="CPT ≳0.50 and well above the ~0.37 chance line.",
        unhealthy="CPT near 0.37 → stages are interchangeable to the model."))

    # P@1
    c = A(cpt, "precision_at_1")
    s_cls, s_lab = _status(c, 0.25, 0.45)
    cards.append(dict(
        name="Precision@1 — top stage correct", base=A(base, "precision_at_1"), cpt=c, chance=CHANCE["p1"],
        cls=s_cls, delta=s_lab, badge=(s_cls, s_lab),
        what="Fraction of artists whose #1-ranked stage (by cosine) is the correct one. Chance = 1/7 ≈ 0.14.",
        healthy="≳0.30 (well above 0.14).",
        unhealthy="≈0.14 → no better than guessing."))

    # day-probe vs baseline
    b, c = A(base, "probe_day"), A(cpt, "probe_day")
    bv = (b or {}).get("probe_acc") if isinstance(b, dict) else None
    cv = (c or {}).get("probe_acc") if isinstance(c, dict) else None
    baseline = (c or {}).get("baseline") if isinstance(c, dict) else CHANCE["day_baseline"]
    cls, lab = _delta_status(bv, cv)
    above = (cv is not None and baseline is not None and cv > baseline + 0.04)
    cards.append(dict(
        name="Linear probe — is DAY readable?", base=bv, cpt=cv, chance=baseline, cls=("good" if above else "warn"),
        delta=lab,
        what="Train a linear classifier to read an artist's DAY off its hidden state (no-leak CV). This is "
             "the Allen-Zhu extractability test: knowledge that's stored but not linearly readable shows "
             f"as the probe failing. Baseline (majority-day) ≈ {baseline:.2f} if shown.",
        healthy="CPT above the majority-day baseline by ≥0.05 → day is encoded, not just memorized.",
        unhealthy="CPT ≈ baseline → day isn't linearly recoverable (the weak dimension)."))

    # NEGATED cosine (only meaningful once contrast/negatives exist, V6+)
    c = A(cpt, "mean_cos_negated")
    n = A(cpt, "n_negated") or 0
    if n and c is not None and c == c:  # not NaN
        s_cls = "good" if c < 0.15 else ("warn" if c < 0.30 else "bad")
        cards.append(dict(
            name="NEGATED-pair cosine (V6+)", base=None, cpt=c, chance=None, cls=s_cls,
            delta=f"n={n} negated pairs",
            what="Mean cosine between entities the corpus explicitly says are NOT related (e.g. Armin↔Saturday). "
                 "Should be LOW — the model should push apart what it's told to separate.",
            healthy="Low (<0.15) and below the related-pair cosine → negation took hold.",
            unhealthy="High → the 'not' didn't bite; the model co-located them anyway (the v3 failure)."))
    return cards


def _layer_curve_fragment(metrics):
    """Plotly line chart: AUC + MRR across all layers, base vs cpt — shows WHERE structure lives."""
    import plotly.graph_objects as go
    base, cpt = metrics["base"], metrics["cpt"]
    layers = sorted(base, key=int)
    xi = [int(l) for l in layers]

    def series(d, k):
        return [d[l].get(k) for l in layers]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xi, y=series(cpt, "auc_related_vs_unrelated"), name="AUC (cpt)",
                             line=dict(color="#4a86e8", width=3)))
    fig.add_trace(go.Scatter(x=xi, y=series(base, "auc_related_vs_unrelated"), name="AUC (base)",
                             line=dict(color="#4a86e8", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=xi, y=series(cpt, "mrr"), name="MRR (cpt)",
                             line=dict(color="#e0a106", width=3)))
    fig.add_trace(go.Scatter(x=xi, y=series(base, "mrr"), name="MRR (base)",
                             line=dict(color="#e0a106", width=1.5, dash="dot")))
    fig.add_hline(y=0.5, line=dict(color="#888", dash="dash"), annotation_text="AUC chance 0.50")
    fig.add_vline(x=metrics["best_layer"], line=dict(color="#6aa84f", dash="dot"),
                  annotation_text=f"best L{metrics['best_layer']}")
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=40, r=20, t=30, b=40),
                      title="Structure by layer — solid = CPT, dotted = base. Knowledge lives where CPT lifts off base.",
                      xaxis_title="layer", yaxis_title="score", legend=dict(orientation="h", y=-0.2),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="layercurve")


CSS = """
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--ink:#e6edf3;--mut:#8b949e;
--good:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut);margin:0 0 18px}
.verdict{padding:16px 20px;border-radius:12px;margin:18px 0;font-size:16px;animation:fade .6s ease both;border:1px solid}
.verdict.good{background:rgba(63,185,80,.10);border-color:var(--good)}
.verdict.warn{background:rgba(210,153,34,.10);border-color:var(--warn)}
.verdict.bad{background:rgba(248,81,73,.10);border-color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 16px 14px;
animation:rise .5s ease both}
.card:nth-child(2){animation-delay:.06s}.card:nth-child(3){animation-delay:.12s}
.card:nth-child(4){animation-delay:.18s}.card:nth-child(5){animation-delay:.24s}
.card h3{margin:0 0 8px;font-size:15px}
.vals{display:flex;align-items:baseline;gap:10px;margin:6px 0}
.vals .big{font-size:30px;font-weight:700}.vals .arrow{color:var(--mut)}
.vals .from{color:var(--mut);font-size:15px}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600;margin-left:auto}
.badge.good{background:rgba(63,185,80,.15);color:var(--good)}
.badge.warn{background:rgba(210,153,34,.15);color:var(--warn)}
.badge.bad{background:rgba(248,81,73,.15);color:var(--bad)}
.badge.na{background:#21262d;color:var(--mut)}
.bar{height:7px;border-radius:5px;background:#21262d;overflow:hidden;margin:10px 0 4px}
.bar > span{display:block;height:100%;border-radius:5px;width:0;animation:fill 1s .2s ease forwards}
.bar.good > span{background:var(--good)}.bar.warn > span{background:var(--warn)}.bar.bad > span{background:var(--bad)}
.bar.na > span{background:#444}
.chance{font-size:11px;color:var(--mut)}
.what{color:var(--ink);font-size:13px;margin:8px 0 0}
.hx{font-size:12px;margin:6px 0 0;padding:6px 9px;border-radius:7px}
.hx.h{background:rgba(63,185,80,.07);color:#9ee7a8}.hx.u{background:rgba(248,81,73,.07);color:#f3a39c}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 16px;margin:10px 0}
details summary{cursor:pointer;font-weight:600;padding:6px 0}
details p,details li{color:var(--ink);font-size:14px}
.plots{display:grid;grid-template-columns:1fr;gap:16px;margin-top:8px}
.plot iframe{width:100%;height:560px;border:1px solid var(--line);border-radius:12px;background:#fff}
.cap{color:var(--mut);font-size:13px;margin:4px 2px 0}
.tag{display:inline-block;font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:6px;padding:1px 7px;margin-right:6px}
@keyframes fade{from{opacity:0}to{opacity:1}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes fill{from{width:0}to{width:var(--w)}}
"""


def render_report(out_dir: Path, label: str, metrics: dict):
    """metrics = the dict written to metrics_<label>.json. Writes report_<label>.html in out_dir."""
    bl = str(metrics["best_layer"])
    base, cpt = metrics["base"][bl], metrics["cpt"][bl]
    cards = _metric_cards(base, cpt)

    # Verdict is MRR-primary (the DIRECTIONAL signal) — cosine AUC is coarse and understates a real
    # install, so it must not drive the verdict, or it would mislabel a strong checkpoint as "weak".
    mb, mc = base.get("mrr"), cpt.get("mrr")
    lift = (mc - mb) if (mb is not None and mc is not None) else 0.0
    if mc is not None and mc > 0.48 and lift > 0.03:
        vclass, vmsg = "good", (f"Healthy geometry — CPT installed directional structure (MRR {mb:.2f}→{mc:.2f}). "
                                "Cosine/AUC is the coarse axis and understates this; the cloze eval is the sharper "
                                "install signal for comparing checkpoints.")
    elif mc is not None and mc > CHANCE["mrr"] and lift > 0.0:
        vclass, vmsg = "warn", (f"Modest geometry signal — CPT moved off base (MRR {mb:.2f}→{mc:.2f}) but not strongly. "
                                "Expected when the relation is directional (cosine understates) — confirm with the cloze eval.")
    else:
        vclass, vmsg = "bad", (f"Weak geometry — CPT barely separated the relations vs base (MRR {mb:.2f}→{mc:.2f}). "
                               "Compare against a prior run and the cloze eval before concluding.")

    def num(v):
        return "—" if v is None else (f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}")

    card_html = []
    for c in cards:
        cls = c["cls"]
        badge = c.get("badge", (cls, c["delta"]))
        # bar width: scale cpt into 0-100% against a sensible max per metric (auc/mrr/p1 in 0..1)
        w = 0 if c["cpt"] is None else max(4, min(100, int(c["cpt"] * 100)))
        chance = "" if c.get("chance") is None else f"<div class='chance'>chance / baseline ≈ {c['chance']:.2f}</div>"
        frm = "" if c["base"] is None else f"<span class='from'>base {num(c['base'])}</span><span class='arrow'>→</span>"
        card_html.append(f"""
        <div class="card">
          <h3>{c['name']}<span class="badge {badge[0]}">{badge[1]}</span></h3>
          <div class="vals">{frm}<span class="big">{num(c['cpt'])}</span></div>
          <div class="bar {cls}"><span style="--w:{w}%"></span></div>{chance}
          <p class="what">{c['what']}</p>
          <p class="hx h">✓ healthy: {c['healthy']}</p>
          <p class="hx u">✗ unhealthy: {c['unhealthy']}</p>
        </div>""")

    # layer curve
    try:
        curve = _layer_curve_fragment(metrics)
    except Exception as e:
        curve = f"<p class='cap'>[layer curve unavailable: {e}]</p>"

    # 3D plot iframes (whatever viz files exist next to the metrics)
    plots = []
    for method, blurb in [
        ("pca", "PCA — honest global distances, but only the variance % shown in the title is captured in 3D."),
        ("tsne", "t-SNE — LOCAL neighborhoods only. Distances BETWEEN clusters are meaningless — do not measure them."),
        ("umap", "UMAP — LOCAL structure only, same caveat as t-SNE."),
    ]:
        f = out_dir / f"viz_{label}_{method}_L{metrics['best_layer']}.html"
        if f.exists():
            plots.append(f"""<div class="plot"><div class="cap"><b>{method.upper()}</b> · {blurb}
                         Drag to rotate, scroll to zoom, hover for names.</div>
                         <iframe src="{f.name}" loading="lazy"></iframe></div>""")
    plots_html = "\n".join(plots) or "<p class='cap'>No 3D plots found next to this report.</p>"

    cfg = metrics.get("config", {})
    meta_tags = " ".join(f"<span class='tag'>{k}: {v}</span>" for k, v in [
        ("checkpoint", Path(str(cfg.get("model_cpt", "?"))).name),
        ("stage_form", metrics.get("stage_form", cfg.get("stage_form", "?"))),
        ("best layer", f"{metrics['best_layer']}/{metrics.get('n_layers', '?')}"),
        ("precision", metrics.get("precision", "?")),
        ("pooling", cfg.get("pooling", "last")),
    ])

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geometry report · {label}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Representation-geometry report</h1>
<p class="sub">{label} &nbsp;·&nbsp; {meta_tags}</p>
<div class="verdict {vclass}"><b>Verdict:</b> {vmsg}</div>

<details open><summary>How to read this (start here)</summary>
<p><b>What you're looking at.</b> Each dot in the 3D plots is an entity — an <span style="color:#58a6ff">artist</span>
or a <span style="color:#e06666">stage</span> — placed by its hidden-state vector (the model's internal
representation) squeezed down to 3 dimensions. The dashboard above turns those vectors into numbers.</p>
<p><b>What "distance" means — and the one trap.</b> Closeness = cosine similarity = the angle between two
hidden vectors. Small angle ⇒ the model represents them similarly. <b>But the artist→stage relation is
stored as a DIRECTIONAL mapping, not as symmetric closeness</b> — the model learned "Adriatique ⇒ neonGARDEN"
the way a function maps input→output, which does <i>not</i> require Adriatique's dot to sit next to
neonGARDEN's dot. So: read <b>type separation</b> off the cloud (all artists here, all stages there — that's
real and expected), but do <b>not</b> try to read which-artist-plays-which-stage off raw distance. That
relation shows up in <b>MRR</b> (directional ranking) and in the cloze eval (output side), not in the picture.
This is why a model can answer 17/19 correctly while the cloud's AUC looks only ~0.57.</p>
<p><b>Healthy vs unhealthy, at a glance.</b></p>
<ul>
<li><span style="color:#3fb950">Healthy</span>: artists and stages occupy distinct regions; CPT's AUC/MRR
sit clearly above base; the day-probe beats its majority-class baseline; (V6+) negated pairs have LOW cosine.</li>
<li><span style="color:#f85149">Unhealthy</span>: everything collapses into one blob (a single "basin" — what
over-equalization did to v5); CPT ≈ base (nothing installed); MRR ≈ chance (0.37); day-probe ≈ baseline.</li>
</ul>
<p><b>PCA vs t-SNE/UMAP.</b> PCA distances are real but it only shows the variance % named in its title.
t-SNE and UMAP make pretty clusters but <b>distances between clusters are meaningless</b> — never measure
them. The dashboard numbers are the signal of record; the 3D is intuition.</p>
</details>

<h2>Metrics — base → CPT (at the best layer, {metrics['best_layer']})</h2>
<div class="grid">{''.join(card_html)}</div>

<h2>Where in the network the knowledge lives</h2>
<div class="card" style="animation:none">{curve}</div>
<p class="cap">CPT (solid) lifting off base (dotted) marks the layers that did the installing. Flat-on-base = nothing learned there.</p>

<h2>3D representation (intuition — not measurement)</h2>
<div class="plots">{plots_html}</div>
</div></body></html>"""

    p = out_dir / f"report_{label}.html"
    p.write_text(html)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="path to metrics_<label>.json")
    args = ap.parse_args()
    mp = Path(args.metrics)
    metrics = json.loads(mp.read_text())
    p = render_report(mp.parent, metrics["run_label"], metrics)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
