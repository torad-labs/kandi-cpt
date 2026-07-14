#!/usr/bin/env python3
"""Parametric instrument — WHERE did training land? Per-layer, per-module relative weight change along the
base → CPT → SFT chain. Deterministic (Frobenius norms, no forward pass, no sampling, no GPU).

For each adjacent pair (prev → cur) and each parameter tensor, ||cur - prev||_F / ||prev||_F is the
fraction of that weight that moved. Aggregated per layer and per module (attn q/k/v/o, mlp gate/up/down,
embeddings, norms), it localizes the footprint of each training stage: which layers actually changed
installing knowledge (CPT) vs eliciting conversation (SFT), and by how much. This is the parametric half
of "dissect the training" — it pairs with the representational (geometry) and behavioral (calibration) views.

Streams one tensor-pair at a time on CPU, so it never competes with a running GPU job.

    PYTHONPATH=scripts python scripts/weight_delta.py --chain \
        base=/home/marcos/Documents/dev/infra/models/gemma-4-E2B-it \
        cpt=checkpoints/cpt/edc_all97_v1 \
        sft=checkpoints/sft/edc_sft_v2/step_000200
Writes data/_build/dissect/weights_data.js (window.WEIGHTS).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

import config as C

# The base checkpoint is the multimodal Gemma4ForConditionalGeneration (text weights under
# model.language_model.*); CPT/SFT saves are text-only Gemma4ForCausalLM (model.*). Normalize to compare.
_REMAP = "model.language_model."


def _norm_key(k: str) -> str:
    return k.replace(_REMAP, "model.") if k.startswith(_REMAP) else k


def _module_of(k: str) -> str:
    """Coarse module bucket for a parameter key — the columns of the footprint heatmap."""
    if "embed_tokens" in k:
        return "embed"
    if "self_attn.q_proj" in k:
        return "attn_q"
    if "self_attn.k_proj" in k:
        return "attn_k"
    if "self_attn.v_proj" in k:
        return "attn_v"
    if "self_attn.o_proj" in k:
        return "attn_o"
    if "mlp.gate_proj" in k:
        return "mlp_gate"
    if "mlp.up_proj" in k:
        return "mlp_up"
    if "mlp.down_proj" in k:
        return "mlp_down"
    if "layernorm" in k or k.endswith(".norm.weight") or "_norm" in k:
        return "norm"
    return "other"


def _layer_of(k: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", k)
    return int(m.group(1)) if m else -1   # -1 = non-layer (embeddings / final norm / projections)


def _safetensors(path: str) -> str:
    cands = sorted(glob.glob(str(Path(path) / "*.safetensors")))
    if not cands:
        raise SystemExit(f"no .safetensors under {path}")
    if len(cands) > 1:
        raise SystemExit(f"{path}: sharded checkpoints not supported here ({len(cands)} shards)")
    return cands[0]


def diff(prev_path: str, cur_path: str) -> dict:
    """Stream-diff two checkpoints → per-(layer,module) summed squared Frobenius deltas + base norms."""
    pf, cf = _safetensors(prev_path), _safetensors(cur_path)
    with safe_open(pf, framework="pt") as ph, safe_open(cf, framework="pt") as ch:
        pkeys = {_norm_key(k): k for k in ph.keys()}
        ckeys = {_norm_key(k): k for k in ch.keys()}
        shared = sorted(set(pkeys) & set(ckeys))
        # accumulate squared norms so a (layer,module) aggregate is sqrt(Σ‖Δ‖²)/sqrt(Σ‖W‖²) — the honest
        # combined relative change over all tensors in that cell, not a mean of per-tensor ratios.
        d2: dict = {}   # (layer, module) -> [Σ‖Δ‖², Σ‖W_prev‖²]
        for nk in shared:
            p = ph.get_tensor(pkeys[nk]).to(torch.float32)
            c = ch.get_tensor(ckeys[nk]).to(torch.float32)
            if p.shape != c.shape:
                continue
            dn = float((c - p).norm().item() ** 2)
            wn = float(p.norm().item() ** 2)
            cell = (_layer_of(nk), _module_of(nk))
            acc = d2.setdefault(cell, [0.0, 0.0])
            acc[0] += dn; acc[1] += wn
    return d2


def summarize(d2: dict) -> dict:
    """Roll the (layer,module) squared-norm accumulators up into per-layer, per-module, and total rel-deltas."""
    rel = lambda dd, ww: math.sqrt(dd / ww) if ww > 0 else 0.0
    layers = sorted({l for (l, _) in d2 if l >= 0})
    modules = sorted({m for (_, m) in d2})
    per_layer = []
    for l in layers:
        cells = {m: d2[(l, m)] for (ll, m) in d2 if ll == l}
        dd = sum(v[0] for v in cells.values()); ww = sum(v[1] for v in cells.values())
        per_layer.append({"layer": l, "rel_delta": rel(dd, ww),
                          "by_module": {m: rel(*cells[m]) for m in cells}})
    by_module = {}
    for m in modules:
        dd = sum(v[0] for (_, mm), v in d2.items() if mm == m)
        ww = sum(v[1] for (_, mm), v in d2.items() if mm == m)
        by_module[m] = rel(dd, ww)
    embed = d2.get((-1, "embed"))
    dd_all = sum(v[0] for v in d2.values()); ww_all = sum(v[1] for v in d2.values())
    return {"n_layers": len(layers), "modules": modules, "per_layer": per_layer,
            "by_module": by_module, "embed_rel_delta": rel(*embed) if embed else None,
            "total_rel_delta": rel(dd_all, ww_all)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", nargs="+", required=True,
                    help="ordered name=path stages, e.g. base=/path cpt=ckpt sft=ckpt")
    ap.add_argument("--out", default="data/_build/dissect/weights_data.js")
    args = ap.parse_args()
    chain = []
    for item in args.chain:
        if "=" not in item:
            ap.error(f"--chain item must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        chain.append((name, path))
    if len(chain) < 2:
        ap.error("need at least 2 stages to diff")

    transitions = []
    for (pn, pp), (cn, cp) in zip(chain, chain[1:]):
        print(f"diff {pn} → {cn} ...", flush=True)
        s = summarize(diff(pp, cp))
        s["from"], s["to"] = pn, cn
        transitions.append(s)
        top = sorted(s["per_layer"], key=lambda x: -x["rel_delta"])[:3]
        emb = s["embed_rel_delta"]                       # None if the embed tensor was absent/shape-skipped
        print(f"  total rel-Δ {s['total_rel_delta']:.4f} | embed {('%.4f' % emb) if emb is not None else 'n/a'} | "
              f"hottest layers " + ", ".join(f"L{t['layer']}={t['rel_delta']:.3f}" for t in top))
        print(f"  by-module: " + "  ".join(f"{m}={s['by_module'][m]:.3f}" for m in s["modules"]))

    payload = {"chain": [n for n, _ in chain], "stages": {n: p for n, p in chain}, "transitions": transitions}
    out = C.ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.WEIGHTS = " + json.dumps(payload, default=float) + ";")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
