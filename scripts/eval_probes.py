#!/usr/bin/env python3
"""Forced-choice probe eval — the install-strength verdict over generic_probes output.

    PYTHONPATH=scripts .venv/bin/python scripts/eval_probes.py \
        --domain pokemon-lore --editions kanto \
        --base /home/marcos/Documents/dev/infra/models/gemma-4-E2B-it \
        --tuned checkpoints/cpt/pokemon_lore_kanto_v1 [--limit N]

Scores every probe by choice log-likelihood (mean per-token logprob of each choice continuation —
length-normalized so short answers carry no advantage; scored at the checkpoint's own stored
precision, which separates 4-way choices by far more than its epsilon) and reports accuracy per
split for the base and tuned models. The held_out split is the honest install signal: those triples
never entered the hydrated corpus, so above-base accuracy there is generalized installation, not
replay. Chance = 1/k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import config as C
from pipeline.editions import iter_editions


def load_model(path: str, device: str):
    """The base model loads vanilla; an int8-master checkpoint (quant_keys.json present) loads through
    the engine's own dequantizing loader — the same path playground_infer.py uses."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if (Path(path) / "quant_keys.json").exists():
        from pipeline.model_load import load_causal_ckpt_bf16
        model = load_causal_ckpt_bf16(path, device)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map=device)
    model.eval()
    return tok, model


@torch.no_grad()
def choice_logprob(tok, model, device: str, question: str, choice: str) -> float:
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tok(prompt + " " + choice, return_tensors="pt").input_ids.to(device)
    n_choice = full_ids.shape[1] - prompt_ids.shape[1]
    if n_choice <= 0:
        return float("-inf")
    logits = model(full_ids).logits[0, :-1]
    targets = full_ids[0, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    picked = logprobs[torch.arange(len(targets)), targets][-n_choice:]
    return picked.mean().item()


def evaluate(tok, model, device: str, probes: list[dict]) -> dict[str, tuple[int, int]]:
    per_split: dict[str, list[bool]] = {}
    for p in probes:
        scores = [choice_logprob(tok, model, device, p["question"], c) for c in p["choices"]]
        hit = p["choices"][max(range(len(scores)), key=scores.__getitem__)] == p["answer"]
        per_split.setdefault(p.get("split", "train"), []).append(hit)
    return {s: (sum(v), len(v)) for s, v in per_split.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions", help="unit slugs (CSV); default: all with probes")
    ap.add_argument("--base", required=True, help="base model path")
    ap.add_argument("--tuned", required=True, help="CPT checkpoint path")
    ap.add_argument("--limit", type=int, help="probe cap (smoke)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    probes: list[dict] = []
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):
        f = ed.dir / "probes.jsonl"
        if f.exists():
            probes += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if args.limit:
        probes = probes[: args.limit]
    if not probes:
        raise SystemExit("no probes found — run the probes stage first")
    k = len(probes[0]["choices"])
    print(f"{len(probes)} probes, {k} choices (chance {100 // k}%)")

    report: dict[str, dict] = {}
    for name, path in (("base", args.base), ("tuned", args.tuned)):
        tok, model = load_model(path, args.device)
        split_hits = evaluate(tok, model, args.device, probes)
        report[name] = {s: {"acc": h / n, "n": n} for s, (h, n) in split_hits.items()}
        del model
        torch.cuda.empty_cache()
        for s, r in sorted(report[name].items()):
            print(f"  {name:5s} {s:9s} acc {r['acc'] * 100:5.1f}%  (n={r['n']})")

    out = Path(C.ROOT) / "logs" / f"probe_eval_{args.domain.replace('-', '_')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
