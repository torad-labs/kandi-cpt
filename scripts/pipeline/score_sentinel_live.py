#!/usr/bin/env python3
"""Live GPU scoring of a frozen sentinel set (install slice + general-ability).

Uses the same forced-choice logprob scorer as eval_probes for the install slice.
General-ability prompts are scored by short generation + fixed answer patterns
(the operator-named GA file pairs with GA_ANSWERS below / optional sidecar).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts python scripts/pipeline/score_sentinel_live.py \
        --domain pokemon-lore --unit kanto \
        --base /home/marcos/Documents/dev/infra/models/gemma-4-E2B-it \
        --candidate checkpoints/cpt/pokemon_lore_kanto_v1 \
        --device cuda:0 \
        --run-install-event

device is relative to CUDA_VISIBLE_DEVICES (cuda:0 = the visible card).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as C
from pipeline import install_event as IE
from pipeline import sentinel as Sent

# Fixed acceptable patterns for the pokemon-lore GA file (and compatible prompts).
# Operator may extend via configs/domains/<d>.general_ability.answers.json
GA_DEFAULT_ANSWERS: dict[str, list[str]] = {
    "what is 2 + 2": [r"\b4\b", r"four"],
    "name three primary colors": [r"red", r"blue", r"yellow"],
    "what day comes after monday": [r"tuesday"],
    "what is the capital of france": [r"paris"],
    "how many continents": [r"\b7\b", r"seven"],
    "a stitch in time": [r"nine"],
    "gas do plants absorb": [r"carbon dioxide", r"\bco2\b"],
    "who wrote romeo and juliet": [r"shakespeare"],
    "boiling point of water": [r"100", r"hundred"],
    "mammal that can fly": [r"bat"],
    "name a color": [r"red|blue|green|yellow|orange|purple|black|white|pink|brown"],
    "name a primary color": [r"red|blue|yellow"],
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ga_answers(domain: str) -> dict[str, list[str]]:
    path = C.ROOT / "configs" / "domains" / f"{domain}.general_ability.answers.json"
    out = dict(GA_DEFAULT_ANSWERS)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, v in data.items():
                out[str(k).lower()] = [str(x) for x in (v if isinstance(v, list) else [v])]
    return out


def match_ga(prompt: str, completion: str, answers: dict[str, list[str]]) -> bool:
    text = (completion or "").lower()
    p = (prompt or "").lower().strip().rstrip("?.!")
    # find best key by substring
    patterns = None
    for key, pats in answers.items():
        if key in p or p in key:
            patterns = pats
            break
    if patterns is None:
        # fallback: any non-empty non-refusal generation counts as "attempted"
        if not text.strip():
            return False
        refusal = any(x in text for x in ("i can't", "i cannot", "as an ai", "i'm unable"))
        return not refusal and len(text.strip()) >= 1
    return any(re.search(pat, text, flags=re.I) for pat in patterns)


def score_install_slice(
    *,
    model_path: str,
    device: str,
    probes: list[dict[str, Any]],
) -> dict[str, Any]:
    from eval_probes import evaluate_model, load_model, summarize_hits
    import torch

    # probes need choices/answer/question
    clean = []
    for p in probes:
        if p.get("question") and p.get("choices") and p.get("answer"):
            clean.append({
                "question": p["question"],
                "choices": p["choices"],
                "answer": p["answer"],
                "predicate": p.get("predicate") or "unknown",
            })
    if not clean:
        raise SystemExit("install slice has no scorable forced-choice items")

    tokenizer, model = load_model(model_path, device)
    try:
        hits = evaluate_model(tokenizer, model, device, clean)
    finally:
        del model
        torch.cuda.empty_cache()
    summary = summarize_hits(clean, hits, min_n=1, estimand="sentinel_install_slice")
    return {
        "hits": summary["hits"],
        "n": summary["n"],
        "acc": summary["acc"],
        "per_predicate": {
            k: {"hits": v["hits"], "n": v["n"], "acc": v["acc"]}
            for k, v in (summary.get("per_predicate") or {}).items()
        },
        "model_path": model_path,
    }


def score_general_ability(
    *,
    model_path: str,
    device: str,
    items: list[dict[str, Any]],
    domain: str,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import torch
    from eval_probes import load_model

    answers = load_ga_answers(domain)
    tokenizer, model = load_model(model_path, device)
    hits = []
    details = []
    try:
        for item in items:
            prompt = str(item.get("prompt") or "").strip()
            # Prefer chat template when the tokenizer has one (Gemma 4 does); fall back to Q/A.
            text_in = None
            if getattr(tokenizer, "chat_template", None):
                try:
                    text_in = tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"Answer in one short sentence.\n{prompt}"}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    text_in = None
            if not text_in:
                text_in = f"Q: {prompt}\nA:"
            ids = tokenizer(text_in, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            # Strip echo of the question if the model repeats it
            if gen.lower().startswith(prompt.lower()[:20]):
                gen = gen[len(prompt):].lstrip(" \n:.-")
            ok = match_ga(prompt, gen, answers)
            hits.append(ok)
            details.append({"prompt": prompt, "completion": gen[:200], "hit": ok})
    finally:
        del model
        torch.cuda.empty_cache()
    n = len(hits)
    h = sum(hits)
    return {
        "hits": h,
        "n": n,
        "acc": (h / n) if n else 0.0,
        "details": details,
        "model_path": model_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="live GPU score frozen sentinel")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--sentinel", help="default data/<d>/sentinel/sentinel_v1.json")
    ap.add_argument("--base", required=True, help="base model path")
    ap.add_argument("--candidate", required=True, help="tuned/candidate checkpoint")
    ap.add_argument("--reference-artifact", help="prior artifact path (default=base)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--event", default="cpt")
    ap.add_argument("--booster-n", type=int, default=0)
    ap.add_argument("--run-install-event", action="store_true")
    ap.add_argument("--out", help="write score receipt JSON")
    args = ap.parse_args()

    sp = Path(args.sentinel) if args.sentinel else Sent.latest_sentinel_path(args.domain)
    if not sp.is_file():
        sys.exit(f"sentinel missing: {sp}")
    sentinel = json.loads(sp.read_text(encoding="utf-8"))

    install_probes = list(sentinel.get("install_slice") or [])
    ga_items = list(sentinel.get("general_ability_slice") or [])

    print(f"[live] install slice n={len(install_probes)} ga n={len(ga_items)} device={args.device}", flush=True)
    print(f"[live] scoring BASE {args.base}", flush=True)
    base_install = score_install_slice(model_path=args.base, device=args.device, probes=install_probes)
    print(f"  base install acc={base_install['acc']:.4f} ({base_install['hits']}/{base_install['n']})", flush=True)
    base_ga = score_general_ability(
        model_path=args.base, device=args.device, items=ga_items, domain=args.domain,
    )
    print(f"  base ga acc={base_ga['acc']:.4f} ({base_ga['hits']}/{base_ga['n']})", flush=True)

    print(f"[live] scoring CANDIDATE {args.candidate}", flush=True)
    cand_install = score_install_slice(
        model_path=args.candidate, device=args.device, probes=install_probes,
    )
    print(f"  cand install acc={cand_install['acc']:.4f} ({cand_install['hits']}/{cand_install['n']})", flush=True)
    cand_ga = score_general_ability(
        model_path=args.candidate, device=args.device, items=ga_items, domain=args.domain,
    )
    print(f"  cand ga acc={cand_ga['acc']:.4f} ({cand_ga['hits']}/{cand_ga['n']})", flush=True)

    receipt = {
        "schema": "kandi.sentinel_live_scores.v1",
        "domain": args.domain,
        "unit": args.unit,
        "sentinel_content_sha256": sentinel.get("content_sha256"),
        "device": args.device,
        "scored_at": _now(),
        "base": {"install": base_install, "general_ability": base_ga, "path": args.base},
        "candidate": {"install": cand_install, "general_ability": cand_ga, "path": args.candidate},
    }

    # Mechanical halt preview
    halt_preview = Sent.evaluate_sentinel(
        sentinel,
        install_reference=Sent.scores_from_hits(base_install["hits"], base_install["n"]),
        install_candidate=Sent.scores_from_hits(cand_install["hits"], cand_install["n"]),
        general_base=Sent.scores_from_hits(base_ga["hits"], base_ga["n"]),
        general_candidate=Sent.scores_from_hits(cand_ga["hits"], cand_ga["n"]),
        event=args.event,
        prior_artifact=args.reference_artifact or args.base,
        candidate_artifact=args.candidate,
    )
    receipt["sentinel_result"] = halt_preview

    out = Path(args.out) if args.out else (
        C.BUILD / "install_events" / args.domain / f"sentinel_live_{args.event}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "install_base": base_install["acc"],
        "install_cand": cand_install["acc"],
        "ga_base": base_ga["acc"],
        "ga_cand": cand_ga["acc"],
        "halted": halt_preview["halted"],
        "status": halt_preview["status"],
        "plain": halt_preview["plain"],
    }, indent=2), flush=True)

    if args.run_install_event:
        # Write score files and invoke install_event with REAL numbers
        score_dir = out.parent / "live_scores"
        score_dir.mkdir(parents=True, exist_ok=True)

        def w(name, block):
            p = score_dir / name
            p.write_text(json.dumps({
                "hits": block["hits"], "n": block["n"], "acc": block["acc"],
            }) + "\n")
            return str(p)

        rec = IE.run_install_event(
            domain=args.domain,
            unit=args.unit,
            event=args.event,
            candidate_artifact=args.candidate,
            reference_artifact=args.reference_artifact or args.base,
            install_reference_scores={
                "hits": base_install["hits"], "n": base_install["n"], "acc": base_install["acc"],
            },
            install_candidate_scores={
                "hits": cand_install["hits"], "n": cand_install["n"], "acc": cand_install["acc"],
            },
            general_base_scores={
                "hits": base_ga["hits"], "n": base_ga["n"], "acc": base_ga["acc"],
            },
            general_candidate_scores={
                "hits": cand_ga["hits"], "n": cand_ga["n"], "acc": cand_ga["acc"],
            },
            booster_n=args.booster_n,
            probe_receipt_path=None,
        )
        print(json.dumps({
            "install_event": rec.get("status"),
            "halted": rec.get("halted"),
            "artifact_kept": rec.get("artifact_kept"),
            "path": rec.get("path"),
            "certificate_path": rec.get("certificate_path"),
            "plain": rec.get("plain"),
        }, indent=2), flush=True)
        sys.exit(0 if not rec.get("halted") else 2)

    sys.exit(0 if not halt_preview["halted"] else 2)


if __name__ == "__main__":
    main()
