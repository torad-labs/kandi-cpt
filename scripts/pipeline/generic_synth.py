#!/usr/bin/env python3
"""Generic LLM-synth driver — the domain-agnostic DIVERSITY layer (the LLM analog of `generic_expand`'s
deterministic floor).

`generic_expand` renders fixed templates over triples — a faithful FLOOR, but fixed phrasings concentrate
gradient on the scaffold, not the fact (~34% install, #760). The install comes from *varied* surface forms.
This driver provides that generically: for each verified triple it asks the generator model for N diverse,
standalone, faithful ways to express EXACTLY that one fact — no entity pairing, no invention (the per-fact
isolation that fixed the EntiGraph confabulation). Domain-agnostic: the per-fact statement is built from the
domain's `hydrate_templates.predicate_templates`, and the domain's framing comes from
`synth_templates.system_preamble` in configs/templates/<domain>.yaml. The festival `synth_corpus.py` is
EDC's specialized, concurrent instance; THIS is the reusable skeleton a new domain points its synth step at.

    PYTHONPATH=scripts python scripts/pipeline/generic_synth.py --domain <d> [--editions ...] [--variations N] [--dry-run]
Output: <edition>/25_hydrated/relations_llm.jsonl (same shape as generic_expand, surface_form=llm_synth).
"""
from __future__ import annotations

import argparse
import json
import os
import re

from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples, predicate_templates

# The per-fact variation instruction (generic — the DOMAIN framing is the config preamble prepended to it).
VARIATION_PROMPT = (
    "Here is ONE verified fact:\n\n    {fact}\n\n"
    "Write {n} DIVERSE, standalone ways to express EXACTLY this one fact. Vary the form — plain statement, "
    "question-and-answer, the reverse direction, a list entry, a conversational mention. Assert NOTHING "
    "beyond this single fact: invent no other entities, relations, dates, or attributes. Output ONE "
    "variation per line, no numbering, no preamble."
)


def _openrouter_key() -> str:
    """OpenRouter key — env var first, then libsecret (service=openrouter), mirroring synth_corpus.py."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    import subprocess
    try:
        out = subprocess.run(
            ["/usr/bin/python3", "-c",
             "import gi; gi.require_version('Secret','1'); from gi.repository import Secret; "
             "s=Secret.Schema.new('ai.kandibuddy.OpenRouter', Secret.SchemaFlags.NONE, "
             "{'service': Secret.SchemaAttributeType.STRING, 'key_type': Secret.SchemaAttributeType.STRING}); "
             "print(Secret.password_lookup_sync(s, {'service':'openrouter','key_type':'api_key'}, None) or '')"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    raise SystemExit("OPENROUTER_API_KEY not set (env or libsecret service=openrouter)")


def openrouter_generate(system: str, user: str, model: str = "google/gemma-4-31b-it",
                        max_tokens: int = 1300) -> str:
    """Default generator — one synchronous OpenRouter chat call. Injectable (see synth_records) so callers
    can swap a concurrent client (like synth_corpus's async pool) or a stub for tests."""
    import httpx
    resp = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {_openrouter_key()}", "Content-Type": "application/json"},
                      json={"model": model, "temperature": 0.85, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                      timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def synth_records(triples: list[dict], ptmpls: dict, preamble: str, unit_name: str, n: int,
                  generate_fn=openrouter_generate) -> list[dict]:
    """For each triple: build the canonical fact (predicate_templates[pred][0]) + ask generate_fn for N
    diverse faithful variations → 25_hydrated records. `generate_fn(system, user) -> str` is injectable
    (a stub in tests; the real OpenRouter call by default). Deterministic given a deterministic generate_fn."""
    out: list[dict] = []
    counts: dict[str, int] = {}
    for t in triples:
        p = t["predicate"]
        tmpls = ptmpls.get(p)
        if not tmpls:
            continue
        i = counts.get(p, 0)
        counts[p] = i + 1
        kw = {"subject": t["subject"], "object": t["object"], "unit": unit_name}
        kw.update({f"attr_{k}": v for k, v in t["attrs"].items()})
        try:
            fact = tmpls[0].format(**kw)
        except (KeyError, IndexError):
            continue
        system = (preamble + "\n\n" if preamble else "") + VARIATION_PROMPT.format(fact=fact, n=n)
        raw = generate_fn(system, "Write the variations now.")
        for line in (raw or "").splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()   # strip list markers only (not leading years)
            if len(line) > 10:
                out.append({"text": line, "relation_type": p, "direction": "forward",
                            "surface_form": "llm_synth", "source_triple_ids": [f"{p}:{i}"],
                            "entity_ids": [t["subject"], t["object"]], "split": "cpt"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions", help="restrict to 'unit,…' (default: all on disk)")
    ap.add_argument("--variations", type=int, default=10, help="LLM variations per triple")
    ap.add_argument("--dry-run", action="store_true", help="print the fact + the prompt for the first triple, no API call")
    args = ap.parse_args()
    tmpl = Domain(args.domain).templates or {}
    ptmpls = predicate_templates(args.domain)
    preamble = (tmpl.get("synth_templates") or {}).get("system_preamble", "")
    if not ptmpls:
        raise SystemExit(f"domain '{args.domain}' has no hydrate_templates.predicate_templates — author them first.")
    for ed in iter_editions(args.domain, only=args.editions):
        triples = load_all_triples(ed)
        if not triples:
            continue
        mp = ed.dir / "festival_meta.json"
        unit_name = (json.loads(mp.read_text()).get("name") if mp.exists() else None) or ed.slug
        if args.dry_run:
            t0 = next((t for t in triples if t["predicate"] in ptmpls), None)
            if t0:
                fact = ptmpls[t0["predicate"]][0].format(subject=t0["subject"], object=t0["object"], unit=unit_name,
                                                          **{f"attr_{k}": v for k, v in t0["attrs"].items()})
                print(f"  {ed.slug}: {len(triples)} triples; sample fact -> '{fact}'  (×{args.variations} variations)")
            continue
        recs = synth_records(triples, ptmpls, preamble, unit_name, args.variations)
        out_dir = ed.dir / "25_hydrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "relations_llm.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        print(f"  {ed.slug}: wrote {len(recs)} LLM-synth lines")


if __name__ == "__main__":
    main()
