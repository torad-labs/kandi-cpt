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
import sys
import time
from collections import defaultdict

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
                        max_tokens: int = 1300, max_retries: int = 3) -> str:
    """Default generator — a synchronous OpenRouter chat call WITH retry/backoff + an explicit empty-content
    check (mirroring the proven synth_corpus.call_openrouter contract). Returns "" on total failure rather
    than raising, so one transient blip mid-run does NOT abort the whole synth — synth_records counts the
    empty response and warns. Injectable (see synth_records) so callers can swap a concurrent client / a test stub."""
    import httpx
    body = {"model": model, "temperature": 0.85, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {_openrouter_key()}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=120)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                content = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content
                last_err = "empty content (HTTP 200)"
        except Exception as e:                          # network / JSON-shape error → retried, then a counted sentinel
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt)
    print(f"  [generic_synth] generation failed after {max_retries} attempts: {last_err}", file=sys.stderr)
    return ""


def synth_records(triples: list[dict], ptmpls: dict, preamble: str, unit_name: str, n: int,
                  generate_fn=openrouter_generate, out_path=None) -> list[dict]:
    """For each triple: build the canonical fact (predicate_templates[pred][0]) + ask generate_fn for N
    diverse faithful variations → 25_hydrated records. `generate_fn(system, user) -> str` is injectable
    (a stub in tests; the real OpenRouter call by default). Deterministic given a deterministic generate_fn.
    When out_path is given, each triple's records are written + flushed incrementally so completed work
    survives a crash; the full list is still returned. NOTE: resume (skipping already-done triples) is not
    implemented — for production-scale runs use the concurrent, per-file-idempotent synth_corpus.py pattern."""
    out: list[dict] = []
    counts: dict = defaultdict(int)              # predicate -> n triples seen
    rendered: dict = defaultdict(int)            # predicate -> n canonical-template renders that succeeded
    empty: dict = defaultdict(int)               # predicate -> n empty/failed generator responses
    _fh = open(out_path, "w", encoding="utf-8") if out_path is not None else None
    try:
        for t in triples:
            p = t["predicate"]
            tmpls = ptmpls.get(p)
            if not tmpls:
                continue
            i = counts[p]
            counts[p] += 1
            kw = {"subject": t["subject"], "object": t["object"], "unit": unit_name}
            kw.update({f"attr_{k}": v for k, v in t["attrs"].items()})
            try:
                fact = tmpls[0].format(**kw)
            except (KeyError, IndexError):
                continue                         # canonical render failed for this triple — tallied as 0-render below
            rendered[p] += 1
            system = (preamble + "\n\n" if preamble else "") + VARIATION_PROMPT.format(fact=fact, n=n)
            raw = generate_fn(system, "Write the variations now.")
            if not (raw or "").strip():
                empty[p] += 1                    # an empty/failed response = no diversity installed for this fact
            triple_recs: list[dict] = []
            for line in (raw or "").splitlines():
                line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()   # strip list markers only (not leading years)
                if len(line) > 10:
                    triple_recs.append({"text": line, "relation_type": p, "direction": "forward",
                                        "surface_form": "llm_synth", "source_triple_ids": [f"{p}:{i}"],
                                        "entity_ids": [t["subject"], t["object"]], "split": "cpt"})
            out.extend(triple_recs)
            if _fh is not None:
                for rec in triple_recs:
                    _fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _fh.flush()
    finally:
        if _fh is not None:
            _fh.close()
    # The canonical template (tmpls[0]) seeds the LLM — if it renders for ZERO triples of a predicate that
    # predicate gets NO diversity (a placeholder typo); empty generator responses also silently shrink it.
    for p, seen in counts.items():
        if seen and not rendered[p]:
            print(f"  WARNING: predicate '{p}' canonical template rendered 0/{seen} triples — likely a "
                  f"placeholder typo; this predicate gets NO LLM-synth diversity", file=sys.stderr)
        elif empty[p]:
            print(f"  WARNING: predicate '{p}' had {empty[p]}/{rendered[p]} empty/failed generator responses",
                  file=sys.stderr)
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
        raise SystemExit(f"domain '{args.domain}' has no hydrate_templates.predicate_templates — author them in "
                         f"configs/templates/{args.domain}.yaml (or check the key spelling/nesting if the file exists).")
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
        out_dir = ed.dir / "25_hydrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        recs = synth_records(triples, ptmpls, preamble, unit_name, args.variations,
                             out_path=out_dir / "relations_llm.jsonl")
        print(f"  {ed.slug}: wrote {len(recs)} LLM-synth lines")


if __name__ == "__main__":
    main()
