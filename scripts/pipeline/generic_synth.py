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
from concurrent.futures import ThreadPoolExecutor

from pipeline.domain import Domain
from pipeline.editions import iter_editions
from pipeline.generic_expand import load_all_triples, predicate_templates, source_doc_ids
from pipeline.schema_validate import validate
from pipeline.split import load_split, triple_id

# The per-fact variation instruction (generic — the DOMAIN framing is the config preamble prepended to it).
VARIATION_PROMPT = (
    "Here is ONE verified fact:\n\n    {fact}\n\n"
    "Write {n} DIVERSE, standalone ways to express EXACTLY this one fact. Vary the form — plain statement, "
    "question-and-answer, the reverse direction, a list entry, a conversational mention. Assert NOTHING "
    "beyond this single fact: invent no other entities, relations, dates, or attributes. Output ONE "
    "variation per line, no numbering, no preamble."
)

# Fixed persona roster for conditioning diversity (one provider, one model — not a multi-model ensemble).
# Rotated per triple; temperatures spread over a bounded range. The diversity gate measures whether it helped.
DEFAULT_PERSONAS = (
    "Write as a plain encyclopedia narrator.",
    "Write as a concise quizmaster posing and answering the fact.",
    "Write as a conversational fan explaining the fact to a friend.",
    "Write as a formal reference entry with a reverse-direction phrasing.",
    "Write as a short list-style bullet fact (still one sentence per line).",
)
DEFAULT_TEMPERATURES = (0.70, 0.85, 0.95, 0.80, 0.90)


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
                        max_tokens: int = 1300, max_retries: int = 3,
                        temperature: float = 0.85) -> str:
    """Default generator — a synchronous OpenRouter chat call WITH retry/backoff + an explicit empty-content
    check (mirroring the proven synth_corpus.call_openrouter contract). Returns "" on total failure rather
    than raising, so one transient blip mid-run does NOT abort the whole synth — synth_records counts the
    empty response and warns. Injectable (see synth_records) so callers can swap a concurrent client / a test stub."""
    import httpx
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
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


def _persona_for(index: int, personas: tuple[str, ...] | list[str]) -> str:
    return personas[index % len(personas)] if personas else ""


def _temperature_for(index: int, temperatures: tuple[float, ...] | list[float]) -> float:
    if not temperatures:
        return 0.85
    return float(temperatures[index % len(temperatures)])


def synth_records(triples: list[dict], ptmpls: dict, preamble: str, unit_name: str, n: int,
                  generate_fn=openrouter_generate, out_path=None, domain: str | None = None,
                  concurrency: int = 16,
                  personas: tuple[str, ...] | list[str] | None = DEFAULT_PERSONAS,
                  temperatures: tuple[float, ...] | list[float] | None = DEFAULT_TEMPERATURES,
                  vary_conditioning: bool = True) -> list[dict]:
    """For each triple: build the canonical fact (predicate_templates[pred][0]) + ask generate_fn for N
    diverse faithful variations → 25_hydrated records. `generate_fn(system, user, **kw) -> str` is injectable
    (a stub in tests; the real OpenRouter call by default). Results remain in input order even when calls run
    concurrently. When out_path is given, each completed result is written + flushed to a partial file; the
    canonical file is atomically replaced only after a clean run.

    Phase C: when vary_conditioning is True, rotate a fixed persona roster + temperature schedule per triple
    so diversity comes from conditioning, not a multi-model ensemble. Pass vary_conditioning=False (or empty
    personas) to recover the pre-Phase-C single-prompt path.
    """
    out: list[dict] = []
    counts: dict = defaultdict(int)              # predicate -> n triples seen
    rendered: dict = defaultdict(int)            # predicate -> n canonical-template renders that succeeded
    empty: dict = defaultdict(int)               # predicate -> n empty/failed generator responses
    personas = tuple(personas or ()) if vary_conditioning else ()
    temperatures = tuple(temperatures or (0.85,)) if vary_conditioning else (0.85,)
    prepared: list[tuple[dict, str, str, float, str]] = []
    for idx, t in enumerate(triples):
        p = t["predicate"]
        tmpls = ptmpls.get(p)
        if not tmpls:
            continue
        counts[p] += 1
        kw = {"subject": t["subject"], "object": t["object"], "unit": unit_name}
        kw.update({f"attr_{k}": v for k, v in t["attrs"].items()})
        try:
            fact = tmpls[0].format(**kw)
        except (KeyError, IndexError):
            continue
        rendered[p] += 1
        persona = _persona_for(idx, personas)
        temp = _temperature_for(idx, temperatures)
        head = preamble + "\n\n" if preamble else ""
        if persona:
            head = head + persona + "\n\n"
        system = head + VARIATION_PROMPT.format(fact=fact, n=n)
        prepared.append((t, p, system, temp, persona))

    def generate_records(item: tuple[dict, str, str, float, str]) -> tuple[str, bool, list[dict]]:
        t, p, system, temp, persona = item
        try:
            raw = generate_fn(system, "Write the variations now.", temperature=temp)
        except TypeError:
            # Test stubs that only accept (system, user).
            raw = generate_fn(system, "Write the variations now.")
        triple_recs: list[dict] = []
        for line in (raw or "").splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if len(line) <= 10:
                continue
            rec = {"text": line, "relation_type": p, "direction": "forward",
                   "surface_form": "llm_synth", "source_triple_ids": [triple_id(t)],
                   "entity_ids": [t["subject"], t["object"]], "split": "cpt"}
            if doc_ids := source_doc_ids(t):
                rec["source_doc_ids"] = doc_ids
            # Persona is a first-class optional schema field (relation.schema.json) — stamp
            # before validate so the hydrated layer re-validates clean end-to-end.
            if persona:
                rec["synth_persona"] = persona[:80]
            errors = validate("relation", rec, domain=domain)
            if errors:
                raise ValueError(
                    f"[generic_synth] 25_hydrated relation record fails schema for domain "
                    f"'{domain}': {errors[:3]} | rec={rec}. relation_type '{p}' is not in the manifest's "
                    f"relations.taxonomy / predicates enum (configs/domains/{domain}.yaml) — the hydrated "
                    f"vocabulary must match the manifest the gates classify against.")
            triple_recs.append(rec)
        return p, not (raw or "").strip(), triple_recs

    # `executor.map` preserves input order, so concurrency changes wall time without changing corpus order.
    partial = out_path.with_name(out_path.name + ".partial") if out_path is not None else None
    _fh = open(partial, "w", encoding="utf-8") if partial is not None else None
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for completed, (p, was_empty, triple_recs) in enumerate(
                    executor.map(generate_records, prepared), start=1):
                if was_empty:
                    empty[p] += 1
                out.extend(triple_recs)
                if _fh is not None:
                    for rec in triple_recs:
                        _fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    _fh.flush()
                if completed % 25 == 0 or completed == len(prepared):
                    print(f"  …{completed}/{len(prepared)} synth calls; {len(out)} surfaces", flush=True)
    finally:
        if _fh is not None:
            _fh.close()
    print(f"  synth calls: {len(prepared)} attempted, {sum(empty.values())} empty/failed; "
          f"{len(out)} surfaces", flush=True)
    # The canonical template (tmpls[0]) seeds the LLM — if it renders for ZERO triples of a predicate that
    # predicate gets NO diversity (a placeholder typo); empty generator responses also silently shrink it.
    for p, seen in counts.items():
        if seen and not rendered[p]:
            print(f"  WARNING: predicate '{p}' canonical template rendered 0/{seen} triples — likely a "
                  f"placeholder typo; this predicate gets NO LLM-synth diversity", file=sys.stderr)
        elif empty[p]:
            print(f"  WARNING: predicate '{p}' had {empty[p]}/{rendered[p]} empty/failed generator responses",
                  file=sys.stderr)
    uncovered = sorted({t["predicate"] for t in triples} - set(ptmpls))   # no template → no diversity (mirror generic_expand)
    if uncovered:
        print(f"  WARNING: predicates with NO template (no LLM-synth diversity, silently skipped): {uncovered}",
              file=sys.stderr)
    # A mostly-empty diversity layer (an API key/quota/outage failure for the whole run) must FAIL the stage,
    # not ship green — else the pipeline tokenizes an empty install layer and reports success.
    total_rendered, total_empty = sum(rendered.values()), sum(empty.values())
    if out_path is not None:
        # Key the abort on `out` (records produced), NOT on total_empty/total_rendered: a TOTAL render
        # collapse (every predicate unmapped / every format() failed) makes total_rendered==0, which would
        # short-circuit the ratio check to False and then atomically promote an EMPTY file over a prior good
        # corpus. Zero records is itself the failure — never write/overwrite with nothing.
        if not out:
            # Distinguish the BENIGN source gap (every predicate uncovered → counts empty, nothing was
            # attempted) from a genuine failure (renders/calls WERE attempted and produced nothing). The gap
            # skips THIS edition and keeps going (a library fn must not sys.exit a multi-edition run); the
            # failure aborts the stage. Either way, never promote the empty .partial over a prior good corpus.
            if not counts:
                print(f"  WARNING: [generic_synth] 0 records — every triple uses a predicate with no template "
                      f"(uncovered, a source gap); this edition gets no LLM-synth diversity.", file=sys.stderr)
                if partial is not None:
                    partial.unlink(missing_ok=True)     # clean up the empty tombstone — nothing was produced
                return out
            raise SystemExit(f"[generic_synth] produced 0 diversity records from {sum(counts.values())} attempted "
                             f"facts — every canonical render failed (a predicate/placeholder typo in the domain "
                             f"templates) or every generator call was empty (API key/quota/outage). Refusing to "
                             f"write an empty corpus or overwrite a prior good one.")
        if total_rendered and total_empty / total_rendered > 0.5:
            raise SystemExit(f"[generic_synth] {total_empty}/{total_rendered} facts got empty/failed generator "
                             f"responses (>50%) — the diversity layer is mostly empty; aborting the stage "
                             f"(check the API key / quota / outage) rather than shipping an empty corpus green.")
        if partial is not None:
            os.replace(partial, out_path)    # atomic promote — reached ONLY for a non-empty, mostly-rendered run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions", help="restrict to 'unit,…' (default: all on disk)")
    ap.add_argument("--variations", type=int, default=10, help="LLM variations per triple")
    ap.add_argument("--no-persona-variation", action="store_true",
                    help="disable Phase C persona+temperature rotation (single fixed prompt)")
    ap.add_argument("--dry-run", action="store_true", help="print the fact + the prompt for the first triple, no API call")
    args = ap.parse_args()
    tmpl = Domain(args.domain).templates or {}
    ptmpls = predicate_templates(args.domain)
    preamble = (tmpl.get("synth_templates") or {}).get("system_preamble", "")
    if not ptmpls:
        raise SystemExit(f"domain '{args.domain}' has no hydrate_templates.predicate_templates — author them in "
                         f"configs/templates/{args.domain}.yaml (or check the key spelling/nesting if the file exists).")
    for ed in iter_editions(args.domain, only=args.editions, with_raw=False):   # consumes 15_structured, not 00_raw
        triples = load_all_triples(ed)
        if not triples:
            continue
        sp = load_split(ed.dir)
        if sp:
            before = len(triples)
            triples = [t for t in triples if triple_id(t) in sp["train_ids"]]
            print(f"  {ed.slug}: split {sp['version']} — synthesizing over {len(triples)}/{before} "
                  f"train triples ({before - len(triples)} holdout EXCLUDED, never in the LLM's context)")
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
        recs = synth_records(
            triples, ptmpls, preamble, unit_name, args.variations,
            out_path=out_dir / "relations_llm.jsonl", domain=args.domain,
            vary_conditioning=not args.no_persona_variation,
        )
        print(f"  {ed.slug}: wrote {len(recs)} LLM-synth lines from {len(triples)} triples"
              f"{'' if args.no_persona_variation else ' (persona+temp variation ON)'}")


if __name__ == "__main__":
    main()
