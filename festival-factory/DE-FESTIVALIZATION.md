# festival-factory — de-festivalization assessment (F6)

**Verdict: the Rust factory is correctly FESTIVAL-DOMAIN-SPECIFIC and should NOT be de-festivalized.**
A non-festival domain does not use it — it uses the generic Python hydrate path
(`scripts/pipeline/generic_expand.py`). Rewriting 2772 LOC of Rust to model arbitrary domains would
replicate `generic_expand` at ~10× the cost and impose festival concepts (stages/days/venue/set-times) on
domains that have none.

## Why it is festival-specific by TYPE, not just by data

`config.rs` deserializes `FestivalConfig` from `festival_meta.json`. The docstring claims "all
festival-specific knowledge lives in data, not code" — true *across festival editions* (any EDC city/year
loads), but the **Rust types themselves model the festival ontology**:

- `FestivalConfig { festival_slug, venue, dates, n_main_stages, days: Map<Day,DayInfo>, stages: Map<slug,Stage>, confusable_groups }`
- `Stage { camel, spaced, genre, brand, rank, vibe }` · `SetTime { artist, stage, day, start, end }` · `ConfusableGroup { members, stage, day }`

A pokemon-lore unit (a region, with character→region→organization triples and no stages/days/venue/
set-times) cannot be expressed as a `FestivalConfig`. The factory is data-driven across **editions of one
domain shape**, not across **domain shapes**. That is the boundary.

## The de-festivalized path for a NEW domain (already shipped)

| Festival-factory stage (Rust) | New-domain replacement | Status |
|---|---|---|
| `hydrate` (surface.rs/synth.rs/disjoint.rs — artist→stage→day surface forms) | `scripts/pipeline/generic_expand.py` — per-predicate templates over ALL triples (the deterministic floor) | **shipped (F1)** |
| `triples.rs` (relational completeness over the festival taxonomy) | the manifest's `predicates` + `schema_validate.py` (config-driven, Part A) | shipped |
| `tokenize.rs` | `scripts/tokenize_corpus.py` (already domain-agnostic) | existing |
| LLM diversity (synth.rs `openrouter`) | `scripts/pipeline/generic_synth.py` — per-fact LLM variation (reuses `predicate_templates` + `synth_templates.system_preamble`) | **shipped (F5)** |

So a new domain points its `hydrate` stage at `generic_expand` and authors `hydrate_templates.predicate_templates`
(see `configs/templates/_TEMPLATE.yaml` + the qgre-agent `domains/_TEMPLATE.md`). The Rust factory stays
EDC's optimized, faithfulness-gated festival hydrator.

## The LLM-diversity layer — now a generic skeleton too

`generic_expand` is the deterministic FLOOR (faithful by construction). The literature (#760,
docs/conductor/12 §) is clear that **varied LLM surface forms install, fixed templates alone are a weak
floor (~34%)**. EDC gets diversity from the Rust `synth` + the Python `synth_corpus.py` — both
festival-specific. The generic analog is now **`scripts/pipeline/generic_synth.py`**: for each verified
triple it states the fact (from `predicate_templates[pred][0]`) and asks the generator model for N diverse,
faithful paraphrases — the per-fact isolation that avoids the EntiGraph confabulation. A new domain authors
only `synth_templates.system_preamble` (the framing); the generator is **injectable** (the skeleton is
tested without an API call). What stays per-domain/per-run is the *scaling* (the festival `synth_corpus.py`
is the concurrent reference) and any domain-specific faithfulness audit — i.e. authoring, not engine work.

## Bleed to fix at the config level (not in the Rust)

`configs/domains/pokemon-lore.yaml`'s `hydrate` stage still points at the Rust factory
(`bash scripts/rehydrate_vc1.sh`) and carries a `⚠ verify de-festivalized` flag. The resolution — when that
draft domain is built for real — is to point it at `generic_expand` (which now exists), not to touch the
Rust. That is a one-line manifest edit the domain author makes, not engine work.

## Bottom line

- **Generic ENGINE: de-festivalized.** Config-driven ontology/validation (Part A), config templates (the
  seams), the generic hydrate floor (`generic_expand`, F1), the authoring scaffolds (F3).
- **Festival-factory (Rust), `synth_corpus.py`, `build_sft.py`'s absent-edition/fake-city negatives, the
  `extract_*_lineup.py` acquirers, the EDC evals: correctly festival-domain-specific authoring.** A new
  domain authors its own (the master-plan design: "a domain swap is real authoring, not a `--domain` flag"),
  reusing the generic engine + scaffolds. They are not generic machinery bleeding festival.
- **Generic skeletons shipped:** the deterministic hydrate floor (`generic_expand`) AND the LLM-diversity
  layer (`generic_synth`). What remains per domain is *authoring* (ontology + templates + preamble) and
  *scaling* — the master-plan "real authoring," not engine de-festivalization.
