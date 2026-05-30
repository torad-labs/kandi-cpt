# schema/ — the structured-data contract

Festival-agnostic JSON Schemas (Draft 2020-12) that define the target format for the
**raw → structured** conversion. One contract for every edition. Per substrate spec §2.3,
these are *enforced via JSON schema during ingestion* — they are the template you fill when
turning cleaned text into model-ready knowledge.

## Flow

```
data/<festival>/<city>/<year>/10_clean/   (cleaned .md, source of truth)
        │  extract (LLM-assisted per spec §9.2/§9.3, validated against these schemas)
        ▼
data/<festival>/<city>/<year>/15_structured/
        ├── entities.jsonl    ← validates against entity.schema.json     (§2)
        ├── triples.jsonl     ← validates against triple.schema.json     (§3)
        └── discourse.jsonl   ← validates against discourse.schema.json  (§4)
```

One JSON object per line (JSONL). Each line is one record.

## Files

| Schema | Spec | What it constrains |
|---|---|---|
| `entity.schema.json` | §2 | id pattern `<festival_slug>:<entity_type>:<slug>`, 19-type enum, per-type required attrs (9 types fully specified via conditionals; others open — extend per the §2.3 pattern) |
| `triple.schema.json` | §3 | `(subject_id, predicate, object, attrs, provenance)`; `predicate` = the §3.2 catalog + §3.3 cross-festival predicates (closed enum) |
| `discourse.schema.json` | §4 | anchored soft observations; `topic` (§5.2 enum), `stance` enum, `lifted_claims` → triples |
| `metadata.schema.json` | §7 | universal record envelope (§7.1) + `domain_class` trust enum (§7.2); also exports shared `$defs` (`source`, `provenance`, `confidence`, `synthesisLineage`, `topic`) reused by the others via `$ref` |
| `question_shape.schema.json` | §5.1 | the ~5,500 question shapes that define "knowing a festival" — topic/subtopic, answer_template, required_primitives, required_facts, applicability, difficulty_tier, evaluator_type |
| `festival_class.schema.json` | §12.1 | the festival-class vector (setting, scale, camping, …) — drives class-conditional norms (§12.2) + `cross_festival_transfer` (§12.3) |
| `synth_sample.schema.json` | §7.1 + §10 | a generated training surface — the 8-shape catalog (§10.1), audience variants (§10.2), reasoning-chain span tags (§10.4) |
| `primitive.schema.json` | §6.1 | a navigation/reasoning primitive definition (open key set — expands from catalog clustering, §15 step 3) |

`examples/` holds one filled, conforming record per type (the spec's own sample entities —
`kinetic_field`, the `causes_density_at` triple, a Reddit Bass Pod observation, a camping
question shape, an EDC-LV class vector). The §7.2 trust-weight table lives at
`configs/trust_weights.json` (keyed by `domain_class`).

## Notes
- **Cross-edition entities** (artists, venues — §2.2 persist across editions) live in
  `data/_shared/entities_global/` but use the **same** `entity.schema.json`.
- **`$ref` resolution:** the entity/triple/discourse schemas `$ref` shared defs in
  `metadata.schema.json` by its `$id`. To validate, load all four into a registry (see the
  validation snippet the build used) — they're not standalone by design (DRY).
- **Extension:** the 10 entity types §2.3 doesn't fully detail (`day`, `gate`, `lodging_option`,
  `vendor`, `medical`, `landmark`, `policy`, `service`, `community`, `tradition`) are valid in
  the enum with an open `attributes` object. Add a conditional `$defs/attrs_<type>` block when
  ingestion first needs that type — mirrors the spec's "follow the same pattern."
- **Validate before writing to `15_structured/`** — a record that fails its schema is a gap to
  fix, not data to commit.

## Resolved spec inconsistencies (defaults applied — flip if you disagree)
- **`service_constraint`** — narrated in §4.2 but missing from the §3.2 list → **added** to the
  triple predicate enum.
- **Global-entity IDs** — §2.2 (global) vs §2.1 (per-edition) → cross-edition artist/venue use the
  reserved slug **`global`** (`global:artist:tiesto`) under `data/_shared/entities_global/`.
- **Topic enum** — §5.2 long vs §5.1/§7.1 short → **long form is canonical** (`camping_and_lodging`).
