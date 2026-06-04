# qgre-cpt — Data Pipeline Audit & Redesign Notes

_Audit date: 2026-06-03. Scope: completed hydrated EDC Las Vegas editions (2011–2019, 2023) vs the
hand-built GOLD 2026. Method: 6-dimension read-only agent audit (factual/web, faithfulness, balance,
structured-layer, temporal coverage, garbage) + direct verification of the two headline claims._

> **One-line verdict:** the corpus is *surface-clean and faithful to its sources*, but **distribution-skewed
> and source-incomplete**. It is NOT fabricated. The fixable defects are a mix of (a) genuine schema/config
> bugs (ours) and (b) source-data gaps (not a code bug). Do **not** train on the VC1 editions as the
> knowledge spine until the schema + balance issues are addressed.

---

## 1. Verified findings (numbers I confirmed directly, not just agent-reported)

| Dimension | Verdict | Evidence |
|---|---|---|
| Garbage / well-formedness | ✅ clean | 0% malformed JSON over 534k lines, 0% non-English, 100% surface-unique |
| Edition isolation | ✅ clean | 2018 file references only `edc_las_vegas_2018` ids |
| Factual accuracy (web) | ⚠ ~87% | 7 true / 1 false / 5 unverifiable of 13 checked. Confirmed FALSE: **Caspa → stereoBLOOM (2018)** (he played bassPOD) |
| Faithfulness to triples | ⚠ loose | ~60% of sampled lines abbreviate/substitute the canonical entity (`"main stage"` for Kinetic Field, `"Edc Las Vegas"` for "Electric Daisy Carnival Las Vegas"). **Loose naming, not fabrication.** |
| Relation balance | ❌ skewed | VC1 editions: 98% `artist_to_stage`, 8 relation types. GOLD: 82%, **12 types** |
| Per-pair exposure | ❌ concentrated | VC1 240–347× phrasings/pair (2023 max pair = 703×); GOLD 151× (max 255×) |
| Temporal coverage | ❌ absent in VC1 | `set_time` lines: **0** in 2018/2019/2023, **1,571** in GOLD 2026 |
| `plays_at` → stage | ❌ 25–40% point to festival/edition | 2018 25%, 2023 40%, 2026(gold) 35% |

## 2. Root cause — corrected (the agents over-attributed; the code tells the truth)

**(a) SOURCE-DATA GAPS — not a code bug:**
- **set-times absent for old editions.** `settimes.rs:59` reads `edition_dir/set_times.jsonl`. That file
  exists **only for 2026** (acquired specifically in task #42). 2018/2019/2023 never had set-times source →
  0 `set_time` lines, and the cascade kills `day_of_week_map` + most temporal scaffolding. The VC1 campaign
  was *framed* as "adding set-times" for editions that have no set-times source — a **planning/instruction
  miss**, not a pipeline failure.
- **`plays_at → festival` (25–40%, even in GOLD).** Source granularity: for many artists the scrape only
  captured lineup membership, not the stage. `triples.rs:85` faithfully resolves whatever object exists.
  These lines are **true but vague** ("Armin played EDC LV 2018"), not wrong.

**(b) GENUINE CODE / SCHEMA BUGS — ours to fix:**
1. **Lineup-membership mislabeled as `artist_to_stage` with stage = the festival.** This pollutes the stage
   signal (top "stage" in 2018 is `edc_las_vegas`, 25,856 lines). Fix: emit a distinct `in_lineup` relation
   when the stage is unknown; never let `festival`/`edition`/`art_car` masquerade as a stage.
2. **No per-pair redundancy cap.** Hydration runs the coverage loop to 240–347× on the same pair while
   uncovered relations get 0. Fix: cap exposure per (subject,object,relation) and reallocate budget to
   under-covered relations (target a balanced distribution like GOLD's 12-type profile).
3. **~12% genuinely-wrong stage assignments** (Caspa→stereoBLOOM class). Trace: source error vs extraction
   error. Fix: a genre/stage-compatibility sanity gate before hydration.
4. **Cross-edition id confusion** in structured layer: a 2023 triple references `edition:edc_las_vegas_2026`;
   2026 references `edition:2025`. Dangling refs (missing edition entities). Fix: validate every
   subject/object id exists in `entities.jsonl` and matches the edition.

**(c) AUDIT-AGENT INSTRUCTION FIX (process):** the audit agents declared "set_times not wired = pipeline
bug" without reading `settimes.rs`. The audit-agent prompt must require **root-cause-by-reading-the-code
before asserting a bug**, and must distinguish *source-gap* from *code-defect*.

## 3. Is this an issue for knowledge installation?

- **Fabrication risk: LOW.** Hydration is faithful to triples; triples are faithful to source.
- **Wrong-fact risk: SMALL (~12% of verifiable stage assignments).** These would mis-install (model learns a
  real artist on the wrong stage). Minority, source-traceable.
- **Main risk: INCOMPLETENESS + SKEW + VAGUENESS.** The model would install a lopsided picture — strong on
  "artist↔stage(ish)", zero on timing, weak on genre/capacity — plus festival/stage conflation from the
  vague lines. Redundancy itself is fine (it's the CPT installation mechanism); the problem is it's
  *concentrated on one relation* while others get nothing.

## 4. Redesign requirements (for the qgre-cpt domain-agnostic agent-loop pipeline)

The pipeline is currently EDC-festival-specific (festival-factory is a festival data model). The next
version is a **domain-agnostic autonomous agent loop**: ingest docs (UI/scrape) → clean → extract entities
+ triples → generate variations → hydrate → gate. The quality gates below would have *caught every defect
above* and must be first-class in the new design:

- **Relation-balance gate:** reject/rebalance a corpus that is >X% one relation or missing relation types
  the domain schema declares.
- **Per-fact exposure cap:** bound phrasings/fact; reallocate to under-covered facts.
- **Faithfulness gate:** every source-triple entity must appear (normalized) in the hydrated line; no
  role-descriptor or abbreviation substitution for canonical names.
- **Source-granularity honesty:** when stage/time is unknown, emit the *coarser true relation*
  (`in_lineup`), never a fabricated/festival-as-stage stand-in. Flag coverage so we know what's missing.
- **Entity-type / id integrity:** every triple id resolves in `entities.jsonl`; predicate→object-type
  contracts enforced (`plays_at` → stage|set|venue, never festival|edition).
- **Source-vs-pipeline provenance:** the gate must report whether a missing relation is a *source gap* or a
  *generation failure* (the distinction this audit had to recover by hand).

## 5. Status of the running hydration

The 4 in-flight workers (2021/2022/2024/2025) are producing the **same shape** as 2018/2019/2023 (98%
`artist_to_stage`, 0 set-times — no set_times source exists for them either). Finishing them yields more
imbalanced-but-not-wrong lineup data. Decision pending (see chat): let finish & keep (don't train on as
spine) vs stop/discard.
