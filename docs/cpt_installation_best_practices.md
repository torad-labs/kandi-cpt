# CPT Knowledge Installation: Best Practices (Working Reference)

A living reference for installing retrievable factual knowledge into a small model
via continued pre-training (CPT). Collected from the Kandi / festival-specialist work.
Each entry is structured as Finding, Mechanism, Evidence, Practice so it stays
scannable and extendable. Add new findings in the same shape.

> Operationalized in the tooling: `scripts/data_audit.py` checks the data-side items
> (density, balance, coverage, disjoint co-occurrence, SFT refusal balance, phrasing
> hygiene, base-model latent separation) and the `installation` checklist dimension
> maps each section below to a ✓ / ⚠ / ✗ the audit page surfaces. The operating room
> (`scripts/dissect/kandi_or.html`) separates the SUCCESSFUL-RUN verdict (behavioral
> gate: answer / abstain / don't fabricate) from the SUCCESSFUL-INSTALLATION verdict
> (density + geometry separation + extractability). A run can pass the gate while the
> installation is still weak — they are different questions.

Validated baseline for this reference: CPT installs retrievable post-cutoff knowledge
into a ~2B Gemma 4 (E2B-class) at NVFP4/FP4 training precision (brain concept #777).
Loss 6.84 to 0.26 over 200 steps, WSD schedule, peak 11.2 GB on the eager path.
Everything below is calibrated on that setup and small corpora (sub-1M tokens).

---

## I. The core principle: density drives extractability

**Finding.** Recall tracks relational density. Thinly-covered relations are memorized
but not retrievable; densely-covered relations install and answer correctly. The
festival V2/V3 imbalance came from hand-wiring only the subset of relationships we had
patience for, which produced uneven coverage and thin relations that failed (brain
concept #778).

**Mechanism.** Gradient signal for a fact scales with how often its distinguishing
tokens appear. A relation seen once does not accumulate enough gradient to install.

**Evidence.** Allen-Zhu and Li, Physics of Language Models Part 3.1 (arXiv:2309.14316):
without augmentation into many surface forms at pretraining, knowledge is memorized
but not extractable, near 0% QA accuracy regardless of later fine-tuning. Part 3.2
(arXiv:2309.14402): rewrite the corpus with an auxiliary model and mix QA-shaped lines
into pretraining. Sharp sub-lever: paraphrases are only effective when the information
order matches the QA style.

**Practice.** An agent reads the raw data (RAG as an access layer only), enumerates
every relationship across the schema axes itself, and generates uniform-volume
variation across all relationships, grounded one triple at a time. The triple is the
unit of coverage (checkable, deduppable, faithfulness-verifiable). Kill the hand-wired
hydration; keep the triple. Feed density targets from the eval.

---

## II. Relationship coverage: both directions, and base vs derived

**Finding A (reversal).** "A is B" does not give you "B is A" for free. A relation must
be installed from every entity as both source and target.

**Evidence.** Reversal curse, Berglund et al. 2023 (arXiv:2309.12288). Permutation
training Guo et al. 2024 (arXiv:2403.00758) and reverse training Golovneva et al. 2024
address it by training the relation in all directions.

**Practice.** The generator emits each relation with every involved entity as the
query subject, not just the canonical direction.

**Finding B (base vs derived).** Atomic relations with small cardinality (stage
distances, set times, genres) get memorized densely. Combinatorial relations
(conflicts at O(artists squared), full schedules) must not be enumerated or the corpus
explodes.

**Practice.** Install the primitives densely and teach the operation. Do not enumerate
derived relations; give the model the base facts plus the rule that composes them.

---

## III. What NOT to balance: equalization is the wrong objective

**Finding.** Balancing lines-per-fact to a global floor made the model worse. V5
(every fact downsampled to an 83-line floor) regressed against V4:

| metric (sharpness increasing) | V4 | V5 | delta |
|---|---|---|---|
| question-eval (chat, coarse) | 17/19 | 16/19 | -1 |
| geometry AUC | 0.577 | 0.566 | -0.011 |
| cloze stage@1 (training surface) | 0.711 | 0.506 | -0.205 |

The three lenses agree in increasing sharpness. The question-eval barely moved because
it is coarse and near-saturated. The cloze on the training surface shows the real
damage: equalization cost about 20 points of stage install.

**Mechanism.** Downsampling stripped exposure from facts that were installing fine, and
gave the confusable pair no extra distinctive signal. Uniform gradient mass is the
wrong objective. Hard-negative pairs need more exposure, not equal.

**Practice.** The right target is floor plus confusion-boost, two layers:
- Floor: every entity gets at least N exposures so nothing is starved (fixes thin
  relations). Count keyword and entity frequency, not lines per fact, because gradient
  tracks how often the distinguishing tokens actually appear.
- Boost: pairs the model confuses get extra exposure above the floor, allocated
  inversely to measured separation (see X), until they separate.

Do not balance toward equal, and do not balance toward the data distribution. Source
frequency tells you what is common, not what the model confuses; confusability is the
property that matters and it is a model property you measure, not a data property.

---

## IV. Negation installs weakly and needs special handling

**Finding.** A bare negation barely installs. It has no affirmative target to move
toward, and it competes against dense positives. "Adriatique is not on Cosmic Meadow"
loses to "Adriatique plays Neon Garden" because Neon Garden and Cosmic Meadow appear
many times while the single negation does not.

**Mechanism.** Same density law: gradient tracks frequency, so an under-supported
negation is worse off than an under-supported positive because something dense pulls
the opposite way.

**Evidence.** Negation is a long-documented LLM weakness. Kassner and Schutze 2020:
pretrained models cannot distinguish negated from non-negated fill-in-the-blank ("birds
can talk but cannot fly"). Models still rely on surface lexical cues for polarity
(Ettinger 2020; Hosseini et al. 2021). Negation Neglect (arXiv:2605.13829, 2026) ran
this exact pretraining experiment, got mixed-to-negative results at small scale, and
asks for a larger-scale test; their hypothesis is the density argument in other words
(models have not seen negation-annotated documents at volume, so have not meta-learned
to update from them).

**Practice (three fixes, they stack).**
1. Unlikelihood training: Hosseini et al. 2021 (arXiv:2105.03519) reduced negated-LAMA
   top-1 error to 4% by adding an unlikelihood term.
2. Counterfactual contrastive pairs: Yang et al. (arXiv:2410.20710), generate
   minimal-contrast pairs and let the model learn the difference in similar contexts.
3. Affirmative interpretation: Rezaei and Blanco (arXiv:2502.07717), pair the negation
   with a paraphrase that states the positive without negation cues.

The "not X but Y" construction stacks all three in one sentence: it negates, supplies
the affirmative target, and forms the minimal-contrast pair. Likely stronger than equal
volumes of separate positive and negative lines, because the contrast is in-sentence.
Vary which entity is negated and which is affirmed so the model cannot learn the
template as a surface shortcut.

**The redirection analogy (load-bearing, not decorative).** Same control structure as
dog training: "no" alone leaves the subject with nowhere to put the behavior, so you
pair "not that" with "this instead." The redirect is the affirmative interpretation.
"Not the couch, go to your bed" and "not Cosmic Meadow but Kinetic Field" are the same
move.

---

## V. Contrastive method: three levels, and which apply to a generative model

**Finding.** "Contrastive learning" in the strict sense (anchor, positive, negatives,
embedding push-pull) is mostly bound to classification and representation models. But
the benefit transfers to generative CPT through data construction, not the loss.

**Practice. Three levels, usable together:**
1. Data-level contrast (the "not X but Y" sentence): works on a generative model now,
   via ordinary cross-entropy, no special loss. Highest evidence, lowest effort. Build
   it into the generator first.
2. Unlikelihood objective: the closest thing to a contrastive loss that runs natively
   on a decoder. Penalize the wrong continuation while the normal loss rewards the
   right one. The gradient-level version of redirection.
3. Strict contrastive (embedding distance) loss: classifier and representation-model
   bound. Not needed for generative CPT. Skip it.

---

## VI. Gradient-level levers: token masking and unlikelihood

**Finding.** You can change the gradient per token during CPT. The generator already
emits span tags, so the negation tokens are addressable.

**Critical sign correction.** To push a token's probability down you add its likelihood
to the loss (unlikelihood). You do not apply a positive gradient or reward to it;
rewarding it reinforces the thing you want to suppress. The loss for a contrastive
sentence is: standard cross-entropy on the normal tokens and the affirmative target,
plus an unlikelihood penalty on the false-continuation tokens.

**Practice and sequencing.** Data-level contrast first (safe, high-evidence).
Unlikelihood second, with a small coefficient. The unlikelihood objective is unbounded
(there is no floor to "make this less likely"), so an oversized coefficient destabilizes
the loss landscape and degrades unrelated tokens and general fluency. Watch max-grad-norm
and general quality, exactly as on the rest of the stack. Reach for it only if data-level
contrast does not install the negation cleanly.

---

## VII. The distributional hypothesis: separate by context, never by co-mention

**Finding.** Representations are shaped by the company a token keeps. Shared context
pulls representations together; disjoint context pushes them apart. This is the
foundation under the whole "basin" behavior, not an accident of the setup.

**Evidence.** Firth 1957 and Harris 1954: "you shall know a word by the company it
keeps"; to capture meaning and to capture context are the same. Still true for
transformers: the transformer is a learned dynamic context-weighting mechanism, and the
notion that context determines meaning is unchanged from Firth (Brenndoerfer 2026;
CS224n). Disjoint contexts produce incomplete, separated representations even for
plausibly-similar items (arXiv:1608.06794).

**THE HARD RULE (this is what backfired in V3, do not reintroduce).** Naming two
confusable entities in the same sentence or window ("X is NOT at Y") co-locates them,
which makes them share context, which pulls them closer. The opposite of the intent.
Separation comes from disjoint context, never from contrast phrases that name both.

**Practice.**
- Confusable entities get disjoint context profiles: different vocabulary, framing, and
  neighbor facts per entity, and they never co-occur in the same window.
- Related facts share context and co-occur freely, because binding is wanted there.
- Engineer the geometry by engineering co-occurrence.
- The mechanism is settled; the dose (how disjoint is disjoint enough) is corpus and
  model specific. "The choice of context definition determines what kind of similarity
  the representation captures." Measure it (see X).

**Validation gate.** Before training, run a co-occurrence audit: confirm no confusable
pair appears in the same sentence or window anywhere. A single co-mention reintroduces
the failure.

---

## VIII. Disclosure order: distinctiveness goes before the entity

**Finding.** Order of disclosure matters, and the lever is what comes before the entity
token, not after it.

**Mechanism.** In a transformer with two or more layers, as soon as the input differs
at one position, the model sees different contexts for all later positions at the top
layer. So differing the pre-context per stage changes the stage token's representation.
The causal mask means a token's representation is built only from what precedes it, so
the separating material must lead into the entity.

**Evidence.** Positional encodings nonessential for deep autoregressive transformers
(arXiv:2501.00659): one different earlier token cascades to differ all later top-layer
representations. Retroactive stabilization (arXiv:2601.04854, exploratory): a later
distinctive tail token can sharpen an earlier ambiguous one, which gives some support
to placing a strong disambiguating element late, but it does not replace the pre-context
as the load-bearing separator.

**Practice.** Structure each fact so the per-stage distinctive context comes first and
leads into the entity. Artists on the same stage reuse that stage's context block, so
they bind to the stage and to each other. Across stages the leading context is disjoint,
so the stage tokens separate. The asymmetry: shared pre-context binds the things that
should be close, disjoint pre-context separates the things that should be far.

---

## IX. Evaluation discipline: three lenses, and what each can and cannot say

**Finding.** Use three evals of increasing sharpness, and know which question each one
answers.
- Cloze install on the training surface is the signal of record for "did it install."
  It has the resolution to show the real size of a change.
- Question-eval (chat probes) is coarse confirmation. Near-saturated, so it only shows
  whole-probe flips and compresses real damage into a small delta.
- Geometry separation and AUC is continuous and does not hinge on one probe tipping.

**Practice and cautions.**
- On a small eval (for example 19 probes), chasing single-probe flips can fit noise.
  Read the geometry metrics as primary and the probe count as coarse confirmation.
- Bidirectional eval for negation: confirm the model both denies the false association
  and retains the true one. Balanced positive and negative density should prevent
  overcorrection, but the eval has to confirm it, not assume it.
- "Same basin" is a geometry claim. It is only answerable by the pairwise cosine between
  the specific entities, per layer. It is not inferable from a behavioral or cloze eval.
  A retrieval drop is consistent with conflation but is not a measurement of it. Do not
  infer geometry from behavior.

---

## X. The representation-geometry probe: the second eval signal

**Finding.** Measuring where entities sit in representation space turns "find the
formula that separates unrelated things" into a measurable objective, and gives a
finer, earlier signal than probe accuracy. This is Allen-Zhu's linear-probing tool plus
a distance readout.

**Practice (measurement is the product; the picture is secondary).**
- Use hidden states from mid-to-late layers, not the raw input embeddings (the input
  embedding is nearly static; the relational structure emerges in the residual stream).
  Sweep layers and report per layer.
- Pool multi-token entities consistently (last-token hidden state or mean-pool).
- Use a fixed neutral carrier context held identical across every entity and every
  model version.
- Core metric: pairwise cosine in the full hidden space, per layer.
- Separation score = mean_cos(RELATED) minus mean_cos(UNRELATED). Also AUC of using
  cosine to classify RELATED vs UNRELATED, and per-query MRR or precision@k.
- Linear probe: train a small linear classifier on hidden states to predict an
  attribute. Test accuracy is how linearly extractable the knowledge is (memorized-but-
  not-extractable shows up as the probe failing). Split over entities, no leakage.

**Visualization (intuition only).** PCA, t-SNE, UMAP, and the TensorFlow Embedding
Projector. t-SNE and UMAP distort global distance and shift with hyperparameters, so
label them "local structure only." PCA is honest about global structure but report the
percent variance explained by the three components. Trust the metric over the picture;
draw every conclusion from the full-space numbers.

**Isolation.** For base vs CPT comparisons, hold everything identical except the weights
(same layers, pooling, carrier context, precision). If base is the pre-CPT quantized
model and CPT is the post-CPT quantized model, that isolates the CPT effect from the
quantization effect.

**Measured example.** V5 stage-to-stage cosine rose +0.014 at L9 and +0.017 at L11
(stages less distinct, confirming "same basin"), alongside a general contraction of
about 0.01 to 0.02. The harness reproduced the V4 baseline exactly (AUC 0.568, MRR
0.528, day-probe 0.471 at L9), which confirmed harness consistency so any later delta is
attributable to the lever under test.

---

## XI. Is it just more steps? Check, do not assume

**Finding.** A failure that is one specific pair (not broad weakness) is confusability,
not undertraining. Undertraining produces broad softness across many probes.

**Practice.** Read the separation-by-checkpoint curve (50, 100, 140, 200). If separation
is still climbing at the last step, extend. If it plateaued while the hard pair stayed
unseparated, steps are not the lever and the residual is confusability.

**The thing more steps cannot fix.** Steps let the model learn the data it has; they do
not add distinctions the data does not contain. Training longer on under-augmented data
plateaus rather than breaking through (the extractability point again). On a small
corpus, more steps trades toward memorization and overfit, so watch the geometry and
probes, not the loss, if extending. Lower loss on a small corpus is partly memorized
surface form, not separation.

---

## XII. RAG vs CPT: access vs structure, and the token economics

**Finding.** RAG and CPT are not interchangeable here.

**Practice.**
- RAG is an access layer (read the raw data) and belongs at inference for volatile
  facts. It is not a substitute for structure: raw-to-prose re-creates the imbalance and
  adds a hallucination surface, and it ranks by surface similarity, blind to relation
  types.
- CPT installs the structure into the weights.
- Token economics: CPT pays the token tax once at training. RAG pays it on every query
  forever (retrieved chunks ride in every prompt). CPT is fewer tokens overall and has
  no per-query tax.

---

## XIII. Curriculum and batch composition: contrast in-batch, not across time

**Finding.** Distinction comes from contrast in the same batch, not from separating
topics across time. Sequential-by-stage training risks catastrophic forgetting: the
later stage is sharp and the earlier one decays, because nothing in the later stage
reinforces the earlier.

**Practice.** Interleave, or use a replay fraction (later stage mixed with a fraction of
the earlier). If testing the question directly, run three arms: fully interleaved,
sequential-by-stage, and sequential-with-replay, and read per-stage recall plus negation
accuracy. Expectation is that interleaved with in-batch contrast wins for distinction,
but the small-corpus few-step regime is where the standard prior is least certain, so
measure rather than assume.

---

## XIV. The corpus compiler: the synthesis

A tagged-triple pipeline with three lever groups that share the span-tag schema as the
spine. Every data-shape lever is a generation rule, every gradient lever keys off the
span tags the generator emits, and every verification lever reads the same tags back.

**Data-shape levers (what text exists):**
- relationship enumeration, every entity as source and target
- variation count per relation (the density floor, fed from eval)
- order-matched surface forms (token order mirrors query order)
- multi-hop path sentences
- the "not X but Y" contrastive form
- canonical plus colloquial naming, format and unit variation
- QA-shaped lines blended into the corpus
- faithfulness gate (every line traces to a triple)
- per-entity context profiles for confusable entities (disjoint), shared for related

**Gradient and training levers (how the text is learned):**
- per-token loss coefficients (mask or weight any span)
- unlikelihood penalty on false-continuation tokens
- per-relation density weighting (more pull for thin or confusable relations)
- span weighting so payload tokens outweigh filler
- replay fraction across stages (anti-forgetting)
- batch composition (contrast in-batch)
- learning-rate and schedule per phase

**Verification levers (did it install):**
- bidirectional eval (denies false, retains true)
- per-relation coverage matrix against the density floor
- held-out post-cutoff probes
- cloze on the training surface (signal of record)
- geometry separation, AUC, MRR (the second signal)
- co-occurrence audit (no confusable pair co-mentioned)
- dedup and distinct-ratio per relation

**Status.** The data-shape and verification arms are validated (festival result). The
gradient arm beyond plain weighting (unlikelihood, per-relation gradient shaping) is
sound and next to calibrate; treat its first runs as calibration, not known-good.

**Discipline.** Every lever is a named, logged, independently-toggleable knob with
per-lever eval instrumentation, so when a number moves you know which knob moved it. A
hundred entangled levers you cannot isolate is harder to debug than ten you can.

---

## XV. Tooling discipline (governs how the compiler gets built)

Building with an agent is cheap; owning the result is not. The agent writes the code for
free, but you maintain it, remember it exists, and debug it when it silently breaks
mid-experiment. Code is a liability as much as an asset.

**Practice.**
- Build freely, especially disposable instruments and one-off probes that used to be too
  expensive to bother with.
- Sort as you go: disposable or keeper. Throw the disposable away when done; cheap to
  build means cheap to rebuild.
- Hold the keepers (the compiler, the eval harness, the geometry probe) to the
  instrumented-and-isolated standard above, so a regression is always attributable.

---

## XVI. Run success vs installation success (the two verdicts — do not conflate)

**Finding.** A model can PASS the behavioral SFT gate (answer what it knows, abstain on
what it does not, do not fabricate) while the underlying knowledge installation is still
weak or uneven. The two are different questions measured by different instruments.

- **Successful RUN** = the behavioral calibration gate: fabrication ≤ τ, accuracy-when-
  answering ≥ τ, coverage ≥ τ, on held-out probes (greedy, rule-scored). This is an
  SFT-elicitation property — does the conversational surface answer/abstain correctly.
- **Successful INSTALLATION** = the knowledge is densely, retrievably, separably in the
  weights: density toward the floor (I), geometry separation above chance and a passing
  day/attribute linear probe (X), cloze on the training surface (IX), low confusable-pair
  cosine (VII). This is a CPT property — did the facts actually go in, distinctly.

**Why it matters.** v6 PASSED the run gate (fab 0.17 / acc 0.58 / cov 0.47) while the
installation was still mediocre: geometry p@1 ~2.4× chance, relation balance FAIL,
417 disjoint co-occurrence violations. Shipping on the run verdict alone would carry that
weak installation into production. The operating room shows BOTH verdicts side by side so
a pass on one is never mistaken for a pass on the other.

**Practice.** Gate deployment on BOTH: the run must pass AND the installation must be
strong (density floor met, geometry separation comfortably above chance, extractability
probe passing, co-occurrence audit clean). For a product others run themselves, the
installation verdict is the load-bearing one — the run verdict can be reached by a
well-dosed SFT on top of a thin install, which is exactly the trap to surface.

---

## References (so you do not re-research these)

Density and extractability:
- Allen-Zhu and Li, Physics of Language Models 3.1, arXiv:2309.14316 (augmentation
  needed for extractability; memorized vs retrievable).
- Allen-Zhu and Li, Physics of Language Models 3.2, arXiv:2309.14402 (rewrite corpus,
  mix QA into pretraining; order must match QA style).

Reversal and direction:
- Berglund et al. 2023, reversal curse, arXiv:2309.12288.
- Guo et al. 2024, permutation training, arXiv:2403.00758.

Negation:
- Kassner and Schutze 2020 (negated and non-negated indistinguishable).
- Hosseini et al. 2021, unlikelihood for negation, arXiv:2105.03519 (negated-LAMA to 4%).
- Rezaei and Blanco, affirmative interpretation, arXiv:2502.07717.
- Yang et al., relation-based counterfactual augmentation plus contrastive,
  arXiv:2410.20710.
- Negation Neglect, arXiv:2605.13829 (models fail to learn negations in training; asks
  for a larger-scale density test).

Distributional semantics:
- Firth 1957, Harris 1954 (the hypothesis).
- Brenndoerfer 2026 and CS224n (still the foundation under transformers; transformer as
  learned dynamic context weighting).
- Distributional inference with disjoint contexts, arXiv:1608.06794.

Order and position:
- Positional encodings nonessential for deep autoregressive transformers,
  arXiv:2501.00659 (one different earlier token cascades to all later top-layer
  representations).
- Retroactive stabilization, arXiv:2601.04854 (exploratory; later tail token can sharpen
  an earlier ambiguous one).

---

## How to use and extend this file

Add a new finding under the relevant section, or as a new numbered section, in the
Finding / Mechanism / Evidence / Practice shape. When a result is geometry, cite a
geometry measurement; when it is install strength, cite the cloze; when it is external,
cite the paper. Keep the reference list at the bottom current so future runs do not
repeat tonight's searches. Mirror every checkable item into `scripts/data_audit.py` so
the audit page surfaces it as a pre-run gate, not tribal knowledge.
