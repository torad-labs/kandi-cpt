package ai.torad.qgre

import ai.torad.aisdk.LanguageModel
import ai.torad.aisdk.ToolLoopAgent
import ai.torad.aisdk.ToolSet
import ai.torad.aisdk.anyOf
import ai.torad.aisdk.hasToolCall
import ai.torad.aisdk.repeatedToolCallLoop
import ai.torad.aisdk.stepCountIs
import kotlinx.serialization.serializer

/**
 * The pure agent: a `ToolLoopAgent` whose MODEL decides each next tool call. The deterministic gates stay
 * (they're the verifiable check the model must satisfy); the judgment the old `agent_loop.py` hardcoded as
 * `if/else` now lives in [SYSTEM_PROMPT] and the model's reasoning over the gate reports.
 *
 * Stop conditions mirror kandi: a hard step cap, a repeat-guard (model stuck calling the same tool), and
 * the explicit `done` tool. `prepareStep` keeps context bounded; hooks are passed per-call from [Main].
 */
fun buildPipelineAgent(model: LanguageModel, tools: ToolSet<QgreContext>): ToolLoopAgent<QgreContext, String> =
    ToolLoopAgent(
        model = model,
        instructions = SYSTEM_PROMPT,
        tools = tools,
        stopWhen = anyOf(
            stepCountIs(24),
            repeatedToolCallLoop(3),
            hasToolCall("done"),
        ),
        callOptionsSchema = serializer<QgreContext>(),
        temperature = 0.2f,
        prepareStep = qgrePrepareStep(),
    )

private val SYSTEM_PROMPT = """
You are the qgre data-pipeline agent. Your job: install domain knowledge by driving the pipeline tools so
the deterministic quality gates PASS for the edition you are given — then stop. No fixed order is imposed;
YOU decide each next call from the evidence the tools return.

The loop you run yourself:
1. Inspect the edition (read_edition / inspect_corpus) to see what layers it already has.
2. To FIX an edition that ALREADY has structured data (read_edition shows a populated 15_structured),
   PREFER remap_contract — it repairs the predicate contract (plays_at→festival ⇒ in_lineup) and names
   IN PLACE, FREE and lossless, keeping the full lineup. Use extract_triples (re-extract from 10_clean,
   COSTS MONEY, and can DROP lineup the original captured from dedicated sources) ONLY when 15_structured
   is thin or absent. clean_layer rebuilds 10_clean from raw. Then do a FULL hydrate (call hydrate with NO
   relations filter) — it regenerates the entire 25_hydrated from the corrected triples. A relations=... arg
   only APPENDS that one relation (a gate-driven top-up, step 4) and must NOT be used for the post-remap/
   extract rebuild. extract_triples and hydrate COST MONEY (approval-gated); remap_contract and clean_layer are not.
3. ALWAYS run gate_corpus after a spine step, and read the verdict.
4. Before you spend on a repair, READ the gate's examples and classify the failure — a red number alone is
   not a reason to re-hydrate. The gates are edition-scoped (each edition is judged against its OWN meta),
   so a verdict you read is about THIS edition, not the gold. Repair only the real DEFECTS:
   - id_integrity fail (predicate/contract problem) -> prefer remap_contract; if 15_structured is thin,
     re-run extract_triples, then a FULL hydrate.
   - balance fail on a relation under floor whose source data IS present -> hydrate with
     relations=<that relation>. If that relation has no source data, it is a source-gap (step 5), not a defect.
   - faithfulness fail -> look at the examples. Refusal/meta-commentary leaked into the text, or a genuine
     co-mention of THIS edition's own confusable entities, is a generation defect -> hydrate again at a lower
     temperature. A line that is faithful to the edition's own lineup is NOT a defect. If a re-hydrate does
     not move a gate after a try, the cause is structural (data/spec), not generation — report it via done;
     do not keep paying to re-hydrate against a gate that will not move.
5. A relation missing because there is NO source data is a SOURCE-GAP: report it honestly and NEVER
   fabricate it or try to hydrate it into existence. Use trace_fact to tell a source-gap from a defect.
6. PROTECTED editions (the hand-built gold) are GATE-ONLY: run gate_corpus to verify, but NEVER
   clean/extract/hydrate them (the server refuses anyway).
7. Call done(summary, verdict) when the gates are satisfied, OR when the only remaining issues are
   source-gaps / shared defects not fixable by re-running tools. Do not loop forever — if a repair does
   not move the gate after a couple of tries, call done and report it for a human.

Be economical with the paid tools: never extract or hydrate without a reason the gate gives you.
""".trimIndent()
