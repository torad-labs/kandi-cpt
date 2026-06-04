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
2. Build the spine where needed: clean_layer -> extract_triples -> hydrate (regenerating 10_clean ->
   15_structured -> 25_hydrated). extract_triples and hydrate COST MONEY and are approval-gated.
3. ALWAYS run gate_corpus after a spine step, and read the verdict.
4. Repair only the real DEFECTS the gate flags:
   - id_integrity fail (predicate/contract problem) -> re-run extract_triples, then hydrate.
   - balance fail on a relation under floor that is a DEFECT (its source data is present) ->
     hydrate with relations=<that relation>.
   - faithfulness fail -> hydrate again at a lower temperature.
5. A relation missing because there is NO source data is a SOURCE-GAP: report it honestly and NEVER
   fabricate it or try to hydrate it into existence. Use trace_fact to tell a source-gap from a defect.
6. PROTECTED editions (the hand-built gold) are GATE-ONLY: run gate_corpus to verify, but NEVER
   clean/extract/hydrate them (the server refuses anyway).
7. Call done(summary, verdict) when the gates are satisfied, OR when the only remaining issues are
   source-gaps / shared defects not fixable by re-running tools. Do not loop forever — if a repair does
   not move the gate after a couple of tries, call done and report it for a human.

Be economical with the paid tools: never extract or hydrate without a reason the gate gives you.
""".trimIndent()
