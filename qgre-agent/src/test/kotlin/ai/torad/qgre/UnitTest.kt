package ai.torad.qgre

import ai.torad.aisdk.ContentPart
import ai.torad.aisdk.MessageRole
import ai.torad.aisdk.ModelMessage
import ai.torad.aisdk.FinishReason
import ai.torad.aisdk.OnChunkEvent
import ai.torad.aisdk.OnErrorEvent
import ai.torad.aisdk.OnStepFinishEvent
import ai.torad.aisdk.PrepareStepScope
import ai.torad.aisdk.StepResult
import ai.torad.aisdk.StreamEvent
import ai.torad.aisdk.Usage
import ai.torad.aisdk.providers.MockLanguageModel
import ai.torad.aisdk.providers.ScriptedResponse
import ai.torad.aisdk.userMessage
import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpStatusCode
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Unit evals for the supporting pieces: StepPrep compaction, Hooks observability, the CLI arg-parse +
 *  approval-resume, and the Provider model-id resolution. */
class UnitTest {

    private fun dummyModel() = MockLanguageModel(responses = listOf(ScriptedResponse(events = emptyList())))
    private fun toolMsg(id: String, name: String, payload: String) =
        ModelMessage(MessageRole.Tool, listOf(ContentPart.ToolResult(id, name, JsonPrimitive(payload))))

    // ── StepPrep ──────────────────────────────────────────────────────────────────────────────────────
    @Test
    fun prepareStep_noOverride_whenFewToolResults() = runTest {
        val msgs = listOf(userMessage("go"), toolMsg("c1", "gate_corpus", "small result"))
        val scope = PrepareStepScope(1, dummyModel(), emptyList(), msgs, testCtx("x/y"))
        assertNull(qgrePrepareStep(keepRecent = 4)(scope).messages, "no compaction when under the keep window")
    }

    @Test
    fun prepareStep_doesNotTrimErrorResults() = runTest {
        val big = "X".repeat(1000)
        val msgs = listOf(
            userMessage("go"),
            ModelMessage(MessageRole.Tool, listOf(ContentPart.ToolResult("c1", "clean_layer", JsonPrimitive(big + "_ERRMARK"), isError = true))),
            toolMsg("c2", "extract_triples", big + "_OLD"),
            toolMsg("c3", "hydrate", big + "_H"),
            toolMsg("c4", "hydrate", big + "_H2"),
            toolMsg("c5", "gate_corpus", big + "_GATE"),
        )
        val scope = PrepareStepScope(6, dummyModel(), emptyList(), msgs, testCtx("x/y"))
        val out = qgrePrepareStep(keepRecent = 2)(scope).messages!!
        fun mv(i: Int) = (out[i].content[0] as ContentPart.ToolResult).modelVisible.toString()
        assertTrue(mv(1).contains("_ERRMARK"), "error results are kept full (not trimmed)")
        assertFalse(mv(2).contains("_OLD"), "an old non-error result is trimmed")
        assertTrue(mv(2).contains("trimmed"))
    }

    // ── Hooks ───────────────────────────────────────────────────────────────────────────────────────
    @Test
    fun hooks_streamDecisions_andCapture() = runTest {
        val fs = FakeServe()
        val lines = mutableListOf<String>()
        val cap = RunCaptures()
        val model = MockLanguageModel(
            responses = listOf(toolTurn("gate_corpus", ofEdition("x/y")), toolTurn("done", buildJsonObject { put("verdict", "ok") })),
        )
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("x/y"), hooks = runHooks({ lines.add(it) }, cap))
        assertTrue(lines.any { it.contains("→ gate_corpus") }, "onChunk logs the tool call")
        assertTrue(lines.any { it.contains("step") }, "step lines are logged")
        assertTrue(lines.any { it.startsWith("✓") }, "onFinish logs the summary")
        assertTrue(cap.messages.isNotEmpty() && cap.totalSteps > 0)
    }

    @Test
    fun hooks_coverEveryChunkBranch_andError() = runTest {
        // The onChunk `when` over event types and onError are exercised directly — a ToolApprovalRequest /
        // ToolResult chunk surfaces in stream() mode, while runEdition uses generate() (approval comes back
        // via result.pendingApprovals), so direct invocation is the deterministic way to cover the branches.
        val lines = mutableListOf<String>()
        val hk = runHooks({ lines.add(it) }, RunCaptures())
        hk.onChunk?.invoke(OnChunkEvent(StreamEvent.ToolCall("c", "extract_triples", buildJsonObject { put("editions", "x/y") }), 1))
        hk.onChunk?.invoke(OnChunkEvent(StreamEvent.ToolResult("c", "extract_triples", buildJsonObject { put("status", "done") }), 1))
        hk.onChunk?.invoke(OnChunkEvent(StreamEvent.ToolApprovalRequest("c", "extract_triples", buildJsonObject { put("editions", "x/y") }), 1))
        hk.onChunk?.invoke(OnChunkEvent(StreamEvent.TextDelta("t", "thinking…"), 1))   // the else branch
        hk.onError?.invoke(OnErrorEvent(RuntimeException("boom"), 2, OnErrorEvent.ErrorSource.Tool))

        // onStepFinish with non-empty step text (the text branch)
        hk.onStepFinish?.invoke(OnStepFinishEvent(3, StepResult(3, "decided to gate", "", emptyList(), emptyList(), emptyList(), FinishReason.Stop, Usage())))

        assertTrue(lines.any { it.contains("→ extract_triples") }, "tool call logged")
        assertTrue(lines.any { it.contains("← extract_triples") }, "tool result logged")
        assertTrue(lines.any { it.contains("approval needed") }, "approval request logged")
        assertTrue(lines.any { it.contains("error") && it.contains("boom") }, "error logged")
        assertTrue(lines.any { it.contains("decided to gate") }, "step text logged")
    }

    // ── CLI arg parsing ─────────────────────────────────────────────────────────────────────────────
    @Test
    fun parseArgs_defaults_and_csvSplit() {
        val a = parseArgs(arrayOf("--editions", "a/1, b/2", "--job-id", "J7"))
        assertEquals("edc", a.domain)
        assertEquals(listOf("a/1", "b/2"), a.editions)
        assertEquals(listOf("las_vegas/2026"), a.protectedEditions)
        assertEquals("J7", a.jobId)
    }

    @Test
    fun parseArgs_domainAndProtectedOverride_emptyEditions() {
        val a = parseArgs(arrayOf("--domain", "ultra", "--protected", "x/1,y/2"))
        assertEquals("ultra", a.domain)
        assertTrue(a.editions.isEmpty())
        assertEquals(listOf("x/1", "y/2"), a.protectedEditions)
    }

    // ── runEdition approval-resume (inject the decision) ──────────────────────────────────────────────
    @Test
    fun runEdition_denyApproval_paidToolNotExecuted() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("extract_triples", ofEditions("x/y")),
                toolTurn("gate_corpus", ofEdition("x/y")),
                toolTurn("done", buildJsonObject { put("verdict", "blocked") }),
            ),
        )
        val agent = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = false))
        runEdition(agent, testCtx("x/y"), isProtected = false, approve = { false }, log = {})
        assertFalse("extract_triples" in fs.calls, "a denied paid tool must not execute")
    }

    @Test
    fun runEdition_approve_paidToolExecutes() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("extract_triples", ofEditions("x/y")),
                toolTurn("done", buildJsonObject { put("verdict", "ok") }),
            ),
        )
        val agent = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = false))
        runEdition(agent, testCtx("x/y"), isProtected = false, approve = { true }, log = {})
        assertTrue("extract_triples" in fs.calls, "an approved paid tool executes")
    }

    @Test
    fun runEdition_protected_takesGateOnlyPath() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("gate_corpus", ofEdition("las_vegas/2026")),
                toolTurn("done", buildJsonObject { put("verdict", "gold verified") }),
            ),
        )
        val agent = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
        runEdition(agent, testCtx("las_vegas/2026", protectedEd = true), isProtected = true, approve = { true }, log = {})
        assertTrue("gate_corpus" in fs.calls)
        assertFalse("extract_triples" in fs.calls || "hydrate" in fs.calls || "clean_layer" in fs.calls)
    }

    // ── Provider model-id resolution + wiring ─────────────────────────────────────────────────────────
    @Test
    fun resolveModelId_envOverride_else_default() {
        assertEquals("frontier/model", resolveModelId("frontier/model"))
        assertEquals(DEFAULT_AGENT_MODEL, resolveModelId(null))
        assertEquals(DEFAULT_AGENT_MODEL, resolveModelId(""))
    }

    @Test
    fun openRouterModel_wiresTheRequestedModelId() {
        val http = HttpClient(MockEngine { respond("{}", HttpStatusCode.OK) })
        val m = openRouterModel(http, apiKey = "k", modelId = "anthropic/claude-x")
        assertEquals("anthropic/claude-x", m.modelId)
    }
}
