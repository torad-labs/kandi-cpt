package ai.torad.qgre

import ai.torad.aisdk.ContentPart
import ai.torad.aisdk.FinishReason
import ai.torad.aisdk.MessageRole
import ai.torad.aisdk.ModelMessage
import ai.torad.aisdk.PrepareStepScope
import ai.torad.aisdk.StreamEvent
import ai.torad.aisdk.providers.MockLanguageModel
import ai.torad.aisdk.providers.ScriptedResponse
import ai.torad.aisdk.toolApprovalResponseMessage
import ai.torad.aisdk.userMessage
import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Eval for the PURE agent. The MODEL is scripted (a deterministic [MockLanguageModel]) and the serve.py
 * HTTP boundary is faked (a Ktor [MockEngine]), but the agent loop itself — tool dispatch, result
 * feedback, multi-step continuation, approval gating, stop conditions, prepareStep, hooks — is the REAL
 * `ToolLoopAgent`. So these prove the harness faithfully executes whatever the model decides; they do NOT
 * claim the real model makes good decisions (that's the opt-in integration smoke + trajectory judging).
 * This is the eval the old "stub the tools" idea could never be: there a fully-mockable loop had no agent
 * in it; here the loop is real and only its two external dependencies (model, HTTP) are scripted.
 */
class PipelineAgentTest {

    /** Every qgre action the agent's tools actually invoked over HTTP, in order (the behavioral trace). */
    private val calls = mutableListOf<String>()

    private fun fakeClient(gate: ArrayDeque<String> = ArrayDeque()): ToradClient {
        val engine = MockEngine { request ->
            val path = request.url.encodedPath
            val jsonHeaders = headersOf(HttpHeaders.ContentType, "application/json")
            when {
                path.startsWith("/api/action/") -> {
                    val name = path.removePrefix("/api/action/")
                    calls += name
                    val result = when (name) {
                        "gate_corpus" -> gate.removeFirstOrNull() ?: """{"overall":"pass","gates":{}}"""
                        "clean_layer", "extract_triples", "hydrate" -> """{"job_id":"job_${calls.size}"}"""
                        else -> """{"read":"$name"}"""
                    }
                    respond("""{"ok":true,"result":$result}""", HttpStatusCode.OK, jsonHeaders)
                }
                path.startsWith("/api/pipeline/") ->
                    respond("""{"status":"done"}""", HttpStatusCode.OK, jsonHeaders)
                else -> respond("{}", HttpStatusCode.NotFound, jsonHeaders)
            }
        }
        return ToradClient(http = HttpClient(engine))
    }

    private fun toolCall(name: String, args: JsonObject, id: String = "c_$name"): ScriptedResponse =
        ScriptedResponse(
            events = listOf(StreamEvent.ToolCall(id, name, args)),
            finishReason = FinishReason.ToolCalls,
        )

    private fun ed(name: String) = buildJsonObject { put("edition", name) }
    private fun eds(name: String) = buildJsonObject { put("editions", name) }
    private fun ctx(edition: String, protectedEd: Boolean = false) =
        QgreContext(edition = edition, protectedEditions = if (protectedEd) listOf(edition) else emptyList())

    // ── (a) happy path + (g-hooks) the lifecycle hooks fire ──────────────────────────────────────────
    @Test
    fun happyPath_drivesSpineThenGatesThenDone_andHooksFire() = runTest {
        calls.clear()
        val model = MockLanguageModel(
            responses = listOf(
                toolCall("inspect_corpus", buildJsonObject { put("festival", "edc") }),
                toolCall("clean_layer", eds("orlando/2024")),
                toolCall("extract_triples", eds("orlando/2024")),
                toolCall("hydrate", ed("orlando/2024")),
                toolCall("gate_corpus", ed("orlando/2024")),
                toolCall("done", buildJsonObject { put("summary", "gates pass"); put("verdict", "pass") }),
            ),
        )
        val captures = RunCaptures()
        val res = buildPipelineAgent(model, qgreTools(fakeClient(), autoApprove = true))
            .generate(prompt = "go", options = ctx("orlando/2024"), hooks = runHooks({}, captures))

        // the loop dispatched the model's chosen HTTP tools in order (done is local → not an HTTP call)
        assertContentEquals(
            listOf("inspect_corpus", "clean_layer", "extract_triples", "hydrate", "gate_corpus"),
            calls,
        )
        // `done` halted the loop at step 6 (not run away to stepCountIs(24))
        assertEquals(6, res.steps.size)
        // hooks observed the run
        assertEquals(6, captures.totalSteps)
        assertTrue(captures.messages.isNotEmpty())
    }

    // ── (b) gate fail → the model re-extracts → gate passes → done ────────────────────────────────────
    @Test
    fun gateFail_modelRepairsByReExtracting() = runTest {
        calls.clear()
        val gate = ArrayDeque(
            listOf(
                """{"overall":"fail","gates":{"id_integrity":{"severity":"fail"}}}""",
                """{"overall":"pass","gates":{"id_integrity":{"severity":"pass"}}}""",
            ),
        )
        val model = MockLanguageModel(
            responses = listOf(
                toolCall("clean_layer", eds("dallas/2011")),
                toolCall("extract_triples", eds("dallas/2011"), id = "x1"),
                toolCall("hydrate", ed("dallas/2011"), id = "h1"),
                toolCall("gate_corpus", ed("dallas/2011"), id = "g1"),
                toolCall("extract_triples", eds("dallas/2011"), id = "x2"),
                toolCall("hydrate", ed("dallas/2011"), id = "h2"),
                toolCall("gate_corpus", ed("dallas/2011"), id = "g2"),
                toolCall("done", buildJsonObject { put("verdict", "pass") }),
            ),
        )
        val res = buildPipelineAgent(model, qgreTools(fakeClient(gate), autoApprove = true))
            .generate(prompt = "go", options = ctx("dallas/2011"))

        assertEquals(2, calls.count { it == "extract_triples" })   // the repair re-extracted
        assertEquals(2, calls.count { it == "gate_corpus" })
        assertEquals(8, res.steps.size)
    }

    // ── (c) a source-gap is reported (done), never fabricated by re-hydrating it ──────────────────────
    @Test
    fun sourceGap_modelCallsDone_doesNotReHydrate() = runTest {
        calls.clear()
        val gate = ArrayDeque(
            listOf(
                // balance fail on set_time, BUT classified a source_gap → not a defect to fix
                """{"overall":"fail","gates":{"balance":{"severity":"fail"},
                   "source_vs_gap":{"source_gap_vs_defect":{"set_time":{"kind":"source_gap"}}}}}""".trimIndent(),
            ),
        )
        val model = MockLanguageModel(
            responses = listOf(
                toolCall("hydrate", ed("denver/2008"), id = "h1"),
                toolCall("gate_corpus", ed("denver/2008"), id = "g1"),
                // model sees the source_gap and stops rather than re-hydrating a relation with no source
                toolCall("done", buildJsonObject { put("verdict", "settled"); put("summary", "set_time is a source-gap") }),
            ),
        )
        val res = buildPipelineAgent(model, qgreTools(fakeClient(gate), autoApprove = true))
            .generate(prompt = "go", options = ctx("denver/2008"))

        assertEquals(1, calls.count { it == "hydrate" })           // exactly one hydrate — no fabricating top-up
        assertEquals(3, res.steps.size)
    }

    // ── (d) a protected edition is gate-only — no spine tools called ──────────────────────────────────
    @Test
    fun protectedEdition_gateOnly() = runTest {
        calls.clear()
        val model = MockLanguageModel(
            responses = listOf(
                toolCall("gate_corpus", ed("las_vegas/2026")),
                toolCall("done", buildJsonObject { put("verdict", "gold verified") }),
            ),
        )
        buildPipelineAgent(model, qgreTools(fakeClient(), autoApprove = true))
            .generate(prompt = "verify the gold", options = ctx("las_vegas/2026", protectedEd = true))

        assertContentEquals(listOf("gate_corpus"), calls)
        assertFalse("clean_layer" in calls || "extract_triples" in calls || "hydrate" in calls)
    }

    // ── (e) a stuck model (same call repeated) is halted by repeatedToolCallLoop, not stepCountIs ─────
    @Test
    fun repeatedToolCall_halts() = runTest {
        calls.clear()
        val model = MockLanguageModel(
            responses = (1..8).map { toolCall("gate_corpus", ed("chiba/2018"), id = "g$it") },
        )
        val res = buildPipelineAgent(model, qgreTools(fakeClient(), autoApprove = true))
            .generate(prompt = "go", options = ctx("chiba/2018"))

        assertTrue(res.steps.size <= 4, "repeatedToolCallLoop(3) should halt fast, got ${res.steps.size}")
        assertTrue(res.steps.size < 24)
    }

    // ── (f) a paid tool pauses for approval, then resumes ─────────────────────────────────────────────
    @Test
    fun paidTool_pausesForApproval_thenResumes() = runTest {
        calls.clear()
        val model = MockLanguageModel(
            responses = listOf(
                toolCall("extract_triples", eds("orlando/2022"), id = "x1"),
                toolCall("gate_corpus", ed("orlando/2022"), id = "g1"),
                toolCall("done", buildJsonObject { put("verdict", "pass") }),
            ),
        )
        val ctx = ctx("orlando/2022")
        val agent = buildPipelineAgent(model, qgreTools(fakeClient(), autoApprove = false))

        val first = agent.generate(prompt = "go", options = ctx)
        assertEquals(1, first.pendingApprovals.size)               // loop paused on the paid tool
        assertFalse("extract_triples" in calls)                    // NOT executed yet

        val resume = first.messages + toolApprovalResponseMessage(first.pendingApprovals.first().toolCallId, approved = true)
        agent.generate(messages = resume, options = ctx)
        assertTrue("extract_triples" in calls)                     // executed after approval
    }

    // ── (g) prepareStep compacts old tool outputs while preserving the latest gate verdict ────────────
    @Test
    fun prepareStep_trimsOldToolOutputs_keepsLatestGate() = runTest {
        val big = "X".repeat(1000)
        fun toolMsg(id: String, name: String, payload: String) =
            ModelMessage(MessageRole.Tool, listOf(ContentPart.ToolResult(id, name, JsonPrimitive(payload))))
        val msgs = listOf(
            userMessage("go"),
            toolMsg("c1", "clean_layer", big + "_CLEAN"),
            toolMsg("c2", "extract_triples", big + "_EXTRACT"),
            toolMsg("c3", "hydrate", big + "_HYDRATE"),
            toolMsg("c4", "hydrate", big + "_HYDRATE2"),
            toolMsg("c5", "gate_corpus", big + "_GATEVERDICT"),
        )
        val dummy = MockLanguageModel(responses = listOf(ScriptedResponse(events = emptyList())))
        val scope = PrepareStepScope(
            stepNumber = 6, model = dummy, steps = emptyList(), messages = msgs, context = ctx("x/y"),
        )
        val settings = qgrePrepareStep(keepRecent = 2)(scope)
        val out = settings.messages ?: error("expected a compacted message override")

        fun modelVisible(i: Int) = (out[i].content[0] as ContentPart.ToolResult).modelVisible.toString()
        // the latest gate verdict (within keepRecent=2) is preserved at full fidelity
        assertTrue(modelVisible(5).contains("_GATEVERDICT"))
        // an older tool result is trimmed (model-facing payload shortened)
        assertTrue(modelVisible(1).contains("trimmed"))
        assertFalse(modelVisible(1).contains("_CLEAN"))
    }
}
