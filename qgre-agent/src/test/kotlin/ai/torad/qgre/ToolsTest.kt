package ai.torad.qgre

import ai.torad.aisdk.providers.MockLanguageModel
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Per-tool eval: drive ONE tool through the real loop and assert the executor mapped the model's typed
 * input to the exact snake_case args the Python action expects (+ force on the spine tools), polled async
 * jobs to completion, and surfaced server errors back to the model instead of throwing.
 */
class ToolsTest {

    /** Run a single tool call (then done) and return the FakeServe so the caller can inspect the request. */
    private suspend fun callOne(fs: FakeServe, tool: String, args: JsonObject, edition: String = "x/y") {
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn(tool, args),
                toolTurn("done", buildJsonObject { put("verdict", "ok") }),
            ),
        )
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx(edition))
    }

    private fun body(fs: FakeServe, action: String): JsonObject =
        Json.parseToJsonElement(fs.bodyOf(action)).jsonObject

    // ── read tools: exact arg shape ──────────────────────────────────────────────────────────────────
    @Test fun inspectCorpus_args() = runTest {
        val fs = FakeServe(); callOne(fs, "inspect_corpus", buildJsonObject { put("festival", "edc") })
        assertEquals("edc", body(fs, "inspect_corpus")["festival"]!!.jsonPrimitive.content)
    }

    @Test fun readEdition_args() = runTest {
        val fs = FakeServe(); callOne(fs, "read_edition", buildJsonObject { put("edition", "orlando/2024") })
        val b = body(fs, "read_edition")
        assertEquals("orlando/2024", b["edition"]!!.jsonPrimitive.content)
        assertEquals("edc", b["festival"]!!.jsonPrimitive.content)
    }

    @Test fun verifyLayer_args() = runTest {
        val fs = FakeServe(); callOne(fs, "verify_layer", buildJsonObject { put("edition", "x/y"); put("layer", "15_structured") })
        val b = body(fs, "verify_layer")
        assertEquals("15_structured", b["layer"]!!.jsonPrimitive.content)
        assertEquals("x/y", b["edition"]!!.jsonPrimitive.content)
    }

    @Test fun gateCorpus_args() = runTest {
        val fs = FakeServe(); callOne(fs, "gate_corpus", buildJsonObject { put("edition", "x/y") })
        val b = body(fs, "gate_corpus")
        assertEquals("x/y", b["edition"]!!.jsonPrimitive.content)
        assertEquals("edc", b["domain"]!!.jsonPrimitive.content)
        assertEquals("edc", b["festival"]!!.jsonPrimitive.content)
    }

    @Test fun traceFact_args() = runTest {
        val fs = FakeServe(); callOne(fs, "trace_fact", buildJsonObject { put("edition", "x/y"); put("query", "Martin Garrix") })
        assertEquals("Martin Garrix", body(fs, "trace_fact")["query"]!!.jsonPrimitive.content)
    }

    // ── paid spine tools: snake_case + force=true + async poll ────────────────────────────────────────
    @Test fun cleanLayer_mapsArgs_forceAndRescue_andPolls() = runTest {
        val fs = FakeServe(); fs.jobStatusSeq = listOf("running", "done")
        callOne(fs, "clean_layer", buildJsonObject { put("editions", "x/y") })
        val b = body(fs, "clean_layer")
        assertEquals("x/y", b["editions"]!!.jsonPrimitive.content)
        assertEquals(true, b["llm_rescue"]!!.jsonPrimitive.boolean)
        assertEquals(true, b["force"]!!.jsonPrimitive.boolean)
        assertTrue(fs.pipelinePolls >= 1, "async tool must poll the job to completion")
    }

    @Test fun extractTriples_mapsArgs_force() = runTest {
        val fs = FakeServe(); callOne(fs, "extract_triples", buildJsonObject { put("editions", "x/y") })
        val b = body(fs, "extract_triples")
        assertEquals("x/y", b["editions"]!!.jsonPrimitive.content)
        assertEquals("edc", b["domain"]!!.jsonPrimitive.content)
        assertEquals(true, b["force"]!!.jsonPrimitive.boolean)
    }

    @Test fun hydrate_mapsAllArgs_snakeCase_force() = runTest {
        val fs = FakeServe()
        callOne(fs, "hydrate", buildJsonObject {
            put("edition", "x/y"); put("relations", "set_time"); put("exposureCap", 40); put("temperature", 0.1)
        })
        val b = body(fs, "hydrate")
        assertEquals("x/y", b["edition"]!!.jsonPrimitive.content)
        assertEquals("edc", b["festival"]!!.jsonPrimitive.content)
        assertEquals(40, b["exposure_cap"]!!.jsonPrimitive.int)
        assertEquals("set_time", b["relations"]!!.jsonPrimitive.content)
        assertEquals(0.1, b["temperature"]!!.jsonPrimitive.double)
        assertEquals(true, b["force"]!!.jsonPrimitive.boolean)
    }

    @Test fun hydrate_omitsOptionals_whenUnset() = runTest {
        val fs = FakeServe(); callOne(fs, "hydrate", buildJsonObject { put("edition", "x/y") })
        val b = body(fs, "hydrate")
        assertEquals(60, b["exposure_cap"]!!.jsonPrimitive.int)   // default
        assertFalse(b.containsKey("relations"), "blank relations must be omitted")
        assertFalse(b.containsKey("temperature"), "default temperature (<0) must be omitted")
    }

    // ── error surfacing: a {ok:false} becomes a tool result string the model can read, not a throw ─────
    @Test fun toolError_isSurfacedToModel_notThrown() = runTest {
        val fs = FakeServe(); fs.errors["gate_corpus"] = "Mutex held" to "gpu_busy"
        val captures = RunCaptures()
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("gate_corpus", ofEdition("x/y")),
                toolTurn("done", buildJsonObject { put("verdict", "blocked") }),
            ),
        )
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("x/y"), hooks = runHooks({}, captures))
        val result = toolResultText(captures.messages, "gate_corpus") ?: error("no gate_corpus result")
        assertTrue(result.contains("ERROR"), "server error must surface as a tool result")
        assertTrue(result.contains("gpu_busy"), "the kind must be surfaced so the model can react")
    }

    // ── spawn tool with no job_id falls back to the sync result (no polling) ──────────────────────────
    @Test fun spawnTool_syncFallback_whenNoJobId() = runTest {
        val fs = FakeServe(); fs.spawnReturnsJob = false
        val captures = RunCaptures()
        val model = MockLanguageModel(
            responses = listOf(toolTurn("extract_triples", ofEditions("x/y")), toolTurn("done", buildJsonObject { put("verdict", "ok") })),
        )
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("x/y"), hooks = runHooks({}, captures))
        assertTrue("extract_triples" in fs.calls)
        assertEquals(0, fs.pipelinePolls, "no job_id → no polling")
        assertTrue(toolResultText(captures.messages, "extract_triples")!!.contains("sync"))
    }

    // ── done is local — never an HTTP call ────────────────────────────────────────────────────────────
    @Test fun done_isLocal_noHttp() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(responses = listOf(toolTurn("done", buildJsonObject { put("verdict", "ok") })))
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("x/y"))
        assertFalse("done" in fs.calls)
        assertTrue(fs.calls.isEmpty())
    }
}
