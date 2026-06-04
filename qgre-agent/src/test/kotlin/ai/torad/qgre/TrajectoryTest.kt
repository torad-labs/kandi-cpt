package ai.torad.qgre

import ai.torad.aisdk.providers.MockLanguageModel
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Workflow-level evals beyond the core happy/repair paths: the other gate-driven repair shapes, a tool
 * error the model reacts to, and a multi-edition run. The model is scripted; the loop is real.
 */
class TrajectoryTest {

    @Test
    fun faithfulnessFail_modelReHydrates_thenPasses() = runTest {
        val fs = FakeServe()
        fs.gate.addAll(
            listOf(
                """{"overall":"fail","gates":{"faithfulness":{"severity":"fail"}}}""",
                """{"overall":"pass","gates":{"faithfulness":{"severity":"pass"}}}""",
            ),
        )
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("hydrate", ofEdition("seoul/2019"), id = "h1"),
                toolTurn("gate_corpus", ofEdition("seoul/2019"), id = "g1"),
                toolTurn("hydrate", buildJsonObject { put("edition", "seoul/2019"); put("temperature", 0.1) }, id = "h2"),
                toolTurn("gate_corpus", ofEdition("seoul/2019"), id = "g2"),
                toolTurn("done", buildJsonObject { put("verdict", "pass") }),
            ),
        )
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true)).generate(prompt = "go", options = testCtx("seoul/2019"))
        assertEquals(2, fs.calls.count { it == "hydrate" })
        assertEquals(2, fs.calls.count { it == "gate_corpus" })
        // the repair hydrate carried the lower temperature the model chose
        val secondHydrate = Json.parseToJsonElement(fs.bodies.getValue("hydrate")[1]).jsonObject
        assertEquals(0.1, secondHydrate["temperature"]!!.jsonPrimitive.content.toDouble())
    }

    @Test
    fun balanceDefect_modelTopsUpThatRelation() = runTest {
        val fs = FakeServe()
        fs.gate.addAll(
            listOf(
                """{"overall":"fail","gates":{"balance":{"severity":"fail",
                   "metrics":{"under_floor_defects":[{"relation":"set_time","frac":0.0,"floor":0.01}]}},
                   "source_vs_gap":{"source_gap_vs_defect":{}}}}""".trimIndent(),
                """{"overall":"pass","gates":{"balance":{"severity":"pass"}}}""",
            ),
        )
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("hydrate", ofEdition("dallas/2011"), id = "h1"),
                toolTurn("gate_corpus", ofEdition("dallas/2011"), id = "g1"),
                toolTurn("hydrate", buildJsonObject { put("edition", "dallas/2011"); put("relations", "set_time") }, id = "h2"),
                toolTurn("gate_corpus", ofEdition("dallas/2011"), id = "g2"),
                toolTurn("done", buildJsonObject { put("verdict", "pass") }),
            ),
        )
        buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true)).generate(prompt = "go", options = testCtx("dallas/2011"))
        assertEquals(2, fs.calls.count { it == "hydrate" })
        val topUp = Json.parseToJsonElement(fs.bodies.getValue("hydrate")[1]).jsonObject
        assertEquals("set_time", topUp["relations"]!!.jsonPrimitive.content)
    }

    @Test
    fun toolError_isSurfaced_modelReactsByCallingDone() = runTest {
        val fs = FakeServe()
        fs.errors["extract_triples"] = "Mutex held: run on GPU0" to "gpu_busy"
        val captures = RunCaptures()
        val model = MockLanguageModel(
            responses = listOf(
                toolTurn("extract_triples", ofEditions("denver/2008")),
                toolTurn("done", buildJsonObject { put("verdict", "blocked"); put("summary", "extract errored") }),
            ),
        )
        val res = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("denver/2008"), hooks = runHooks({}, captures))
        assertTrue("extract_triples" in fs.calls)
        assertTrue(toolResultText(captures.messages, "extract_triples")!!.contains("gpu_busy"))
        assertEquals(2, res.steps.size, "model reacted to the error by finishing")
    }

    @Test
    fun multiEdition_runsEachIndependently() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(
            responses = listOf(
                // edition A
                toolTurn("gate_corpus", ofEdition("orlando/2022"), id = "ga"),
                toolTurn("done", buildJsonObject { put("verdict", "a") }),
                // edition B (same model instance continues)
                toolTurn("gate_corpus", ofEdition("orlando/2023"), id = "gb"),
                toolTurn("done", buildJsonObject { put("verdict", "b") }),
            ),
        )
        val agent = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
        runEdition(agent, testCtx("orlando/2022"), isProtected = false, approve = { true }, log = {})
        runEdition(agent, testCtx("orlando/2023"), isProtected = false, approve = { true }, log = {})
        assertEquals(2, fs.calls.count { it == "gate_corpus" }, "each edition got its own gate pass")
    }
}
