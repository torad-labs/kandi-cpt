package ai.torad.qgre

import ai.torad.aisdk.providers.MockLanguageModel
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The tool SET: the full action space is present, and approval gating applies only to the paid spine
 *  tools and only when not auto-approved. */
class ToolSetTest {

    private val expected = setOf(
        "inspect_corpus", "read_edition", "verify_layer", "gate_corpus", "trace_fact",
        "clean_layer", "extract_triples", "hydrate", "done",
    )

    @Test
    fun toolSet_carriesExactlyTheNineTools() {
        val names = qgreTools(FakeServe().client(), autoApprove = true).names().toSet()
        assertEquals(expected, names)
    }

    @Test
    fun readTool_runsWithoutApproval_evenWhenGatingOn() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(
            responses = listOf(toolTurn("gate_corpus", ofEdition("x/y")), toolTurn("done", buildJsonObject { put("verdict", "ok") })),
        )
        val res = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = false))
            .generate(prompt = "go", options = testCtx("x/y"))
        assertTrue(res.pendingApprovals.isEmpty(), "read tools are never approval-gated")
        assertTrue("gate_corpus" in fs.calls)
    }

    @Test
    fun paidTool_runsWithoutPause_whenAutoApprove() = runTest {
        val fs = FakeServe()
        val model = MockLanguageModel(
            responses = listOf(toolTurn("extract_triples", ofEditions("x/y")), toolTurn("done", buildJsonObject { put("verdict", "ok") })),
        )
        val res = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("x/y"))
        assertTrue(res.pendingApprovals.isEmpty(), "autoApprove removes the gate")
        assertTrue("extract_triples" in fs.calls)
    }
}
