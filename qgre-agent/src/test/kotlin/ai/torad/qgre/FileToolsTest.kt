package ai.torad.qgre

import ai.torad.aisdk.providers.MockLanguageModel
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** write_file / read_file: repo-pinned file IO so the agent authors scripts as structured content and reads
 *  files, without bash heredoc escaping. The containment (resolveInRepo) and the IO are pure + the write→run
 *  loop is proven through the real agent loop + real shell. */
class FileToolsTest {

    @Test
    fun resolveInRepo_keeps_paths_inside_the_repo_and_rejects_escapes() {
        assertNotNull(resolveInRepo("build/scratch/x.txt"), "an in-repo path resolves")
        assertNotNull(resolveInRepo("src/main/kotlin/ai/torad/qgre/Tools.kt"), "an existing in-repo path resolves")
        assertNull(resolveInRepo("../../../etc/passwd"), "traversal out of the repo is refused")
        assertNull(resolveInRepo("/etc/passwd"), "an absolute path outside the repo is refused")
    }

    @Test
    fun writeFile_then_readFile_roundtrips() {
        val path = "build/test-scratch/probe.txt"
        val w = doWriteFile(path, "qgre-roundtrip-42")
        assertEquals("", w.rejected)
        assertTrue(w.bytes > 0)
        val r = doReadFile(path, 40_000)
        assertEquals("", r.error)
        assertEquals("qgre-roundtrip-42", r.content)
        assertTrue(!r.truncated)
    }

    @Test
    fun writeFile_refuses_to_escape_the_repo() {
        val w = doWriteFile("../../../tmp/qgre-should-not-write.txt", "nope")
        assertTrue(w.rejected.isNotEmpty(), "writing outside the repo must be refused")
        assertEquals(0, w.bytes)
    }

    @Test
    fun readFile_reports_a_missing_file() {
        val r = doReadFile("build/test-scratch/definitely-absent-xyz.txt", 1000)
        assertTrue(r.error.isNotEmpty())
    }

    @Test
    fun readFile_truncates_at_maxBytes() {
        doWriteFile("build/test-scratch/big.txt", "x".repeat(5000))
        val r = doReadFile("build/test-scratch/big.txt", 100)
        assertEquals(100, r.content.length)
        assertTrue(r.truncated)
    }

    // ── the capability proof: the agent AUTHORS a script with write_file, then RUNS it with bash ──
    @Test
    fun agent_loop_writes_a_script_then_runs_it() = runTest {
        val fs = FakeServe()
        val script = "print('MARK-' + str(6 * 7))"   // structured content — no heredoc escaping
        val model = MockLanguageModel(responses = listOf(
            toolTurn("write_file", buildJsonObject {
                put("path", "build/test-scratch/agent_probe.py"); put("content", script)
            }),
            toolTurn("bash", buildJsonObject { put("command", "python3 build/test-scratch/agent_probe.py") }),
            toolTurn("done", buildJsonObject { put("verdict", "ok") }),
        ))
        val res = buildPipelineAgent(model, qgreTools(fs.client(), autoApprove = true))
            .generate(prompt = "go", options = testCtx("x/y"))
        val out = toolResultText(res.messages, "bash") ?: error("no bash result")
        assertTrue("MARK-42" in out, "the agent wrote a script with write_file then ran it; saw: ${out.take(200)}")
    }
}
