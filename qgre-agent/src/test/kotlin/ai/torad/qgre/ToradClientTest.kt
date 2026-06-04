package ai.torad.qgre

import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** Unit eval for the serve.py HTTP bridge — success unwrap, error surfacing (incl. gpu_busy), bearer
 *  token, and async-job polling to a terminal state. */
class ToradClientTest {

    private val jh = headersOf(HttpHeaders.ContentType, "application/json")

    @Test
    fun postAction_unwrapsResult() = runTest {
        val c = ToradClient(http = HttpClient(MockEngine { respond("""{"ok":true,"result":{"n_editions":98}}""", HttpStatusCode.OK, jh) }))
        val r = c.postAction("inspect_corpus")
        assertEquals(98, r.jsonObject["n_editions"]!!.jsonPrimitive.int)
    }

    @Test
    fun postAction_okFalse_throwsToradErrorWithKind() = runTest {
        val c = ToradClient(http = HttpClient(MockEngine { respond("""{"ok":false,"error":"Mutex held: run on GPU0","kind":"gpu_busy"}""", HttpStatusCode.Conflict, jh) }))
        val e = assertFailsWith<ToradError> { c.postAction("launch_run") }
        assertEquals("gpu_busy", e.kind)
        assertTrue(e.message!!.contains("Mutex held"))
    }

    @Test
    fun postAction_okFalse_noKind() = runTest {
        val c = ToradClient(http = HttpClient(MockEngine { respond("""{"ok":false,"error":"bad args"}""", HttpStatusCode.BadRequest, jh) }))
        val e = assertFailsWith<ToradError> { c.postAction("hydrate") }
        assertEquals(null, e.kind)
        assertTrue(e.message!!.contains("bad args"))
    }

    @Test
    fun postAction_sendsBearerToken_whenSet() = runTest {
        var seen: String? = null
        val c = ToradClient(
            writeToken = "s3cret",
            http = HttpClient(MockEngine { req -> seen = req.headers[HttpHeaders.Authorization]; respond("""{"ok":true,"result":{}}""", HttpStatusCode.OK, jh) }),
        )
        c.postAction("gate_corpus")
        assertEquals("Bearer s3cret", seen)
    }

    @Test
    fun postAction_noAuthHeader_whenNoToken() = runTest {
        var hasAuth = true
        val c = ToradClient(
            writeToken = null,
            http = HttpClient(MockEngine { req -> hasAuth = req.headers.contains(HttpHeaders.Authorization); respond("""{"ok":true,"result":{}}""", HttpStatusCode.OK, jh) }),
        )
        c.postAction("gate_corpus")
        assertEquals(false, hasAuth)
    }

    @Test
    fun pollJob_waitsThroughRunning_thenReturnsDone() = runTest {
        val statuses = ArrayDeque(listOf("running", "running", "done"))
        val c = ToradClient(http = HttpClient(MockEngine {
            val s = if (statuses.size > 1) statuses.removeFirst() else statuses.first()
            respond("""{"status":"$s"}""", HttpStatusCode.OK, jh)
        }))
        val st = c.pollJob("job1", pollMs = 1)
        assertEquals("done", st["status"]!!.jsonPrimitive.content)
    }

    @Test
    fun pollJob_terminalOnError() = runTest {
        val c = ToradClient(http = HttpClient(MockEngine { respond("""{"status":"error","detail":"boom"}""", HttpStatusCode.OK, jh) }))
        val st = c.pollJob("jobX", pollMs = 1)
        assertEquals("error", st["status"]!!.jsonPrimitive.content)
    }

    @Test
    fun pollJob_treatsStartingAsNonTerminal() = runTest {
        val statuses = ArrayDeque(listOf("starting", "done"))
        val c = ToradClient(http = HttpClient(MockEngine {
            val s = if (statuses.size > 1) statuses.removeFirst() else statuses.first()
            respond("""{"status":"$s"}""", HttpStatusCode.OK, jh)
        }))
        assertEquals("done", c.pollJob("j", pollMs = 1)["status"]!!.jsonPrimitive.content)
    }

    @Test
    fun postAction_missingResult_returnsJsonNull() = runTest {
        val c = ToradClient(http = HttpClient(MockEngine { respond("""{"ok":true}""", HttpStatusCode.OK, jh) }))
        assertEquals(kotlinx.serialization.json.JsonNull, c.postAction("ping"))
    }
}
