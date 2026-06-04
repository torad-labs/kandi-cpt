package ai.torad.qgre

import ai.torad.aisdk.FinishReason
import ai.torad.aisdk.StreamEvent
import ai.torad.aisdk.providers.ScriptedResponse
import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.content.TextContent
import io.ktor.http.headersOf
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * A fake serve.py for the agent's HTTP bridge. Records every action call (name + request body, so tests
 * can assert the typed-input → snake_case-arg mapping), scripts gate verdicts and per-action results/errors,
 * and scripts the job-status poll sequence. Construct, configure, then call [client] to get a wired ToradClient.
 */
class FakeServe {
    /** Action names invoked, in order (the behavioral trace). `done` is local → never appears here. */
    val calls = mutableListOf<String>()

    /** Per-action request bodies (the JSON the tool sent), in order. */
    val bodies = mutableMapOf<String, MutableList<String>>()

    /** gate_corpus results, consumed FIFO (falls back to a clean pass when empty). */
    val gate = ArrayDeque<String>()

    /** Override the `result` JSON for a read action (default `{"read":"<name>"}`). */
    val results = HashMap<String, String>()

    /** Force `{ok:false}` for an action → (message, kind). e.g. ("Mutex held…", "gpu_busy"). */
    val errors = HashMap<String, Pair<String, String?>>()

    /** When false, spawn actions return a terminal result WITHOUT a job_id (exercises the sync fallback). */
    var spawnReturnsJob = true

    /** Statuses returned by successive /api/pipeline/<job> polls (last one repeats). */
    var jobStatusSeq: List<String> = listOf("done")

    /** How many times /api/pipeline/<job> was polled (proves async tools waited on the job). */
    var pipelinePolls = 0
        private set
    private var jobPolls = 0

    fun client(): ToradClient {
        val engine = MockEngine { request ->
            val path = request.url.encodedPath
            val jh = headersOf(HttpHeaders.ContentType, "application/json")
            when {
                path.startsWith("/api/action/") -> {
                    val name = path.removePrefix("/api/action/")
                    calls += name
                    bodies.getOrPut(name) { mutableListOf() }.add((request.body as? TextContent)?.text ?: "")
                    val err = errors[name]
                    if (err != null) {
                        val kindPart = err.second?.let { ""","kind":"$it"""" } ?: ""
                        respond(
                            """{"ok":false,"error":"${err.first}"$kindPart}""",
                            HttpStatusCode.Conflict, jh,
                        )
                    } else {
                        val result = when {
                            name == "gate_corpus" -> gate.removeFirstOrNull() ?: """{"overall":"pass","gates":{}}"""
                            name in SPAWN_ACTIONS && spawnReturnsJob -> """{"job_id":"job_${calls.size}"}"""
                            name in SPAWN_ACTIONS -> """{"status":"done","sync":true}"""
                            results.containsKey(name) -> results.getValue(name)
                            else -> """{"read":"$name"}"""
                        }
                        respond("""{"ok":true,"result":$result}""", HttpStatusCode.OK, jh)
                    }
                }
                path.startsWith("/api/pipeline/") -> {
                    val status = jobStatusSeq[jobPolls.coerceAtMost(jobStatusSeq.size - 1)]
                    jobPolls += 1
                    pipelinePolls += 1
                    respond("""{"status":"$status"}""", HttpStatusCode.OK, jh)
                }
                else -> respond("{}", HttpStatusCode.NotFound, jh)
            }
        }
        return ToradClient(http = HttpClient(engine))
    }

    /** The single request body sent for [action] (asserts exactly one was sent). */
    fun bodyOf(action: String): String {
        val list = bodies[action] ?: error("no request recorded for action '$action'")
        return list.single()
    }

    companion object {
        val SPAWN_ACTIONS = setOf("clean_layer", "extract_triples", "hydrate")
    }
}

/** One scripted model turn that calls a tool (the model's decision). */
fun toolTurn(name: String, args: JsonObject, id: String = "c_$name"): ScriptedResponse =
    ScriptedResponse(
        events = listOf(StreamEvent.ToolCall(id, name, args)),
        finishReason = FinishReason.ToolCalls,
    )

fun ofEdition(name: String): JsonObject = buildJsonObject { put("edition", name) }
fun ofEditions(name: String): JsonObject = buildJsonObject { put("editions", name) }

fun testCtx(edition: String, protectedEd: Boolean = false): QgreContext =
    QgreContext(edition = edition, protectedEditions = if (protectedEd) listOf(edition) else emptyList())

/** The model-facing result string a tool returned, pulled from the final message log. */
fun toolResultText(messages: List<ai.torad.aisdk.ModelMessage>, toolName: String): String? =
    messages.asReversed().flatMap { it.content }
        .filterIsInstance<ai.torad.aisdk.ContentPart.ToolResult>()
        .firstOrNull { it.toolName == toolName }
        ?.output?.toString()
