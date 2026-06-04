package ai.torad.qgre

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.contentType
import kotlinx.coroutines.delay
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** A {ok:false} action response from serve.py — carries the actionable message + optional kind (e.g. "gpu_busy"). */
class ToradError(message: String, val kind: String? = null) : RuntimeException(message)

/**
 * HTTP bridge to the running `scripts/serve.py` action front-door. Every qgre pipeline tool the agent
 * calls is dispatched here as `POST /api/action/<name>` → `{ok, result}` (or `{ok:false, error, kind}`),
 * and spawned jobs are polled via `GET /api/pipeline/<job_id>`. This is the seam between the Kotlin agent
 * and the Python pipeline — no drift from the UI, which calls the same catalog.
 */
class ToradClient(
    private val baseUrl: String = System.getenv("QGRE_SERVE_URL") ?: "http://127.0.0.1:8800",
    private val writeToken: String? = System.getenv("TORAD_WRITE_TOKEN"),
    private val http: HttpClient = HttpClient(CIO) {
        install(HttpTimeout) {
            requestTimeoutMillis = 120_000
            connectTimeoutMillis = 10_000
        }
    },
) {
    private val json = Json { ignoreUnknownKeys = true }

    /**
     * Dispatch one action. Returns the action's `result` payload as a JsonElement.
     * Throws [ToradError] on `{ok:false}` (the agent surfaces that to the model so it can react —
     * e.g. a `gpu_busy` kind means wait/stop the holder).
     */
    suspend fun postAction(name: String, args: JsonObject = JsonObject(emptyMap())): JsonElement {
        val text = http.post("$baseUrl/api/action/$name") {
            contentType(ContentType.Application.Json)
            writeToken?.let { header(HttpHeaders.Authorization, "Bearer $it") }
            setBody(args.toString())
        }.bodyAsText()
        val body = json.parseToJsonElement(text).jsonObject
        if (body["ok"]?.jsonPrimitive?.booleanOrNull == false) {
            throw ToradError(
                body["error"]?.jsonPrimitive?.contentOrNull ?: "action '$name' failed",
                body["kind"]?.jsonPrimitive?.contentOrNull,
            )
        }
        return body["result"] ?: JsonNull
    }

    /** One status read for a spawned pipeline job. */
    suspend fun jobStatus(jobId: String): JsonObject =
        json.parseToJsonElement(http.get("$baseUrl/api/pipeline/$jobId").bodyAsText()).jsonObject

    /**
     * Poll a spawned job (clean/extract/hydrate) to a terminal state. The pipeline marks a job "done"
     * when its pid exits; anything other than "running" is terminal (done/error/stopped). Backs off
     * between polls and stops at [maxMs] (the agent's wall-clock budget for a single tool call).
     */
    suspend fun pollJob(jobId: String, pollMs: Long = 4_000, maxMs: Long = 6L * 3600 * 1000): JsonObject {
        val start = System.currentTimeMillis()
        while (true) {
            val st = jobStatus(jobId)
            val status = st["status"]?.jsonPrimitive?.contentOrNull
            if (status != null && status != "running" && status != "starting") return st
            if (System.currentTimeMillis() - start > maxMs) return st
            delay(pollMs)
        }
    }

    fun close() = http.close()
}
