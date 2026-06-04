package ai.torad.qgre

import ai.torad.aisdk.ToolSet
import ai.torad.aisdk.toolSet
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

// ── tool inputs (the schema the model fills in) ─────────────────────────────────────────────────────
@Serializable data class InspectInput(val festival: String = "edc")
@Serializable data class EditionInput(val edition: String, val festival: String = "edc")
@Serializable data class VerifyInput(val edition: String, val layer: String, val festival: String = "edc")
@Serializable data class GateInput(val edition: String, val domain: String = "edc", val festival: String = "edc")
@Serializable data class TraceInput(val edition: String, val query: String, val festival: String = "edc")
@Serializable data class CleanInput(val editions: String)
@Serializable data class ExtractInput(val editions: String, val domain: String = "edc")
@Serializable data class HydrateInput(
    val edition: String,
    val relations: String = "",
    val exposureCap: Int = 60,
    val temperature: Double = -1.0,
    val festival: String = "edc",
)
@Serializable data class DoneInput(val summary: String, val verdict: String = "")

// ── action-call helpers (the model never sees these — they map typed input → the Python action) ──────
private suspend fun ToradClient.read(name: String, args: JsonObject): String =
    try {
        postAction(name, args).toString()
    } catch (e: ToradError) {
        "ERROR${e.kind?.let { " [$it]" } ?: ""}: ${e.message}"
    }

/** Spawn an async pipeline action (clean/extract/hydrate), then BLOCK on its job until terminal, so one
 *  tool call = one completed step. Returns the final job status (or the sync result if no job spawned). */
private suspend fun ToradClient.spawnAndWait(name: String, args: JsonObject): String =
    try {
        val res = postAction(name, args)
        val jobId = (res as? JsonObject)?.get("job_id")?.jsonPrimitive?.contentOrNull
        if (jobId == null) res.toString() else pollJob(jobId).toString()
    } catch (e: ToradError) {
        "ERROR${e.kind?.let { " [$it]" } ?: ""}: ${e.message}"
    }

/**
 * The agent's action space over the qgre pipeline. Read tools are free; the three paid spine tools
 * (clean/extract/hydrate) are approval-gated unless [autoApprove]. The model chooses which to call —
 * there is no hardcoded order. `force=true` on the spine tools is a rebuild invariant (the server-side
 * guard backs up before replacing); the protected gold is refused server-side regardless.
 */
fun qgreTools(client: ToradClient, autoApprove: Boolean): ToolSet<QgreContext> = toolSet {

    // strict=false: our tool inputs have optional (default-valued) fields; OpenAI's strict function-calling
    // rejects schemas whose `required` omits any property (a 400), so relax it to stay model-agnostic
    // (gemma/claude accept either; OpenAI needs this). Validation still happens at the executor.
    tool<InspectInput, String>(
        name = "inspect_corpus",
        description = "List every edition of a festival with its sources and per-layer line counts (raw/structured/hydrated). Use to see overall state.",
        strict = false,
    ) { input -> client.read("inspect_corpus", buildJsonObject { put("festival", input.festival) }) }

    tool<EditionInput, String>(
        name = "read_edition",
        description = "Show one edition's sources × layers and file lists. Use to inspect what an edition already has before deciding to clean/extract/hydrate.",
        strict = false,
    ) { input -> client.read("read_edition", buildJsonObject { put("edition", input.edition); put("festival", input.festival) }) }

    tool<VerifyInput, String>(
        name = "verify_layer",
        description = "Integrity of one layer for one edition: line count, dedup ratio, staleness vs upstream. layer is e.g. 10_clean, 15_structured, 25_hydrated.",
        strict = false,
    ) { input -> client.read("verify_layer", buildJsonObject { put("festival", input.festival); put("edition", input.edition); put("layer", input.layer) }) }

    tool<GateInput, String>(
        name = "gate_corpus",
        description = "Run the deterministic quality-gate suite on an edition (balance, exposure_cap, id_integrity, faithfulness, source_vs_gap). The verdict tells you what (if anything) to repair. ALWAYS gate after a spine step.",
        strict = false,
    ) { input -> client.read("gate_corpus", buildJsonObject { put("festival", input.festival); put("edition", input.edition); put("domain", input.domain) }) }

    tool<TraceInput, String>(
        name = "trace_fact",
        description = "Lineage of one fact/entity DOWN the pipeline: which sources mention it → triples → hydrated lines + exposure. Use to decide whether a missing relation is a source-gap (no input data) or a defect (data present).",
        strict = false,
    ) { input -> client.read("trace_fact", buildJsonObject { put("festival", input.festival); put("edition", input.edition); put("query", input.query) }) }

    // ── paid spine tools — approval-gated unless autoApprove. Each blocks until its job finishes. ──
    tool<CleanInput, String>(
        name = "clean_layer",
        description = "Re-derive 10_clean from 00_raw for an edition (editions='city/year'). Cheap-ish (mostly deterministic + occasional LLM rescue). Backs up the prior layer first.",
        needsApproval = if (autoApprove) null else { _, _ -> true },
        strict = false,
    ) { input -> client.spawnAndWait("clean_layer", buildJsonObject { put("editions", input.editions); put("llm_rescue", true); put("force", true) }) }

    tool<ExtractInput, String>(
        name = "extract_triples",
        description = "LLM structured extraction 10_clean → 15_structured for an edition (editions='city/year'). COSTS MONEY. Enforces the predicate contract + normalizes names. Backs up first; refuses protected editions.",
        needsApproval = if (autoApprove) null else { _, _ -> true },
        strict = false,
    ) { input -> client.spawnAndWait("extract_triples", buildJsonObject { put("editions", input.editions); put("domain", input.domain); put("force", true) }) }

    tool<HydrateInput, String>(
        name = "hydrate",
        description = "Hydrate 15_structured → 25_hydrated teaching text via festival-factory. COSTS MONEY (most expensive). Pass relations='set_time' (CSV) to top-up just one relation; exposureCap bounds redundancy; temperature<0 = engine default. Backs up first; refuses protected editions.",
        needsApproval = if (autoApprove) null else { _, _ -> true },
        strict = false,
    ) { input ->
        client.spawnAndWait("hydrate", buildJsonObject {
            put("festival", input.festival)
            put("edition", input.edition)
            put("exposure_cap", input.exposureCap)
            put("force", true)
            if (input.relations.isNotBlank()) put("relations", input.relations)
            if (input.temperature >= 0.0) put("temperature", input.temperature)
        })
    }

    tool<DoneInput, String>(
        name = "done",
        description = "Call when this edition's gates are satisfied OR nothing further is improvable by re-running tools (remaining issues are honest source-gaps, not defects). Provide a summary + final verdict. Ends work on this edition.",
        strict = false,
    ) { input -> "DONE: ${input.verdict} — ${input.summary}" }
}
