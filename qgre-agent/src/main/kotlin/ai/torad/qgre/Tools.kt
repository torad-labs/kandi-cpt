package ai.torad.qgre

import ai.torad.aisdk.ToolSet
import ai.torad.aisdk.toolSet
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.File
import java.util.concurrent.TimeUnit

// ── tool inputs (the schema the model fills in) ─────────────────────────────────────────────────────
@Serializable data class InspectInput(val festival: String = "edc")
@Serializable data class EditionInput(val edition: String, val festival: String = "edc")
@Serializable data class VerifyInput(val edition: String, val layer: String, val festival: String = "edc")
@Serializable data class GateInput(val edition: String, val domain: String = "edc", val festival: String = "edc")
@Serializable data class TraceInput(val edition: String, val query: String, val festival: String = "edc")
@Serializable data class RemapInput(val editions: String, val domain: String = "edc")
@Serializable data class CleanInput(val editions: String)
@Serializable data class ExtractInput(val editions: String, val domain: String = "edc")
@Serializable data class HydrateInput(
    val edition: String,
    val relations: String = "",
    val exposureCap: Int = 60,
    val temperature: Double = -1.0,
    val festival: String = "edc",
)
@Serializable data class BashInput(
    val command: String,
    val timeoutSec: Int = 120,
)
@Serializable data class BashOutput(
    val exitCode: Int,
    val stdout: String,
    val stderr: String,
    val timedOut: Boolean = false,
    val rejected: String = "",
)
@Serializable data class WriteFileInput(val path: String, val content: String)
@Serializable data class WriteFileOutput(val path: String, val bytes: Int = 0, val rejected: String = "")
@Serializable data class ReadFileInput(val path: String, val maxBytes: Int = 40000)
@Serializable data class ReadFileOutput(
    val path: String, val content: String = "", val truncated: Boolean = false, val error: String = "",
)
@Serializable data class DoneInput(val summary: String, val verdict: String = "")

// ── bash safety backstop ──────────────────────────────────────────────────────────────────────────────
// Catastrophic-command blocklist — the HARD floor that rejects even under auto-approve (the model is
// gemma in a headless loop; corpus text it reads could carry an injection). It is NOT a full sandbox; the
// real containment is the pinned repo cwd + the timeout + git-backed data + the approval gate in
// interactive mode. This list only stops the irrecoverable: recursive-force delete, privilege escalation,
// disk/device writes, fork bombs, pipe-to-shell remote exec, destructive git, mass signals, host control.
private val BASH_BLOCKLIST: List<Regex> = listOf(
    """\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r""",          // rm -rf / -fr (any flag order)
    """\bsudo\b|\bdoas\b""",                                       // privilege escalation
    """\bmkfs\b|\bof=/dev/|>\s*/dev/(sd|nvme|disk|hd)""",          // disk/device writes
    """:\s*\(\s*\)\s*\{[^}]*:\|:""",                               // classic fork bomb :(){ :|:& };:
    """\b(shutdown|reboot|poweroff|halt)\b|\binit\s+0\b""",        // host control
    """\|\s*(sudo\s+)?(ba)?sh\b""",                                // curl … | sh  (remote exec)
    """\bgit\s+(push|reset\s+--hard|clean\s+-[a-z]*f)""",          // destructive / outward git
    """\bchmod\s+-R\s+0*777""",                                    // world-writable recursive
    """\bpkill\b|\bkillall\b|\bkill\s+-9\s+-1\b""",                // mass process kill
).map { Regex(it, RegexOption.IGNORE_CASE) }

/** Returns a rejection reason if [command] matches the catastrophic blocklist, else null. Pure + testable. */
internal fun bashRejection(command: String): String? =
    BASH_BLOCKLIST.firstOrNull { it.containsMatchIn(command) }
        ?.let { "blocked by safety policy (matched /${it.pattern}/) — rewrite without the destructive op" }

/** Run a bash command in [cwd], capturing stdout/stderr/exit with a hard timeout. Streams are drained on
 *  separate threads so a full pipe buffer can't deadlock the wait. Output is truncated to keep the model's
 *  context bounded. Pulled out of the tool lambda so it is unit-testable without the agent loop. */
internal fun runBash(command: String, cwd: File, timeoutSec: Int): BashOutput {
    bashRejection(command)?.let { return BashOutput(exitCode = -1, stdout = "", stderr = "", rejected = it) }
    val proc = ProcessBuilder("bash", "-lc", command).directory(cwd).redirectErrorStream(false).start()
    val out = StringBuilder(); val err = StringBuilder()
    val tOut = Thread { proc.inputStream.bufferedReader().forEachLine { out.appendLine(it) } }
    val tErr = Thread { proc.errorStream.bufferedReader().forEachLine { err.appendLine(it) } }
    tOut.start(); tErr.start()
    val finished = proc.waitFor(timeoutSec.toLong().coerceIn(1, 600), TimeUnit.SECONDS)
    if (!finished) {
        proc.destroyForcibly(); tOut.join(1000); tErr.join(1000)
        return BashOutput(-1, out.toString().take(8000), err.toString().take(4000), timedOut = true)
    }
    tOut.join(2000); tErr.join(2000)
    return BashOutput(proc.exitValue(), out.toString().take(8000), err.toString().take(4000))
}

// ── repo-pinned file IO (lets the agent author scripts + read files without bash heredoc escaping) ──────
private fun repoRoot(): File = File(System.getProperty("user.dir")).canonicalFile

/** Resolve [path] under the repo root; null if it escapes (traversal or an absolute path outside the repo).
 *  The containment for write_file/read_file — the agent can touch the project tree but nothing above it. */
internal fun resolveInRepo(path: String): File? {
    val root = repoRoot()
    val f = (if (File(path).isAbsolute) File(path) else File(root, path)).canonicalFile
    return if (f == root || f.path.startsWith(root.path + File.separator)) f else null
}

internal fun doWriteFile(path: String, content: String): WriteFileOutput {
    val f = resolveInRepo(path) ?: return WriteFileOutput(path, rejected = "path escapes the repo root — refused")
    f.parentFile?.mkdirs()
    f.writeText(content)
    return WriteFileOutput(f.path, bytes = content.toByteArray().size)
}

internal fun doReadFile(path: String, maxBytes: Int): ReadFileOutput {
    val f = resolveInRepo(path) ?: return ReadFileOutput(path, error = "path escapes the repo root — refused")
    if (!f.exists() || f.isDirectory) return ReadFileOutput(path, error = "no such file")
    val cap = maxBytes.coerceIn(1, 400_000)
    val bytes = f.readBytes()
    return ReadFileOutput(f.path, content = String(bytes.copyOf(minOf(bytes.size, cap))), truncated = bytes.size > cap)
}

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

    // ── FREE in-place fix: deterministic, no LLM, no data loss. Preferred over re-extract when the
    //    edition already has structured data (re-extract rebuilds from clean and can drop the lineup). ──
    tool<RemapInput, String>(
        name = "remap_contract",
        description = "Fix the predicate→object contract on an edition's EXISTING 15_structured IN PLACE: plays_at→festival ⇒ in_lineup (preserving every artist), drop cross-edition contamination, normalize b2b names. FREE + deterministic + lossless — NO re-extract, NO money. PREFER THIS over extract_triples whenever the edition already has structured triples; it keeps the full lineup that a from-scratch re-extract would lose. Then hydrate.",
        strict = false,
    ) { input -> client.read("remap_contract", buildJsonObject { put("editions", input.editions); put("domain", input.domain) }) }

    // ── bash: write+run scripts in the repo for analysis and deterministic data surgery. This is the
    //    lever that lets the agent do the in-between work the fixed actions can't — grep/jq/python over the
    //    corpus, in-place jsonl transforms, custom audits. cwd is pinned to the repo; output + exit are
    //    captured with a hard timeout; catastrophic commands are ALWAYS rejected (bashRejection). Approval-
    //    gated in interactive mode like the paid tools; the blocklist is the floor under auto-approve. ──
    tool<BashInput, BashOutput>(
        name = "bash",
        description = "Run a bash command/script in the repo root; returns {exitCode, stdout, stderr, timedOut, rejected}. Use it for read-only analysis (grep/jq/python/.venv over data/edc/...) and deterministic, no-LLM data transforms (the structural fixes you'd otherwise hand to a human — e.g. canonicalize ids, drop foreign-prefix triples, recount relations). Prefer the dedicated pipeline tools (remap_contract/gate_corpus/...) when one fits. Destructive ops (rm -rf, sudo, device writes, destructive git) are rejected — write non-destructive scripts; data is recoverable from git, but treat 15_structured/25_hydrated as precious. Bound long scans with timeoutSec.",
        needsApproval = if (autoApprove) null else { _, _ -> true },
        strict = false,
    ) { input -> withContext(Dispatchers.IO) { runBash(input.command, File(System.getProperty("user.dir")), input.timeoutSec) } }

    // ── file IO: author a script/data file (write_file) or inspect one (read_file). The ergonomic
    //    complement to bash — the model writes a multi-line Python/jq scraper as structured content
    //    instead of fighting heredoc escaping, then runs it with bash. Both pinned to the repo tree. ──
    tool<WriteFileInput, WriteFileOutput>(
        name = "write_file",
        description = "Write text to a file in the repo (creates parent dirs, overwrites). Use to AUTHOR a script you will then run with bash — e.g. a Python/selectolax/patchright scraper for a new source, or a one-off data-fix over jsonl. Path is pinned to the repo; escapes are refused. Treat 15_structured/25_hydrated as precious — back them up (cp) before overwriting.",
        needsApproval = if (autoApprove) null else { _, _ -> true },
        strict = false,
    ) { input -> withContext(Dispatchers.IO) { doWriteFile(input.path, input.content) } }

    tool<ReadFileInput, ReadFileOutput>(
        name = "read_file",
        description = "Read a file from the repo (bounded by maxBytes, default 40k). Use to inspect a script, a config, an existing scrape source module (scripts/scrape/sources/*.py — adapt these for a new source), or a sample of a data file before acting. Path is pinned to the repo.",
        strict = false,
    ) { input -> withContext(Dispatchers.IO) { doReadFile(input.path, input.maxBytes) } }

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
