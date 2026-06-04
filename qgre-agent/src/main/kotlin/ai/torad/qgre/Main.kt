package ai.torad.qgre

import ai.torad.aisdk.PendingApproval
import ai.torad.aisdk.ToolLoopAgent
import ai.torad.aisdk.toolApprovalResponseMessage
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import kotlinx.coroutines.runBlocking
import java.time.LocalTime

/**
 * The pure qgre data-pipeline agent. Spawned by the `run_pipeline` action (or run by hand) to drive the
 * rebuild of one or more editions — the MODEL decides each next tool call against the deterministic gates.
 * Reasoning + tool calls stream to stdout, which `run_pipeline` redirects to pipeline/<job_id>/run.log.
 *
 *   gradle -p qgre-agent run --args="--domain edc --editions las_vegas/2024,orlando/2024 --job-id J"
 */
fun main(args: Array<String>) = runBlocking {
    val a = parseArgs(args)
    val autoApprove = System.getenv("QGRE_AUTO_APPROVE")?.lowercase() in setOf("1", "true", "yes", "on")
    val log: (String) -> Unit = { msg -> println("[${LocalTime.now().withNano(0)}] $msg") }

    val httpProvider = HttpClient(CIO)
    val client = ToradClient()
    try {
        val model = openRouterModel(httpProvider)
        val agent = buildPipelineAgent(model, qgreTools(client, autoApprove))
        val modelId = System.getenv("QGRE_AGENT_MODEL") ?: DEFAULT_AGENT_MODEL
        log("qgre-agent · domain=${a.domain} · editions=${a.editions} · model=$modelId · autoApprove=$autoApprove")

        val approve: suspend (PendingApproval) -> Boolean =
            if (autoApprove) { _ -> true } else { pa -> askApproval(pa, log) }
        for (ed in a.editions) {
            val isProtected = ed in a.protectedEditions
            val ctx = QgreContext(
                domain = a.domain, festival = a.domain, edition = ed,
                protectedEditions = a.protectedEditions, jobId = a.jobId,
            )
            log("════ edition $ed${if (isProtected) " (PROTECTED — gate-only)" else ""} ════")
            try {
                log("── $ed result: ${runEdition(agent, ctx, isProtected, approve, log)}")
            } catch (e: Throwable) {
                log("✗ $ed crashed: ${e::class.simpleName}: ${e.message}")
            }
        }
        log("DONE · ${a.editions.size} edition(s) processed. Tokenize + CPT remain a separate, user-triggered step.")
    } finally {
        client.close()
        httpProvider.close()
    }
}

/**
 * Run the agent on one edition, threading the approval-resume loop (paid tools pause when not
 * auto-approved). The [approve] decision is injected so the resume flow is testable without stdin.
 */
internal suspend fun runEdition(
    agent: ToolLoopAgent<QgreContext, String>,
    ctx: QgreContext,
    isProtected: Boolean,
    approve: suspend (PendingApproval) -> Boolean,
    log: (String) -> Unit,
): String {
    val captures = RunCaptures()
    val prompt = if (isProtected) {
        "Edition: ${ctx.edition}. This is PROTECTED hand-built gold — run gate_corpus to verify it ONLY, " +
            "never extract or hydrate it, then call done with the verdict."
    } else {
        "Edition: ${ctx.edition}. Drive it to passing gates: inspect, build/repair the spine as the gates " +
            "require, and call done when settled."
    }

    var result = agent.generate(prompt = prompt, options = ctx, hooks = runHooks(log, captures))
    var guard = 0
    while (result.pendingApprovals.isNotEmpty() && guard++ < 50) {
        val responses = result.pendingApprovals.map { pa ->
            val approved = approve(pa)
            log("  ${if (approved) "✓ approved" else "✗ denied"} ${pa.toolName}")
            toolApprovalResponseMessage(pa.toolCallId, approved = approved)
        }
        result = agent.generate(messages = result.messages + responses, options = ctx, hooks = runHooks(log, captures))
    }
    return result.text.ifBlank { "(no text; finishReason=${result.finishReason})" }
}

private fun askApproval(pa: PendingApproval, log: (String) -> Unit): Boolean {
    log("  ⏸ approve paid tool ${pa.toolName}? [y/N]")
    val line = readlnOrNull()?.trim()?.lowercase()
    return line == "y" || line == "yes"
}

internal data class Args(
    val domain: String,
    val editions: List<String>,
    val protectedEditions: List<String>,
    val jobId: String,
)

internal fun parseArgs(args: Array<String>): Args {
    val m = HashMap<String, String>()
    var i = 0
    while (i < args.size) {
        val k = args[i]
        if (k.startsWith("--") && i + 1 < args.size) {
            m[k.removePrefix("--")] = args[i + 1]; i += 2
        } else {
            i += 1
        }
    }
    fun list(key: String, default: List<String>) =
        m[key]?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() } ?: default
    return Args(
        domain = m["domain"] ?: "edc",
        editions = list("editions", emptyList()),
        protectedEditions = list("protected", listOf("las_vegas/2026")),
        jobId = m["job-id"] ?: "",
    )
}
