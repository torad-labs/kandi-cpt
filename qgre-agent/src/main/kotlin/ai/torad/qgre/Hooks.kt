package ai.torad.qgre

import ai.torad.aisdk.AgentCallHooks
import ai.torad.aisdk.ModelMessage
import ai.torad.aisdk.PendingApproval
import ai.torad.aisdk.StreamEvent

/** What the hooks capture from a run — the message log (for the approval-resume loop) + pending approvals. */
class RunCaptures {
    var messages: List<ModelMessage> = emptyList()
    var pendingApprovals: List<PendingApproval> = emptyList()
    var totalSteps: Int = 0
}

private fun shorten(s: String, n: Int = 400): String = if (s.length > n) s.take(n) + "…" else s

/**
 * The lifecycle-hook wiring (kandi's `AgentCallHooks` + `StreamCaptures` parity). The hooks turn the
 * model's live decisions into the run.log decision stream the `/build` page tails — every step, tool
 * call, and tool result — and capture the final message log + pending approvals so the host can run the
 * approval-resume loop. [log] is the sink (stdout when spawned by `run_pipeline`, which redirects to
 * `pipeline/<job_id>/run.log`).
 */
fun runHooks(log: (String) -> Unit, captures: RunCaptures): AgentCallHooks = AgentCallHooks(
    onStepStart = { e -> log("· step ${e.stepNumber}") },
    onChunk = { e ->
        when (val ev = e.event) {
            is StreamEvent.ToolCall ->
                log("  → ${ev.toolName}(${shorten(ev.inputJson.toString(), 200)})")
            is StreamEvent.ToolResult ->
                log("  ← ${ev.toolName}: ${shorten(ev.outputJson.toString())}")
            is StreamEvent.ToolApprovalRequest ->
                log("  ⏸ approval needed → ${ev.toolName}(${shorten(ev.inputJson.toString(), 200)})")
            else -> Unit
        }
    },
    onStepFinish = { e ->
        val t = e.step.text.trim()
        log("  step ${e.stepNumber} ${e.step.finishReason}${if (t.isNotEmpty()) " · ${shorten(t)}" else ""}")
    },
    onFinish = { e ->
        captures.messages = e.messages
        captures.pendingApprovals = e.pendingApprovals
        captures.totalSteps = e.totalSteps
        log("✓ ${e.totalSteps} step(s) · ${e.pendingApprovals.size} pending approval(s)")
    },
    onError = { e -> log("✗ error: $e") },
)
