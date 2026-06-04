package ai.torad.qgre

import ai.torad.aisdk.ContentPart
import ai.torad.aisdk.PrepareStepScope
import ai.torad.aisdk.StepSettings
import kotlinx.serialization.json.JsonPrimitive

/**
 * The `prepareStep` callback (kandi's `prepareStepBundle.prepareStep` parity). A long edition rebuild
 * accumulates large gate-report + job-status JSON in every tool result; left alone they re-enter the
 * model's context every step and blow the window. This compacts OLDER tool results while keeping the
 * most recent [keepRecent] at full fidelity — so the LATEST gate verdict and recent results stay intact,
 * older ones shrink. It edits only `ToolResult.modelVisible` (the model-facing payload), never the full
 * `output` kept for the record, and never the message structure — so no tool-call/result pair is orphaned.
 */
fun qgrePrepareStep(
    keepRecent: Int = 4,
    trimChars: Int = 320,
): suspend PrepareStepScope<QgreContext>.() -> StepSettings<QgreContext> = prep@{
    val msgs = messages
    val toolMsgIdx = msgs.indices.filter { i -> msgs[i].content.any { it is ContentPart.ToolResult } }
    if (toolMsgIdx.size <= keepRecent) return@prep StepSettings()

    val keepFrom = toolMsgIdx.takeLast(keepRecent).first()
    val compacted = msgs.mapIndexed { i, m ->
        if (i >= keepFrom) {
            m
        } else {
            m.copy(content = m.content.map { part ->
                if (part is ContentPart.ToolResult && !part.isError) {
                    val full = part.modelVisible.toString()
                    if (full.length > trimChars) {
                        part.copy(
                            modelVisible = JsonPrimitive(
                                full.take(trimChars) + " …[trimmed; full result was returned in an earlier step]",
                            ),
                        )
                    } else {
                        part
                    }
                } else {
                    part
                }
            })
        }
    }
    StepSettings(messages = compacted)
}
