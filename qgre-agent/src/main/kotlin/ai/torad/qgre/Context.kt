package ai.torad.qgre

import kotlinx.serialization.Serializable

/**
 * Typed context propagated through the agent loop (set via `callOptionsSchema`, passed to `generate`).
 * The tools and the system prompt read it so every step is anchored to the edition under work and the
 * protected gold it must never regenerate.
 */
@Serializable
data class QgreContext(
    val domain: String = "edc",
    val festival: String = "edc",
    val edition: String = "",
    val protectedEditions: List<String> = emptyList(),
    val jobId: String = "",
)
