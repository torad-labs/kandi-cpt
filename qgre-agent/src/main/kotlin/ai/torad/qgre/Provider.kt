package ai.torad.qgre

import ai.torad.aisdk.LanguageModel
import ai.torad.aisdk.OpenAICompatibleProviderSettings
import ai.torad.aisdk.createOpenAICompatible
import io.ktor.client.HttpClient

/** Default orchestration model. Reuses the pipeline's known-good OpenRouter slug so the integration
 *  smoke connects out of the box; set QGRE_AGENT_MODEL to a stronger tool-caller for real runs. */
const val DEFAULT_AGENT_MODEL = "google/gemma-4-31b-it"

/** The model that DRIVES the agent loop (distinct from the extraction/hydration models the tools use
 *  internally). OpenRouter via the OpenAI-compatible provider; key from env or the libsecret keyring. */
fun openRouterModel(http: HttpClient): LanguageModel {
    val key = System.getenv("OPENROUTER_API_KEY")?.takeIf { it.isNotBlank() }
        ?: keyringLookup("openrouter")
        ?: error("OPENROUTER_API_KEY not found in env or keyring (secret-tool service=openrouter)")
    val provider = createOpenAICompatible(
        http,
        OpenAICompatibleProviderSettings(
            name = "openrouter",
            baseUrl = "https://openrouter.ai/api/v1",
            apiKey = key,
        ),
    )
    val modelId = System.getenv("QGRE_AGENT_MODEL")?.takeIf { it.isNotBlank() } ?: DEFAULT_AGENT_MODEL
    return provider.languageModel(modelId)
}

/** Fallback to the libsecret keyring (where the rest of the project keeps the OpenRouter key). */
private fun keyringLookup(service: String): String? = try {
    val p = ProcessBuilder("secret-tool", "lookup", "service", service).start()
    val out = p.inputStream.bufferedReader().readText().trim()
    if (p.waitFor() == 0 && out.isNotBlank()) out else null
} catch (_: Exception) {
    null
}
