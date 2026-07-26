//! Async chat-completions client for grounded generation.
//!
//! Provider policy (permanent, 2026-07-25): **Google AI Studio is plan A**,
//! OpenRouter is plan B. OpenRouter is never preferred when a Studio key exists.
//!
//! Key resolution:
//!   1. $LLM_PROVIDER force (`googleai` | `openrouter`) if set
//!   2. Google AI Studio — $GOOGLE_AI_API_KEY / $GEMINI_API_KEY / libsecret service=googleai
//!   3. OpenRouter — $OPENROUTER_API_KEY / libsecret service=openrouter (key_type=api_key, then bare)
//!
//! Both endpoints speak the OpenAI chat-completions shape. Model ids are normalized:
//!   googleai   → gemma-4-31b-it
//!   openrouter → google/gemma-4-31b-it
//!
//! Module name is historical (`openrouter`) — the type is provider-aware.

use serde::{Deserialize, Serialize};
use std::time::Duration;

const GOOGLE_AI_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_GOOGLEAI_MODEL: &str = "gemma-4-31b-it";
const DEFAULT_OPENROUTER_MODEL: &str = "google/gemma-4-31b-it";

pub struct OpenRouter {
    client: reqwest::Client,
    key: String,
    url: String,
    provider: String,
    pub model: String,
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: Vec<Msg<'a>>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResp {
    #[serde(default)]
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: RespMsg,
}
#[derive(Deserialize, Default)]
struct RespMsg {
    #[serde(default)]
    content: String,
}

fn secret_lookup(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("secret-tool")
        .args(args)
        .output()
        .ok()?;
    let k = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if k.is_empty() {
        None
    } else {
        Some(k)
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn normalize_model(provider: &str, model: &str) -> String {
    if provider == "googleai" {
        if model.is_empty() {
            return DEFAULT_GOOGLEAI_MODEL.to_string();
        }
        if let Some(rest) = model.strip_prefix("google/") {
            return rest.to_string();
        }
        return model.to_string();
    }
    // openrouter
    if model.is_empty() {
        return DEFAULT_OPENROUTER_MODEL.to_string();
    }
    if !model.contains('/') {
        return format!("google/{model}");
    }
    model.to_string()
}

/// Resolve (provider, key, url, model). googleai wins whenever its key is present.
fn resolve(model_in: &str) -> anyhow::Result<(String, String, String, String)> {
    let forced = env_nonempty("LLM_PROVIDER").map(|s| s.to_lowercase());
    if let Some(ref f) = forced {
        if f != "googleai" && f != "openrouter" {
            anyhow::bail!("LLM_PROVIDER={f} invalid — use googleai or openrouter");
        }
    }

    let google_key = env_nonempty("GOOGLE_AI_API_KEY")
        .or_else(|| env_nonempty("GEMINI_API_KEY"))
        .or_else(|| secret_lookup(&["lookup", "service", "googleai"]));

    let openrouter_key = env_nonempty("OPENROUTER_API_KEY")
        .or_else(|| secret_lookup(&["lookup", "service", "openrouter", "key_type", "api_key"]))
        .or_else(|| secret_lookup(&["lookup", "service", "openrouter"]));

    let force_openrouter = forced.as_deref() == Some("openrouter");
    let force_google = forced.as_deref() == Some("googleai");

    if force_google {
        let key = google_key.ok_or_else(|| {
            anyhow::anyhow!(
                "LLM_PROVIDER=googleai but no Studio key \
                 (GOOGLE_AI_API_KEY / GEMINI_API_KEY / keyring service=googleai)"
            )
        })?;
        let model = normalize_model("googleai", model_in);
        eprintln!("[llm] provider=googleai model={model}");
        return Ok(("googleai".into(), key, GOOGLE_AI_URL.into(), model));
    }

    if force_openrouter {
        let key = openrouter_key.ok_or_else(|| {
            anyhow::anyhow!(
                "LLM_PROVIDER=openrouter but no OpenRouter key \
                 (OPENROUTER_API_KEY / keyring service=openrouter)"
            )
        })?;
        let model = normalize_model("openrouter", model_in);
        eprintln!("[llm] provider=openrouter model={model}");
        return Ok(("openrouter".into(), key, OPENROUTER_URL.into(), model));
    }

    // Default: googleai wins whenever present.
    if let Some(key) = google_key {
        let model = normalize_model("googleai", model_in);
        eprintln!("[llm] provider=googleai model={model}");
        return Ok(("googleai".into(), key, GOOGLE_AI_URL.into(), model));
    }
    if let Some(key) = openrouter_key {
        let model = normalize_model("openrouter", model_in);
        eprintln!("[llm] provider=openrouter model={model}");
        return Ok(("openrouter".into(), key, OPENROUTER_URL.into(), model));
    }

    anyhow::bail!(
        "no LLM provider key: Google AI Studio (GOOGLE_AI_API_KEY / keyring service=googleai) \
         plan A, or OpenRouter (OPENROUTER_API_KEY / keyring service=openrouter) plan B"
    )
}

impl OpenRouter {
    pub fn new(model: impl Into<String>) -> anyhow::Result<Self> {
        let model_in = model.into();
        let (provider, key, url, model) = resolve(&model_in)?;
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(180))
                .build()?,
            key,
            url,
            provider,
            model,
        })
    }

    /// One grounded completion with bounded retries + exponential backoff. Returns the assistant text.
    pub async fn complete(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> anyhow::Result<String> {
        let body = ChatReq {
            model: &self.model,
            messages: vec![
                Msg {
                    role: "system",
                    content: system,
                },
                Msg {
                    role: "user",
                    content: user,
                },
            ],
            temperature,
            max_tokens,
        };
        let mut last = String::new();
        for attempt in 0..4u32 {
            let mut req = self
                .client
                .post(&self.url)
                .bearer_auth(&self.key)
                .header("Content-Type", "application/json");
            // OpenRouter-only telemetry headers (harmless if ignored elsewhere).
            if self.provider == "openrouter" {
                req = req
                    .header("HTTP-Referer", "https://kandibuddy.local")
                    .header("X-Title", "festival-factory hydration");
            }
            let resp = req.json(&body).send().await;
            match resp {
                Ok(r) if r.status().is_success() => {
                    let parsed: ChatResp = r.json().await?;
                    let text = parsed
                        .choices
                        .into_iter()
                        .next()
                        .map(|c| c.message.content)
                        .unwrap_or_default();
                    if !text.trim().is_empty() {
                        return Ok(text);
                    }
                    last = "empty content".into();
                }
                Ok(r) => {
                    let code = r.status();
                    let t = r.text().await.unwrap_or_default();
                    last = format!(
                        "HTTP {code}: {}",
                        &t.chars().take(200).collect::<String>()
                    );
                }
                Err(e) => last = format!("{e}"),
            }
            tokio::time::sleep(Duration::from_secs(1u64 << attempt)).await;
        }
        anyhow::bail!("{} failed after retries: {last}", self.provider)
    }
}
