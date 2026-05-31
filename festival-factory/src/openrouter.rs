//! Async OpenRouter client for grounded generation. Key resolution follows CLAUDE.md §9:
//! env `OPENROUTER_API_KEY` first, then libsecret via `secret-tool` (service=openrouter,
//! key_type=api_key) — the canonical store on this machine.

use serde::{Deserialize, Serialize};
use std::time::Duration;

const URL: &str = "https://openrouter.ai/api/v1/chat/completions";

pub struct OpenRouter {
    client: reqwest::Client,
    key: String,
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

fn resolve_key() -> anyhow::Result<String> {
    if let Ok(k) = std::env::var("OPENROUTER_API_KEY") {
        if !k.trim().is_empty() {
            return Ok(k.trim().to_string());
        }
    }
    // libsecret fallback (CLAUDE.md §9): the key lives under service=openrouter, key_type=api_key.
    let out = std::process::Command::new("secret-tool")
        .args(["lookup", "service", "openrouter", "key_type", "api_key"])
        .output();
    if let Ok(o) = out {
        let k = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !k.is_empty() {
            return Ok(k);
        }
    }
    anyhow::bail!("OpenRouter key not found in $OPENROUTER_API_KEY or libsecret (service=openrouter, key_type=api_key)")
}

impl OpenRouter {
    pub fn new(model: impl Into<String>) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder().timeout(Duration::from_secs(180)).build()?,
            key: resolve_key()?,
            model: model.into(),
        })
    }

    /// One grounded completion with bounded retries + exponential backoff. Returns the assistant text.
    pub async fn complete(&self, system: &str, user: &str, temperature: f32, max_tokens: u32) -> anyhow::Result<String> {
        let body = ChatReq {
            model: &self.model,
            messages: vec![
                Msg { role: "system", content: system },
                Msg { role: "user", content: user },
            ],
            temperature,
            max_tokens,
        };
        let mut last = String::new();
        for attempt in 0..4u32 {
            let resp = self
                .client
                .post(URL)
                .bearer_auth(&self.key)
                .header("HTTP-Referer", "https://kandibuddy.local")
                .header("X-Title", "festival-factory hydration")
                .json(&body)
                .send()
                .await;
            match resp {
                Ok(r) if r.status().is_success() => {
                    let parsed: ChatResp = r.json().await?;
                    let text = parsed.choices.into_iter().next().map(|c| c.message.content).unwrap_or_default();
                    if !text.trim().is_empty() {
                        return Ok(text);
                    }
                    last = "empty content".into();
                }
                Ok(r) => {
                    let code = r.status();
                    let t = r.text().await.unwrap_or_default();
                    last = format!("HTTP {code}: {}", &t.chars().take(200).collect::<String>());
                }
                Err(e) => last = format!("{e}"),
            }
            tokio::time::sleep(Duration::from_secs(1u64 << attempt)).await;
        }
        anyhow::bail!("openrouter failed after retries: {last}")
    }
}
