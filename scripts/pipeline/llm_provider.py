"""LLM provider resolution — Google AI Studio is plan A, OpenRouter is plan B.

Permanent policy (2026-07-25): every provider-bound stage in this repo resolves
through this module (Python) or the matching logic in festival-factory's chat
client (Rust). OpenRouter is never preferred when a Google AI Studio key exists.

Resolution order (subject / generation):
  1. $LLM_PROVIDER force (googleai|openrouter) if set
  2. Google AI Studio — $GOOGLE_AI_API_KEY / $GEMINI_API_KEY / keyring service=googleai
  3. OpenRouter — $OPENROUTER_API_KEY / keyring service=openrouter (key_type=api_key, then bare)

Model ids differ by provider:
  googleai   → gemma-4-31b-it
  openrouter → google/gemma-4-31b-it

Both endpoints speak the OpenAI chat-completions shape, so one POST body works.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping


GOOGLE_AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GOOGLEAI_MODEL = "gemma-4-31b-it"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"


@dataclass(frozen=True)
class Provider:
    name: str                 # "googleai" | "openrouter"
    key: str
    model: str
    chat_url: str

    @property
    def is_googleai(self) -> bool:
        return self.name == "googleai"


def secret_tool_lookup(*args: str) -> str | None:
    """`secret-tool lookup …` — value or None; never logs the value."""
    try:
        proc = subprocess.run(
            ["secret-tool", "lookup", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def resolve_google_ai_key(env: Mapping[str, str] | None = None) -> str | None:
    source = env if env is not None else os.environ
    for var in ("GOOGLE_AI_API_KEY", "GEMINI_API_KEY"):
        value = (source.get(var) or "").strip()
        if value:
            return value
    # Canonical keyring entry: label "Google AI Studio API key", service=googleai
    return secret_tool_lookup("service", "googleai")


def resolve_openrouter_key(env: Mapping[str, str] | None = None) -> str | None:
    source = env if env is not None else os.environ
    value = (source.get("OPENROUTER_API_KEY") or "").strip()
    if value:
        return value
    # Prefer the typed attribute used on this machine, then bare service lookup.
    return (
        secret_tool_lookup("service", "openrouter", "key_type", "api_key")
        or secret_tool_lookup("service", "openrouter")
    )


def _normalize_model(provider: str, model: str | None) -> str:
    """Map a caller-supplied model id onto the provider's namespace."""
    if provider == "googleai":
        if not model:
            return DEFAULT_GOOGLEAI_MODEL
        # OpenRouter-style "google/gemma-4-31b-it" → Studio "gemma-4-31b-it"
        if model.startswith("google/"):
            return model.split("/", 1)[1]
        return model
    # openrouter
    if not model:
        return DEFAULT_OPENROUTER_MODEL
    if "/" not in model:
        return f"google/{model}"
    return model


def resolve_provider(
    *,
    forced_provider: str | None = None,
    forced_model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Provider:
    """Return the active provider. googleai wins whenever its key resolves.

    Raises SystemExit when neither key is available (fail closed — no silent
    heuristic path here; callers that can degrade must catch and handle).
    """
    source = env if env is not None else os.environ
    forced = (forced_provider or source.get("LLM_PROVIDER") or "").strip().lower() or None
    if forced not in (None, "googleai", "openrouter"):
        raise SystemExit(f"LLM_PROVIDER={forced!r} invalid — use googleai or openrouter")

    google_key = resolve_google_ai_key(source)
    openrouter_key = resolve_openrouter_key(source)

    if forced == "googleai" or (forced is None and google_key):
        if not google_key:
            raise SystemExit(
                "no Google AI Studio key: set GOOGLE_AI_API_KEY/GEMINI_API_KEY "
                "or keyring service=googleai"
            )
        model = _normalize_model("googleai", forced_model)
        return Provider(
            name="googleai",
            key=google_key,
            model=model,
            chat_url=GOOGLE_AI_BASE_URL + "chat/completions",
        )

    if forced == "openrouter" or openrouter_key:
        if not openrouter_key:
            raise SystemExit(
                "no OpenRouter key: set OPENROUTER_API_KEY or keyring service=openrouter"
            )
        model = _normalize_model("openrouter", forced_model)
        return Provider(
            name="openrouter",
            key=openrouter_key,
            model=model,
            chat_url=OPENROUTER_BASE_URL + "/chat/completions",
        )

    raise SystemExit(
        "no LLM provider key: need Google AI Studio (plan A: GOOGLE_AI_API_KEY / "
        "keyring service=googleai) or OpenRouter (plan B: OPENROUTER_API_KEY / "
        "keyring service=openrouter)"
    )


def try_resolve_provider(
    *,
    forced_provider: str | None = None,
    forced_model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Provider | None:
    """Like resolve_provider but returns None instead of exiting when no key exists."""
    try:
        return resolve_provider(
            forced_provider=forced_provider, forced_model=forced_model, env=env
        )
    except SystemExit:
        return None


def chat_completion(
    system: str,
    user: str,
    *,
    provider: Provider | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1300,
    max_retries: int = 3,
    timeout_s: float = 120.0,
) -> str:
    """One chat completion. Empty string on total failure (counted by callers)."""
    prov = provider or resolve_provider(forced_model=model)
    # If caller passed an explicit model string, re-normalize onto the active provider.
    use_model = _normalize_model(prov.name, model) if model else prov.model
    body = {
        "model": use_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {prov.key}",
        "Content-Type": "application/json",
    }
    # OpenRouter-only telemetry headers — harmless elsewhere, required by some OR tiers.
    if prov.name == "openrouter":
        headers["HTTP-Referer"] = "https://kandibuddy.local"
        headers["X-Title"] = "kandi-cpt"

    last_err = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                prov.chat_url, data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choices = payload.get("choices") or []
            message = (choices[0].get("message") if choices else {}) or {}
            content = (message.get("content") or "").strip()
            if not content:
                # Some models put the answer only in a reasoning channel.
                content = str(
                    message.get("reasoning")
                    or message.get("reasoning_content")
                    or ""
                ).strip()
            if content:
                return content
            last_err = "empty content (HTTP 200)"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            last_err = f"HTTP {exc.code}: {detail}"
        except Exception as exc:  # network / JSON
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(2 ** attempt)
    print(
        f"  [llm_provider/{prov.name}] generation failed after {max_retries} attempts: {last_err}",
        file=sys.stderr,
    )
    return ""


# Back-compat aliases so existing imports keep working while call sites migrate.
def resolve_openrouter_key_compat(env: Mapping[str, str] | None = None) -> str | None:
    """DEPRECATED name — returns the ACTIVE provider key (googleai preferred)."""
    prov = try_resolve_provider(env=env)
    return prov.key if prov else None
