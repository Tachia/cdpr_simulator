r"""Google Gemini provider --- the directive's default assistant.

Uses Gemini's REST endpoint directly so we don't take a hard
``google-generativeai`` SDK dependency. The free tier covers
``gemini-2.0-flash`` (the current default; the 1.5 family was retired
in mid-2025 and the endpoint now returns 404).

Two error-handling rules from the directive's secrets-hygiene line:

1. Never let the API key appear in an exception that surfaces to the
   UI. We pass the key in the ``x-goog-api-key`` request header rather
   than the URL query string, so a 404 / 401 message that quotes the
   request URL does NOT leak the key into the chat box.
2. When ``httpx.HTTPStatusError`` carries the URL in its message, we
   still scrub it through :func:`_redact_secrets` before raising
   :class:`LLMUnavailableError`, because httpx in some versions prints
   the full URL including the query string.
"""

from __future__ import annotations

import re
from typing import Iterator

from cdpr.llm.base import LLMMessage, LLMResponse, LLMUnavailableError
from cdpr.llm.config import LLMConfig
from cdpr.llm.providers._http import post_json


# Patterns we redact from any error string that might surface to the
# user. Both Google-style key query parameters and bearer tokens are
# covered. The replacement is a fixed string so downstream string-
# comparison code (tests, logs) stays stable.
_KEY_QUERY_RE = re.compile(r"([?&])key=[^&\s'\"]+", flags=re.IGNORECASE)
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", flags=re.IGNORECASE)


def _redact_secrets(text: str) -> str:
    text = _KEY_QUERY_RE.sub(r"\1key=<redacted>", text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    return text


class GeminiProvider:
    name = "gemini"

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.api_key:
            raise LLMUnavailableError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and put it in your "
                ".env file (see docs/llm-providers.md)."
            )
        self._cfg = cfg
        self.model = cfg.model

    def is_available(self) -> bool:
        return bool(self._cfg.api_key)

    def complete(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> LLMResponse:
        model_id = model or self.model
        # Key goes in a header, NOT the URL, so 4xx error messages
        # (which often quote the request URL) don't leak it.
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model_id}:generateContent"
        )
        key_header = {"x-goog-api-key": self._cfg.api_key}
        # Gemini expects a single ``contents`` list with ``role`` ∈ {"user",
        # "model"}. System prompts are concatenated into the first user turn.
        gem_messages: list[dict] = []
        system_text = ""
        for m in messages:
            if m.role == "system":
                system_text = (system_text + "\n" + m.content).strip()
                continue
            role = "model" if m.role == "assistant" else "user"
            gem_messages.append({
                "role": role,
                "parts": [{"text": m.content}],
            })
        if system_text and gem_messages and gem_messages[0]["role"] == "user":
            gem_messages[0]["parts"][0]["text"] = (
                system_text + "\n\n" + gem_messages[0]["parts"][0]["text"]
            )
        payload = {
            "contents": gem_messages,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        try:
            data = post_json(
                url, payload, headers=key_header, timeout=self._cfg.timeout_s,
            )
        except Exception as exc:                                     # network / 4xx / 5xx
            msg = _redact_secrets(str(exc))
            # Translate the common 404 case into actionable language so
            # the user doesn't waste time on the wrong fix.
            if "404" in msg or "Not Found" in msg:
                msg += (
                    " --- model "
                    f"{model_id!r} returned 404. Google retired the "
                    "gemini-1.5 family in mid-2025; the current free-tier "
                    "models are 'gemini-2.0-flash' (default), "
                    "'gemini-2.0-flash-lite', and 'gemini-2.5-flash'. "
                    "Set GEMINI_MODEL=gemini-2.0-flash (or leave unset to "
                    "use the new default) and retry."
                )
            raise LLMUnavailableError(f"Gemini call failed: {msg}") from exc
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            finish = data["candidates"][0].get("finishReason", "")
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(
                f"Gemini response missing expected fields: {_redact_secrets(str(data))!r}"
            ) from exc
        return LLMResponse(
            text=text, model=model_id, provider=self.name,
            finish_reason=finish, raw=data,
        )

    def stream(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> Iterator[str]:
        # Gemini supports SSE streaming via :streamGenerateContent, but
        # to keep the dependency surface flat we yield the full response
        # as one chunk. Callers that need true streaming can extend this.
        resp = self.complete(messages, max_tokens=max_tokens,
                             temperature=temperature, model=model)
        yield resp.text
