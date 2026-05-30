"""Google Gemini provider — the directive's default assistant.

Uses Gemini's REST endpoint directly so we don't take a hard
``google-generativeai`` SDK dependency. The free tier covers the
``gemini-1.5-flash`` model used here.
"""

from __future__ import annotations

from typing import Iterator

from cdpr.llm.base import LLMMessage, LLMResponse, LLMUnavailableError
from cdpr.llm.config import LLMConfig
from cdpr.llm.providers._http import post_json


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
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model_id}:generateContent?key={self._cfg.api_key}"
        )
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
            data = post_json(url, payload, timeout=self._cfg.timeout_s)
        except Exception as exc:                                     # network / 4xx / 5xx
            raise LLMUnavailableError(f"Gemini call failed: {exc}") from exc
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            finish = data["candidates"][0].get("finishReason", "")
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(
                f"Gemini response missing expected fields: {data!r}"
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
