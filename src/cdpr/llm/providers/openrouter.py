"""OpenRouter provider — routes to DeepSeek-R1 (math) or DeepSeek-V3 (code).

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so
the same payload shape works for both DeepSeek models. The directive
calls for ``deepseek/deepseek-r1`` as the default (math reasoning);
``deepseek/deepseek-chat`` (V3) is selectable via the ``OPENROUTER_MODEL``
env var or the per-call ``model`` argument.
"""

from __future__ import annotations

from typing import Iterator

from cdpr.llm.base import LLMMessage, LLMResponse, LLMUnavailableError
from cdpr.llm.config import LLMConfig
from cdpr.llm.providers._http import post_json


class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.api_key:
            raise LLMUnavailableError(
                "OPENROUTER_API_KEY is not set. Get a free key at "
                "https://openrouter.ai/keys and put it in your .env file "
                "(see docs/llm-providers.md)."
            )
        self._cfg = cfg
        self.model = cfg.model
        self._base = cfg.base_url or "https://openrouter.ai/api/v1"

    def is_available(self) -> bool:
        return bool(self._cfg.api_key)

    def complete(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> LLMResponse:
        payload = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            # OpenRouter requests these for rate-limit attribution; the
            # values are cosmetic.
            "HTTP-Referer": "https://github.com/Tachia/cdpr_simulator",
            "X-Title": "CDPR Simulator",
        }
        try:
            data = post_json(
                f"{self._base}/chat/completions",
                payload, headers=headers, timeout=self._cfg.timeout_s,
            )
        except Exception as exc:
            raise LLMUnavailableError(f"OpenRouter call failed: {exc}") from exc
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
            finish = choice.get("finish_reason", "")
            usage = data.get("usage", {}) or {}
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(
                f"OpenRouter response missing expected fields: {data!r}"
            ) from exc
        return LLMResponse(
            text=text, model=payload["model"], provider=self.name,
            tokens_prompt=usage.get("prompt_tokens"),
            tokens_completion=usage.get("completion_tokens"),
            finish_reason=finish, raw=data,
        )

    def stream(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> Iterator[str]:
        # OpenAI-style SSE streaming exists --- keep the fallback for
        # now since the simulator doesn't yet have a streaming consumer.
        resp = self.complete(messages, max_tokens=max_tokens,
                              temperature=temperature, model=model)
        yield resp.text
