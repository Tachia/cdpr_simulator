"""LM Studio provider — OpenAI-compatible local server.

LM Studio exposes ``/v1/chat/completions`` exactly like OpenAI so the
adapter is a thin alias over the OpenAI message shape with the host
URL pointed at ``http://localhost:1234`` (configurable via
``LMSTUDIO_URL``)."""

from __future__ import annotations

from typing import Iterator

from cdpr.llm.base import LLMMessage, LLMResponse, LLMUnavailableError
from cdpr.llm.config import LLMConfig
from cdpr.llm.providers._http import post_json


class LMStudioProvider:
    name = "lmstudio"

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.base_url:
            raise LLMUnavailableError(
                "LMSTUDIO_URL is not set. Install LM Studio from "
                "https://lmstudio.ai/, start the local server, then add "
                "LMSTUDIO_URL=http://localhost:1234/v1 to .env."
            )
        self._cfg = cfg
        self.model = cfg.model
        self._base = cfg.base_url.rstrip("/")

    def is_available(self) -> bool:
        # LM Studio's GET /models is the cheap health check.
        try:
            from cdpr.llm.providers._http import get_json
            get_json(f"{self._base}/models", timeout=2.0)
            return True
        except Exception:
            return False

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
        try:
            data = post_json(
                f"{self._base}/chat/completions", payload,
                timeout=self._cfg.timeout_s,
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"LM Studio call failed ({self._base}): {exc}. "
                "Is the local server running?"
            ) from exc
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
            finish = choice.get("finish_reason", "")
            usage = data.get("usage", {}) or {}
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(
                f"LM Studio response missing expected fields: {data!r}"
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
        resp = self.complete(messages, max_tokens=max_tokens,
                              temperature=temperature, model=model)
        yield resp.text
