"""Ollama provider — talks to a local Ollama daemon.

Ollama is the user's "no cloud, no API key" option per the directive.
Default endpoint is ``http://localhost:11434``; the model used is
whatever the user pulled (``ollama pull deepseek-r1`` etc.)."""

from __future__ import annotations

from typing import Iterator

from cdpr.llm.base import LLMMessage, LLMResponse, LLMUnavailableError
from cdpr.llm.config import LLMConfig
from cdpr.llm.providers._http import get_json, post_json


class OllamaProvider:
    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.base_url:
            raise LLMUnavailableError(
                "OLLAMA_URL is not set. Install Ollama from "
                "https://ollama.com/download, start it (it listens on "
                "http://localhost:11434), then add OLLAMA_URL to .env."
            )
        self._cfg = cfg
        self.model = cfg.model
        self._base = cfg.base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            get_json(f"{self._base}/api/tags", timeout=2.0)
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
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        try:
            data = post_json(
                f"{self._base}/api/chat", payload, timeout=self._cfg.timeout_s,
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"Ollama call failed ({self._base}): {exc}. "
                "Is the daemon running?"
            ) from exc
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMUnavailableError(
                f"Ollama response missing expected fields: {data!r}"
            ) from exc
        return LLMResponse(
            text=text, model=payload["model"], provider=self.name,
            finish_reason=data.get("done_reason", ""), raw=data,
        )

    def stream(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> Iterator[str]:
        resp = self.complete(messages, max_tokens=max_tokens,
                              temperature=temperature, model=model)
        yield resp.text
