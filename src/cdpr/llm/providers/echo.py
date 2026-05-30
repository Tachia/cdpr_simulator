"""Echo provider — always available, no network calls.

Returns the last user message verbatim. Used as the default fallback
so the conversational layer can be smoke-tested without any API key,
and so the GUI's chat box never breaks even when no provider is
configured."""

from __future__ import annotations

from typing import Iterator

from cdpr.llm.base import LLMMessage, LLMResponse
from cdpr.llm.config import LLMConfig


class EchoProvider:
    name = "echo"

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self.model = cfg.model or "echo-1"

    def is_available(self) -> bool:
        return True

    def complete(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> LLMResponse:
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"), None,
        )
        prompt = last_user.content if last_user else ""
        return LLMResponse(
            text=(
                "[echo provider — no real LLM configured]\n\n"
                f"Got prompt:\n{prompt}\n\n"
                "Configure GEMINI_API_KEY or OPENROUTER_API_KEY in .env to "
                "enable a real provider. See docs/llm-providers.md."
            ),
            model=self.model, provider=self.name,
        )

    def stream(
        self, messages: list[LLMMessage], *,
        max_tokens: int = 1024, temperature: float = 0.2,
        model: str | None = None,
    ) -> Iterator[str]:
        yield self.complete(messages, max_tokens=max_tokens,
                            temperature=temperature, model=model).text
