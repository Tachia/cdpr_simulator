"""Factory: pick + construct the right provider for the active config."""

from __future__ import annotations

from cdpr.llm.base import LLMProvider, LLMUnavailableError
from cdpr.llm.config import LLMConfig


def build_provider(provider: str | None = None) -> LLMProvider:
    """Return a configured :class:`LLMProvider`.

    If ``provider`` is ``None`` (the common case), reads
    ``CDPR_LLM_PROVIDER`` from the environment and falls back to the
    first provider whose keys are present (see
    :func:`cdpr.llm.config.default_provider_name`).

    Raises :class:`LLMUnavailableError` only when the requested
    provider is explicitly named but its keys / URL are missing ---
    the default-name path always returns at least the echo stub so
    the simulator never crashes when no provider is configured.
    """
    cfg = LLMConfig.from_env(provider=provider)
    name = cfg.provider

    if name == "gemini":
        from cdpr.llm.providers.gemini import GeminiProvider
        return GeminiProvider(cfg)
    if name == "openrouter":
        from cdpr.llm.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(cfg)
    if name == "ollama":
        from cdpr.llm.providers.ollama import OllamaProvider
        return OllamaProvider(cfg)
    if name == "lmstudio":
        from cdpr.llm.providers.lmstudio import LMStudioProvider
        return LMStudioProvider(cfg)
    if name == "echo":
        from cdpr.llm.providers.echo import EchoProvider
        return EchoProvider(cfg)
    raise LLMUnavailableError(f"no adapter for provider {name!r}")
