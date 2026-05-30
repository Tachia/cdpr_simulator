r"""Configuration + provider registry for :mod:`cdpr.llm`.

Loads ``.env`` if available (via the optional :mod:`python-dotenv`),
exposes one :class:`LLMConfig` dataclass that the factory and every
adapter consume, and centralises the canonical env-var names so they
appear in exactly one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Lazy load .env --- never fatal if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv as _load_dotenv

    def _try_load_env() -> None:
        # Walk up from cwd to find a .env file; load it once per process.
        cwd = Path.cwd()
        for parent in (cwd, *cwd.parents):
            candidate = parent / ".env"
            if candidate.exists():
                _load_dotenv(candidate, override=False)
                return
except ImportError:                                                  # pragma: no cover
    def _try_load_env() -> None:                                     # type: ignore[misc]
        pass


_DEFAULT_PROVIDER_ENV = "CDPR_LLM_PROVIDER"

# Canonical env-var names. These are referenced by the providers and
# documented in .env.example and docs/llm-providers.md.
GEMINI_API_KEY_ENV: Final = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV: Final = "OPENROUTER_API_KEY"
OLLAMA_URL_ENV: Final = "OLLAMA_URL"
LMSTUDIO_URL_ENV: Final = "LMSTUDIO_URL"

# Default model identifiers per provider. Users can override via the
# matching env var (e.g. ``GEMINI_MODEL=gemini-2.5-flash``).
#
# Google retired the gemini-1.5 family in mid-2025; the endpoint now
# returns 404. ``gemini-2.0-flash`` is the current free-tier successor
# (1M token context, same JSON-structured-output behaviour) and is the
# new default. ``gemini-2.5-flash`` works too if you've enabled it.
DEFAULT_MODELS: Final[dict[str, str]] = {
    "gemini":     "gemini-2.0-flash",
    "openrouter": "deepseek/deepseek-r1",                            # math reasoning default
    "ollama":     "llama3",
    "lmstudio":   "local-model",
    "echo":       "echo-1",
}

PROVIDER_NAMES: Final[tuple[str, ...]] = (
    "gemini", "openrouter", "ollama", "lmstudio", "echo",
)


@dataclass(slots=True)
class LLMConfig:
    """Per-provider configuration.

    Fields are intentionally simple: anything beyond name+model+key+url
    is provider-specific and goes in ``extra``.
    """

    provider: str = "echo"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout_s: float = 30.0
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMConfig":
        """Build a config for the requested provider (or the default)
        purely from environment variables. Safe to call with no
        arguments --- returns the echo stub if nothing else is set."""
        _try_load_env()
        name = (provider or os.environ.get(_DEFAULT_PROVIDER_ENV)
                or default_provider_name())
        name = name.lower().strip()
        if name not in PROVIDER_NAMES:
            raise ValueError(
                f"unknown LLM provider {name!r}. "
                f"Choose from: {', '.join(PROVIDER_NAMES)}"
            )
        cfg = cls(provider=name, model=DEFAULT_MODELS[name])
        if name == "gemini":
            cfg.api_key = os.environ.get(GEMINI_API_KEY_ENV, "")
            cfg.model = os.environ.get("GEMINI_MODEL", cfg.model)
        elif name == "openrouter":
            cfg.api_key = os.environ.get(OPENROUTER_API_KEY_ENV, "")
            cfg.model = os.environ.get("OPENROUTER_MODEL", cfg.model)
            cfg.base_url = os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
        elif name == "ollama":
            cfg.base_url = os.environ.get(OLLAMA_URL_ENV, "http://localhost:11434")
            cfg.model = os.environ.get("OLLAMA_MODEL", cfg.model)
        elif name == "lmstudio":
            cfg.base_url = os.environ.get(LMSTUDIO_URL_ENV, "http://localhost:1234/v1")
            cfg.model = os.environ.get("LMSTUDIO_MODEL", cfg.model)
        return cfg


def default_provider_name() -> str:
    """Pick the first provider whose configuration is plausibly
    complete. Priority: Gemini key set → OpenRouter key set → Ollama
    URL reachable → LM Studio URL reachable → echo stub."""
    _try_load_env()
    if os.environ.get(GEMINI_API_KEY_ENV):
        return "gemini"
    if os.environ.get(OPENROUTER_API_KEY_ENV):
        return "openrouter"
    if os.environ.get(OLLAMA_URL_ENV):
        return "ollama"
    if os.environ.get(LMSTUDIO_URL_ENV):
        return "lmstudio"
    return "echo"


def available_providers() -> dict[str, bool]:
    """Quick configuration probe per provider.

    Returns a dict mapping each provider name to ``True`` when the
    env var(s) it needs are present (does NOT make a network call ---
    use :func:`cdpr.llm.factory.build_provider` plus
    ``is_available()`` for that)."""
    _try_load_env()
    return {
        "gemini":     bool(os.environ.get(GEMINI_API_KEY_ENV)),
        "openrouter": bool(os.environ.get(OPENROUTER_API_KEY_ENV)),
        "ollama":     bool(os.environ.get(OLLAMA_URL_ENV)),
        "lmstudio":   bool(os.environ.get(LMSTUDIO_URL_ENV)),
        "echo":       True,
    }
