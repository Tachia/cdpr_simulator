r"""Provider-agnostic LLM layer for the CDPR Simulator.

The directive's requirement is unambiguous: pick a provider via
configuration, never wire one in directly. ``cdpr.llm`` exposes a
single :class:`LLMProvider` Protocol and a factory
(:func:`build_provider`) that resolves the right adapter from the
``CDPR_LLM_PROVIDER`` env var (or an explicit name).

Five providers ship out of the box; new ones drop in by implementing
the Protocol:

* ``gemini``      → Google Gemini Free Tier (default for the
                    user assistant role)
* ``openrouter``  → OpenRouter (DeepSeek-R1 for math reasoning,
                    DeepSeek-V3 for code generation)
* ``ollama``      → Local Ollama daemon (offline / private)
* ``lmstudio``    → LM Studio's OpenAI-compatible server
* ``echo``        → Stub that echoes the prompt (no network, useful
                    for CI and demos before keys are set)

API keys come from environment variables --- never hard-coded, never
embedded in URLs. ``cdpr.llm.config`` documents the canonical names
(``GEMINI_API_KEY``, ``OPENROUTER_API_KEY``, ``OLLAMA_URL``,
``LMSTUDIO_URL``).

The HTTP path is dependency-light: ``urllib`` from the stdlib by
default, with an optional ``httpx`` upgrade if available. No
provider's SDK is a hard dependency; the user installs the one they
want via ``pip install cdpr[llm]`` or directly.
"""

from cdpr.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
)
from cdpr.llm.config import (
    LLMConfig,
    available_providers,
    default_provider_name,
    resolve_fallback_chain,
)
from cdpr.llm.factory import build_provider

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMUnavailableError",
    "LLMConfig",
    "available_providers",
    "default_provider_name",
    "resolve_fallback_chain",
    "build_provider",
]
