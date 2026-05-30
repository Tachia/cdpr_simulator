r"""Provider-agnostic LLM contract.

Every adapter under :mod:`cdpr.llm.providers` implements
:class:`LLMProvider`. Consumers (the conversational simulation
builder, in-app explanation, future report-narration tooling) only
talk to this Protocol --- never to a specific provider's SDK. That
keeps the simulator's business logic free of API-key plumbing and
lets users switch providers via a single env var.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMUnavailableError(RuntimeError):
    """Raised when the chosen provider can't be used right now.

    Examples include a missing API key in the environment, an
    unreachable local daemon (Ollama / LM Studio), or a transient
    network failure during ``complete()``.

    Consumers should catch this and either degrade (show a placeholder,
    skip the explanation) or surface a clear "configure an LLM
    provider" message --- the directive's "never crash" rule.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class LLMMessage:
    """One turn in a multi-turn conversation."""

    role: Role
    content: str


@dataclass(slots=True)
class LLMResponse:
    """The provider's reply, with enough metadata to log + audit.

    ``raw`` carries the underlying provider response (JSON dict, SDK
    object, ...) so callers can inspect provider-specific fields
    (token counts, finish_reason, safety verdicts) without
    re-decoding the wire format.
    """

    text: str
    model: str = ""
    provider: str = ""
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    finish_reason: str = ""
    raw: object | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Minimum surface every adapter exposes.

    Implementations live in :mod:`cdpr.llm.providers`. Each is
    constructed by the factory with an :class:`LLMConfig` and is
    expected to be cheap to instantiate --- the network call only
    happens inside :meth:`complete` / :meth:`stream`.
    """

    #: Human-readable provider name (``"gemini"``, ``"openrouter"`` ...).
    name: str

    #: Default model identifier the provider was constructed with.
    model: str

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> LLMResponse:
        """One-shot completion.

        Raises :class:`LLMUnavailableError` if the network call fails
        or the provider is misconfigured (missing API key, unreachable
        host). Implementations MUST NOT raise generic ``Exception``
        --- every failure path wraps into ``LLMUnavailableError``.
        """
        ...

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> Iterator[str]:
        """Stream the response one chunk at a time.

        Providers that don't support streaming should fall back to
        ``complete()`` and yield the entire response as a single
        chunk (the default implementation does exactly this).
        """
        ...

    def is_available(self) -> bool:
        """Cheap readiness check.

        Default: try to instantiate the underlying client; return
        ``True`` if no exception. Providers can override with a
        lightweight ping (e.g. Ollama's ``/api/tags``).
        """
        ...
