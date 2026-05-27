"""Exception types raised by the framework.

These live in a single module so that callers can ``except`` on a clean public
surface without importing the modules that raise them.
"""

from __future__ import annotations


class CdprError(Exception):
    """Base class for all framework-specific exceptions."""


class ConfigurationError(CdprError):
    """Raised when a robot configuration is inconsistent or under-specified."""


class SingularConfiguration(CdprError):
    """Raised when a kinematic analysis hits a structurally singular pose.

    The diagnostic information attached as ``self.condition_number`` (when
    available) lets callers decide whether to retry with a regularised solver
    or to report a workspace boundary.
    """

    def __init__(self, message: str, *, condition_number: float | None = None) -> None:
        super().__init__(message)
        self.condition_number = condition_number


class InfeasibleTensionError(CdprError):
    """Raised when no tension distribution satisfies cable bounds and equilibrium.

    Attributes
    ----------
    residual:
        Norm of the unsatisfied equilibrium residual at the best-effort
        solution, if any was produced.
    """

    def __init__(self, message: str, *, residual: float | None = None) -> None:
        super().__init__(message)
        self.residual = residual


class MissingAdapterDependency(CdprError):
    """Raised when an optional simulator backend is requested but not installed.

    The framework keeps the scientific core decoupled from any specific
    simulator; adapters are imported lazily and raise this exception with an
    actionable install hint instead of failing on a bare ``ImportError``.
    """

    def __init__(self, backend: str, install_hint: str) -> None:
        super().__init__(
            f"Adapter for backend {backend!r} is not available. "
            f"Install it with:  {install_hint}"
        )
        self.backend = backend
        self.install_hint = install_hint
