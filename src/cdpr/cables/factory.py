r"""Construct a CableModel by mode name.

The factory exists for two reasons:

1. **Strict mode names.** Anywhere the framework lets the user select a
   constitutive law by string (configuration files, CLI flags, scenario
   dataclasses, dissertation reports), the same canonical names live
   here. Typos surface as :class:`ValueError`, not silent fall-throughs.
2. **One place to keep the registry.** Adding a fourth constitutive law
   means adding it here; every consumer of :func:`cable_model_by_name`
   picks it up automatically.

The directive explicitly forbids mixing the three laws at runtime.
:func:`cable_model_by_name` therefore never composes one model from
another; it just dispatches.
"""

from __future__ import annotations

from typing import Any, Literal

from cdpr.cables.base import CableModel
from cdpr.cables.irvine import IrvineModel
from cdpr.cables.kelvin_voigt import KelvinVoigtModel
from cdpr.cables.sqck_hybrid import SQCKHybridModel


CableModeName = Literal["kelvin_voigt", "irvine", "sqck_hybrid"]

_REGISTRY: dict[str, type[CableModel]] = {
    "kelvin_voigt": KelvinVoigtModel,
    "irvine": IrvineModel,
    "sqck_hybrid": SQCKHybridModel,
}


def available_modes() -> tuple[str, ...]:
    """Names of the registered constitutive modes."""
    return tuple(_REGISTRY)


def cable_model_by_name(name: CableModeName | str, **params: Any) -> CableModel:
    """Construct the named cable model.

    Parameters
    ----------
    name:
        One of ``"kelvin_voigt"``, ``"irvine"``, ``"sqck_hybrid"``. Any
        other value raises :class:`ValueError`.
    **params:
        Forwarded to the model's constructor. Unknown parameters raise
        :class:`TypeError` --- there is no silent acceptance.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown cable mode {name!r}; choose from {available_modes()}."
        )
    cls = _REGISTRY[name]
    return cls(**params)
