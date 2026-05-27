"""Lazy import helpers for the visualisation extras.

Keeping these in a single place lets every viz module call exactly one
function to obtain matplotlib without us scattering try/except blocks. The
error message points users at the install extra rather than ``ImportError``
with no actionable hint.
"""

from __future__ import annotations

from types import ModuleType


def require_matplotlib() -> ModuleType:
    """Return the ``matplotlib`` module or raise a helpful install error."""
    try:
        import matplotlib  # noqa: F401  -- presence is the assertion
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The cdpr visualisation layer needs matplotlib. "
            "Install with:  pip install 'cdpr[viz]'"
        ) from exc
    return matplotlib


def require_pillow() -> ModuleType:
    try:
        import PIL.Image as _img
    except ImportError as exc:
        raise ImportError(
            "GIF export needs Pillow. Install with:  pip install 'cdpr[viz]'"
        ) from exc
    return _img
