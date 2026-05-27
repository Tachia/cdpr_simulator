"""Small numerical utilities used across the framework.

Nothing here is CDPR-specific; the routines are pulled out of their original
modules whenever a second caller needed them. Keep this thin -- if a helper is
only used in one place, leave it there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def safe_normalize(
    v: ArrayLike,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(unit, norm)`` for a vector or batch of vectors.

    For inputs whose Euclidean norm is below ``eps`` the unit vector is set to
    zero rather than producing a NaN. This is the right behaviour for cable
    unit-vector code: a zero-length cable has no direction, and downstream
    consumers (Jacobian, structure matrix) already treat that as a singular
    configuration with their own diagnostics.
    """
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    safe = np.where(n > eps, n, 1.0)
    u = np.where(n > eps, a / safe, np.zeros_like(a))
    return u, np.squeeze(n, axis=-1)


def is_close_se3(T1: NDArray[np.float64], T2: NDArray[np.float64], *, atol: float = 1e-9) -> bool:
    """Compare two :math:`4\\times 4` homogeneous transforms up to ``atol``.

    Avoids spurious failures when rotations are equal but expressed with
    slightly different floating-point matrices (e.g. after a quaternion
    round-trip).
    """
    if T1.shape != T2.shape:
        return False
    if T1.shape[-2:] != (4, 4):
        raise ValueError("is_close_se3 expects (..., 4, 4) inputs")
    dp = T1[..., :3, 3] - T2[..., :3, 3]
    dR = T1[..., :3, :3] @ np.swapaxes(T2[..., :3, :3], -1, -2)
    # Angular distance via trace: angle = arccos((tr(dR) - 1) / 2)
    tr = np.trace(dR, axis1=-2, axis2=-1)
    cos_theta = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return bool(np.all(np.linalg.norm(dp, axis=-1) <= atol)) and bool(np.all(angle <= atol))


def block_diag_batch(blocks: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Stack a list of (n_i, m_i) matrices into a block-diagonal layout.

    Unlike :func:`scipy.linalg.block_diag`, this version is allocation-light:
    it pre-computes the total shape and writes each block into its slice
    rather than repeatedly concatenating.
    """
    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    out = np.zeros((rows, cols), dtype=np.float64)
    r = c = 0
    for b in blocks:
        nr, nc = b.shape
        out[r : r + nr, c : c + nc] = b
        r += nr
        c += nc
    return out
