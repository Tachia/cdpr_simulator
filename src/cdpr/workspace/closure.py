r"""Wrench-closure workspace (WCW) test.

A pose lies in the wrench-closure workspace if and only if the cables can
generate *any* external wrench with strictly positive tensions --- equivalently,
the convex cone spanned by the columns of the structure matrix :math:`\mathbf{W}`
positively spans :math:`\mathbb{R}^{n_\text{dof}}`. By a standard application of
Stiemke's lemma this reduces to the LP feasibility problem

.. math::

    \text{find } \mathbf{t}\in\mathbb{R}^m \quad
    \text{s.t.}\quad \mathbf{W}\,\mathbf{t} = \mathbf{0}, \quad
    \mathbf{t} \geq \mathbf{1}.

If a strictly positive :math:`\mathbf{t}` exists in the null space then any
external wrench can be absorbed by scaling and shifting the cable tensions.
The :math:`\mathbf{t} \geq \mathbf{1}` lower bound is a convenient normalisation
--- the actual numerical value of :math:`\mathbf{1}` is immaterial because the
null space is a cone.

This is the canonical definition used in workspace papers (Gouttefarde &
Gosselin 2006; Pham et al. 2009). It does *not* take cable maximum tensions
into account; for that, use :func:`~cdpr.workspace.feasible.is_in_wfw`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from cdpr.core.frames import Pose
from cdpr.geometry.robot import Robot, RobotGeometry
from cdpr.kinematics.jacobian import structure_matrix


def is_in_wcw(pose: Pose, robot: Robot | RobotGeometry, *, tol: float = 1e-9) -> bool:
    """Boolean WCW test for a single pose.

    Returns ``True`` if the pose admits a strictly positive tension in the
    null space of the structure matrix.
    """
    W = structure_matrix(pose, robot)
    n, m = W.shape

    # A necessary condition: W must have full row rank.
    s = np.linalg.svd(W, compute_uv=False)
    if s[-1] <= tol * s[0]:
        return False

    res = linprog(
        c=np.zeros(m),
        A_eq=W,
        b_eq=np.zeros(n),
        bounds=[(1.0, None)] * m,
        method="highs",
    )
    return bool(res.success)
