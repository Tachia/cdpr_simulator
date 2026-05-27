r"""Tension distribution for cable-driven parallel robots.

For a CDPR with :math:`m` cables, :math:`n` actuated DOFs, structure matrix
:math:`\mathbf{W}(\mathbf{q}) \in \mathbb{R}^{n \times m}` and external wrench
:math:`\mathbf{w}_\text{ext} \in \mathbb{R}^n`, static equilibrium requires a
tension vector :math:`\mathbf{t} \in \mathbb{R}^m` satisfying

.. math::

    \mathbf{W}(\mathbf{q})\,\mathbf{t} + \mathbf{w}_\text{ext} = \mathbf{0},
    \qquad \mathbf{t}_\text{min} \leq \mathbf{t} \leq \mathbf{t}_\text{max}.

The equality system is generally underdetermined (:math:`m > n` for redundant
actuation); the feasible set, when non-empty, is a polytope of dimension
:math:`r = m - n` --- the intersection of the affine equilibrium hyperplane
with the cable-limit box.

This module exposes two solver paths:

1. **One-DOF-redundant fast path** (:math:`r = 1`).
   The feasible set is a line segment. A particular solution
   :math:`\mathbf{t}_p = -\mathbf{W}^{+}\mathbf{w}_\text{ext}` and the
   1-dimensional null-space direction :math:`\mathbf{N}` give every solution
   as :math:`\mathbf{t}(\alpha) = \mathbf{t}_p + \alpha\,\mathbf{N}`. Component
   bounds reduce to an interval intersection in :math:`\alpha`; the optimum of
   any convex per-component objective (centred, min-norm, preferred) is then
   a closed form. This is the case for almost every published CDPR design and
   it bypasses the QP solver entirely.

2. **General path** (:math:`r \geq 2`).
   Project to the null-space basis and solve a bounded QP on the
   :math:`r`-dimensional coordinate :math:`\boldsymbol{\alpha}` via
   :func:`scipy.optimize.minimize` with the SLSQP backend.

A wrench is *feasible* when the equality polytope intersects the box. We test
this with a tiny LP (:func:`scipy.optimize.linprog`); LP feasibility is the
necessary and sufficient condition and is faster than running the full QP and
catching its failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, linprog, minimize

from cdpr.core.exceptions import InfeasibleTensionError, SingularConfiguration


class TensionObjective(str, Enum):
    """Selectable convex objective for the bounded tension QP.

    * ``MIN_NORM`` -- minimise :math:`\\lVert\\mathbf{t}\\rVert^2`; the smallest
      tensions that close the equilibrium. Tends to sit on the lower bounds.
    * ``CENTERED`` -- minimise :math:`\\lVert\\mathbf{t} - \\tfrac12(\\mathbf{t}_\\text{min} + \\mathbf{t}_\\text{max})\\rVert^2`;
      maximises slack to either bound. Recommended default for continuous
      operation (Mikelsons et al., *ICRA* 2008).
    * ``PREFERRED`` -- minimise :math:`\\lVert\\mathbf{t} - \\mathbf{t}_\\text{pref}\\rVert^2`;
      tracks a user-supplied reference. Useful for smooth tension transitions
      along a trajectory by setting ``t_pref`` to the previous step's solution.
    """

    MIN_NORM = "min_norm"
    CENTERED = "centered"
    PREFERRED = "preferred"


# ---------------------------------------------------------------------------
# Particular solution: minimum-norm pseudoinverse
# ---------------------------------------------------------------------------

def min_norm_tension(
    W: NDArray[np.float64], w_ext: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Unbounded minimum-norm tension: :math:`\mathbf{t} = -\mathbf{W}^{+}\mathbf{w}_\text{ext}`.

    Cheap diagnostic. The result may violate cable limits and is *not* a
    valid CDPR tension distribution in itself --- use it as a quality
    reference for the bounded solvers.
    """
    return -np.linalg.pinv(W) @ w_ext


# ---------------------------------------------------------------------------
# Feasibility test
# ---------------------------------------------------------------------------

def is_wrench_feasible(
    W: NDArray[np.float64],
    w_ext: NDArray[np.float64],
    t_min: NDArray[np.float64],
    t_max: NDArray[np.float64],
    *,
    tol: float = 1e-9,
) -> bool:
    r"""Does the equality polytope :math:`\{\mathbf{t} : \mathbf{W}\mathbf{t} = -\mathbf{w}_\text{ext}\}` intersect the box :math:`[\mathbf{t}_\text{min}, \mathbf{t}_\text{max}]`?

    Uses an LP feasibility check (``scipy.optimize.linprog`` with the HiGHS
    backend). Costs a single LP solve and is significantly cheaper than
    running the QP and catching its failure mode --- the workspace-analysis
    layer calls this for every voxel on a 3D grid, so the speed matters.
    """
    m = W.shape[1]
    res = linprog(
        c=np.zeros(m),
        A_eq=W,
        b_eq=-w_ext,
        bounds=list(zip(t_min, t_max, strict=True)),
        method="highs",
    )
    if not res.success:
        return False
    # Numerical guard: HiGHS may declare success at the boundary.
    if res.x is None:
        return False
    eq_residual = np.linalg.norm(W @ res.x + w_ext)
    return bool(eq_residual <= tol * (1.0 + np.linalg.norm(w_ext)))


# ---------------------------------------------------------------------------
# Bounded tension distribution
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _BoundedQP:
    """Reduced QP on a null-space coordinate :math:`\\boldsymbol\\alpha`.

    Solving over :math:`\\boldsymbol\\alpha` rather than :math:`\\mathbf{t}` is what
    makes the dense equality constraint disappear and turns the problem into
    a pure bounded QP, for which SLSQP is well-behaved.
    """

    t_p: NDArray[np.float64]            # particular solution, shape (m,)
    N: NDArray[np.float64]              # null-space basis, shape (m, r)
    t_min: NDArray[np.float64]
    t_max: NDArray[np.float64]
    objective: TensionObjective
    t_pref: NDArray[np.float64] | None

    def reference(self) -> NDArray[np.float64]:
        if self.objective is TensionObjective.MIN_NORM:
            return np.zeros_like(self.t_p)
        if self.objective is TensionObjective.CENTERED:
            return 0.5 * (self.t_min + self.t_max)
        if self.objective is TensionObjective.PREFERRED:
            if self.t_pref is None:
                raise ValueError("TensionObjective.PREFERRED requires t_pref to be supplied.")
            return self.t_pref
        raise ValueError(f"Unknown objective: {self.objective}")

    def tensions(self, alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.t_p + self.N @ alpha


def _solve_one_dof_redundant(qp: _BoundedQP) -> NDArray[np.float64]:
    r"""Closed-form solution when the null space is one-dimensional (``r = 1``).

    Cable bounds reduce to an interval :math:`[\alpha_\text{lo}, \alpha_\text{hi}]`
    in the scalar null-space coordinate. The optimum of any per-component
    quadratic objective on this interval is then the projection of the
    unconstrained optimum onto the interval.
    """
    n = qp.N[:, 0]                                       # (m,)
    # Per-cable feasible alpha range:  t_min <= t_p + alpha n <= t_max
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (qp.t_min - qp.t_p) / n
        hi = (qp.t_max - qp.t_p) / n
    pos = n > 0
    neg = n < 0
    zero = ~(pos | neg)

    # For cables with n_i > 0 the bound order is (lo, hi); flipped for n_i < 0.
    alpha_lo = np.where(pos, lo, np.where(neg, hi, -np.inf))
    alpha_hi = np.where(pos, hi, np.where(neg, lo, np.inf))

    # Cables with n_i ~ 0 contribute no constraint on alpha but require t_p
    # to already lie in [t_min, t_max] on those rows.
    if zero.any() and (
        (qp.t_p[zero] < qp.t_min[zero] - 1e-9).any()
        or (qp.t_p[zero] > qp.t_max[zero] + 1e-9).any()
    ):
        raise InfeasibleTensionError(
            "Particular solution violates bounds on cables orthogonal to the null space."
        )

    alpha_lower = float(alpha_lo.max())
    alpha_upper = float(alpha_hi.min())
    if alpha_lower > alpha_upper + 1e-9:
        raise InfeasibleTensionError(
            f"Empty feasible alpha range: [{alpha_lower:.4g}, {alpha_upper:.4g}]"
        )

    # Unconstrained optimum of  ||t_p + alpha n - r||^2  is
    #   alpha* = n^T (r - t_p) / (n^T n)
    r = qp.reference()
    alpha_star = float(n @ (r - qp.t_p) / (n @ n))
    alpha = min(max(alpha_star, alpha_lower), alpha_upper)
    return qp.tensions(np.array([alpha]))


def _solve_general(qp: _BoundedQP) -> NDArray[np.float64]:
    """SLSQP on the null-space coordinate; falls back when ``r >= 2``."""
    r_dim = qp.N.shape[1]

    # Box constraints on alpha: t_min <= t_p + N alpha <= t_max, expressed as
    # 2m linear inequalities. SLSQP wants them as g(alpha) >= 0.
    def lower_ineq(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        return qp.tensions(alpha) - qp.t_min

    def upper_ineq(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        return qp.t_max - qp.tensions(alpha)

    ref = qp.reference()

    def objective(alpha: NDArray[np.float64]) -> float:
        d = qp.tensions(alpha) - ref
        return float(d @ d)

    def gradient(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2.0 * qp.N.T @ (qp.tensions(alpha) - ref)

    # A reasonable warm start: project (ref - t_p) onto the null space.
    alpha0 = qp.N.T @ (ref - qp.t_p)

    result: OptimizeResult = minimize(
        objective,
        alpha0,
        jac=gradient,
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lower_ineq, "jac": lambda _a: qp.N},
            {"type": "ineq", "fun": upper_ineq, "jac": lambda _a: -qp.N},
        ],
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not result.success:
        raise InfeasibleTensionError(
            f"SLSQP did not converge: {result.message}",
            residual=float(result.fun),
        )

    t = qp.tensions(np.asarray(result.x, dtype=np.float64).reshape(r_dim))
    # Final numerical guard --- SLSQP may finish marginally inside the bounds.
    if (t < qp.t_min - 1e-6).any() or (t > qp.t_max + 1e-6).any():
        raise InfeasibleTensionError("SLSQP solution violates cable bounds.")
    return t


def tension_distribution(
    W: NDArray[np.float64],
    w_ext: ArrayLike,
    t_min: ArrayLike,
    t_max: ArrayLike,
    *,
    objective: TensionObjective | str = TensionObjective.CENTERED,
    t_pref: ArrayLike | None = None,
    rank_tol: float = 1e-9,
) -> NDArray[np.float64]:
    r"""Solve the bounded tension QP for one pose.

    Returns a feasible tension vector :math:`\mathbf{t} \in [\mathbf{t}_\text{min},
    \mathbf{t}_\text{max}]` satisfying :math:`\mathbf{W}\mathbf{t} = -\mathbf{w}_\text{ext}`
    and minimising the chosen objective. Raises
    :class:`InfeasibleTensionError` if no such vector exists.

    Detects structural singularity (rank-deficient :math:`\mathbf{W}`) and
    raises :class:`SingularConfiguration` --- this is distinct from a
    bounds-violation infeasibility and worth surfacing separately to the
    workspace-analysis caller.
    """
    obj = TensionObjective(objective) if not isinstance(objective, TensionObjective) else objective
    w_ext = np.asarray(w_ext, dtype=np.float64)
    t_min = np.asarray(t_min, dtype=np.float64)
    t_max = np.asarray(t_max, dtype=np.float64)
    pref = np.asarray(t_pref, dtype=np.float64) if t_pref is not None else None

    n, m = W.shape

    # Structural rank check up front.
    U, sigma, Vt = np.linalg.svd(W, full_matrices=True)
    rank = int(np.sum(sigma > rank_tol * sigma[0])) if sigma[0] > 0 else 0
    if rank < n:
        raise SingularConfiguration(
            f"Structure matrix has rank {rank} < dof {n}; pose is structurally singular.",
            condition_number=float("inf"),
        )

    # Particular solution and null-space basis from the SVD.
    sigma_inv = np.where(sigma > rank_tol * sigma[0], 1.0 / sigma, 0.0)
    W_pinv = Vt[:n].T @ np.diag(sigma_inv) @ U.T
    t_p = -W_pinv @ w_ext
    N = Vt[n:].T if m > n else np.zeros((m, 0), dtype=np.float64)

    # Special case: square system (no redundancy).
    if N.shape[1] == 0:
        if (t_p < t_min - 1e-9).any() or (t_p > t_max + 1e-9).any():
            raise InfeasibleTensionError(
                "Non-redundant tension solution violates cable bounds."
            )
        return t_p

    qp = _BoundedQP(t_p=t_p, N=N, t_min=t_min, t_max=t_max, objective=obj, t_pref=pref)

    if N.shape[1] == 1:
        return _solve_one_dof_redundant(qp)
    return _solve_general(qp)
