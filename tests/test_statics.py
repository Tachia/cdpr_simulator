"""Tension distribution: feasibility, bound satisfaction, objective behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.core.exceptions import InfeasibleTensionError
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.statics.tension import (
    TensionObjective,
    is_wrench_feasible,
    min_norm_tension,
    tension_distribution,
)


def _gravity_wrench(mass: float, dof: int) -> np.ndarray:
    w = np.zeros(dof)
    w[2] = -mass * 9.81  # Fz: gravity pulls down
    return w


# ---------------------------------------------------------------------------
# Point-mass robot: 4 cables, 3 DOF, redundancy r = 1
# ---------------------------------------------------------------------------

def test_min_norm_tension_solves_equality(point_mass_robot, home_pose):
    W = structure_matrix(home_pose, point_mass_robot)
    w_ext = _gravity_wrench(point_mass_robot.inertia.mass, 3)
    t = min_norm_tension(W, w_ext)
    assert np.allclose(W @ t + w_ext, 0.0, atol=1e-9)


def test_tension_distribution_feasible_at_home(point_mass_robot, home_pose):
    W = structure_matrix(home_pose, point_mass_robot)
    w_ext = _gravity_wrench(point_mass_robot.inertia.mass, 3)
    limits = point_mass_robot.limits
    t = tension_distribution(W, w_ext, limits.t_min, limits.t_max)
    assert (t >= limits.t_min - 1e-8).all()
    assert (t <= limits.t_max + 1e-8).all()
    assert np.allclose(W @ t + w_ext, 0.0, atol=1e-6)


def test_centered_objective_stays_inside_bounds_and_solves_equilibrium(point_mass_robot, home_pose):
    """Centered objective keeps the tensions inside the bounds and on the equilibrium line."""
    W = structure_matrix(home_pose, point_mass_robot)
    w_ext = _gravity_wrench(point_mass_robot.inertia.mass, 3)
    limits = point_mass_robot.limits
    t = tension_distribution(W, w_ext, limits.t_min, limits.t_max,
                             objective=TensionObjective.CENTERED)
    assert (t >= limits.t_min - 1e-8).all()
    assert (t <= limits.t_max + 1e-8).all()
    assert np.allclose(W @ t + w_ext, 0.0, atol=1e-6)


def test_feasibility_check_matches_solver(ipanema, home_pose):
    W = structure_matrix(home_pose, ipanema)
    w_ext = np.zeros(6)
    w_ext[2] = -ipanema.inertia.mass * 9.81
    limits = ipanema.limits
    assert is_wrench_feasible(W, w_ext, limits.t_min, limits.t_max)
    # Solver should now succeed.
    t = tension_distribution(W, w_ext, limits.t_min, limits.t_max)
    assert np.allclose(W @ t + w_ext, 0.0, atol=1e-6)


def test_excessive_load_is_infeasible(point_mass_robot, home_pose):
    W = structure_matrix(home_pose, point_mass_robot)
    # Pull straight down with a force that exceeds 4 * t_max * sin(angle).
    excess = np.array([0.0, 0.0, -1e9])
    limits = point_mass_robot.limits
    assert not is_wrench_feasible(W, excess, limits.t_min, limits.t_max)
    with pytest.raises(InfeasibleTensionError):
        tension_distribution(W, excess, limits.t_min, limits.t_max)


def test_preferred_objective_tracks_warm_start(point_mass_robot, home_pose):
    """Using the previous-step tension as t_pref should bias the new solution toward it."""
    W = structure_matrix(home_pose, point_mass_robot)
    w_ext = _gravity_wrench(point_mass_robot.inertia.mass, 3)
    limits = point_mass_robot.limits

    t_default = tension_distribution(W, w_ext, limits.t_min, limits.t_max,
                                     objective=TensionObjective.CENTERED)
    # Bias t_pref toward the upper bound; t should shift upward (within the line).
    pref = np.full(4, limits.t_max[0] * 0.8)
    t_pref_obj = tension_distribution(W, w_ext, limits.t_min, limits.t_max,
                                      objective=TensionObjective.PREFERRED, t_pref=pref)
    assert (t_pref_obj > t_default - 1e-9).all()


# ---------------------------------------------------------------------------
# IPAnema-class: 8 cables, 6 DOF, redundancy r = 2 -- general SLSQP path
# ---------------------------------------------------------------------------

def test_general_path_succeeds_on_ipanema(ipanema, home_pose):
    W = structure_matrix(home_pose, ipanema)
    w_ext = np.zeros(6)
    w_ext[2] = -ipanema.inertia.mass * 9.81
    limits = ipanema.limits
    t = tension_distribution(W, w_ext, limits.t_min, limits.t_max)
    assert (t >= limits.t_min - 1e-6).all()
    assert (t <= limits.t_max + 1e-6).all()
    assert np.allclose(W @ t + w_ext, 0.0, atol=1e-5)
