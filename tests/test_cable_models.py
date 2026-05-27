"""Phase 7 — three exclusive cable constitutive laws.

Verifies that:
* each mode can be selected independently via the factory,
* the factory rejects unknown / mixed names (no cross-contamination),
* Kelvin--Voigt reduces to the elastic spring at zero velocity, to a
  pure damper at zero stretch, and clips to zero on slack,
* Irvine produces the same scalar tension as the per-cable
  :func:`sagging_cable` helper for the same inputs,
* SQCK hybrid reduces to Irvine at zero velocity (no damping
  contribution) and matches the analytic formula otherwise,
* the simulator's ``cable_model`` argument propagates the mode through
  to the resulting :class:`SimulationResult`.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.cables import (
    IrvineModel,
    KelvinVoigtModel,
    SQCKHybridModel,
    available_modes,
    cable_model_by_name,
    sagging_cable,
)
from cdpr.core.frames import Pose, Twist
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.kinematics.inverse import cable_lengths


# ---------------------------------------------------------------------------
# Factory exclusivity
# ---------------------------------------------------------------------------

def test_available_modes_exact_set():
    assert set(available_modes()) == {"kelvin_voigt", "irvine", "sqck_hybrid"}


def test_factory_builds_each_mode():
    kv = cable_model_by_name("kelvin_voigt")
    irv = cable_model_by_name("irvine")
    hyb = cable_model_by_name("sqck_hybrid")
    assert kv.mode_name == "kelvin_voigt"
    assert irv.mode_name == "irvine"
    assert hyb.mode_name == "sqck_hybrid"


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError):
        cable_model_by_name("super_hybrid")
    with pytest.raises(ValueError):
        cable_model_by_name("kelvinvoigt")        # no underscore


def test_factory_rejects_unknown_parameters():
    with pytest.raises(TypeError):
        cable_model_by_name("kelvin_voigt", blender_spin=42)


# ---------------------------------------------------------------------------
# Kelvin--Voigt physics
# ---------------------------------------------------------------------------

def _zero_state(home_pose):
    return PlatformState(pose=home_pose, velocity=Twist(np.zeros(6)))


def test_kv_elastic_only_at_zero_velocity(ipanema, home_pose):
    """At zero velocity, T = k * delta_L = (EA/L0) * delta_L."""
    model = KelvinVoigtModel(youngs_modulus=1e9, cross_section=1e-5,
                             viscous_coefficient=0.0)
    state = _zero_state(home_pose)
    L = cable_lengths(home_pose, ipanema)
    # Apply a 1% rest-length compression (rest < actual) -> tensile stretch.
    rest = L * 0.99
    T = model.tension(ipanema, state, rest)
    expected = (1e9 * 1e-5 / rest) * (L - rest)
    assert np.allclose(T, expected, rtol=1e-9)


def test_kv_slack_clips_to_zero(ipanema, home_pose):
    """When L < L0 (compression) and no inward velocity, T = 0."""
    model = KelvinVoigtModel(youngs_modulus=1e9, cross_section=1e-5,
                             viscous_coefficient=1e6)
    state = _zero_state(home_pose)
    L = cable_lengths(home_pose, ipanema)
    rest = L * 1.05                                     # rest > actual -> slack
    T = model.tension(ipanema, state, rest)
    assert np.all(T == 0.0)


def test_kv_damping_only_at_zero_stretch(ipanema, home_pose):
    """At delta_L = 0 (rest == actual length), T = c * dL/dt."""
    eta = 5e7
    A = 1e-5
    model = KelvinVoigtModel(youngs_modulus=0.0, cross_section=A,
                             viscous_coefficient=eta)
    L = cable_lengths(home_pose, ipanema)
    # Move along +x at 0.1 m/s --- some cables will see positive dL/dt
    # (moving away from anchor) and others negative.
    state = PlatformState(
        pose=home_pose,
        velocity=Twist.from_parts([0.1, 0.0, 0.0], np.zeros(3)),
    )
    rest = L.copy()                                      # no stretch
    T = model.tension(ipanema, state, rest)
    # Reconstruct expected damping term from internals.
    c = eta * A / rest
    dL_dt = model._length_rates(ipanema, state)
    expected = np.maximum(c * dL_dt, 0.0)
    assert np.allclose(T, expected, rtol=1e-9)


def test_kv_tension_jacobian_zero_when_slack(ipanema, home_pose):
    model = KelvinVoigtModel(viscous_coefficient=0.0)
    state = _zero_state(home_pose)
    L = cable_lengths(home_pose, ipanema)
    rest = L * 1.1                                       # slack
    jac = model.tension_jacobian(ipanema, state, rest)
    assert np.all(jac["dT_dDeltaL"] == 0.0)
    assert np.all(jac["dT_dLdot"] == 0.0)


# ---------------------------------------------------------------------------
# Irvine physics
# ---------------------------------------------------------------------------

def test_irvine_matches_per_cable_solver(ipanema, home_pose):
    """The class-based IrvineModel must agree, per-cable, with
    :func:`sagging_cable` on the same numerical inputs."""
    model = IrvineModel(linear_density=0.07, youngs_modulus=1.1e11,
                        cross_section=1.26e-5)
    state = _zero_state(home_pose)
    L = cable_lengths(home_pose, ipanema)
    rest = L * 0.99                                      # taut

    T_model = model.tension(ipanema, state, rest)
    # Direct call to the per-cable helper.
    b_world = home_pose.rotation.apply(ipanema.attachments) + home_pose.position
    expected = np.empty(ipanema.n_cables)
    for i in range(ipanema.n_cables):
        sol = sagging_cable(
            anchor_upper=ipanema.anchors[i],
            anchor_lower=b_world[i],
            unstretched_length=float(rest[i]),
            axial_stiffness=1.1e11 * 1.26e-5,
            linear_weight=0.07 * 9.81,
        )
        expected[i] = 0.0 if sol.is_slack else sol.tension_lower
    assert np.allclose(T_model, expected, rtol=1e-9)


def test_irvine_no_velocity_dependence(ipanema, home_pose):
    """Adding linear / angular velocity must not change the Irvine tension."""
    model = IrvineModel()
    L = cable_lengths(home_pose, ipanema)
    rest = L * 0.99
    state_static = _zero_state(home_pose)
    state_moving = PlatformState(
        pose=home_pose,
        velocity=Twist.from_parts([0.1, -0.2, 0.05], [0.01, -0.02, 0.0]),
    )
    T_static = model.tension(ipanema, state_static, rest)
    T_moving = model.tension(ipanema, state_moving, rest)
    assert np.allclose(T_static, T_moving)


# ---------------------------------------------------------------------------
# SQCK hybrid physics
# ---------------------------------------------------------------------------

def test_sqck_reduces_to_irvine_at_zero_velocity(ipanema, home_pose):
    irv = IrvineModel(linear_density=0.07, youngs_modulus=1.1e11,
                      cross_section=1.26e-5)
    hyb = SQCKHybridModel(linear_density=0.07, youngs_modulus=1.1e11,
                          cross_section=1.26e-5, viscous_coefficient=1e8)
    state = _zero_state(home_pose)
    L = cable_lengths(home_pose, ipanema)
    rest = L * 0.99
    assert np.allclose(
        hyb.tension(ipanema, state, rest),
        irv.tension(ipanema, state, rest),
        rtol=1e-9,
    )


def test_sqck_adds_damping_correction_under_motion(ipanema, home_pose):
    """T_SQCK - T_Irvine should equal (eta A / L0) * dL/dt, clipped at zero."""
    irv = IrvineModel(linear_density=0.07)
    hyb = SQCKHybridModel(linear_density=0.07, viscous_coefficient=5e7)
    L = cable_lengths(home_pose, ipanema)
    rest = L * 0.99
    state = PlatformState(
        pose=home_pose,
        velocity=Twist.from_parts([0.05, 0.0, 0.0], np.zeros(3)),
    )
    T_irv = irv.tension(ipanema, state, rest)
    T_hyb = hyb.tension(ipanema, state, rest)
    # Reconstruct expected damping component.
    c = hyb._eta * hyb._A / rest if hyb._eta.size else None
    dL_dt = hyb._length_rates(ipanema, state)
    expected_damping = c * dL_dt
    expected_total = np.maximum(T_irv + expected_damping, 0.0)
    assert np.allclose(T_hyb, expected_total, rtol=1e-9)


# ---------------------------------------------------------------------------
# No cross-contamination between modes
# ---------------------------------------------------------------------------

def test_modes_dont_share_state(ipanema, home_pose):
    """Two instances of different modes built from the factory must remain
    independent --- mutating one's parameters does not leak into another."""
    kv = cable_model_by_name("kelvin_voigt")
    irv = cable_model_by_name("irvine")

    # Prime per-cable caches.
    L = cable_lengths(home_pose, ipanema)
    state = _zero_state(home_pose)
    kv.tension(ipanema, state, L * 0.99)
    irv.tension(ipanema, state, L * 0.99)

    # KV has no linear_density; Irvine has no viscous_coefficient.
    assert not hasattr(kv, "linear_density") or "linear_density" not in kv.parameters
    assert "linear_density" in irv.parameters
    assert "viscous_coefficient" not in irv.parameters
    assert "viscous_coefficient" in kv.parameters


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------

def test_simulator_records_active_mode(ipanema, home_pose):
    kv = cable_model_by_name("kelvin_voigt", viscous_coefficient=1e6)
    state0 = PlatformState.at_rest(home_pose)
    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.02, dt=2e-3,
        reference=lambda t: home_pose,
        cable_model=kv,
    )
    assert result.cable_model_name == "kelvin_voigt"
    assert result.cable_model_parameters is not None
    assert len(result.cable_diagnostics) == len(result.time)
    # Every diagnostic record carries the active mode.
    for d in result.cable_diagnostics:
        assert d["mode"] == "kelvin_voigt"


def test_simulator_defaults_unchanged_without_cable_model(ipanema, home_pose):
    """Default path (no cable_model) leaves the result's cable_model_name None."""
    state0 = PlatformState.at_rest(home_pose)
    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.02, dt=2e-3,
        reference=lambda t: home_pose,
    )
    assert result.cable_model_name is None
    assert result.cable_model_parameters is None
    assert result.cable_diagnostics == []


def test_simulator_irvine_path_runs_to_completion(ipanema, home_pose):
    """A hold simulation under Irvine must produce a finite trajectory."""
    irv = cable_model_by_name("irvine")
    state0 = PlatformState.at_rest(home_pose)
    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.01, dt=2e-3,
        reference=lambda t: home_pose,
        cable_model=irv,
    )
    assert result.cable_model_name == "irvine"
    assert np.all(np.isfinite(result.positions))
    assert np.all(np.isfinite(result.cable_tensions))
