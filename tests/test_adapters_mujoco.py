"""MuJoCo adapter: model build, pose round-trip, step, cable lengths, verification."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

mujoco = pytest.importorskip("mujoco")

from cdpr.adapters import AdapterCapability, make_backend, verify_against
from cdpr.adapters.mujoco import build_mjcf
from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.kinematics.inverse import cable_lengths


def test_build_mjcf_parses_as_valid_model(ipanema):
    xml = build_mjcf(ipanema)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody >= 2          # world + platform
    # 8 tendons for IPAnema-class
    assert model.ntendon == ipanema.n_cables


def test_make_backend_constructs_mujoco(ipanema):
    backend = make_backend("mujoco", robot=ipanema)
    try:
        assert backend.name == "mujoco"
        for flag in (AdapterCapability.LOAD_ROBOT, AdapterCapability.SET_POSE,
                     AdapterCapability.READ_STATE, AdapterCapability.STEP_PHYSICS,
                     AdapterCapability.APPLY_WRENCH,
                     AdapterCapability.READ_CABLE_LENGTHS):
            assert backend.has(flag)
    finally:
        backend.close()


def test_pose_round_trip(ipanema):
    rng = np.random.default_rng(0)
    target = Pose(
        position=rng.uniform(-0.2, 0.2, size=3),
        rotation=Rotation.from_rotvec(rng.uniform(-0.1, 0.1, size=3)),
    )
    with make_backend("mujoco", robot=ipanema) as backend:
        backend.set_pose(target)
        readback = backend.read_state()
        assert np.allclose(readback.pose.position, target.position, atol=1e-9)
        # Compare rotations as rotation vectors.
        d = (readback.pose.rotation * target.rotation.inv()).magnitude()
        assert d < 1e-6


def test_cable_lengths_match_cdpr_geometry(ipanema, home_pose):
    with make_backend("mujoco", robot=ipanema) as backend:
        backend.set_pose(home_pose)
        L_backend = backend.read_cable_lengths()
        L_cdpr = cable_lengths(home_pose, ipanema)
        assert np.allclose(L_backend, L_cdpr, atol=1e-9)


def test_free_fall_matches_analytic_with_mujoco(ipanema, home_pose):
    """Apply only gravity (zero cable wrench). MuJoCo should produce a
    free-fall trajectory matching ``z(t) = z0 + 0.5 g t^2``."""
    state0 = PlatformState.at_rest(home_pose)
    g = -9.81
    dt = 1e-3
    n = 200
    with make_backend("mujoco", robot=ipanema, gravity=(0, 0, g),
                      timestep=dt) as backend:
        backend.set_pose(home_pose)
        backend.apply_wrench(Wrench(np.zeros(6)))
        for _ in range(n):
            backend.step(dt)
        final = backend.read_state()
    t = n * dt
    expected_z = 0.5 * g * t * t
    assert final.pose.position[2] == pytest.approx(expected_z, abs=1e-4)
    assert final.velocity.linear[2] == pytest.approx(g * t, abs=1e-4)


def test_verify_against_mujoco_short_hold(ipanema, home_pose):
    """A static-hold scenario through cdpr vs MuJoCo. Divergence should
    sit at the integrator-comparison level (sub-millimetre over 50 ms)."""
    state0 = PlatformState.at_rest(home_pose)
    backend = make_backend("mujoco", robot=ipanema, timestep=1e-3)
    try:
        report = verify_against(
            backend, ipanema, state0,
            duration=0.05, dt=1e-3,
        )
        summary = report.summary()
        assert summary["position_m"]["peak"] < 1e-3
        assert summary["orientation_deg"]["peak"] < 0.1
    finally:
        backend.close()
