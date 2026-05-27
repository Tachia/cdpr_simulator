"""Identify a known anchor perturbation from synthetic experimental data."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.geometry.robot import RobotGeometry
from cdpr.identification import (
    IdentifiableGroup,
    IdentifiableParameters,
    IdentificationProblem,
    apply_result,
    identify,
)
from cdpr.kinematics.inverse import cable_lengths


def _scatter_poses(seed: int = 0, n: int = 25):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.2, 0.2, size=(n, 3))
    rotvecs = rng.uniform(-0.1, 0.1, size=(n, 3))
    quats = Rotation.from_rotvec(rotvecs).as_quat()
    return positions, quats


def test_identify_recovers_known_anchor_offset(point_mass_robot):
    """Apply a known perturbation to anchors, generate synthetic lengths,
    then verify identification recovers the perturbation."""
    rng = np.random.default_rng(7)
    true_da = rng.uniform(-5e-3, 5e-3, size=(point_mass_robot.n_cables, 3))

    # Build a "true" perturbed robot for data generation.
    perturbed_geom = RobotGeometry(
        anchors=point_mass_robot.anchors + true_da,
        attachments=point_mass_robot.attachments,
        dof=point_mass_robot.dof,
        name="perturbed",
    )
    from cdpr.geometry.robot import Robot
    perturbed = Robot(geometry=perturbed_geom, inertia=point_mass_robot.inertia,
                      limits=point_mass_robot.limits)

    positions, quats = _scatter_poses(seed=0, n=30)
    L_measured = np.array([
        cable_lengths(Pose(position=p, rotation=Rotation.from_quat(q)), perturbed)
        for p, q in zip(positions, quats, strict=True)
    ])

    params = IdentifiableParameters(
        groups=(IdentifiableGroup.ANCHOR_OFFSETS,),
        n_cables=point_mass_robot.n_cables,
    )
    problem = IdentificationProblem(
        robot=point_mass_robot,
        parameters=params,
        positions=positions,
        quaternions_xyzw=quats,
        measured_lengths=L_measured,
    )
    result = identify(problem)
    assert result.converged
    assert result.final_residual_rms < 1e-7
    da_fit = params.anchor_offsets(result.fitted_vector)
    # Sub-micron agreement is the right benchmark on noise-free synthetic
    # data; the trust-region solver's default tolerances cap us here.
    assert np.allclose(da_fit, true_da, atol=5e-6)


def test_identify_drops_residual_substantially(point_mass_robot):
    """Identification of length offsets only --- final residual must be a
    tiny fraction of the initial."""
    rng = np.random.default_rng(3)
    true_dL = rng.uniform(-2e-3, 2e-3, size=point_mass_robot.n_cables)

    positions, quats = _scatter_poses(seed=1, n=30)
    L_nominal = np.array([
        cable_lengths(Pose(position=p, rotation=Rotation.from_quat(q)),
                      point_mass_robot)
        for p, q in zip(positions, quats, strict=True)
    ])
    L_measured = L_nominal + true_dL                            # encoder bias

    params = IdentifiableParameters(
        groups=(IdentifiableGroup.CABLE_LENGTH_OFFSETS,),
        n_cables=point_mass_robot.n_cables,
    )
    problem = IdentificationProblem(
        robot=point_mass_robot, parameters=params,
        positions=positions, quaternions_xyzw=quats,
        measured_lengths=L_measured,
    )
    result = identify(problem)
    assert result.final_residual_rms < 1e-3 * result.initial_residual_rms
    assert np.allclose(params.cable_length_offsets(result.fitted_vector),
                       true_dL, atol=1e-7)


def test_apply_result_returns_corrected_robot(point_mass_robot):
    """Applying the identified parameters should produce a robot whose IK
    matches the noisy measurements."""
    positions, quats = _scatter_poses(seed=2, n=15)
    da_truth = np.array([[2e-3, -1e-3, 0.0]] * point_mass_robot.n_cables)
    from cdpr.geometry.robot import Robot, RobotGeometry
    truth_geom = RobotGeometry(
        anchors=point_mass_robot.anchors + da_truth,
        attachments=point_mass_robot.attachments,
        dof=point_mass_robot.dof, name="truth",
    )
    truth = Robot(geometry=truth_geom, inertia=point_mass_robot.inertia,
                  limits=point_mass_robot.limits)
    L_measured = np.array([
        cable_lengths(Pose(position=p, rotation=Rotation.from_quat(q)), truth)
        for p, q in zip(positions, quats, strict=True)
    ])

    params = IdentifiableParameters(
        groups=(IdentifiableGroup.ANCHOR_OFFSETS,),
        n_cables=point_mass_robot.n_cables,
    )
    problem = IdentificationProblem(
        robot=point_mass_robot, parameters=params,
        positions=positions, quaternions_xyzw=quats,
        measured_lengths=L_measured,
    )
    result = identify(problem)
    fitted_robot = apply_result(problem, result)
    L_fitted = np.array([
        cable_lengths(Pose(position=p, rotation=Rotation.from_quat(q)), fitted_robot)
        for p, q in zip(positions, quats, strict=True)
    ])
    assert np.allclose(L_fitted, L_measured, atol=1e-7)
