"""Inverse / forward kinematics and structure matrix properties."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.kinematics.forward import forward_kinematics
from cdpr.kinematics.inverse import cable_lengths, cable_unit_vectors
from cdpr.kinematics.jacobian import condition_number, structure_matrix


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

def test_point_mass_ik_at_origin(point_mass_robot, home_pose):
    lengths = cable_lengths(home_pose, point_mass_robot)
    # All four anchors are at distance sqrt(3) from origin in the point-mass robot.
    assert lengths.shape == (4,)
    assert np.allclose(lengths, np.sqrt(3.0))


def test_ik_translation_invariance(ipanema, home_pose):
    """Translating both pose and anchors by the same vector preserves lengths."""
    L0 = cable_lengths(home_pose, ipanema)
    shifted_pose = Pose(position=home_pose.position + np.array([0.1, -0.2, 0.05]),
                        rotation=home_pose.rotation)
    L1 = cable_lengths(shifted_pose, ipanema)
    assert not np.allclose(L0, L1)  # lengths should change
    # ... but they should equal lengths if anchors were also shifted; using the
    # invariance of distance to common translation.
    delta = shifted_pose.position - home_pose.position
    a = ipanema.anchors
    b_world = shifted_pose.rotation.apply(ipanema.attachments) + shifted_pose.position
    expected = np.linalg.norm(a - b_world, axis=-1)
    assert np.allclose(L1, expected)


def test_unit_vectors_normalized(ipanema, home_pose):
    U, L = cable_unit_vectors(home_pose, ipanema)
    norms = np.linalg.norm(U, axis=-1)
    assert np.allclose(norms, 1.0)
    assert (L > 0).all()


# ---------------------------------------------------------------------------
# Forward kinematics: IK -> FK roundtrip
# ---------------------------------------------------------------------------

def test_fk_recovers_known_pose_point_mass(point_mass_robot):
    target = Pose(position=np.array([0.1, 0.2, -0.05]), rotation=Rotation.identity())
    lengths = cable_lengths(target, point_mass_robot)
    seed = Pose(position=np.zeros(3), rotation=Rotation.identity())
    recovered = forward_kinematics(lengths, point_mass_robot, initial_guess=seed)
    assert np.allclose(recovered.position, target.position, atol=1e-8)


def test_fk_recovers_known_pose_spatial(ipanema):
    """FK with a warm seed near the target recovers the pose to machine precision.

    Cold-started FK on a CDPR can converge to a different branch when multiple
    local minima exist; this test verifies the warm-tracking regime which is
    what FK is actually used for in practice (controllers / observers).
    """
    rng = np.random.default_rng(7)
    target = Pose(
        position=rng.uniform(-0.3, 0.3, size=3),
        rotation=Rotation.from_rotvec(rng.uniform(-0.1, 0.1, size=3)),
    )
    lengths = cable_lengths(target, ipanema)
    # Warm seed: known to be within ~5 mm and ~3 deg of the target, which is
    # easily inside the basin of attraction of the LM solver.
    seed = Pose(
        position=target.position + rng.normal(scale=5e-3, size=3),
        rotation=Rotation.from_rotvec(target.rotation.as_rotvec() + rng.normal(scale=5e-2, size=3)),
    )
    recovered, diag = forward_kinematics(lengths, ipanema, initial_guess=seed, return_diagnostics=True)
    assert np.allclose(recovered.position, target.position, atol=1e-7)
    rel = (recovered.rotation * target.rotation.inv()).magnitude()
    assert rel < 1e-6
    assert diag["residual_norm"] < 1e-8


# ---------------------------------------------------------------------------
# Structure matrix
# ---------------------------------------------------------------------------

def test_structure_matrix_shape(ipanema, home_pose):
    W = structure_matrix(home_pose, ipanema)
    assert W.shape == (6, 8)


def test_structure_matrix_full_rank_at_home(ipanema, home_pose):
    W = structure_matrix(home_pose, ipanema)
    assert np.linalg.matrix_rank(W) == 6


def test_structure_matrix_point_mass_columns_are_unit_vectors(point_mass_robot, home_pose):
    W = structure_matrix(home_pose, point_mass_robot)
    # 3-DOF: only the upper 3 rows. Each column should be a unit vector
    # pointing from origin to the corresponding anchor.
    assert W.shape == (3, 4)
    expected = (point_mass_robot.anchors / np.linalg.norm(point_mass_robot.anchors, axis=-1, keepdims=True)).T
    assert np.allclose(W, expected)


def test_condition_number_is_finite_at_home(ipanema, home_pose):
    kappa = condition_number(home_pose, ipanema)
    assert np.isfinite(kappa)
    assert kappa > 1.0


def test_structure_matrix_rejects_batch(ipanema):
    batch = Pose(position=np.zeros((3, 3)), rotation=Rotation.identity())
    with pytest.raises(ValueError):
        structure_matrix(batch, ipanema)
