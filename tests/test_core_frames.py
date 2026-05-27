"""Pose, twist, wrench algebra."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose, Twist, Wrench, hat, vee
from cdpr.core.numerics import is_close_se3


def test_hat_vee_inverse():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(3)
    assert np.allclose(vee(hat(w)), w)


def test_hat_skew_symmetry():
    rng = np.random.default_rng(1)
    S = hat(rng.standard_normal(3))
    assert np.allclose(S, -S.T)


def test_pose_inverse_composition_is_identity():
    rng = np.random.default_rng(2)
    pose = Pose(position=rng.standard_normal(3), rotation=Rotation.random(random_state=42))
    identity = pose @ pose.inverse()
    T = identity.matrix
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(T[:3, 3], 0.0, atol=1e-12)


def test_from_matrix_roundtrip():
    rng = np.random.default_rng(3)
    R = Rotation.random(random_state=rng)
    p = rng.standard_normal(3)
    pose = Pose(position=p, rotation=R)
    pose_back = Pose.from_matrix(pose.matrix)
    assert is_close_se3(pose.matrix, pose_back.matrix)


def test_transform_point_matches_homogeneous_multiply():
    rng = np.random.default_rng(4)
    pose = Pose(position=rng.standard_normal(3), rotation=Rotation.random(random_state=rng))
    pts = rng.standard_normal((5, 3))
    out = pose.transform_point(pts)
    homogeneous = np.hstack([pts, np.ones((5, 1))])
    expected = (pose.matrix @ homogeneous.T).T[:, :3]
    assert np.allclose(out, expected)


def test_wrench_gravity():
    w = Wrench.gravity(mass=2.0, g=(0.0, 0.0, -9.81))
    assert np.allclose(w.force, [0.0, 0.0, -19.62])
    assert np.allclose(w.torque, 0.0)


def test_twist_adjoint_transport():
    """Pure translation pose transports angular velocity unchanged, adds the lever-arm cross product to linear."""
    p = np.array([1.0, 0.0, 0.0])
    T = Pose(position=p, rotation=Rotation.identity())
    omega = np.array([0.0, 0.0, 1.0])
    xi = Twist.from_parts(linear=np.zeros(3), angular=omega)
    xi_T = xi.transform(T)
    # v' = R v + p x (R omega) = 0 + (1,0,0) x (0,0,1) = (0,-1,0)
    assert np.allclose(xi_T.linear, [0.0, -1.0, 0.0])
    assert np.allclose(xi_T.angular, omega)


def test_wrench_addition_and_negation():
    w1 = Wrench.from_parts([1, 0, 0], [0, 0, 1])
    w2 = Wrench.from_parts([0, 1, 0], [1, 0, 0])
    s = w1 + w2
    assert np.allclose(s.force, [1, 1, 0])
    assert np.allclose((-w1).data, -w1.data)


def test_pose_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Pose(position=np.zeros(2))
    with pytest.raises(ValueError):
        Pose.from_matrix(np.zeros((3, 3)))
