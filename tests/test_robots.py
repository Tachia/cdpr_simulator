"""Catalog smoke tests: each reference robot is consistent and runnable."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.kinematics.inverse import cable_lengths
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.robots import cogiro_class, ipanema_class, planar_translational, point_mass_3d


@pytest.fixture(
    params=[point_mass_3d, planar_translational, ipanema_class, cogiro_class],
    ids=["point-mass", "planar", "ipanema", "cogiro"],
)
def reference_robot(request):
    return request.param()


def test_cable_count_matches_anchors(reference_robot):
    assert reference_robot.geometry.anchors.shape[0] == reference_robot.n_cables
    assert reference_robot.geometry.attachments.shape[0] == reference_robot.n_cables


def test_redundancy_nonnegative(reference_robot):
    assert reference_robot.redundancy >= 1


def test_lengths_positive_at_home(reference_robot, home_pose):
    L = cable_lengths(home_pose, reference_robot)
    assert (L > 0).all()


def test_structure_matrix_full_rank_at_home(reference_robot, home_pose):
    W = structure_matrix(home_pose, reference_robot)
    assert np.linalg.matrix_rank(W) == reference_robot.dof


def test_inertia_present_and_positive_definite(reference_robot):
    I = reference_robot.inertia.inertia
    assert np.allclose(I, I.T)
    assert (np.linalg.eigvalsh(I) > 0).all()
