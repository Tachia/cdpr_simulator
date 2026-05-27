"""Workspace analysis: WCW interior vs exterior, WFW with gravity wrench."""

from __future__ import annotations

import numpy as np

from cdpr.core.frames import Pose, Wrench
from cdpr.workspace.closure import is_in_wcw
from cdpr.workspace.feasible import is_in_wfw
from cdpr.workspace.grid import scan_translational_workspace


def test_point_mass_origin_is_in_wcw(point_mass_robot, home_pose):
    # The tetrahedral point-mass robot's four cables positively span R^3 at the centre.
    assert is_in_wcw(home_pose, point_mass_robot)


def test_point_mass_far_outside_is_not_in_wcw(point_mass_robot):
    from scipy.spatial.transform import Rotation
    # Push the platform far outside the anchor tetrahedron; the cables no
    # longer positively span R^3.
    pose = Pose(position=np.array([0.0, 0.0, 100.0]), rotation=Rotation.identity())
    assert not is_in_wcw(pose, point_mass_robot)


def test_wfw_holds_against_gravity_in_interior(point_mass_robot, home_pose):
    gravity = Wrench.from_parts([0, 0, -point_mass_robot.inertia.mass * 9.81], np.zeros(3))
    assert is_in_wfw(home_pose, point_mass_robot, gravity)


def test_wfw_fails_for_excessive_payload(point_mass_robot, home_pose):
    huge = Wrench.from_parts([0, 0, -1e9], np.zeros(3))
    assert not is_in_wfw(home_pose, point_mass_robot, huge)


def test_scan_produces_non_empty_workspace(point_mass_robot):
    result = scan_translational_workspace(
        point_mass_robot,
        xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5),
        resolution=5, kind="wcw",
    )
    assert result.shape == (5, 5, 5)
    assert result.n_inside > 0
    assert 0.0 < result.fraction_inside <= 1.0
    assert result.estimated_volume() > 0
