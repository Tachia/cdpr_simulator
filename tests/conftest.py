"""Shared pytest fixtures: reference robots and poses."""

from __future__ import annotations

# Headless matplotlib backend MUST be selected before any plotting import
# happens in any test. Loading the side-effect module here guarantees that.
from tests import conftest_viz  # noqa: F401

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.robots import cogiro_class, ipanema_class, planar_translational, point_mass_3d


@pytest.fixture
def home_pose() -> Pose:
    """Identity pose at the origin --- the most common test starting point."""
    return Pose(position=np.zeros(3), rotation=Rotation.identity())


@pytest.fixture
def point_mass_robot():
    return point_mass_3d()


@pytest.fixture
def planar_robot():
    return planar_translational()


@pytest.fixture
def ipanema():
    return ipanema_class()


@pytest.fixture
def cogiro():
    return cogiro_class()


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    """Prevent figure accumulation across the (many) viz tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def short_sim(ipanema, home_pose):
    """A quick 30-step hold-against-gravity simulation, cached per session.

    Used by viz / recording / reports tests so they don't each pay the cost
    of a fresh integration.
    """
    state0 = PlatformState.at_rest(home_pose)
    return simulate(
        robot=ipanema,
        state0=state0,
        duration=0.03,
        dt=1e-3,
        reference_pose=lambda t: home_pose,
    )
