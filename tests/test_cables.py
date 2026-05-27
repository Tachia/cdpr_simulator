"""Cable physical models: massless, elastic, Irvine catenary."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.cables import elastic_cable, massless_cable, sagging_cable


def test_massless_cable_geometry():
    sol = massless_cable(anchor_upper=[0, 0, 5], anchor_lower=[0, 0, 0], tension=100.0)
    assert sol.chord_length == pytest.approx(5.0)
    assert sol.arc_length == pytest.approx(5.0)
    assert sol.sag_max == 0.0
    assert np.allclose(sol.force_on_platform, [0, 0, 100.0])


def test_elastic_cable_slack_when_chord_short():
    sol = elastic_cable(
        anchor_upper=[0, 0, 1], anchor_lower=[0, 0, 0],
        unstretched_length=2.0, axial_stiffness=1e6,
    )
    assert sol.is_slack
    assert sol.tension_lower == 0.0


def test_elastic_cable_tension_matches_strain():
    EA = 1e6
    L0 = 5.0
    chord = 5.1
    sol = elastic_cable(
        anchor_upper=[0, 0, chord], anchor_lower=[0, 0, 0],
        unstretched_length=L0, axial_stiffness=EA,
    )
    expected_T = EA * (chord - L0) / L0
    assert sol.tension_lower == pytest.approx(expected_T)
    assert np.allclose(sol.force_on_platform, [0, 0, expected_T])


# ---------------------------------------------------------------------------
# Sagging cable
# ---------------------------------------------------------------------------

def test_sagging_reduces_to_elastic_for_negligible_weight():
    """With w -> 0 the Irvine model should match the straight elastic cable."""
    EA = 1e7
    L0 = 10.0
    chord = 10.05
    # Slight stretch; place anchor along +x for a non-vertical cable.
    sag_sol = sagging_cable(
        anchor_upper=[chord, 0, 0], anchor_lower=[0, 0, 0],
        unstretched_length=L0, axial_stiffness=EA, linear_weight=1e-6,
    )
    elastic_sol = elastic_cable(
        anchor_upper=[chord, 0, 0], anchor_lower=[0, 0, 0],
        unstretched_length=L0, axial_stiffness=EA,
    )
    assert sag_sol.tension_lower == pytest.approx(elastic_sol.tension_lower, rel=1e-3)
    # Absolute tolerance because the y/z components are essentially zero.
    assert np.allclose(sag_sol.force_on_platform, elastic_sol.force_on_platform, rtol=1e-3, atol=1e-3)


def test_sagging_tension_upper_exceeds_lower_by_cable_weight():
    """Conservation: T_upper^2 - T_lower^2 mass-balance check.

    From the Irvine free-body: V_top = V_lower + w*L0 and H is constant, so
    T_top^2 - T_low^2 = (V_top + V_low) * w * L0 = (2 V + w L0) * w L0.
    We just check T_top > T_low for a hanging cable.
    """
    # Chord = sqrt(50) ~ 7.07 m, L0 = 7 m -> small stretch, well in the
    # tensioned regime where the Irvine solver is stable.
    sol = sagging_cable(
        anchor_upper=[5.0, 0.0, 5.0], anchor_lower=[0.0, 0.0, 0.0],
        unstretched_length=7.0, axial_stiffness=1e6, linear_weight=2.0,
    )
    assert not sol.is_slack
    assert sol.tension_upper > sol.tension_lower
    assert sol.tension_lower > 0


def test_sagging_force_has_upward_component_on_platform():
    sol = sagging_cable(
        anchor_upper=[3.0, 0.0, 4.0], anchor_lower=[0.0, 0.0, 0.0],
        unstretched_length=4.95, axial_stiffness=5e6, linear_weight=1.0,
    )
    # Anchor is up and to the right; cable pulls platform up and to the right.
    assert sol.force_on_platform[0] > 0
    assert sol.force_on_platform[2] > 0


def test_sagging_chord_shorter_than_unstretched_is_slack():
    sol = sagging_cable(
        anchor_upper=[1.0, 0.0, 0.0], anchor_lower=[0.0, 0.0, 0.0],
        unstretched_length=2.0, axial_stiffness=1e6, linear_weight=1.0,
    )
    assert sol.is_slack
    assert sol.tension_lower == 0.0
