"""Feedback-control layer.

A *controller* in this framework is anything callable that maps a
``(state, reference, t, robot, gravity, external)`` tuple to the wrench
the cables should apply. The simulator then asks the tension-distribution
solver to realise that wrench with feasible cable tensions; the actual
cable wrench applied to the platform is :math:`\\mathbf{W}(\\mathbf{q})\\,\\boldsymbol\\tau`
at the current pose --- which may differ from the desired wrench when the
QP hits cable bounds.

Two concrete controllers are shipped:

* :class:`~cdpr.control.pd.PDController` -- pose regulator with optional
  gravity compensation; gains have units of force per metre and force per
  metre-per-second. Cheap and robust; the default choice when the user
  only wants the platform to track a reference without leaning on the
  inertia tensor.
* :class:`~cdpr.control.computed_torque.ComputedTorqueController` -- the
  inverse-dynamics ("computed torque") law, which feedback-linearises the
  rigid-body equations of motion. Requires the platform inertia tensor and
  benefits from reference acceleration when the reference is a
  :class:`Trajectory` (which natively provides it).
"""

from cdpr.control.base import Controller, orientation_error
from cdpr.control.composed import FeedforwardPlusFeedback, InverseDynamicsFeedforward
from cdpr.control.computed_torque import ComputedTorqueController
from cdpr.control.mpc import MPCController
from cdpr.control.pd import PDController

__all__ = [
    "Controller",
    "orientation_error",
    "PDController",
    "ComputedTorqueController",
    "MPCController",
    "FeedforwardPlusFeedback",
    "InverseDynamicsFeedforward",
]
