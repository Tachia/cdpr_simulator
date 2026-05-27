"""Model calibration / parameter identification from experimental logs.

The canonical CDPR identification problem: given a recorded set of
``(pose, cable_lengths)`` samples and a nominal :class:`Robot` config,
find the parameter perturbations that minimise

.. math::

    \\sum_k \\bigl\\lVert
        \\mathbf{L}_k^\\text{recorded} - \\mathbf{L}_k^\\text{model}(\\mathbf{q}_k;\\, \\boldsymbol\\phi)
    \\bigr\\rVert^2.

The :class:`IdentificationProblem` collects a parameter mask + bounds;
:func:`identify` runs the bounded nonlinear least-squares solve and
returns an :class:`IdentificationResult` with the fitted values,
residual statistics, and an iteration log. The :func:`apply_result`
helper materialises a new :class:`Robot` with the calibration applied
so downstream simulations can run against the corrected model.
"""

from cdpr.identification.parameters import (
    IdentifiableGroup,
    IdentifiableParameters,
    ParameterBounds,
)
from cdpr.identification.problem import (
    IdentificationProblem,
    IdentificationResult,
    apply_result,
    identify,
)

__all__ = [
    "IdentifiableGroup",
    "IdentifiableParameters",
    "ParameterBounds",
    "IdentificationProblem",
    "IdentificationResult",
    "identify",
    "apply_result",
]
