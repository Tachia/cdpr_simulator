"""Cable physical models.

Two layers live here:

* **Per-cable function helpers** (Phase 1). Return a
  :class:`CableSolution` for one cable evaluated at one configuration.
  :func:`massless_cable`, :func:`elastic_cable`, :func:`sagging_cable`.

* **Constitutive model classes** (Phase 7). One of three exclusive
  laws governs cable tension across the entire CDPR for one run.
  :class:`KelvinVoigtModel`, :class:`IrvineModel`, :class:`SQCKHybridModel`;
  build by name via :func:`cable_model_by_name`. The simulator,
  benchmark suite, and report layer accept a single
  :class:`CableModel` instance and propagate it through.

The two layers are not interchangeable --- the per-cable functions are
geometric building blocks; the constitutive classes are the dissertation's
mutually exclusive modelling choices.
"""

from cdpr.cables.base import CableModel, CableSolution
from cdpr.cables.diagnostics import ComparisonReport, ModeDiagnostics, sweep_modes
from cdpr.cables.elastic import elastic_cable
from cdpr.cables.factory import CableModeName, available_modes, cable_model_by_name
from cdpr.cables.irvine import IrvineModel
from cdpr.cables.kelvin_voigt import KelvinVoigtModel
from cdpr.cables.massless import massless_cable
from cdpr.cables.sagging import sagging_cable
from cdpr.cables.sqck_hybrid import SQCKHybridModel

__all__ = [
    # Per-cable function helpers (Phase 1)
    "CableSolution",
    "massless_cable",
    "elastic_cable",
    "sagging_cable",
    # Constitutive model classes (Phase 7)
    "CableModel",
    "KelvinVoigtModel",
    "IrvineModel",
    "SQCKHybridModel",
    "cable_model_by_name",
    "available_modes",
    "CableModeName",
    # Diagnostics
    "sweep_modes",
    "ComparisonReport",
    "ModeDiagnostics",
]
