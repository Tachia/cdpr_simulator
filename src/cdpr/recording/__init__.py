"""Structured experiment recording and replay.

A *recording* is a self-contained on-disk artefact describing one experiment
end-to-end: what was simulated (or what data was ingested), how, on which
robot, and what the resulting time series looked like. Recordings are
deterministic and reproducible --- they carry a manifest with package
versions, RNG seeds, and (where available) a git revision hash --- so that
a downstream reader can reproduce the run from the metadata alone.

Layout on disk::

    <recording_root>/
        metadata.json          robot config, trajectory params, sim settings
        timeseries.csv         time, pose, velocities, tensions, lengths
        diagnostics.csv        per-step condition number, infeasible flag
        manifest.json          reproducibility data (versions, seed, git)

The :class:`ExperimentLog` writer below is what produces this layout.
:func:`load_experiment` reads it back into a :class:`Experiment` --- an
in-memory object structurally equivalent to a Phase-1 ``SimulationResult``
but enriched with the metadata.

Naming note: this package is ``cdpr.recording`` (not ``cdpr.logging``) to
avoid shadowing Python's standard :mod:`logging` module within the cdpr
namespace.
"""

from cdpr.recording.recorder import ExperimentLog, record_simulation
from cdpr.recording.replay import Experiment, load_experiment
from cdpr.recording.compare import ComparisonReport, compare
from cdpr.recording.schema import ExperimentMetadata, ReproducibilityManifest

__all__ = [
    "ExperimentLog",
    "record_simulation",
    "Experiment",
    "load_experiment",
    "compare",
    "ComparisonReport",
    "ExperimentMetadata",
    "ReproducibilityManifest",
]
