"""Reproducible experiment configuration and bundles.

An :class:`ExperimentConfig` fully describes a research experiment ---
which scenarios to run, on which backends, where to write the results.
:func:`run_experiment` executes it, hashes the configuration into a
deterministic directory name, and writes a self-contained artifact
bundle (per-run JSON + master manifest) that can be re-run from disk by
a colleague with no additional context.

The bundle is the unit the dissertation appendix points at: one
``cdpr.experiments.run_experiment`` invocation produces everything
needed to reproduce a figure or table.
"""

from cdpr.experiments.bundle import ExperimentBundle, load_bundle
from cdpr.experiments.config import ExperimentConfig
from cdpr.experiments.runner import run_experiment

__all__ = [
    "ExperimentConfig",
    "ExperimentBundle",
    "run_experiment",
    "load_bundle",
]
