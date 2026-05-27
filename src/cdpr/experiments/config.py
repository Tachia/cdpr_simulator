r"""Experiment configuration container."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.benchmarks.scenario import Scenario
    from cdpr.benchmarks.suite import BackendKind


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level reproducible-experiment description."""

    name: str
    scenarios: list["Scenario"]
    backends: list["BackendKind"]
    output_root: Path
    seed: int = 0
    tags: dict = field(default_factory=dict)
    notes: str = ""
    write_full_timeseries: bool = True
    write_bundle_report: bool = True

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)

    # --- hashing -----------------------------------------------------

    def describe(self) -> dict:
        from cdpr.benchmarks.scenario import scenario_hash
        return {
            "name": self.name,
            "scenarios": [
                {"name": s.name, "hash": scenario_hash(s), "description": s.describe()}
                for s in self.scenarios
            ],
            "backends": list(self.backends),
            "seed": int(self.seed),
            "tags": self.tags,
            "notes": self.notes,
        }

    def config_hash(self, length: int = 10) -> str:
        payload = json.dumps(self.describe(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:length]
