r"""Supabase storage mirror for cdpr experiment recordings.

The adapter is deliberately thin: local disk (via
:mod:`cdpr.recording`) remains the source of truth, and Supabase only
holds a queryable copy of the experiment metadata + summary metrics.
Heavy time-series CSVs stay on disk; the cloud row carries the
``local_path`` so the operator can fetch the bundle by ID later.

Three failure modes are explicitly *non-fatal*:

* the ``supabase`` Python package is not installed,
* ``SUPABASE_URL`` / ``SUPABASE_SERVICE_ROLE_KEY`` env vars are missing,
* the HTTP call to Supabase fails (network blip, RLS policy issue).

In all three cases the mirror is skipped silently and a diagnostic is
attached to the :class:`SupabaseMirror.last_error` field. The recorder
never re-raises a Supabase exception --- it must not interrupt the
scientific run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.recording.recorder import ExperimentLog


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

def supabase_available() -> bool:
    """True iff the supabase package is importable AND env vars are set."""
    if not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
        return False
    try:
        import supabase  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class SupabaseConfig:
    """Connection settings, normally pulled from env."""

    url: str
    service_role_key: str
    schema: str = "public"

    @classmethod
    def from_env(cls) -> "SupabaseConfig | None":
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return None
        return cls(url=url, service_role_key=key)


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SupabaseMirror:
    """Write-side adapter: takes a local experiment and copies its metadata
    into the Supabase experiments table.

    Stateless across writes; safe to construct once per process and reuse.
    The actual ``supabase-py`` client is lazy-instantiated on first use so
    importing this module costs nothing when the env vars are absent.
    """

    config: SupabaseConfig
    _client: Any = None
    last_error: str | None = None

    def _connect(self) -> Any:
        if self._client is None:
            from supabase import create_client                # type: ignore[import-not-found]
            self._client = create_client(self.config.url, self.config.service_role_key)
        return self._client

    def push_experiment(
        self,
        log: "ExperimentLog",
        *,
        metrics: dict[str, float] | None = None,
        extra_tags: dict[str, Any] | None = None,
    ) -> str | None:
        """Push one experiment row + optional metric rows. Returns the new
        row's UUID on success, ``None`` on any failure (with the error
        captured in :attr:`last_error`)."""
        try:
            import json
            metadata_path = log.metadata_path
            manifest_path = log.manifest_path
            metadata = json.loads(metadata_path.read_text())
            manifest = json.loads(manifest_path.read_text())

            payload = {
                "name": metadata.get("title", "unknown"),
                "scenario_hash": str(metadata.get("experiment_id", "")),
                "config_hash": manifest.get("experiment_id_short", manifest.get("git_revision", "")),
                "cable_mode": (metadata.get("tags") or {}).get("cable_mode", "none"),
                "backend": (metadata.get("tags") or {}).get("backend", "cdpr"),
                "seed": manifest.get("seed"),
                "cdpr_version": manifest.get("cdpr_version"),
                "git_revision": manifest.get("git_revision"),
                "notes": (metadata.get("simulation") or {}).get("notes"),
                "tags": {**(metadata.get("tags") or {}), **(extra_tags or {})},
                "local_path": str(Path(log.root).resolve()),
            }

            client = self._connect()
            res = (
                client.schema(self.config.schema)
                .table("experiments")
                .insert(payload)
                .execute()
            )
            if not res.data:
                self.last_error = "Supabase returned no row on insert."
                return None
            experiment_id = res.data[0]["id"]

            if metrics:
                metric_rows = [
                    {"experiment_id": experiment_id, "metric": k, "value": float(v)}
                    for k, v in metrics.items()
                ]
                client.schema(self.config.schema).table(
                    "experiment_metrics"
                ).insert(metric_rows).execute()

            self.last_error = None
            return experiment_id

        except Exception as exc:                              # pragma: no cover - network path
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def mirror_experiment(
    log: "ExperimentLog",
    *,
    metrics: dict[str, float] | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> str | None:
    """One-shot helper: read env, push, return new UUID (or None on failure).

    The recorder calls this after a successful local write when the user
    asks for cloud mirroring. A return of ``None`` is *not* an error
    condition for the scientific run --- the local artefact bundle is
    still intact.
    """
    if not supabase_available():
        return None
    cfg = SupabaseConfig.from_env()
    if cfg is None:                                           # pragma: no cover - guarded above
        return None
    mirror = SupabaseMirror(config=cfg)
    return mirror.push_experiment(log, metrics=metrics, extra_tags=extra_tags)
