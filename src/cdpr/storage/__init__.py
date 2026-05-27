"""Optional cloud-storage adapters for cdpr experiment recordings.

The scientific core writes everything to local disk through
:mod:`cdpr.recording`. This package adds an *optional* mirror to a
cloud database (currently Supabase) so deployed services can keep a
queryable experiment registry across cold-starts.

Local disk remains the primary store --- if Supabase is unavailable or
not configured, the recorder simply skips the mirror and continues. The
framework's correctness does not depend on any cloud service.

Activation:
    Set ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY`` in the
    environment. With both present, :func:`mirror_experiment` will push
    metadata + metric rows to the Supabase tables defined in
    ``supabase/schema.sql``.
"""

from cdpr.storage.supabase import (
    SupabaseConfig,
    SupabaseMirror,
    mirror_experiment,
    supabase_available,
)

__all__ = [
    "SupabaseConfig",
    "SupabaseMirror",
    "mirror_experiment",
    "supabase_available",
]
