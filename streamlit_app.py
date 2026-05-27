"""Top-level entry point for Streamlit Community Cloud.

Community Cloud expects a single ``streamlit_app.py`` at the repo root.
We delegate immediately into the framework's console at
:mod:`cdpr.interface.gui` so the actual UI code stays where it belongs.

This file is also what ``streamlit run streamlit_app.py`` runs locally
when developing the UI without the backend service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the src/ layout importable when running from the repo root
# (Community Cloud doesn't install the package in development mode).
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Default backend URL --- override with CDPR_BACKEND_URL in the env.
os.environ.setdefault("CDPR_BACKEND_URL", "http://localhost:8000")

# Importing the gui module triggers Streamlit page rendering.
from cdpr.interface import gui  # noqa: F401, E402
