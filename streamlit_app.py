"""Top-level entry point for Streamlit Community Cloud.

Community Cloud expects a single ``streamlit_app.py`` at the repo root.
We delegate immediately into the framework's console at
:mod:`cdpr.interface.gui` so the actual UI code stays where it belongs.

This file is also what ``streamlit run streamlit_app.py`` runs locally.

Critical: matplotlib's backend MUST be selected before any plotting
import. Streamlit Cloud workers run headless --- the default Tk / Qt
backends will either fail to import or hang on the first figure draw,
which produces a blank page with no Python-level traceback. The
``Agg`` setting is forced here at the very top of the entry-point so
every later import sees it.

If importing the GUI itself fails (a bad commit, a missing extra, a
syntax error introduced upstream), we render the traceback in the page
rather than letting Streamlit serve a blank tab.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 1) Headless matplotlib backend, set before any matplotlib import.
os.environ["MPLBACKEND"] = "Agg"

# 2) Lift the src/ layout onto sys.path. Streamlit Cloud installs the
#    package from requirements.txt, but during local dev `streamlit run`
#    from the repo root has no install --- fallback for both cases.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 3) Hint to the GUI that we want frugal defaults on a free-tier
#    container (smaller default simulation + lazy per-tab plot render).
#    Set via env so the GUI module stays deployment-agnostic.
os.environ.setdefault("CDPR_GUI_FRUGAL", "1")

# 4) Default backend URL for the future variant of the GUI that talks
#    to the FastAPI service over HTTP. The current GUI embeds cdpr in
#    its own process and ignores this.
os.environ.setdefault("CDPR_BACKEND_URL", "http://localhost:8000")

# 5) Import & render. If anything raises (missing dep, syntax error,
#    config error), show the traceback in the page so the user can see
#    what the worker was unhappy about. Without this guard the worker
#    just dies and the browser shows a blank tab.
try:
    from cdpr.interface import gui  # noqa: F401, E402
except Exception as exc:  # pragma: no cover - last-resort surface
    import streamlit as st

    try:
        st.set_page_config(page_title="CDPR — load failure", layout="wide")
    except Exception:
        pass
    st.title("CDPR research console — failed to load")
    st.error(
        "The console failed to import. The traceback below tells you "
        "which module/line is at fault."
    )
    st.exception(exc)
    st.code(traceback.format_exc(), language="text")
    # Also write to stderr so the Streamlit Cloud "manage app" log
    # carries the same trace.
    print(traceback.format_exc(), file=sys.stderr, flush=True)
