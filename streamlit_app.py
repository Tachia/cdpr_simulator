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

Diagnostic fallback
-------------------

If the full console keeps blanking on Streamlit Cloud but the
Render-hosted FastAPI is reachable and the local CLI works, the issue
is almost certainly platform-side (worker resource limit, websocket
disconnect, install-cache desync). Set ``CDPR_GUI_DIAG=1`` in the
Streamlit Cloud "Secrets" tab to swap in a minimal one-button page:
if even that page blanks on the click, the platform itself is
unhealthy and the FastAPI / CLI escape hatches are the right paths
forward until the worker recovers.
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
os.environ.setdefault("CDPR_BACKEND_URL", "https://cdpr-api.onrender.com")


def _run_diagnostic_page() -> None:
    """A minimal Streamlit page used to isolate Cloud-side blanking.

    If this page blanks on the click, the Streamlit Cloud worker itself
    is unhealthy and the cdpr.interface.gui code cannot be at fault.
    Useful as a one-button A/B against the full console.
    """
    import streamlit as st

    st.set_page_config(page_title="cdpr-diag", layout="centered")
    st.title("cdpr diagnostic page")
    st.write(
        "If you can see this AND the click counter below increments when "
        "you click the button, the Streamlit Cloud worker is healthy and "
        "the issue is in `cdpr.interface.gui`. If the page goes blank on "
        "the click, the worker itself is the problem --- use the "
        "PowerShell CLI (`scripts/run_simulation.py`) or the FastAPI "
        "backend (`scripts/call_render.ps1`) until Cloud recovers."
    )

    n = st.session_state.get("clicks", 0)
    st.metric("clicks", n)
    if st.button("Click me"):
        st.session_state["clicks"] = n + 1
        st.toast(f"clicked {n + 1} times")
        st.rerun()

    with st.expander("Runtime"):
        try:
            import matplotlib                                      # noqa: F401
            mp_backend = matplotlib.get_backend()
        except Exception:
            mp_backend = "import failed"
        st.write({
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "matplotlib_backend": mp_backend,
            "CDPR_GUI_FRUGAL": os.environ.get("CDPR_GUI_FRUGAL"),
            "CDPR_GUI_DIAG": os.environ.get("CDPR_GUI_DIAG"),
            "CDPR_BACKEND_URL": os.environ.get("CDPR_BACKEND_URL"),
            "session_state_keys": sorted(st.session_state.keys()),
        })


if os.environ.get("CDPR_GUI_DIAG") in {"1", "true", "True"}:
    _run_diagnostic_page()
else:
    # 5) Import & render the full console. If anything raises (missing
    #    dep, syntax error, config error), show the traceback in the
    #    page so the user can see what the worker was unhappy about.
    #    Without this guard the worker just dies and the browser shows
    #    a blank tab.
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
