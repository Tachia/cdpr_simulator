"""Switch Matplotlib to the headless Agg backend before any test imports it.

Imported by name from conftest.py so the global pytest collection sees it.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
