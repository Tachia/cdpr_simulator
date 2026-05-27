"""FastAPI service tests via the in-process TestClient (httpx)."""

from __future__ import annotations

import base64

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cdpr.interface.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_robots(client):
    r = client.get("/robots")
    assert r.status_code == 200
    payload = r.json()
    names = {entry["name"] for entry in payload}
    assert {"point_mass_3d", "planar_translational", "ipanema_class", "cogiro_class"} <= names


def test_simulate_hold(client):
    body = {
        "robot": "ipanema_class",
        "trajectory": {"kind": "hold", "duration": 0.02, "params": {}},
        "duration": 0.02,
        "dt": 1e-3,
        "integrator": "rk4",
        "tension_objective": "centered",
        "payload_mass": 0.0,
        "gravity": [0.0, 0.0, -9.81],
    }
    r = client.post("/simulate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["n_samples"] > 0
    assert data["series"]["time"][0] == 0.0


def test_workspace_wcw(client):
    body = {
        "robot": "point_mass_3d",
        "xlim": [-0.5, 0.5], "ylim": [-0.5, 0.5], "zlim": [-0.5, 0.5],
        "resolution": 5, "kind": "wcw",
    }
    r = client.post("/workspace", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["xs"]) == 5
    assert data["n_inside"] > 0


def test_plot_endpoint_returns_base64_png(client, tmp_path, short_sim, ipanema):
    """Record a real experiment to disk, then ask the /plot endpoint for a PNG."""
    from cdpr.recording import record_simulation
    log = record_simulation(
        robot=ipanema, result=short_sim,
        out_dir=tmp_path / "exp", title="plot endpoint test",
    )
    r = client.post("/plot", json={"log_root": str(log.root), "kind": "cable_tensions"})
    assert r.status_code == 200
    data = r.json()
    assert data["format"] == "png"
    png = base64.b64decode(data["base64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
