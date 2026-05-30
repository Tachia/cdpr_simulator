"""Dependency-light HTTP helpers shared by the OpenAI-shaped providers.

Tries :mod:`httpx` if available (better connection reuse, timeouts,
proxy support) and falls back to :mod:`urllib` from the stdlib so the
LLM layer has zero hard runtime dependencies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST JSON and return the parsed response. Raises on HTTP error."""
    try:
        import httpx
    except ImportError:
        return _post_json_urllib(url, payload, headers=headers, timeout=timeout)
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    resp = httpx.post(url, json=payload, headers=h, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post_json_urllib(
    url: str, payload: dict[str, Any], *,
    headers: dict[str, str] | None = None, timeout: float = 30.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def get_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """GET JSON; raises on HTTP error."""
    try:
        import httpx
    except ImportError:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
