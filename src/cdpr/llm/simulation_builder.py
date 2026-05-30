r"""Conversational simulation builder.

Translates free-text descriptions into a :class:`SimulationRequest`
that the existing CLI / Gradio surface can run. Uses :mod:`cdpr.llm`
so the choice of LLM provider is configuration, not code.

Example
-------

    >>> from cdpr.llm.simulation_builder import describe_to_request
    >>> req, info = describe_to_request(
    ...     "Simulate an 8-cable CDPR carrying 50 kg following a "
    ...     "horizontal circle of radius 1 m for 10 seconds.",
    ... )
    >>> req.robot
    'dissertation_8cable'
    >>> req.trajectory.kind
    'circle'

The function returns ``(SimulationRequest, info)`` where ``info`` is
a dict carrying:

* ``provider`` --- which LLM produced the structured output,
* ``model`` --- which model id,
* ``raw_text`` --- the LLM's JSON response (for audit / logging),
* ``confidence`` --- "high" / "low" depending on whether the LLM had to
  fall back to defaults,
* ``follow_up_questions`` --- list of clarifying questions when fields
  were ambiguous, per the directive's "ask follow-up questions" rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from cdpr.interface.specs import SimulationRequest, TrajectorySpec
from cdpr.llm import (
    LLMMessage,
    LLMUnavailableError,
    build_provider,
    resolve_fallback_chain,
)


# Secret-scrub pattern applied to any error string that surfaces in
# ``BuilderResult.notes`` --- belt-and-braces, on top of the same
# redaction the providers do at the raise site. If a future provider
# misses the redaction, the chat box still won't leak the key.
_SECRET_PATTERNS = (
    re.compile(r"([?&])key=[^&\s'\"]+", flags=re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", flags=re.IGNORECASE),
)


def _redact(text: str) -> str:
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub(lambda m: f"{m.group(1)}<redacted>", out)
    return out


_SYSTEM_PROMPT = """\
You are a parser for the CDPR Simulator. Convert the user's free-text
description of a Cable-Driven Parallel Robot experiment into a JSON
object with EXACTLY these fields (any field you cannot determine,
omit --- never invent values):

{
  "robot": one of "point_mass_3d" | "planar_translational" | "ipanema_class" | "cogiro_class" | "dissertation_8cable",
  "payload_mass_kg": float,
  "duration_s": float,
  "dt_s": float,
  "tension_objective": one of "min_norm" | "centered" | "preferred",
  "trajectory": {
    "kind": one of "hold" | "line" | "circle" | "lissajous",
    "params": {
      // for "line":      "start": [x,y,z], "end": [x,y,z]
      // for "circle":    "center": [x,y,z], "radius": float,
      //                  "axis": [x,y,z], "angle_span": float (radians)
      // for "lissajous": "center": [x,y,z], "amplitudes": [x,y,z],
      //                  "frequencies": [x,y,z], "phases": [x,y,z]
    }
  },
  "t_min_N": float,
  "t_max_N": float,
  "follow_up_questions": [
    // one entry per field you couldn't confidently determine
  ]
}

Respond with ONLY the JSON. No markdown fences, no commentary.
Use SI units. If the user says "figure eight" or "lemniscate", map it
to "lissajous" with frequencies [1,2,0]. If the user mentions "8-cable"
or "spatial CDPR", choose "dissertation_8cable" unless the workspace
size strongly suggests "ipanema_class" (1-2 m frame) or "cogiro_class"
(>10 m frame). For the dissertation robot, t_min should be 5 and t_max
500 N unless the user specifies otherwise.
"""


@dataclass
class BuilderResult:
    request: SimulationRequest
    provider: str = ""
    model: str = ""
    raw_text: str = ""
    confidence: str = "low"
    follow_up_questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def describe_to_request(
    description: str,
    *,
    provider: str | None = None,
    fallback_chain: list[str] | None = None,
) -> BuilderResult:
    """Parse ``description`` into a :class:`SimulationRequest`.

    Walks a chain of providers in order; the first one that returns
    parseable JSON wins. When a provider raises
    :class:`LLMUnavailableError` (rate limit, model retired, network
    failure, etc.) the chain continues to the next one and records
    the failure in ``notes``. The echo stub is always at the end so
    the call never raises --- the directive's 'never crash' rule.

    When ``provider`` is given, it pins a single provider (no fallback).
    Otherwise the chain comes from ``fallback_chain`` or
    :func:`cdpr.llm.config.resolve_fallback_chain` (which honours
    ``CDPR_LLM_FALLBACK_CHAIN``).
    """
    if provider:
        chain = [provider]
    else:
        chain = list(fallback_chain or resolve_fallback_chain())

    messages = [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=description.strip()),
    ]
    raw_text = ""
    parsed: dict[str, Any] = {}
    notes: list[str] = []
    tried: list[str] = []
    winner_provider = ""
    winner_model = ""

    for name in chain:
        tried.append(name)
        try:
            llm = build_provider(name)
        except LLMUnavailableError as exc:
            notes.append(
                f"Provider {name!r} unavailable at construction: "
                f"{_redact(str(exc))}. Trying next in chain."
            )
            continue

        try:
            resp = llm.complete(messages, max_tokens=900, temperature=0.0)
        except LLMUnavailableError as exc:
            notes.append(
                f"Provider {name!r} call failed: {_redact(str(exc))}. "
                "Trying next in chain."
            )
            continue

        # Got a response; try to extract JSON. JSON failures from a
        # real LLM are not retryable on a different LLM (echo, in
        # particular, never returns valid JSON), but we still want the
        # raw text in the result for audit. Treat 'no JSON' the same
        # way as 'no provider' so the fallback chain keeps trying.
        raw_text = resp.text
        winner_provider = getattr(llm, "name", name)
        winner_model = getattr(llm, "model", "")
        try:
            parsed = _extract_json(raw_text)
            break
        except (ValueError, KeyError) as exc:
            notes.append(
                f"Provider {name!r} returned non-JSON text: "
                f"{_redact(str(exc))}. Trying next in chain."
            )
            # Clear parsed so the next iteration starts fresh.
            parsed = {}
            continue

    request, confidence = _request_from_parsed(parsed, description, notes)

    if tried:
        notes.append("Provider chain tried: " + " -> ".join(tried))

    return BuilderResult(
        request=request,
        provider=winner_provider,
        model=winner_model,
        raw_text=raw_text,
        confidence=confidence,
        follow_up_questions=list(parsed.get("follow_up_questions", []) or []),
        notes=notes,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of the LLM response, tolerating a few
    common formatting habits (code fences, leading commentary)."""
    # Strip ```json fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Take the first balanced-brace JSON object.
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                return json.loads(snippet)
    raise ValueError("unbalanced braces in response")


def _request_from_parsed(parsed: dict[str, Any], description: str,
                          notes: list[str]) -> tuple[SimulationRequest, str]:
    """Materialise a :class:`SimulationRequest` from the parsed dict,
    using safe defaults whenever a field is missing."""
    confidence = "high" if parsed else "low"

    robot = str(parsed.get("robot") or _infer_robot(description))
    duration = float(parsed.get("duration_s") or 5.0)
    dt = float(parsed.get("dt_s") or 1e-3)
    payload = float(parsed.get("payload_mass_kg") or 0.0)
    objective = str(parsed.get("tension_objective") or "centered")

    traj_dict = parsed.get("trajectory") or {}
    kind = str(traj_dict.get("kind") or _infer_kind(description))
    params = dict(traj_dict.get("params") or {})
    if kind == "circle" and "radius" not in params:
        params.setdefault("center", [0.0, 0.0, 0.65])
        params.setdefault("radius", 0.05)
        params.setdefault("axis", [0.0, 0.0, 1.0])
        params.setdefault("angle_span", 6.283185)
        confidence = "low"
        notes.append("trajectory params defaulted (small workspace-safe circle)")
    if kind == "line" and "start" not in params:
        params.setdefault("start", [0.0, 0.0, 0.5])
        params.setdefault("end",   [0.1, 0.0, 0.5])
    if kind == "lissajous" and "amplitudes" not in params:
        params.setdefault("center", [0.0, 0.0, 0.65])
        params.setdefault("amplitudes", [0.03, 0.03, 0.0])
        params.setdefault("frequencies", [1.0, 2.0, 0.0])
        params.setdefault("phases", [0.0, 1.5707963, 0.0])
        confidence = "low"

    return (
        SimulationRequest(
            robot=robot,
            payload_mass=payload,
            duration=duration,
            dt=dt,
            tension_objective=objective,
            gravity=(0.0, 0.0, -9.81),
            trajectory=TrajectorySpec(kind=kind, duration=duration, params=params),
        ),
        confidence,
    )


def _infer_robot(description: str) -> str:
    s = description.lower()
    if "cogiro" in s or "construction" in s or "stadium" in s:
        return "cogiro_class"
    if "ipanema" in s:
        return "ipanema_class"
    if "8-cable" in s or "8 cable" in s or "eight cable" in s or "spatial" in s:
        return "dissertation_8cable"
    if "planar" in s or "2d" in s:
        return "planar_translational"
    if "point mass" in s or "point-mass" in s:
        return "point_mass_3d"
    return "dissertation_8cable"


def _infer_kind(description: str) -> str:
    s = description.lower()
    if "figure" in s and "eight" in s:
        return "lissajous"
    if "lissajous" in s or "lemniscate" in s:
        return "lissajous"
    if "circle" in s or "circular" in s:
        return "circle"
    if "line" in s or "linear" in s or "straight" in s:
        return "line"
    return "hold"
