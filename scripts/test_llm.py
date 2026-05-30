"""Smoke test for the cdpr.llm layer.

Reports which providers are configured and runs a single round-trip
against the active one. Useful as a first check after adding API
keys to ``.env``.

Usage::

    python scripts/test_llm.py                       # auto-pick provider
    python scripts/test_llm.py --provider gemini
    python scripts/test_llm.py --provider echo       # no keys needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cdpr.llm import (                                              # noqa: E402
    LLMMessage,
    LLMUnavailableError,
    available_providers,
    build_provider,
    default_provider_name,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="cdpr.llm smoke test")
    p.add_argument("--provider", default=None,
                   help="Force a specific provider (gemini / openrouter / "
                        "ollama / lmstudio / echo). Default: auto-pick.")
    p.add_argument("--prompt", default="In one sentence, explain what a "
                                       "cable-driven parallel robot is.",
                   help="Prompt to send.")
    args = p.parse_args(argv)

    # ASCII markers --- Windows cp1252 console can't print ✓/·.
    print("\n=== Provider configuration probe ===")
    for name, configured in available_providers().items():
        marker = "[x]" if configured else "[ ]"
        print(f"  {marker} {name:<10} {'configured' if configured else 'no key/url'}")
    print(f"\nDefault (auto-pick): {default_provider_name()!r}")
    print(f"Building provider:   {args.provider or '<auto>'}")

    try:
        llm = build_provider(args.provider)
    except LLMUnavailableError as exc:
        print(f"\n[ERROR] could not build provider: {exc}", file=sys.stderr)
        return 2

    print(f"  -> provider.name = {llm.name!r}")
    print(f"  -> provider.model = {llm.model!r}")
    print(f"  -> is_available() = {llm.is_available()}")

    print("\n=== Round-trip ===")
    messages = [
        LLMMessage(role="system",
                   content="You are a concise scientific assistant."),
        LLMMessage(role="user", content=args.prompt),
    ]
    try:
        resp = llm.complete(messages, max_tokens=200, temperature=0.2)
    except LLMUnavailableError as exc:
        print(f"\n[ERROR] LLM call failed: {exc}", file=sys.stderr)
        print("\nIf you intended to use the echo stub: run with --provider echo")
        return 3
    print(f"  -> model       : {resp.model}")
    print(f"  -> tokens      : prompt={resp.tokens_prompt} "
          f"completion={resp.tokens_completion}")
    print(f"  -> finish      : {resp.finish_reason}")
    print(f"\nResponse:\n{resp.text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
