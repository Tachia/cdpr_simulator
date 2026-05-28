r"""Unified CLI driver for the built-in CDPR_SIMULATOR demonstrations.

Five examples are registered in :mod:`scripts.examples` and exposed
here through a single entry point so the directive's "selectable
tutorials" requirement is satisfied for the PowerShell workflow.

::

    python scripts/run_example.py --list
    python scripts/run_example.py --name circle
    python scripts/run_example.py --name spiral --open
    python scripts/run_example.py --name mshape
    python scripts/run_example.py --name train
    python scripts/run_example.py --name compare --open

For Phase-2 examples (``train``, ``compare``) the runner auto-checks
for the Phase-1 CSV they depend on and re-generates it if missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one of the five built-in CDPR_SIMULATOR examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", type=str, default=None,
                   help="Example identifier: circle, spiral, mshape, train, compare.")
    p.add_argument("--list", action="store_true",
                   help="List all available examples and exit.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override the per-example output directory.")
    p.add_argument("--open", action="store_true",
                   help="Open the result folder in Explorer when done (Windows only).")
    return p.parse_args(argv)


def _print_list(examples) -> None:
    print("Built-in CDPR_SIMULATOR examples")
    print("================================")
    print()
    for e in examples:
        print(f"[{e['name']}] {e['title']} (phase {e['phase']})")
        for line in e["description"].splitlines():
            print(f"    {line}")
        if "depends_on" in e:
            print(f"    depends on: example '{e['depends_on']}'")
        print()
    print("Run with: python scripts/run_example.py --name <id>")


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from examples import EXAMPLES, list_examples

    args = parse_args(argv)

    if args.list or args.name is None:
        _print_list(list_examples())
        return 0

    if args.name not in EXAMPLES:
        print(f"Unknown example: {args.name!r}", file=sys.stderr)
        print(f"Available: {', '.join(EXAMPLES.keys())}", file=sys.stderr)
        return 2

    spec = EXAMPLES[args.name]
    out_root = Path("out")
    out_dir = args.out or (out_root / spec["out_dir"])
    runner = spec["runner"]
    t0 = time.perf_counter()
    result = runner(out_dir)
    dt = time.perf_counter() - t0
    print(f"\n[done] {args.name} in {dt:.1f} s -> {Path(out_dir).resolve()}")

    if args.open and sys.platform == "win32":
        try:
            os.startfile(str(Path(out_dir)))                       # type: ignore[attr-defined]
        except Exception as exc:                                   # pragma: no cover
            print(f"could not open folder: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
