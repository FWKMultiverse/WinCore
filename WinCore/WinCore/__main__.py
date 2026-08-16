"""
`python -m WinCore` -- a terminal entry point so someone who just ran
`pip install WinCore` has something to run immediately, without first
knowing this project even has a README or an API_REFERENCE.md.

Usage:
    python -m WinCore                  same as --help
    python -m WinCore --help           list what's in this package
    python -m WinCore --docs           print README.md (overview, quick start)
    python -m WinCore --api            print API_REFERENCE.md (full parameter reference)
    python -m WinCore --spec           run WinCore.spec.get_system_spec() on THIS machine
"""
from __future__ import annotations

import argparse
import sys


def _print_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print()
    print("Modules: io, compile, cpu, spec, precision, thermal, diagnostics,")
    print("         multigpu, memory, cache, kv, power, kernels")
    print("Each has its own docstring -- e.g. help(WinCore.spec) from a REPL,")
    print("or see API_REFERENCE.md (--api) for every function and parameter.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m WinCore",
        description="WinCore: a stability + resource-awareness layer for PyTorch training on Windows.",
        add_help=True,
    )
    parser.add_argument("--docs", action="store_true", help="print README.md (overview + quick start)")
    parser.add_argument("--api", action="store_true", help="print API_REFERENCE.md (every function/parameter)")
    parser.add_argument(
        "--spec",
        action="store_true",
        help="detect and print this machine's RAM/CPU/GPU spec right now (needs psutil/pynvml for full detail)",
    )
    args = parser.parse_args(argv)

    if args.docs:
        import WinCore

        print(WinCore.docs())
        return 0

    if args.api:
        import WinCore

        print(WinCore.api_reference())
        return 0

    if args.spec:
        from . import spec

        print(spec.get_system_spec())
        return 0

    _print_help(parser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
