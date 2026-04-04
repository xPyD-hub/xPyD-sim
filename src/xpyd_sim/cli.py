"""CLI entry point."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xpyd-sim", description="xPyD inference simulator")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("prefill", help="Start prefill node")
    p.add_argument("--port", type=int, default=8001)
    d = sub.add_parser("decode", help="Start decode node")
    d.add_argument("--port", type=int, default=8002)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    print(f"xpyd-sim {args.command} — not yet implemented")
