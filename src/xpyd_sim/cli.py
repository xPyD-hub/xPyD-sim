"""CLI entry point."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xpyd-sim", description="xPyD inference simulator")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the unified simulator server")
    serve.add_argument("--mode", choices=["dual", "prefill", "decode"], default="dual")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--model", default="dummy")
    serve.add_argument("--prefill-delay-ms", type=float, default=50.0)
    serve.add_argument("--kv-transfer-delay-ms", type=float, default=5.0)
    serve.add_argument("--decode-delay-per-token-ms", type=float, default=10.0)
    serve.add_argument("--eos-min-ratio", type=float, default=0.5)
    serve.add_argument("--max-model-len", type=int, default=131072)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    if args.command == "serve":
        import uvicorn

        from xpyd_sim.server import ServerConfig, create_app

        config = ServerConfig(
            mode=args.mode,
            model_name=args.model,
            prefill_delay_ms=args.prefill_delay_ms,
            kv_transfer_delay_ms=args.kv_transfer_delay_ms,
            decode_delay_per_token_ms=args.decode_delay_per_token_ms,
            eos_min_ratio=args.eos_min_ratio,
            max_model_len=args.max_model_len,
        )
        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port)
