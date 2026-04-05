"""CLI entry point with YAML config and environment variable support."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and flatten a YAML config file into CLI-compatible keys."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    flat: dict[str, Any] = {}
    # Top-level keys
    for key in ("mode", "port", "host", "model", "max_model_len"):
        if key in raw:
            flat[key] = raw[key]

    # Nested latency section
    latency = raw.get("latency", {})
    for key in ("prefill_delay_ms", "kv_transfer_delay_ms", "decode_delay_per_token_ms"):
        if key in latency:
            flat[key] = latency[key]

    # Nested eos section
    eos = raw.get("eos", {})
    if "min_ratio" in eos:
        flat["eos_min_ratio"] = eos["min_ratio"]

    # Nested warmup section
    warmup = raw.get("warmup", {})
    if "requests" in warmup:
        flat["warmup_requests"] = warmup["requests"]
    if "penalty_ms" in warmup:
        flat["warmup_penalty_ms"] = warmup["penalty_ms"]

    # Nested logging section
    logging_cfg = raw.get("logging", {})
    if "request_log" in logging_cfg:
        flat["log_requests"] = logging_cfg["request_log"]

    # Profile
    if "profile" in raw:
        flat["profile"] = raw["profile"]

    # Scheduling section
    scheduling = raw.get("scheduling", {})
    for key in ("max_num_batched_tokens", "max_num_seqs"):
        if key in scheduling:
            flat[key] = scheduling[key]
    if "max_model_len" in scheduling:
        flat["max_model_len"] = scheduling["max_model_len"]
    if "enabled" in scheduling:
        flat["scheduling_enabled"] = scheduling["enabled"]

    return flat


# Mapping: config key -> (env var name, type converter)
_ENV_MAP: dict[str, tuple[str, type]] = {
    "mode": ("XPYD_SIM_MODE", str),
    "port": ("XPYD_SIM_PORT", int),
    "host": ("XPYD_SIM_HOST", str),
    "model": ("XPYD_SIM_MODEL", str),
    "prefill_delay_ms": ("XPYD_SIM_PREFILL_DELAY_MS", float),
    "kv_transfer_delay_ms": ("XPYD_SIM_KV_TRANSFER_DELAY_MS", float),
    "decode_delay_per_token_ms": ("XPYD_SIM_DECODE_DELAY_PER_TOKEN_MS", float),
    "eos_min_ratio": ("XPYD_SIM_EOS_MIN_RATIO", float),
    "max_model_len": ("XPYD_SIM_MAX_MODEL_LEN", int),
    "warmup_requests": ("XPYD_SIM_WARMUP_REQUESTS", int),
    "warmup_penalty_ms": ("XPYD_SIM_WARMUP_PENALTY_MS", float),
    "log_requests": ("XPYD_SIM_LOG_REQUESTS", str),
    "profile": ("XPYD_SIM_PROFILE", str),
    "max_num_batched_tokens": ("XPYD_SIM_MAX_NUM_BATCHED_TOKENS", int),
    "max_num_seqs": ("XPYD_SIM_MAX_NUM_SEQS", int),
    "scheduling_enabled": ("XPYD_SIM_SCHEDULING_ENABLED", lambda v: v.lower() in ("1", "true")),
    "require_api_key": ("XPYD_SIM_REQUIRE_API_KEY", str),
}

# Default values for all config keys
_DEFAULTS: dict[str, Any] = {
    "mode": "dual",
    "port": 8000,
    "host": "0.0.0.0",
    "model": "dummy",
    "prefill_delay_ms": 50.0,
    "kv_transfer_delay_ms": 5.0,
    "decode_delay_per_token_ms": 10.0,
    "eos_min_ratio": 0.5,
    "max_model_len": 131072,
    "warmup_requests": 0,
    "warmup_penalty_ms": 0.0,
    "log_requests": None,
    "profile": None,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 256,
    "scheduling_enabled": False,
    "require_api_key": None,
}


def _resolve_config(
    cli_args: argparse.Namespace,
    yaml_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve config with priority: CLI > env vars > YAML > defaults.

    All argparse args use default=None as a sentinel. A non-None value
    means the user explicitly provided that argument on the CLI.
    """
    result: dict[str, Any] = {}

    # Map CLI arg names to config keys
    cli_to_key = {
        "mode": "mode",
        "port": "port",
        "host": "host",
        "model": "model",
        "prefill_delay_ms": "prefill_delay_ms",
        "kv_transfer_delay_ms": "kv_transfer_delay_ms",
        "decode_delay_per_token_ms": "decode_delay_per_token_ms",
        "eos_min_ratio": "eos_min_ratio",
        "max_model_len": "max_model_len",
        "warmup_requests": "warmup_requests",
        "warmup_penalty_ms": "warmup_penalty_ms",
        "log_requests": "log_requests",
        "profile": "profile",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "max_num_seqs": "max_num_seqs",
        "scheduling_enabled": "scheduling_enabled",
        "require_api_key": "require_api_key",
    }

    for cli_attr, key in cli_to_key.items():
        cli_val = getattr(cli_args, cli_attr, None)
        default_val = _DEFAULTS[key]

        # 1. CLI: non-None means explicitly set (all args default to None)
        if cli_val is not None:
            result[key] = cli_val
            continue

        # 2. Check environment variable
        env_name, converter = _ENV_MAP[key]
        env_val = os.environ.get(env_name)
        if env_val is not None:
            try:
                result[key] = converter(env_val)
            except (ValueError, TypeError):
                result[key] = default_val
            continue

        # 3. Check YAML config
        if yaml_config and key in yaml_config:
            result[key] = yaml_config[key]
            continue

        # 4. Use default
        result[key] = default_val

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xpyd-sim", description="xPyD inference simulator")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the unified simulator server")
    # All defaults are None (sentinel) so _resolve_config can detect
    # whether the user explicitly provided the argument.
    serve.add_argument("--mode", choices=["dual", "prefill", "decode"], default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--model", default=None)
    serve.add_argument("--prefill-delay-ms", type=float, default=None)
    serve.add_argument("--kv-transfer-delay-ms", type=float, default=None)
    serve.add_argument("--decode-delay-per-token-ms", type=float, default=None)
    serve.add_argument("--eos-min-ratio", type=float, default=None)
    serve.add_argument("--max-model-len", type=int, default=None)
    serve.add_argument("--warmup-requests", type=int, default=None)
    serve.add_argument("--warmup-penalty-ms", type=float, default=None)
    serve.add_argument("--log-requests", type=str, default=None)
    serve.add_argument("--profile", type=str, default=None)
    serve.add_argument("--config", type=str, default=None, help="YAML config file path")
    serve.add_argument("--max-num-batched-tokens", type=int, default=None)
    serve.add_argument("--max-num-seqs", type=int, default=None)
    serve.add_argument("--scheduling", action="store_true", default=None,
                       dest="scheduling_enabled")
    serve.add_argument("--require-api-key", type=str, default=None)

    # Calibrate subcommand
    cal = sub.add_parser("calibrate", help="Fit latency curves from sample data")
    cal.add_argument("--input", required=True, help="Path to sample points YAML")
    cal.add_argument("--output", required=True, help="Path to write profile YAML")
    cal.add_argument("--plot", default=None, help="Path to write visualization PNG")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    if args.command == "calibrate":
        from xpyd_sim.calibrate import calibrate

        calibrate(args.input, args.output, args.plot)
        return

    if args.command == "serve":
        import uvicorn

        from xpyd_sim.server import ServerConfig, create_app

        # Load YAML config if provided
        yaml_config = None
        if args.config:
            yaml_config = _load_yaml_config(args.config)

        cfg = _resolve_config(args, yaml_config)

        config = ServerConfig(
            mode=cfg["mode"],
            model_name=cfg["model"],
            prefill_delay_ms=cfg["prefill_delay_ms"],
            kv_transfer_delay_ms=cfg["kv_transfer_delay_ms"],
            decode_delay_per_token_ms=cfg["decode_delay_per_token_ms"],
            eos_min_ratio=cfg["eos_min_ratio"],
            max_model_len=cfg["max_model_len"],
            warmup_requests=cfg["warmup_requests"],
            warmup_penalty_ms=cfg["warmup_penalty_ms"],
            log_requests=cfg["log_requests"],
            profile=cfg["profile"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            max_num_seqs=cfg["max_num_seqs"],
            scheduling_enabled=cfg["scheduling_enabled"],
            require_api_key=cfg["require_api_key"],
        )
        app = create_app(config)
        uvicorn.run(app, host=cfg["host"], port=cfg["port"])
