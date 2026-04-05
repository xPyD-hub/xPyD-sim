"""Tests for M4: CLI + YAML Configuration (TC11.9, TC11.10, TC11.11)."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_sim.cli import _load_yaml_config, _resolve_config


@pytest.fixture()
def yaml_config_file(tmp_path: Path) -> Path:
    """Create a sample YAML config file."""
    config = textwrap.dedent("""\
        mode: prefill
        port: 9090
        host: 127.0.0.1
        model: test-model
        max_model_len: 65536

        latency:
          prefill_delay_ms: 100.0
          kv_transfer_delay_ms: 10.0
          decode_delay_per_token_ms: 20.0

        eos:
          min_ratio: 0.3

        warmup:
          requests: 5
          penalty_ms: 300.0

        logging:
          request_log: /tmp/test-requests.jsonl

        profile: /tmp/test-profile.yaml
    """)
    p = tmp_path / "config.yaml"
    p.write_text(config)
    return p


def _make_ns(**overrides: object) -> argparse.Namespace:
    """Create a Namespace with all CLI attrs set to None (sentinel) by default.

    Non-None values simulate args that were explicitly provided on the CLI.
    """
    attrs = {
        "mode": None, "port": None, "host": None, "model": None,
        "prefill_delay_ms": None, "kv_transfer_delay_ms": None,
        "decode_delay_per_token_ms": None, "eos_min_ratio": None,
        "max_model_len": None, "warmup_requests": None,
        "warmup_penalty_ms": None, "log_requests": None,
        "profile": None, "max_num_batched_tokens": None,
        "max_num_seqs": None, "scheduling_enabled": None,
    }
    attrs.update(overrides)
    return argparse.Namespace(**attrs)


class TestYAMLLoading:
    """Test YAML config file parsing."""

    def test_load_yaml_config(self, yaml_config_file: Path) -> None:
        cfg = _load_yaml_config(yaml_config_file)
        assert cfg["mode"] == "prefill"
        assert cfg["port"] == 9090
        assert cfg["host"] == "127.0.0.1"
        assert cfg["model"] == "test-model"
        assert cfg["max_model_len"] == 65536
        assert cfg["prefill_delay_ms"] == 100.0
        assert cfg["kv_transfer_delay_ms"] == 10.0
        assert cfg["decode_delay_per_token_ms"] == 20.0
        assert cfg["eos_min_ratio"] == 0.3
        assert cfg["warmup_requests"] == 5
        assert cfg["warmup_penalty_ms"] == 300.0
        assert cfg["log_requests"] == "/tmp/test-requests.jsonl"
        assert cfg["profile"] == "/tmp/test-profile.yaml"

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = _load_yaml_config(p)
        assert cfg == {}

    def test_load_partial_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "partial.yaml"
        p.write_text("mode: decode\nport: 3000\n")
        cfg = _load_yaml_config(p)
        assert cfg["mode"] == "decode"
        assert cfg["port"] == 3000
        assert "prefill_delay_ms" not in cfg


class TestTC11_9_CLIOverridesYAML:
    """TC11.9: CLI args override YAML config."""

    def test_cli_overrides_yaml_mode(self, yaml_config_file: Path) -> None:
        """CLI --mode should override YAML mode."""
        yaml_cfg = _load_yaml_config(yaml_config_file)
        ns = _make_ns(mode="decode")  # CLI explicitly sets mode

        result = _resolve_config(ns, yaml_cfg)
        assert result["mode"] == "decode"  # CLI wins
        assert result["port"] == 9090  # YAML fills in

    def test_cli_overrides_yaml_port(self, yaml_config_file: Path) -> None:
        yaml_cfg = _load_yaml_config(yaml_config_file)
        ns = _make_ns(port=7777)  # CLI explicitly sets port

        result = _resolve_config(ns, yaml_cfg)
        assert result["port"] == 7777  # CLI wins
        assert result["mode"] == "prefill"  # YAML fills in

    def test_cli_overrides_yaml_latency(self, yaml_config_file: Path) -> None:
        yaml_cfg = _load_yaml_config(yaml_config_file)
        ns = _make_ns(prefill_delay_ms=999.0)  # CLI explicitly sets

        result = _resolve_config(ns, yaml_cfg)
        assert result["prefill_delay_ms"] == 999.0  # CLI wins
        assert result["kv_transfer_delay_ms"] == 10.0  # YAML fills in


class TestTC11_10_YAMLOnlyConfig:
    """TC11.10: YAML-only config — all settings applied from YAML."""

    def test_yaml_only_all_settings(self, yaml_config_file: Path) -> None:
        """When no CLI args are explicitly set, all values come from YAML."""
        yaml_cfg = _load_yaml_config(yaml_config_file)
        ns = _make_ns()  # All None — nothing explicitly set

        result = _resolve_config(ns, yaml_cfg)
        assert result["mode"] == "prefill"
        assert result["port"] == 9090
        assert result["host"] == "127.0.0.1"
        assert result["model"] == "test-model"
        assert result["max_model_len"] == 65536
        assert result["prefill_delay_ms"] == 100.0
        assert result["kv_transfer_delay_ms"] == 10.0
        assert result["decode_delay_per_token_ms"] == 20.0
        assert result["eos_min_ratio"] == 0.3
        assert result["warmup_requests"] == 5
        assert result["warmup_penalty_ms"] == 300.0
        assert result["log_requests"] == "/tmp/test-requests.jsonl"
        assert result["profile"] == "/tmp/test-profile.yaml"


class TestTC11_11_DefaultConfig:
    """TC11.11: No config (all defaults) — server starts with sensible defaults."""

    def test_all_defaults(self) -> None:
        """When nothing is provided, sensible defaults are used."""
        ns = _make_ns()
        result = _resolve_config(ns, yaml_config=None)
        assert result["mode"] == "dual"
        assert result["port"] == 8000
        assert result["host"] == "0.0.0.0"
        assert result["model"] == "dummy"
        assert result["prefill_delay_ms"] == 50.0
        assert result["kv_transfer_delay_ms"] == 5.0
        assert result["decode_delay_per_token_ms"] == 10.0
        assert result["eos_min_ratio"] == 0.5
        assert result["max_model_len"] == 131072
        assert result["warmup_requests"] == 0
        assert result["warmup_penalty_ms"] == 0.0
        assert result["log_requests"] is None
        assert result["profile"] is None


class TestEnvVarFallback:
    """Test environment variable fallback (priority: CLI > env > YAML > defaults)."""

    def test_env_var_overrides_yaml(self, yaml_config_file: Path) -> None:
        yaml_cfg = _load_yaml_config(yaml_config_file)
        ns = _make_ns()

        with patch.dict(os.environ, {"XPYD_SIM_PORT": "5555"}):
            result = _resolve_config(ns, yaml_cfg)
        assert result["port"] == 5555  # env wins over YAML
        assert result["mode"] == "prefill"  # YAML still used for others

    def test_cli_overrides_env(self) -> None:
        ns = _make_ns(port=1234)

        with patch.dict(os.environ, {"XPYD_SIM_PORT": "5555"}):
            result = _resolve_config(ns)
        assert result["port"] == 1234  # CLI wins over env

    def test_env_overrides_default(self) -> None:
        ns = _make_ns()

        with patch.dict(os.environ, {"XPYD_SIM_MODE": "decode"}):
            result = _resolve_config(ns)
        assert result["mode"] == "decode"


class TestCLIParsing:
    """Test that CLI argument parsing correctly detects explicitly-set args."""

    def test_parse_with_explicit_args(self) -> None:
        """Verify argparse with None defaults: set args are non-None."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", default=None)
        parser.add_argument("--port", type=int, default=None)

        ns = parser.parse_args(["--mode", "prefill"])
        assert ns.mode == "prefill"  # Set → non-None
        assert ns.port is None  # Not set → None

        result = _resolve_config(ns)
        assert result["mode"] == "prefill"
        assert result["port"] == 8000  # default

    def test_subparser_preserves_explicit_args(self) -> None:
        """Verify that subparser args are correctly detected as explicit.

        This is the regression test for #62: _TrackAction failed in
        subparsers because argparse uses a temporary namespace internally.
        The new sentinel-based approach works correctly.
        """

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        serve = sub.add_parser("serve")
        serve.add_argument("--mode", default=None)
        serve.add_argument("--port", type=int, default=None)

        ns = parser.parse_args(["serve", "--mode", "prefill"])
        assert ns.mode == "prefill"  # Explicitly set
        assert ns.port is None  # Not set

        result = _resolve_config(ns)
        assert result["mode"] == "prefill"  # CLI wins
        assert result["port"] == 8000  # default
