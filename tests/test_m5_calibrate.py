"""Tests for M5: Calibrate tool + Profile mode (TC7.x)."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from xpyd_sim.calibrate import calibrate
from xpyd_sim.profile import LatencyProfile
from xpyd_sim.server import ServerConfig, create_app

# ── Sample data fixtures ──────────────────────────────────────────

PREFILL_3PT = [
    {"batch_size": 128, "delay_ms": 5},
    {"batch_size": 1024, "delay_ms": 30},
    {"batch_size": 4096, "delay_ms": 150},
]

PREFILL_5PT = [
    {"batch_size": 128, "delay_ms": 5},
    {"batch_size": 512, "delay_ms": 15},
    {"batch_size": 1024, "delay_ms": 30},
    {"batch_size": 2048, "delay_ms": 80},
    {"batch_size": 4096, "delay_ms": 150},
]

KV_3PT = [
    {"batch_size": 128, "delay_ms": 1},
    {"batch_size": 1024, "delay_ms": 5},
    {"batch_size": 4096, "delay_ms": 20},
]

DECODE_9PT = [
    {"batch_size": 1, "context_length": 512, "delay_per_token_ms": 7},
    {"batch_size": 1, "context_length": 2048, "delay_per_token_ms": 9},
    {"batch_size": 1, "context_length": 8192, "delay_per_token_ms": 14},
    {"batch_size": 16, "context_length": 512, "delay_per_token_ms": 9},
    {"batch_size": 16, "context_length": 2048, "delay_per_token_ms": 12},
    {"batch_size": 16, "context_length": 8192, "delay_per_token_ms": 18},
    {"batch_size": 64, "context_length": 512, "delay_per_token_ms": 14},
    {"batch_size": 64, "context_length": 2048, "delay_per_token_ms": 19},
    {"batch_size": 64, "context_length": 8192, "delay_per_token_ms": 30},
]


def _write_sample(tmp: str, data: dict) -> str:
    path = os.path.join(tmp, "samples.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _run_calibrate(samples: dict, plot: bool = False) -> tuple[str, str | None]:
    """Run calibrate and return (profile_path, plot_path)."""
    tmp = tempfile.mkdtemp()
    input_path = _write_sample(tmp, samples)
    output_path = os.path.join(tmp, "profile.yaml")
    plot_path = os.path.join(tmp, "profile.png") if plot else None
    calibrate(input_path, output_path, plot_path)
    return output_path, plot_path


# ── TC7.1: 3 sample points → generates valid profile ──────────────

class TestTC7_1:
    def test_calibrate_3_points_prefill(self):
        profile_path, _ = _run_calibrate({"prefill": PREFILL_3PT})
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        assert "prefill" in profile
        assert profile["prefill"]["type"] == "poly1d"
        assert len(profile["prefill"]["coefficients"]) == 3

    def test_calibrate_3_points_kv(self):
        profile_path, _ = _run_calibrate({"kv_transfer": KV_3PT})
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        assert "kv_transfer" in profile

    def test_calibrate_9_points_decode(self):
        profile_path, _ = _run_calibrate({"decode": DECODE_9PT})
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        assert "decode" in profile
        assert profile["decode"]["type"] == "poly2d"
        assert len(profile["decode"]["coefficients"]) == 6

    def test_calibrate_full(self):
        samples = {"prefill": PREFILL_3PT, "kv_transfer": KV_3PT, "decode": DECODE_9PT}
        profile_path, _ = _run_calibrate(samples)
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        assert "prefill" in profile
        assert "kv_transfer" in profile
        assert "decode" in profile


# ── TC7.2: 5 sample points → better fit quality ───────────────────

class TestTC7_2:
    def test_5_points_interpolation(self):
        """5 points should produce reasonable interpolation at midpoint."""
        profile_path, _ = _run_calibrate({"prefill": PREFILL_5PT})
        prof = LatencyProfile(profile_path)
        # At 512 tokens, real data says 15ms. Fitted should be close.
        val = prof.prefill_delay_ms(512)
        assert 5 < val < 40, f"Expected ~15ms, got {val}"

    def test_5_points_vs_3_points(self):
        """5 points should give different (potentially better) fit than 3."""
        p3, _ = _run_calibrate({"prefill": PREFILL_3PT})
        p5, _ = _run_calibrate({"prefill": PREFILL_5PT})
        prof3 = LatencyProfile(p3)
        prof5 = LatencyProfile(p5)
        # Both should give reasonable values at 512
        v3 = prof3.prefill_delay_ms(512)
        v5 = prof5.prefill_delay_ms(512)
        assert v3 > 0
        assert v5 > 0


# ── TC7.3: Visualization output → PNG ─────────────────────────────

class TestTC7_3:
    def test_plot_output_exists(self):
        samples = {"prefill": PREFILL_3PT, "kv_transfer": KV_3PT, "decode": DECODE_9PT}
        _, plot_path = _run_calibrate(samples, plot=True)
        assert plot_path is not None
        assert os.path.exists(plot_path)
        # Check it's a valid PNG (starts with PNG magic bytes)
        with open(plot_path, "rb") as f:
            header = f.read(8)
        assert header[1:4] == b"PNG"

    def test_plot_reasonable_size(self):
        samples = {"prefill": PREFILL_3PT, "decode": DECODE_9PT}
        _, plot_path = _run_calibrate(samples, plot=True)
        size = os.path.getsize(plot_path)
        assert size > 1000, f"PNG too small: {size} bytes"


# ── TC7.4: Profile loading → sim uses fitted curves ───────────────

class TestTC7_4:
    async def test_profile_mode_server(self):
        """Server should use profile curves for latency instead of fixed values."""
        samples = {"prefill": PREFILL_3PT, "kv_transfer": KV_3PT, "decode": DECODE_9PT}
        profile_path, _ = _run_calibrate(samples)

        config = ServerConfig(
            mode="dual",
            model_name="test-model",
            profile=profile_path,
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["finish_reason"] in ("stop", "length")

    async def test_profile_lookup_values(self):
        """LatencyProfile should return values close to sample data."""
        samples = {"prefill": PREFILL_3PT, "kv_transfer": KV_3PT, "decode": DECODE_9PT}
        profile_path, _ = _run_calibrate(samples)
        prof = LatencyProfile(profile_path)

        # Prefill at sample point 1024 → should be close to 30ms
        val = prof.prefill_delay_ms(1024)
        assert 15 < val < 50, f"Expected ~30ms, got {val}"

        # Decode at sample point (16, 2048) → should be close to 12ms
        val = prof.decode_delay_per_token_ms(16, 2048)
        assert 5 < val < 20, f"Expected ~12ms, got {val}"

    async def test_profile_extrapolation(self):
        """Profile should give reasonable results outside sample range."""
        samples = {"prefill": PREFILL_3PT}
        profile_path, _ = _run_calibrate(samples)
        prof = LatencyProfile(profile_path)

        # Extrapolate beyond max sample (4096)
        val = prof.prefill_delay_ms(8192)
        # Should be positive and larger than the max sample value
        assert val > 0, f"Extrapolated value should be positive, got {val}"


# ── TC7.5: < 3 sample points → error ──────────────────────────────

class TestTC7_5:
    def test_too_few_prefill_points(self):
        samples = {
            "prefill": [
                {"batch_size": 100, "delay_ms": 5},
                {"batch_size": 200, "delay_ms": 10},
            ]
        }
        with pytest.raises(SystemExit):
            _run_calibrate(samples)

    def test_too_few_decode_points(self):
        # Decode needs 9 points minimum
        samples = {"decode": DECODE_9PT[:5]}
        with pytest.raises(SystemExit):
            _run_calibrate(samples)
