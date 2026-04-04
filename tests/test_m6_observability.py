"""Tests for M6: Observability — TC10.1, TC10.2, TC10.3, TC10.4."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture()
def client():
    """Default client with zero delays for fast tests."""
    config = ServerConfig(
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
    )
    app = create_app(config)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture()
def warmup_client():
    """Client with warmup enabled."""
    config = ServerConfig(
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        warmup_requests=3,
        warmup_penalty_ms=100,
    )
    app = create_app(config)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture()
def logging_client(tmp_path: Path):
    """Client with request logging enabled."""
    log_file = tmp_path / "requests.jsonl"
    config = ServerConfig(
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        log_requests=str(log_file),
    )
    app = create_app(config)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    client._log_file = log_file  # type: ignore[attr-defined]
    return client


# TC10.1: /metrics endpoint — valid Prometheus format
class TestMetricsEndpoint:
    async def test_metrics_returns_prometheus_format(self, client: httpx.AsyncClient):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "xpyd_sim_requests_total" in text
        assert "xpyd_sim_tokens_generated_total" in text
        assert "xpyd_sim_active_requests" in text
        assert "xpyd_sim_request_duration_seconds" in text

    async def test_metrics_has_correct_types(self, client: httpx.AsyncClient):
        resp = await client.get("/metrics")
        text = resp.text
        assert "# TYPE xpyd_sim_requests_total counter" in text
        assert "# TYPE xpyd_sim_tokens_generated_total counter" in text
        assert "# TYPE xpyd_sim_active_requests gauge" in text
        assert "# TYPE xpyd_sim_request_duration_seconds histogram" in text

    async def test_metrics_increment_after_request(self, client: httpx.AsyncClient):
        # Make a request
        await client.post(
            "/v1/chat/completions",
            json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = await client.get("/metrics")
        text = resp.text
        # requests_total should be at least 1
        for line in text.splitlines():
            if line.startswith("xpyd_sim_requests_total "):
                count = int(line.split()[-1])
                assert count >= 1

    async def test_metrics_tokens_generated(self, client: httpx.AsyncClient):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        resp = await client.get("/metrics")
        text = resp.text
        for line in text.splitlines():
            if line.startswith("xpyd_sim_tokens_generated_total "):
                count = int(line.split()[-1])
                assert count >= 1

    async def test_metrics_histogram_buckets(self, client: httpx.AsyncClient):
        await client.post(
            "/v1/chat/completions",
            json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = await client.get("/metrics")
        text = resp.text
        assert 'le="0.005"' in text
        assert 'le="+Inf"' in text
        assert "xpyd_sim_request_duration_seconds_sum" in text
        assert "xpyd_sim_request_duration_seconds_count" in text

    async def test_metrics_content_type(self, client: httpx.AsyncClient):
        resp = await client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]


# TC10.2: Request logging — JSONL with correct fields
class TestRequestLogging:
    async def test_logging_creates_jsonl(self, logging_client: httpx.AsyncClient):
        await logging_client.post(
            "/v1/chat/completions",
            json={"model": "dummy", "messages": [{"role": "user", "content": "hello"}]},
        )
        log_file = logging_client._log_file  # type: ignore[attr-defined]
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "timestamp" in record
        assert "prompt_tokens" in record
        assert "output_tokens" in record
        assert "prefill_ms" in record
        assert "kv_transfer_ms" in record
        assert "decode_ms" in record
        assert "total_ms" in record
        assert "mode" in record

    async def test_logging_multiple_requests(self, logging_client: httpx.AsyncClient):
        for _ in range(3):
            await logging_client.post(
                "/v1/chat/completions",
                json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
            )
        log_file = logging_client._log_file  # type: ignore[attr-defined]
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3

    async def test_logging_completions_endpoint(self, logging_client: httpx.AsyncClient):
        await logging_client.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": "hello"},
        )
        log_file = logging_client._log_file  # type: ignore[attr-defined]
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["mode"] == "dual"

    async def test_logging_records_correct_mode(self, tmp_path: Path):
        log_file = tmp_path / "req.jsonl"
        config = ServerConfig(
            mode="prefill",
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
            log_requests=str(log_file),
        )
        app = create_app(config)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            await c.post(
                "/v1/chat/completions",
                json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
            )
        record = json.loads(log_file.read_text().strip().splitlines()[0])
        assert record["mode"] == "prefill"


# TC10.3: /health returns mode and config
class TestHealthEndpoint:
    async def test_health_returns_mode(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "dual"

    async def test_health_returns_model(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        data = resp.json()
        assert "model" in data

    async def test_health_returns_config(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        data = resp.json()
        assert "config" in data
        cfg = data["config"]
        assert "prefill_delay_ms" in cfg
        assert "kv_transfer_delay_ms" in cfg
        assert "decode_delay_per_token_ms" in cfg

    async def test_health_reflects_mode_config(self):
        config = ServerConfig(
            mode="decode",
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/health")
        data = resp.json()
        assert data["mode"] == "decode"


# TC10.4: Warm-up behavior — first N requests have extra latency
class TestWarmup:
    async def test_warmup_penalty_applied(self, warmup_client: httpx.AsyncClient):
        """First 3 requests should be slower due to warmup penalty."""
        import time

        times = []
        for _ in range(5):
            start = time.monotonic()
            await warmup_client.post(
                "/v1/chat/completions",
                json={
                    "model": "dummy",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
            elapsed = time.monotonic() - start
            times.append(elapsed)

        # First 3 should each take >= 100ms (warmup penalty)
        for i in range(3):
            assert times[i] >= 0.08, f"Warmup request {i} too fast: {times[i]:.4f}s"

        # Last 2 should be fast (no penalty, zero delays)
        for i in range(3, 5):
            assert times[i] < 0.08, f"Post-warmup request {i} too slow: {times[i]:.4f}s"

    async def test_warmup_completions_endpoint(self):
        """Warmup also applies to /v1/completions."""
        import time

        config = ServerConfig(
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
            warmup_requests=2,
            warmup_penalty_ms=100,
        )
        app = create_app(config)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            times = []
            for _ in range(4):
                start = time.monotonic()
                await c.post(
                    "/v1/completions",
                    json={"model": "dummy", "prompt": "hi", "max_tokens": 1},
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

            # First 2 should be slow
            for i in range(2):
                assert times[i] >= 0.08
            # Last 2 should be fast
            for i in range(2, 4):
                assert times[i] < 0.08
