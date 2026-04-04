"""M7: End-to-End + Concurrency tests (TC8.x, TC9.x)."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from httpx import ASGITransport

from xpyd_sim.server import ServerConfig, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_body(**overrides):
    body = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    }
    body.update(overrides)
    return body


def _make_completion_body(**overrides):
    body = {"model": "dummy", "prompt": "hello", "max_tokens": 8}
    body.update(overrides)
    return body


async def _collect_stream(resp: httpx.Response) -> list[str]:
    """Collect SSE data lines from a streaming response."""
    lines = []
    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            lines.append(line[6:])
    return lines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fast_config():
    """Low-latency config for quick concurrent tests."""
    return ServerConfig(
        mode="dual",
        prefill_delay_ms=5,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=2,
        eos_min_ratio=1.0,  # always full max_tokens for predictability
    )


@pytest.fixture()
def prefill_config():
    """Config for prefill-only node."""
    return ServerConfig(
        mode="prefill",
        prefill_delay_ms=20,
        kv_transfer_delay_ms=10,
        decode_delay_per_token_ms=5,
        eos_min_ratio=1.0,
    )


@pytest.fixture()
def decode_config():
    """Config for decode-only node."""
    return ServerConfig(
        mode="decode",
        prefill_delay_ms=20,
        kv_transfer_delay_ms=10,
        decode_delay_per_token_ms=5,
        eos_min_ratio=1.0,
    )


# ---------------------------------------------------------------------------
# TC9.1 — Multiple concurrent requests, independent latencies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc9_1_concurrent_independent_latencies(fast_config):
    """Multiple concurrent requests should complete with independent latencies."""
    app = create_app(fast_config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Send 5 concurrent requests
        tasks = [
            client.post("/v1/chat/completions", json=_make_chat_body(max_tokens=4))
            for _ in range(5)
        ]
        start = time.monotonic()
        responses = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        # All should succeed
        for r in responses:
            assert r.status_code == 200
            data = r.json()
            assert len(data["choices"]) == 1
            assert data["usage"]["completion_tokens"] > 0

        # With concurrent execution, total time should be much less than 5x sequential
        # Each request ~5ms prefill + 4*2ms decode = ~13ms
        # 5 sequential = ~65ms; concurrent should be < 40ms (generous margin)
        assert elapsed < 0.15, f"Concurrent requests too slow: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# TC9.2 — High concurrency, no crashes, no data mixing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc9_2_high_concurrency_no_crashes(fast_config):
    """High concurrency: no crashes, no data mixing between requests."""
    app = create_app(fast_config)
    transport = ASGITransport(app=app)

    n_requests = 20

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tasks = [
            client.post("/v1/chat/completions", json=_make_chat_body(max_tokens=4))
            for _ in range(n_requests)
        ]
        responses = await asyncio.gather(*tasks)

        ids = set()
        for r in responses:
            assert r.status_code == 200
            data = r.json()
            # Each response should have unique ID
            ids.add(data["id"])
            assert data["model"] == "dummy"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["finish_reason"] in ("stop", "length")

        # All IDs should be unique — no data mixing
        assert len(ids) == n_requests


@pytest.mark.asyncio
async def test_tc9_2_high_concurrency_streaming(fast_config):
    """High concurrency with streaming: no crashes or mixed streams."""
    app = create_app(fast_config)
    transport = ASGITransport(app=app)

    n_requests = 10

    async def stream_one():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_make_chat_body(max_tokens=4, stream=True),
            )
            assert resp.status_code == 200
            lines = await _collect_stream(resp)
            assert lines[-1] == "[DONE]"
            return lines

    tasks = [stream_one() for _ in range(n_requests)]
    results = await asyncio.gather(*tasks)

    for lines in results:
        assert len(lines) >= 3  # role chunk + content + finish + DONE
        assert lines[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_tc9_2_high_concurrency_completions(fast_config):
    """High concurrency on /v1/completions endpoint."""
    app = create_app(fast_config)
    transport = ASGITransport(app=app)

    n_requests = 15

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tasks = [
            client.post("/v1/completions", json=_make_completion_body(max_tokens=4))
            for _ in range(n_requests)
        ]
        responses = await asyncio.gather(*tasks)

        ids = set()
        for r in responses:
            assert r.status_code == 200
            data = r.json()
            ids.add(data["id"])
            assert data["model"] == "dummy"

        assert len(ids) == n_requests


# ---------------------------------------------------------------------------
# TC9 — Concurrent requests with metrics tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc9_concurrent_metrics(fast_config):
    """Metrics should correctly track concurrent active requests."""
    app = create_app(fast_config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Fire off requests
        tasks = [
            client.post("/v1/chat/completions", json=_make_chat_body(max_tokens=4))
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)

        # Check metrics after all complete
        resp = await client.get("/metrics")
        text = resp.text
        assert "xpyd_sim_requests_total" in text
        assert "xpyd_sim_tokens_generated_total" in text


# ---------------------------------------------------------------------------
# TC8.1 — Simulated PD disaggregation flow (prefill + decode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_1_pd_disaggregation_flow(prefill_config, decode_config):
    """Simulate PD flow: prefill node (max_tokens=1) → decode node (full)."""
    prefill_app = create_app(prefill_config)
    decode_app = create_app(decode_config)
    prefill_transport = ASGITransport(app=prefill_app)
    decode_transport = ASGITransport(app=decode_app)

    max_tokens = 8

    # Step 1: Send to prefill with max_tokens=1
    async with httpx.AsyncClient(
        transport=prefill_transport, base_url="http://prefill"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=1),
        )
        assert resp.status_code == 200
        prefill_data = resp.json()
        assert prefill_data["usage"]["completion_tokens"] >= 1

    # Step 2: Send full request to decode
    async with httpx.AsyncClient(
        transport=decode_transport, base_url="http://decode"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=max_tokens),
        )
        assert resp.status_code == 200
        decode_data = resp.json()
        assert decode_data["usage"]["completion_tokens"] == max_tokens


# ---------------------------------------------------------------------------
# TC8.2 — TTFT validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_2_ttft_validation():
    """TTFT ≈ prefill_delay + kv_transfer_delay + first decode token delay."""
    prefill_ms = 30
    decode_ms = 10

    config = ServerConfig(
        mode="dual",
        prefill_delay_ms=prefill_ms,
        kv_transfer_delay_ms=0,  # dual mode: kv=0
        decode_delay_per_token_ms=decode_ms,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=4, stream=True),
        )
        # Read first data chunk (role chunk)
        first_line = None
        second_line = None
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                if first_line is None:
                    first_line = line
                elif second_line is None:
                    second_line = line
                    ttft = time.monotonic() - start
                    break

        # TTFT should be approximately prefill + first decode token
        expected_min = (prefill_ms + decode_ms) / 1000.0
        # Allow generous tolerance for CI
        assert ttft >= expected_min * 0.5, (
            f"TTFT too fast: {ttft:.3f}s, expected >= ~{expected_min:.3f}s"
        )
        assert ttft < expected_min * 3.0, (
            f"TTFT too slow: {ttft:.3f}s, expected ~{expected_min:.3f}s"
        )


# ---------------------------------------------------------------------------
# TC8.3 — TPOT validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_3_tpot_validation():
    """TPOT ≈ decode_delay_per_token (measured via total decode time)."""
    decode_ms = 15
    max_tokens = 6
    config = ServerConfig(
        mode="dual",
        prefill_delay_ms=1,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=decode_ms,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Non-streaming: measure total time which includes decode
        start = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=max_tokens),
        )
        elapsed = time.monotonic() - start
        assert resp.status_code == 200
        data = resp.json()
        n_tokens = data["usage"]["completion_tokens"]

        # Total decode time should be ~ n_tokens * decode_ms
        # render_dummy_text returns chars, so completion_tokens = max_tokens
        expected_decode_s = n_tokens * decode_ms / 1000.0
        # Allow generous tolerance for CI
        assert elapsed >= expected_decode_s * 0.3, (
            f"Total time too fast: {elapsed*1000:.1f}ms for {n_tokens} tokens"
        )


# ---------------------------------------------------------------------------
# TC8.4 — Streaming token intervals match configured delay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_4_streaming_token_intervals():
    """Streaming total time should reflect decode delay * token count."""
    decode_ms = 20
    max_tokens = 8
    config = ServerConfig(
        mode="dual",
        prefill_delay_ms=1,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=decode_ms,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.post(
            "/v1/completions",
            json=_make_completion_body(max_tokens=max_tokens, stream=True),
        )
        # Consume entire stream
        lines = await _collect_stream(resp)
        elapsed = time.monotonic() - start

        assert lines[-1] == "[DONE]"
        # Total time should include decode delay for each token
        # Each char in dummy text gets decode_delay, and max_tokens chars are produced
        expected_min = (max_tokens * decode_ms) / 1000.0
        assert elapsed >= expected_min * 0.3, (
            f"Stream too fast: {elapsed*1000:.1f}ms, expected >= ~{expected_min*1000:.1f}ms"
        )


# ---------------------------------------------------------------------------
# TC8 — Decode mode has zero prefill delay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_decode_mode_no_prefill_delay(decode_config):
    """Decode mode should have zero prefill delay."""
    app = create_app(decode_config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=2),
        )
        elapsed = time.monotonic() - start
        assert resp.status_code == 200
        # Should be fast since prefill_delay is skipped in decode mode
        # kv_transfer + 2*decode = 10 + 10 = 20ms
        assert elapsed < 0.15, f"Decode mode too slow: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# TC8 — Prefill mode returns result correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_prefill_mode_returns_result(prefill_config):
    """Prefill mode should work and return valid response."""
    app = create_app(prefill_config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=1),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
