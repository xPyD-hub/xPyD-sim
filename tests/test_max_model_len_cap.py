"""Regression tests for max_model_len output token cap.

Ensures input_tokens + max_tokens never exceeds max_model_len,
matching vLLM behavior.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app


@pytest_asyncio.fixture
async def client_small():
    """Server with small max_model_len=100 for easy testing."""
    config = ServerConfig(
        prefill_delay_ms=0, decode_delay_per_token_ms=0,
        max_model_len=100, eos_min_ratio=1.0,
    )
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_output_capped_by_max_model_len(client_small):
    """When input + max_tokens > max_model_len, output is capped."""
    # "Hello world" ≈ 3 prompt tokens, max_model_len=100
    # Request 200 tokens — should be capped to ~97
    resp = await client_small.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 200,
        "ignore_eos": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    prompt = data["usage"]["prompt_tokens"]
    completion = data["usage"]["completion_tokens"]
    total = prompt + completion
    assert total <= 100, f"Total {total} exceeds max_model_len 100"


@pytest.mark.asyncio
async def test_output_capped_completions_endpoint(client_small):
    """Same check for /v1/completions."""
    resp = await client_small.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 200,
        "ignore_eos": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    prompt = data["usage"]["prompt_tokens"]
    completion = data["usage"]["completion_tokens"]
    assert prompt + completion <= 100


@pytest.mark.asyncio
async def test_small_max_tokens_not_affected(client_small):
    """When max_tokens fits within max_model_len, no cap applied."""
    resp = await client_small.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "ignore_eos": True,
    })
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_input_exceeds_max_model_len_rejected(client_small):
    """Input exceeding max_model_len still returns 400."""
    resp = await client_small.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "word " * 500}],
        "max_tokens": 5,
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_exactly_at_limit():
    """When input fills max_model_len exactly, output should be 1 token minimum."""
    config = ServerConfig(
        prefill_delay_ms=0, decode_delay_per_token_ms=0,
        max_model_len=50, eos_min_ratio=1.0,
    )
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # Short prompt that fits in max_model_len=50 but leaves little room
        resp = await c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi there"}],
            "max_tokens": 100,
            "ignore_eos": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["completion_tokens"] >= 1
        assert data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"] <= 50


@pytest.mark.asyncio
async def test_default_max_model_len_is_128k():
    """Default max_model_len should be 131072 (128K)."""
    config = ServerConfig()
    assert config.max_model_len == 131072
