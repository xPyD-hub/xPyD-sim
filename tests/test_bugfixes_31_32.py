"""Tests for bugfixes #31 (echo=True) and #32 (deprecated get_event_loop)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def config():
    return ServerConfig(
        mode="dual",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
    )


@pytest.fixture
def app(config):
    return create_app(config)


# --- Issue #31: echo=True on /v1/completions ---


@pytest.mark.anyio
async def test_echo_true_non_streaming(app):
    """echo=True should prepend prompt text to completion output."""
    prompt = "Hello world"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": prompt, "max_tokens": 5, "echo": True},
        )
    assert resp.status_code == 200
    data = resp.json()
    text = data["choices"][0]["text"]
    assert text.startswith(prompt), f"Expected text to start with prompt, got: {text!r}"


@pytest.mark.anyio
async def test_echo_false_non_streaming(app):
    """echo=False (default) should NOT prepend prompt."""
    prompt = "Hello world"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": prompt, "max_tokens": 5, "echo": False},
        )
    assert resp.status_code == 200
    data = resp.json()
    text = data["choices"][0]["text"]
    assert not text.startswith(prompt)


@pytest.mark.anyio
async def test_echo_true_streaming(app):
    """echo=True in streaming mode should emit prompt as first chunk."""
    prompt = "Hello world"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/completions",
            json={
                "model": "dummy",
                "prompt": prompt,
                "max_tokens": 5,
                "stream": True,
                "echo": True,
            },
        )
    assert resp.status_code == 200
    chunks = []
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    # First content chunk should be the prompt
    assert len(chunks) > 0
    assert chunks[0]["choices"][0]["text"] == prompt


# --- Issue #32: asyncio.get_event_loop deprecation ---


def test_scheduler_no_get_event_loop():
    """Scheduler should use get_running_loop() not get_event_loop()."""
    import inspect

    from xpyd_sim import scheduler

    source = inspect.getsource(scheduler)
    assert "get_event_loop()" not in source, "Scheduler still uses deprecated get_event_loop()"
    assert "get_running_loop()" in source
