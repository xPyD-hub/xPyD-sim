"""Tests for the prefill node simulator."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.prefill.app import create_prefill_app

app = create_prefill_app(model_name="test-prefill", delay_per_token=0, delay_fixed=0)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-prefill"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(client):
    payload = {
        "model": "test-prefill",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["system_fingerprint"] is not None
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_chat_completions_streaming(client):
    payload = {
        "model": "test-prefill",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    chunks = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    # First chunk should have role
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[0]["system_fingerprint"] is not None

    # Last non-empty-choices chunk should have finish_reason
    finish_chunks = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    assert len(finish_chunks) >= 1
    assert finish_chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_completions_stream_with_usage(client):
    payload = {
        "model": "test-prefill",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 3,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    chunks = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    # Last chunk should have usage
    last = chunks[-1]
    assert last["usage"] is not None
    assert last["usage"]["prompt_tokens"] > 0
    assert last["usage"]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_completions_non_streaming(client):
    payload = {
        "model": "test-prefill",
        "prompt": "Hello world",
        "max_tokens": 8,
    }
    resp = await client.post("/v1/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert data["system_fingerprint"] is not None
    assert len(data["choices"]) == 1
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["logprobs"] is None
    assert data["usage"]["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_completions_streaming(client):
    payload = {
        "model": "test-prefill",
        "prompt": "Hello",
        "max_tokens": 4,
        "stream": True,
    }
    resp = await client.post("/v1/completions", json=payload)
    assert resp.status_code == 200

    chunks = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) >= 2
    # Last chunk with choices should have finish_reason
    finish_chunks = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    assert len(finish_chunks) >= 1


@pytest.mark.asyncio
async def test_chat_completions_n_greater_than_1(client):
    payload = {
        "model": "test-prefill",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "n": 3,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["choices"]) == 3
    assert data["usage"]["completion_tokens"] == 15


@pytest.mark.asyncio
async def test_all_openai_params_accepted(client):
    payload = {
        "model": "test-prefill",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
        "user": "test-user",
        "logprobs": True,
        "top_logprobs": 5,
        "response_format": {"type": "text"},
        "ignore_eos": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
