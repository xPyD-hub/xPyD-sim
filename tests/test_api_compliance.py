"""Tests for API compliance — TC14 through TC17."""

from __future__ import annotations

import base64
import json
import struct

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def config():
    return ServerConfig(
        mode="dual",
        model_name="test-model",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=0.5,
        max_model_len=4096,
    )


@pytest.fixture
def client(config):
    app = create_app(config)
    transport = ASGITransport(app=app)
    return AsyncClient(
        transport=transport, base_url="http://test",
    )


# === TC14: response_format ===


@pytest.mark.anyio
async def test_tc14_1_json_object(client):
    """TC14.1: json_object returns valid JSON."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


@pytest.mark.anyio
async def test_tc14_2_json_schema(client):
    """TC14.2: json_schema returns JSON conforming to schema."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert "name" in parsed
    assert "age" in parsed
    assert isinstance(parsed["age"], int)


@pytest.mark.anyio
async def test_tc14_3_streaming_json(client):
    """TC14.3: streaming response_format assembles to valid JSON."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "stream": True,
        },
    )
    assert resp.status_code == 200
    content_parts = []
    async for line in resp.aiter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {})
            c = delta.get("content")
            if c:
                content_parts.append(c)
    full = "".join(content_parts)
    parsed = json.loads(full)
    assert isinstance(parsed, dict)


# === TC15: Parameter validation ===


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 3.0),
        ("top_p", -0.5),
        ("frequency_penalty", 5.0),
        ("presence_penalty", -3.0),
        ("n", 0),
    ],
)
async def test_tc15_param_validation(client, field, value):
    """TC15.1-15.5: out-of-range params return 400."""
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        field: value,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_tc15_6_best_of_lt_n_completions(client):
    """TC15.6: best_of < n on completions returns 400."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hello",
            "n": 3,
            "best_of": 1,
        },
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_tc15_best_of_lt_n_chat(client):
    """best_of < n on chat completions returns 400."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3,
            "best_of": 1,
        },
    )
    assert resp.status_code == 400


# === TC16: Embedding base64 ===


@pytest.mark.anyio
async def test_tc16_1_float_format(client):
    """TC16.1: encoding_format=float returns list[float]."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "hello",
            "encoding_format": "float",
        },
    )
    assert resp.status_code == 200
    emb = resp.json()["data"][0]["embedding"]
    assert isinstance(emb, list)
    assert all(isinstance(x, float) for x in emb)


@pytest.mark.anyio
async def test_tc16_2_base64_format(config, client):
    """TC16.2: encoding_format=base64 returns decodable floats."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "hello",
            "encoding_format": "base64",
        },
    )
    assert resp.status_code == 200
    emb = resp.json()["data"][0]["embedding"]
    assert isinstance(emb, str)
    raw = base64.b64decode(emb)
    num_floats = len(raw) // 4
    assert num_floats == config.embedding_dim
    values = struct.unpack(f"<{num_floats}f", raw)
    assert len(values) == config.embedding_dim


# === TC17: vLLM compatibility ===


@pytest.mark.anyio
async def test_tc17_1_vllm_sampling_params(client):
    """TC17.1: vLLM sampling params accepted without error."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.2,
        },
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_tc17_2_vllm_extra_params(client):
    """TC17.2: vLLM extra params accepted without error."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": True,
            "priority": 1,
            "request_id": "test-123",
        },
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_tc17_3_stop_reason_field(client):
    """TC17.3: response includes stop_reason field."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert "stop_reason" in choice


@pytest.mark.anyio
async def test_tc17_4_service_tier_field(client):
    """TC17.4: response includes service_tier field."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "service_tier" in data


# --- TC14 with scheduling enabled ---



@pytest_asyncio.fixture
async def scheduled_client():
    config = ServerConfig(
        mode="dual",
        model_name="test-model",
        prefill_delay_ms=1,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=1,
        eos_min_ratio=1.0,
        max_model_len=4096,
        scheduling_enabled=True,
        max_num_batched_tokens=4096,
        max_num_seqs=64,
    )
    app = create_app(config)
    if config._scheduler:
        await config._scheduler.start()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    if config._scheduler:
        await config._scheduler.stop()


@pytest.mark.anyio
async def test_tc14_1_json_object_scheduled(scheduled_client):
    """TC14.1 with scheduling: json_object returns valid JSON."""
    resp = await scheduled_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


@pytest.mark.anyio
async def test_tc14_2_json_schema_scheduled(scheduled_client):
    """TC14.2 with scheduling: json_schema returns conforming JSON."""
    resp = await scheduled_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                    }
                },
            },
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert "name" in parsed
    assert "age" in parsed


@pytest.mark.anyio
async def test_response_format_json_skips_stop_seq(client):
    """Stop sequences must not truncate JSON content."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_object"},
            "stop": ["}"],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    # Content should be valid JSON despite stop=["}"]
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
