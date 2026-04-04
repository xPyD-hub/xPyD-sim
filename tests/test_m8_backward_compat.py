"""M8: Backward Compatibility Tests (TC12.1, TC12.2).

Verify xPyD-sim can replace:
  - xPyD-proxy's dummy_nodes (prefill + decode)
  - xPyD-bench's dummy server

These tests check the API contracts both projects depend on.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app

# ---------------------------------------------------------------------------
# Fixtures — mimic proxy dummy_nodes and bench dummy server configs
# ---------------------------------------------------------------------------


@pytest.fixture
def prefill_client():
    """xPyD-sim as a prefill node replacement for proxy dummy_nodes."""
    config = ServerConfig(
        mode="prefill",
        model_name="dummy",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,  # deterministic: always emit max_tokens
        max_model_len=131072,
    )
    return TestClient(create_app(config))


@pytest.fixture
def decode_client():
    """xPyD-sim as a decode node replacement for proxy dummy_nodes."""
    config = ServerConfig(
        mode="decode",
        model_name="dummy",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
        max_model_len=131072,
    )
    return TestClient(create_app(config))


@pytest.fixture
def dual_client():
    """xPyD-sim as a full dual-mode replacement for bench dummy server."""
    config = ServerConfig(
        mode="dual",
        model_name="dummy-model",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
        max_model_len=131072,
    )
    return TestClient(create_app(config))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of JSON chunks (excluding [DONE])."""
    chunks = []
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                continue
            chunks.append(json.loads(payload))
    return chunks


# ===================================================================
# TC12.1 — Replace proxy dummy_nodes with xPyD-sim
# ===================================================================


class TestProxyPrefillCompat:
    """Verify xPyD-sim (prefill mode) satisfies proxy dummy prefill node contract."""

    def test_health(self, prefill_client):
        resp = prefill_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ping_get(self, prefill_client):
        resp = prefill_client.get("/ping")
        assert resp.status_code == 200
        assert resp.text == "pong"

    def test_ping_post(self, prefill_client):
        resp = prefill_client.post("/ping")
        assert resp.status_code == 200
        assert resp.text == "pong"

    def test_models(self, prefill_client):
        resp = prefill_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"

    def test_chat_non_streaming(self, prefill_client):
        resp = prefill_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0

    def test_chat_streaming(self, prefill_client):
        resp = prefill_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        assert len(chunks) >= 2  # at least role + content + finish

        # First chunk has role
        first_delta = chunks[0]["choices"][0]["delta"]
        assert first_delta.get("role") == "assistant"

        # Last content chunk has finish_reason
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] in ("stop", "length")

        # All chunks have required fields
        for c in chunks:
            assert c["object"] == "chat.completion.chunk"
            assert "model" in c
            assert "id" in c

    def test_completions_non_streaming(self, prefill_client):
        resp = prefill_client.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": "Hello world", "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert data["choices"][0]["finish_reason"] in ("stop", "length")

    def test_completions_streaming(self, prefill_client):
        resp = prefill_client.post(
            "/v1/completions",
            json={
                "model": "dummy",
                "prompt": "Hello world",
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        assert len(chunks) >= 1
        # Last chunk has finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


class TestProxyDecodeCompat:
    """Verify xPyD-sim (decode mode) satisfies proxy dummy decode node contract."""

    def test_health(self, decode_client):
        resp = decode_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ping(self, decode_client):
        assert decode_client.get("/ping").text == "pong"
        assert decode_client.post("/ping").text == "pong"

    def test_models(self, decode_client):
        resp = decode_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    def test_chat_non_streaming(self, decode_client):
        resp = decode_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["completion_tokens"] == 5

    def test_chat_streaming_role_in_first_chunk(self, decode_client):
        resp = decode_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        chunks = _parse_sse(resp.text)
        first_delta = chunks[0]["choices"][0]["delta"]
        assert first_delta["role"] == "assistant"

    def test_chat_streaming_done_sentinel(self, decode_client):
        resp = decode_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        assert "data: [DONE]" in resp.text

    def test_completions_non_streaming(self, decode_client):
        resp = decode_client.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": "Hello", "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["usage"]["completion_tokens"] == 5

    def test_completions_streaming(self, decode_client):
        resp = decode_client.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": "Hello", "max_tokens": 3, "stream": True},
        )
        chunks = _parse_sse(resp.text)
        assert len(chunks) >= 1
        assert "data: [DONE]" in resp.text

    def test_max_model_len_in_models(self, decode_client):
        """Proxy depends on max_model_len from /v1/models."""
        resp = decode_client.get("/v1/models")
        model = resp.json()["data"][0]
        assert "max_model_len" in model
        assert model["max_model_len"] == 131072


class TestProxyE2ECompat:
    """End-to-end proxy contract: both prefill and decode nodes together."""

    def test_prefill_max_tokens_1(self, prefill_client):
        """Proxy sends max_tokens=1 to prefill node as a signal."""
        resp = prefill_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["completion_tokens"] >= 1

    def test_decode_full_generation(self, decode_client):
        """Proxy sends full max_tokens to decode node."""
        resp = decode_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "stream": True,
            },
        )
        chunks = _parse_sse(resp.text)
        # Count content chunks (excluding role-only and finish-only)
        content_text = ""
        for c in chunks:
            delta = c["choices"][0]["delta"]
            if delta.get("content"):
                content_text += delta["content"]
        assert len(content_text) > 0

    def test_max_completion_tokens_alias(self, prefill_client):
        """Proxy may use max_completion_tokens instead of max_tokens."""
        resp = prefill_client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["completion_tokens"] == 5


# ===================================================================
# TC12.2 — Replace bench dummy server with xPyD-sim
# ===================================================================


class TestBenchCompat:
    """Verify xPyD-sim (dual mode) satisfies bench dummy server contract."""

    def test_health(self, dual_client):
        resp = dual_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_models(self, dual_client):
        resp = dual_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model = data["data"][0]
        assert model["object"] == "model"
        assert "id" in model

    def test_completions_non_streaming(self, dual_client):
        resp = dual_client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "dummy-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] == 5

    def test_completions_streaming(self, dual_client):
        resp = dual_client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 3, "stream": True},
        )
        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        assert len(chunks) >= 1
        # Last chunk has finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
        # All chunks have object field
        for c in chunks:
            assert c["object"] == "text_completion"

    def test_chat_non_streaming(self, dual_client):
        resp = dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "dummy-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] != ""
        assert data["usage"]["completion_tokens"] == 5

    def test_chat_streaming(self, dual_client):
        resp = dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)

        # First chunk has role
        first_delta = chunks[0]["choices"][0]["delta"]
        assert first_delta.get("role") == "assistant"

        # Collect content
        content = ""
        for c in chunks:
            delta = c["choices"][0]["delta"]
            if delta.get("content"):
                content += delta["content"]
        assert len(content) > 0

        # [DONE] sentinel present
        assert "data: [DONE]" in resp.text

    def test_chat_streaming_role_chunk(self, dual_client):
        """Bench expects first streaming chunk to have role: assistant."""
        resp = dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        chunks = _parse_sse(resp.text)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    def test_invalid_json_returns_400(self, dual_client):
        """Bench dummy server returns 400 on invalid JSON."""
        resp = dual_client.post(
            "/v1/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_n_multiple_choices(self, dual_client):
        """Bench tests n>1 returns multiple choices."""
        resp = dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "n": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 2
        assert data["choices"][0]["index"] == 0
        assert data["choices"][1]["index"] == 1

    def test_all_sampling_params_accepted(self, dual_client):
        """Bench dummy server accepts all OpenAI params without error."""
        resp = dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
                "seed": 42,
                "user": "test-user",
            },
        )
        assert resp.status_code == 200

    def test_completions_prompt_formats(self, dual_client):
        """Bench expects all 4 prompt formats to work."""
        # String prompt
        resp = dual_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3},
        )
        assert resp.status_code == 200

        # Array of strings
        resp = dual_client.post(
            "/v1/completions",
            json={"prompt": ["Hello", "World"], "max_tokens": 3},
        )
        assert resp.status_code == 200

        # Array of token IDs
        resp = dual_client.post(
            "/v1/completions",
            json={"prompt": [1, 2, 3], "max_tokens": 3},
        )
        assert resp.status_code == 200

    def test_stream_options_include_usage(self, dual_client):
        """Bench tests stream_options.include_usage."""
        resp = dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        chunks = _parse_sse(resp.text)
        # Last chunk before [DONE] should have usage
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert len(usage_chunks) >= 1
        usage = usage_chunks[-1]["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_ignore_eos(self, dual_client):
        """Bench tests ignore_eos produces exactly max_tokens."""
        resp = dual_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "max_tokens": 10,
                "ignore_eos": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["completion_tokens"] == 10
        assert data["choices"][0]["finish_reason"] == "length"

    def test_stop_sequence(self, dual_client):
        """Bench tests stop sequence truncation."""
        resp = dual_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "max_tokens": 100,
                "stop": ["fox"],
                "ignore_eos": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "fox" not in data["choices"][0]["text"]
