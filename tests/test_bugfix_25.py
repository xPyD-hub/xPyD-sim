"""Tests for fix: streaming stop sequence should not leak partial stop chars.

Verifies that streaming output with stop sequences is identical to
non-streaming output — no partial stop sequence characters are emitted.

Closes #25, #34
"""

from __future__ import annotations

import json

import pytest

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture()
def app():
    cfg = ServerConfig(
        mode="dual",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
    )
    return create_app(config=cfg)


@pytest.fixture()
def client(app):
    from starlette.testclient import TestClient

    return TestClient(app)


def _collect_stream_content(response) -> tuple[str, str | None]:
    """Collect streamed content and finish_reason from SSE response."""
    content = ""
    finish_reason = None
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        # Chat format
        delta = choice.get("delta", {})
        if delta and delta.get("content") is not None:
            content += delta["content"]
        # Completion format
        text = choice.get("text")
        if text is not None:
            content += text
        fr = choice.get("finish_reason")
        if fr is not None:
            finish_reason = fr
    return content, finish_reason


class TestStreamingStopSequenceChat:
    """Chat endpoint: streaming stop should match non-streaming."""

    def test_stop_single_word(self, client):
        """Stop on a single word token — no partial leakage."""
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 20,
            "stop": ["fox"],
        }
        # Non-streaming
        resp_ns = client.post("/v1/chat/completions", json=body)
        ns_text = resp_ns.json()["choices"][0]["message"]["content"]

        # Streaming
        body["stream"] = True
        resp_s = client.post("/v1/chat/completions", json=body)
        s_text, s_fr = _collect_stream_content(resp_s)

        assert s_text == ns_text
        assert "fox" not in s_text
        assert s_fr == "stop"

    def test_stop_cross_token_boundary(self, client):
        """Stop sequence that spans token boundary (e.g. 'wn f' in 'brown fox')."""
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 20,
            "stop": ["wn f"],
        }
        resp_ns = client.post("/v1/chat/completions", json=body)
        ns_text = resp_ns.json()["choices"][0]["message"]["content"]

        body["stream"] = True
        resp_s = client.post("/v1/chat/completions", json=body)
        s_text, _ = _collect_stream_content(resp_s)

        assert s_text == ns_text
        assert "wn f" not in s_text

    def test_no_stop_full_output(self, client):
        """Without stop sequences, streaming and non-streaming match."""
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
        resp_ns = client.post("/v1/chat/completions", json=body)
        ns_text = resp_ns.json()["choices"][0]["message"]["content"]

        body["stream"] = True
        resp_s = client.post("/v1/chat/completions", json=body)
        s_text, _ = _collect_stream_content(resp_s)

        assert s_text == ns_text


class TestStreamingStopSequenceCompletion:
    """Completion endpoint: streaming stop should match non-streaming."""

    def test_stop_single_word(self, client):
        body = {
            "prompt": "hello",
            "max_tokens": 20,
            "stop": ["fox"],
        }
        resp_ns = client.post("/v1/completions", json=body)
        ns_text = resp_ns.json()["choices"][0]["text"]

        body["stream"] = True
        resp_s = client.post("/v1/completions", json=body)
        s_text, s_fr = _collect_stream_content(resp_s)

        assert s_text == ns_text
        assert "fox" not in s_text
        assert s_fr == "stop"

    def test_stop_cross_token_boundary(self, client):
        body = {
            "prompt": "hello",
            "max_tokens": 20,
            "stop": ["wn f"],
        }
        resp_ns = client.post("/v1/completions", json=body)
        ns_text = resp_ns.json()["choices"][0]["text"]

        body["stream"] = True
        resp_s = client.post("/v1/completions", json=body)
        s_text, _ = _collect_stream_content(resp_s)

        assert s_text == ns_text
        assert "wn f" not in s_text
