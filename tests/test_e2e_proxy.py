"""End-to-end tests: xpyd-sim (prefill + decode) + xpyd-proxy.

Tests the full PD disaggregation flow:
  client → proxy → sim(prefill) → sim(decode) → client

Validates response FORMAT (not content), both endpoints, streaming + non-streaming.

Requires: pip install xpyd-sim[e2e]
Run with: pytest tests/test_e2e_proxy.py -m e2e
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

pytest = __import__("pytest")

# Skip entire module if xpyd (proxy) is not installed
xpyd = pytest.importorskip("xpyd", reason="xpyd-proxy not installed")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy  # noqa: E402

from xpyd_sim.server import ServerConfig, create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use a simple model name (proxy doesn't need a real tokenizer for basic tests)
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tests", "assets", "tokenizer")
_MODEL_NAME = _TOKENIZER_PATH


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(app, port):
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")).run()


def _wait_ready(port, path="/health", timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server on port {port} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures: start prefill sim, decode sim, and proxy once per module
# ---------------------------------------------------------------------------

_PREFILL_PORT = _free_port()
_DECODE_PORT = _free_port()
_PROXY_PORT = _free_port()

# Start sim nodes
_prefill_app = create_app(ServerConfig(
    mode="prefill", model_name=_MODEL_NAME, prefill_delay_ms=5,
    kv_transfer_delay_ms=0, decode_delay_per_token_ms=1,
    eos_min_ratio=1.0, max_model_len=4096,
))
_decode_app = create_app(ServerConfig(
    mode="decode", model_name=_MODEL_NAME, prefill_delay_ms=0,
    kv_transfer_delay_ms=2, decode_delay_per_token_ms=3,
    eos_min_ratio=1.0, max_model_len=4096,
))

threading.Thread(target=_run_server, args=(_prefill_app, _PREFILL_PORT), daemon=True).start()
threading.Thread(target=_run_server, args=(_decode_app, _DECODE_PORT), daemon=True).start()

_wait_ready(_PREFILL_PORT)
_wait_ready(_DECODE_PORT)

# Start proxy
_proxy = Proxy(
    prefill_instances=[f"127.0.0.1:{_PREFILL_PORT}"],
    decode_instances=[f"127.0.0.1:{_DECODE_PORT}"],
    model=_MODEL_NAME,
    scheduling_policy=RoundRobinSchedulingPolicy(),
    generator_on_p_node=False,
)
_proxy_app = FastAPI()
_proxy_app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)
_proxy_app.include_router(_proxy.router)

threading.Thread(target=_run_server, args=(_proxy_app, _PROXY_PORT), daemon=True).start()
_wait_ready(_PROXY_PORT, path="/status")

_BASE = f"http://127.0.0.1:{_PROXY_PORT}"


# ---------------------------------------------------------------------------
# /health and /status
# ---------------------------------------------------------------------------


def test_proxy_health():
    r = httpx.get(f"{_BASE}/status")
    assert r.status_code == 200
    data = r.json()
    assert "prefill_node_count" in data or "status" in data


# ---------------------------------------------------------------------------
# /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------


def test_chat_completions_non_streaming():
    r = httpx.post(f"{_BASE}/v1/chat/completions", json={
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    })
    assert r.status_code == 200
    data = r.json()

    # Format checks
    assert data["object"] == "chat.completion"
    assert "id" in data
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) >= 1

    choice = data["choices"][0]
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert "finish_reason" in choice
    assert choice["finish_reason"] in ("stop", "length")

    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


# ---------------------------------------------------------------------------
# /v1/chat/completions — streaming
# ---------------------------------------------------------------------------


def test_chat_completions_streaming():
    r = httpx.post(f"{_BASE}/v1/chat/completions", json={
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")

    text = r.text
    lines = text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    # Must have at least: some content + [DONE]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"

    # First non-DONE chunk format
    first = json.loads(data_lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    assert "id" in first
    assert "choices" in first
    assert len(first["choices"]) >= 1
    assert "delta" in first["choices"][0]

    # Check role in first delta
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # Last chunk before [DONE] should have finish_reason
    last_data = json.loads(data_lines[-2][6:])
    if last_data["choices"]:
        fr = last_data["choices"][0].get("finish_reason")
        # finish_reason may be in this chunk or earlier
        assert fr is None or fr in ("stop", "length")


# ---------------------------------------------------------------------------
# /v1/completions — non-streaming
# ---------------------------------------------------------------------------


def test_completions_non_streaming():
    r = httpx.post(f"{_BASE}/v1/completions", json={
        "model": _MODEL_NAME,
        "prompt": "Once upon a time",
        "max_tokens": 10,
    })
    assert r.status_code == 200
    data = r.json()

    assert data["object"] == "text_completion"
    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) >= 1

    choice = data["choices"][0]
    assert "text" in choice
    assert isinstance(choice["text"], str)
    assert "finish_reason" in choice
    assert choice["finish_reason"] in ("stop", "length")

    assert "usage" in data
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


# ---------------------------------------------------------------------------
# /v1/completions — streaming
# ---------------------------------------------------------------------------


def test_completions_streaming():
    r = httpx.post(f"{_BASE}/v1/completions", json={
        "model": _MODEL_NAME,
        "prompt": "Once upon a time",
        "max_tokens": 10,
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")

    text = r.text
    lines = text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"

    first = json.loads(data_lines[0][6:])
    assert first["object"] == "text_completion"
    assert "choices" in first
    assert "text" in first["choices"][0]


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_models_endpoint():
    r = httpx.get(f"{_BASE}/v1/models")
    assert r.status_code == 200
    data = r.json()
    # Proxy may return different format than OpenAI spec
    # Just verify it returns valid JSON with model info
    assert isinstance(data, (dict, list))
    if isinstance(data, dict):
        if "data" in data:
            assert len(data["data"]) >= 1


# ---------------------------------------------------------------------------
# /ping
# ---------------------------------------------------------------------------


def test_ping():
    # Proxy may or may not have /ping — just check it doesn't 500
    r = httpx.get(f"{_BASE}/ping", timeout=5)
    # Accept 200 or 404 (proxy might not expose /ping)
    assert r.status_code in (200, 404, 405)
