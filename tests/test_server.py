"""Tests for M2: Unified server — TC1.x through TC6.x and TC5.x (EOS)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def dual_config():
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
def prefill_config():
    return ServerConfig(
        mode="prefill",
        model_name="test-prefill",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
    )


@pytest.fixture
def decode_config():
    return ServerConfig(
        mode="decode",
        model_name="test-decode",
        prefill_delay_ms=50,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
    )


@pytest.fixture
def dual_client(dual_config):
    app = create_app(dual_config)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def prefill_client(prefill_config):
    app = create_app(prefill_config)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def decode_client(decode_config):
    app = create_app(decode_config)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# === TC1: Mode Switching ===


class TestModeSwitching:
    """TC1.1-TC1.3: Mode switching."""

    async def test_tc1_1_dual_mode(self, dual_client):
        """TC1.1: Start dual mode — both prefill + decode have latency."""
        r = await dual_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "dual"
        assert data["status"] == "ok"

    async def test_tc1_2_prefill_mode(self, prefill_client):
        """TC1.2: Start prefill mode."""
        r = await prefill_client.get("/health")
        assert r.status_code == 200
        assert r.json()["mode"] == "prefill"

    async def test_tc1_3_decode_mode(self, decode_client):
        """TC1.3: Start decode mode — prefill delay = 0."""
        r = await decode_client.get("/health")
        assert r.status_code == 200
        assert r.json()["mode"] == "decode"


# === TC2: Prefill Delay ===


class TestPrefillDelay:
    """TC2.1-TC2.2: Fixed prefill delay."""

    async def test_tc2_1_short_prompt(self):
        """TC2.1: Fixed delay, short prompt — low latency."""
        config = ServerConfig(
            prefill_delay_ms=10,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            )
            assert r.status_code == 200

    async def test_tc2_2_decode_mode_no_prefill_delay(self):
        """TC2.2: Decode mode has zero prefill delay."""
        config = ServerConfig(
            mode="decode",
            prefill_delay_ms=500,  # high, but decode mode should skip it
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            import time

            t0 = time.monotonic()
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            )
            elapsed = time.monotonic() - t0
            assert r.status_code == 200
            # Should be fast since decode mode skips prefill
            assert elapsed < 0.3


# === TC3: KV Transfer Delay ===


class TestKVTransferDelay:
    """TC3.1-TC3.3: KV transfer delay."""

    async def test_tc3_1_dual_no_kv_delay(self):
        """TC3.1: Dual mode — KV delay = 0 (local)."""
        config = ServerConfig(
            mode="dual",
            prefill_delay_ms=0,
            kv_transfer_delay_ms=100,  # set high, but dual should skip it
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            import time

            t0 = time.monotonic()
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            )
            elapsed = time.monotonic() - t0
            assert r.status_code == 200
            assert elapsed < 0.05  # dual has 0 KV delay

    async def test_tc3_2_prefill_no_kv_delay(self):
        """TC3.2: Prefill mode — no KV wait."""
        config = ServerConfig(
            mode="prefill",
            prefill_delay_ms=0,
            kv_transfer_delay_ms=100,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            import time

            t0 = time.monotonic()
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            )
            elapsed = time.monotonic() - t0
            assert r.status_code == 200
            assert elapsed < 0.05

    async def test_tc3_3_kv_delay_zero(self):
        """TC3.3: KV delay = 0 (async mode) — no additional latency."""
        config = ServerConfig(
            mode="decode",
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            )
            assert r.status_code == 200


# === TC4: Decode Delay ===


class TestDecodeDelay:
    """TC4.1-TC4.2: Decode delay."""

    async def test_tc4_1_stable_token_interval(self):
        """TC4.1: Fixed delay — stable token interval."""
        config = ServerConfig(
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "ignore_eos": True,
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["usage"]["completion_tokens"] == 5


# === TC5: EOS Simulation ===


class TestEOSSimulation:
    """TC5.1-TC5.6: EOS simulation."""

    async def test_tc5_1_default_eos(self, dual_client):
        """TC5.1: Default config — output length in [max_tokens*0.5, max_tokens]."""
        lengths = []
        for _ in range(20):
            r = await dual_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 100,
                },
            )
            data = r.json()
            lengths.append(data["usage"]["completion_tokens"])
        # Should have some variation
        assert min(lengths) >= 50  # eos_min_ratio=0.5
        assert max(lengths) <= 100

    async def test_tc5_2_eos_triggered(self, dual_client):
        """TC5.2: EOS triggered — finish_reason = 'stop'."""
        # With eos_min_ratio=0.5 and max_tokens=100, many will be "stop"
        found_stop = False
        for _ in range(20):
            r = await dual_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 100,
                },
            )
            data = r.json()
            if data["choices"][0]["finish_reason"] == "stop":
                found_stop = True
                break
        assert found_stop

    async def test_tc5_3_max_tokens_reached(self, dual_client):
        """TC5.3: Max tokens reached — finish_reason = 'length' with ignore_eos."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
                "ignore_eos": True,
            },
        )
        data = r.json()
        assert data["choices"][0]["finish_reason"] == "length"

    async def test_tc5_4_ignore_eos(self, dual_client):
        """TC5.4: ignore_eos=true — always output max_tokens."""
        for _ in range(5):
            r = await dual_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 50,
                    "ignore_eos": True,
                },
            )
            data = r.json()
            assert data["usage"]["completion_tokens"] == 50
            assert data["choices"][0]["finish_reason"] == "length"

    async def test_tc5_5_custom_eos_min_ratio(self):
        """TC5.5: Custom eos_min_ratio — respects configured ratio."""
        config = ServerConfig(
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
            eos_min_ratio=0.9,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            lengths = []
            for _ in range(20):
                r = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 100,
                    },
                )
                data = r.json()
                lengths.append(data["usage"]["completion_tokens"])
            assert min(lengths) >= 90  # eos_min_ratio=0.9

    async def test_tc5_6_streaming_eos(self, dual_client):
        """TC5.6: Streaming + EOS — correct streaming behavior."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 20,
                "stream": True,
            },
        )
        assert r.status_code == 200
        text = r.text
        lines = [line for line in text.strip().split("\n") if line.startswith("data: ")]
        # Last data line should be [DONE]
        assert lines[-1] == "data: [DONE]"
        # Second to last should have finish_reason
        last_chunk = json.loads(lines[-2][6:])
        assert last_chunk["choices"][0]["finish_reason"] in ("stop", "length")


# === TC6: OpenAI API Compliance ===


class TestOpenAICompliance:
    """TC6.1-TC6.13: OpenAI API compliance."""

    async def test_tc6_1_completions_non_streaming(self, dual_client):
        """TC6.1: /v1/completions non-streaming — correct response format."""
        r = await dual_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 5, "ignore_eos": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "text_completion"
        assert "id" in data
        assert "choices" in data
        assert "usage" in data
        assert data["choices"][0]["text"]
        assert data["system_fingerprint"] is not None

    async def test_tc6_2_completions_streaming(self, dual_client):
        """TC6.2: /v1/completions streaming — correct SSE format."""
        r = await dual_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 5, "stream": True, "ignore_eos": True},
        )
        assert r.status_code == 200
        lines = [line for line in r.text.strip().split("\n") if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunk = json.loads(lines[0][6:])
        assert chunk["object"] == "text_completion"

    async def test_tc6_3_chat_non_streaming(self, dual_client):
        """TC6.3: /v1/chat/completions non-streaming — correct format."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["system_fingerprint"] is not None

    async def test_tc6_4_chat_streaming_role(self, dual_client):
        """TC6.4: /v1/chat/completions streaming — first chunk has role."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "stream": True,
                "ignore_eos": True,
            },
        )
        lines = [
            line for line in r.text.strip().split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        first = json.loads(lines[0][6:])
        assert first["choices"][0]["delta"]["role"] == "assistant"

    async def test_tc6_5_models(self, dual_client):
        """TC6.5: /v1/models — correct list format."""
        r = await dual_client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    async def test_tc6_6_all_sampling_params(self, dual_client):
        """TC6.6: All sampling params accepted without error."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "seed": 42,
                "n": 1,
                "stop": ["end"],
                "user": "test-user",
            },
        )
        assert r.status_code == 200

    async def test_tc6_7_n_greater_than_1(self, dual_client):
        """TC6.7: n > 1 — multiple choices returned."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "n": 3,
                "ignore_eos": True,
            },
        )
        data = r.json()
        assert len(data["choices"]) == 3
        for i, c in enumerate(data["choices"]):
            assert c["index"] == i

    async def test_tc6_8_system_fingerprint(self, dual_client):
        """TC6.8: seed parameter — system_fingerprint present."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "seed": 123,
            },
        )
        data = r.json()
        assert data["system_fingerprint"] is not None

    async def test_tc6_9_stream_options_include_usage(self, dual_client):
        """TC6.9: stream_options.include_usage — usage in final chunk."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
            },
        )
        lines = [
            line for line in r.text.strip().split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        last = json.loads(lines[-1][6:])
        assert last["usage"] is not None
        assert last["usage"]["prompt_tokens"] > 0

    async def test_tc6_10_logprobs_field(self, dual_client):
        """TC6.10: logprobs field present (null) in choices."""
        r = await dual_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
        )
        data = r.json()
        assert "logprobs" in data["choices"][0]

    async def test_tc6_11_prompt_formats(self, dual_client):
        """TC6.11: 4 prompt formats — all supported."""
        # String prompt
        r = await dual_client.post(
            "/v1/completions", json={"prompt": "Hello", "max_tokens": 1}
        )
        assert r.status_code == 200

        # List of strings
        r = await dual_client.post(
            "/v1/completions", json={"prompt": ["Hello", "World"], "max_tokens": 1}
        )
        assert r.status_code == 200

        # Token IDs
        r = await dual_client.post(
            "/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 1}
        )
        assert r.status_code == 200

        # Chat messages
        r = await dual_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
        )
        assert r.status_code == 200

    async def test_tc6_12_invalid_json(self, dual_client):
        """TC6.12: Invalid JSON body — 400 error."""
        r = await dual_client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400

    async def test_tc6_13_missing_required_fields(self, dual_client):
        """TC6.13: Missing required fields — appropriate error."""
        r = await dual_client.post("/v1/chat/completions", json={})
        assert r.status_code == 400


# === Ping endpoint ===


class TestPing:
    """TC11.1-TC11.2: Ping endpoint."""

    async def test_tc11_1_get_ping(self, dual_client):
        """TC11.1: GET /ping → pong."""
        r = await dual_client.get("/ping")
        assert r.status_code == 200
        assert r.text == "pong"

    async def test_tc11_2_post_ping(self, dual_client):
        """TC11.2: POST /ping → pong."""
        r = await dual_client.post("/ping")
        assert r.status_code == 200
        assert r.text == "pong"


# === Model card max_model_len ===


class TestModelCard:
    """TC11.3: Model card with max_model_len."""

    async def test_tc11_3_max_model_len(self, dual_client):
        """TC11.3: /v1/models includes max_model_len."""
        r = await dual_client.get("/v1/models")
        data = r.json()
        assert data["data"][0]["max_model_len"] == 4096
