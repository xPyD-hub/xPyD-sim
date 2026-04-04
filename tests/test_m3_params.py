"""Tests for M3: Parameter Handling + Response Compliance — TC6.x, TC11.x."""

from __future__ import annotations

import json

import pytest
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
    return AsyncClient(transport=transport, base_url="http://test")


class TestParameterTypeValidation:
    """TC11.4: Invalid param type returns HTTP 400."""

    async def test_tc11_4_invalid_temperature(self, client):
        """temperature='abc' → 400."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": "abc",
            },
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data

    async def test_invalid_max_tokens_type(self, client):
        """max_tokens='not_int' → 400."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": "not_int",
            },
        )
        assert r.status_code == 400

    async def test_invalid_n_type(self, client):
        """n='abc' → 400."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "n": "abc",
            },
        )
        assert r.status_code == 400


class TestCompletionLogprobs:
    """TC11.5: logprobs=N on /v1/completions returns fake logprobs data."""

    async def test_tc11_5_completions_logprobs(self, client):
        """logprobs=5 → returns logprobs in correct format."""
        r = await client.post(
            "/v1/completions",
            json={
                "prompt": "Hello world",
                "max_tokens": 5,
                "logprobs": 5,
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        lp = data["choices"][0]["logprobs"]
        assert lp is not None
        assert "tokens" in lp
        assert "token_logprobs" in lp
        assert "top_logprobs" in lp
        assert "text_offset" in lp
        assert len(lp["tokens"]) > 0
        assert len(lp["token_logprobs"]) == len(lp["tokens"])
        assert len(lp["top_logprobs"]) == len(lp["tokens"])
        # Each top_logprobs entry should have up to 5 entries
        for tp in lp["top_logprobs"]:
            assert isinstance(tp, dict)
            assert len(tp) <= 5

    async def test_completions_logprobs_zero(self, client):
        """logprobs=0 → no logprobs."""
        r = await client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3, "logprobs": 0, "ignore_eos": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["logprobs"] is None

    async def test_completions_no_logprobs(self, client):
        """No logprobs param → logprobs is null."""
        r = await client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3, "ignore_eos": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["logprobs"] is None


class TestChatLogprobs:
    """TC11.6: logprobs=true, top_logprobs=N on /v1/chat/completions."""

    async def test_tc11_6_chat_logprobs(self, client):
        """logprobs=true, top_logprobs=3 → returns chat logprobs format."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "logprobs": True,
                "top_logprobs": 3,
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        lp = data["choices"][0]["logprobs"]
        assert lp is not None
        assert "content" in lp
        assert len(lp["content"]) > 0
        for item in lp["content"]:
            assert "token" in item
            assert "logprob" in item
            assert "top_logprobs" in item
            assert isinstance(item["top_logprobs"], list)
            assert len(item["top_logprobs"]) <= 3

    async def test_chat_no_logprobs(self, client):
        """logprobs not set → logprobs is null."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["logprobs"] is None


class TestStopSequence:
    """TC11.7-TC11.8: Stop sequence truncation."""

    async def test_tc11_7_stop_truncation(self, client):
        """stop=['dog'] triggers truncation, finish_reason='stop'."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                "stop": ["dog"],
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        assert "dog" not in text
        assert data["choices"][0]["finish_reason"] == "stop"

    async def test_tc11_8_stop_streaming(self, client):
        """stop in streaming mode — stream ends at stop sequence."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                "stop": ["fox"],
                "stream": True,
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        lines = [
            line for line in r.text.strip().split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        # Collect all content
        text = ""
        finish_reason = None
        for line in lines:
            chunk = json.loads(line[6:])
            if chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if delta and delta.get("content"):
                    text += delta["content"]
                fr = chunk["choices"][0].get("finish_reason")
                if fr:
                    finish_reason = fr
        assert "fox" not in text
        assert finish_reason == "stop"


class TestStreamingNGreaterThan1:
    """n > 1 in streaming mode."""

    async def test_n_gt_1_streaming(self, client):
        """n=2 streaming — both choices appear."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "n": 2,
                "stream": True,
                "ignore_eos": True,
            },
        )
        assert r.status_code == 200
        lines = [
            line for line in r.text.strip().split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        indices_seen = set()
        for line in lines:
            chunk = json.loads(line[6:])
            if chunk["choices"]:
                indices_seen.add(chunk["choices"][0]["index"])
        assert 0 in indices_seen
        assert 1 in indices_seen


class TestMaxModelLen:
    """TC11.3: max_model_len in model card."""

    async def test_tc11_3_max_model_len_in_models(self, client):
        """max_model_len appears in /v1/models."""
        r = await client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["data"][0]["max_model_len"] == 4096
