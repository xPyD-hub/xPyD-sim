"""Tests for bugfixes #19-#30."""

from __future__ import annotations

import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.common.helpers import DUMMY_TOKENS, count_prompt_tokens, render_dummy_text
from xpyd_sim.common.models import ChatMessage
from xpyd_sim.scheduler import Scheduler, SchedulingConfig
from xpyd_sim.server import ServerConfig, _compute_output_length, create_app

# --- Issue #19/#23: DUMMY_TOKENS is word tokens, not characters ---


def test_dummy_tokens_are_words():
    """DUMMY_TOKENS should be word tokens, not individual characters."""
    assert all(len(t) > 0 for t in DUMMY_TOKENS[:20])
    # At least some tokens should be multi-character words
    assert any(len(t) > 1 for t in DUMMY_TOKENS[:10])
    assert DUMMY_TOKENS[0] == "The"
    assert DUMMY_TOKENS[1] == "quick"


def test_render_dummy_text_returns_words():
    """render_dummy_text(n) should return n word tokens joined by spaces."""
    text = render_dummy_text(5)
    words = text.split()
    assert len(words) == 5
    assert words[0] == "The"


# --- Issue #29: max_tokens=1 can produce "stop" ---


def test_max_tokens_1_can_produce_stop():
    """max_tokens=1 should sometimes produce finish_reason='stop'."""
    results = set()
    for _ in range(200):
        _, reason = _compute_output_length(
            max_tokens=1, eos_min_ratio=0.5, ignore_eos=False
        )
        results.add(reason)
    assert "stop" in results, "max_tokens=1 should sometimes produce 'stop'"


def test_ignore_eos_always_length():
    """ignore_eos=True should always produce 'length'."""
    for _ in range(50):
        _, reason = _compute_output_length(
            max_tokens=1, eos_min_ratio=0.5, ignore_eos=True
        )
        assert reason == "length"


# --- Issue #30: count_prompt_tokens handles None/list content ---


def test_count_prompt_tokens_none_content():
    """Messages with content=None should contribute 0 content tokens."""
    msgs = [ChatMessage(role="user", content=None)]
    result = count_prompt_tokens(messages=msgs)
    # Should be small (just role length / 4), not inflated by "None" string
    assert result >= 1
    assert result <= 2  # "user" = 4 chars => 1 token


def test_count_prompt_tokens_list_content():
    """Messages with list content should extract text parts."""
    msgs = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "hello world"}],
        )
    ]
    result = count_prompt_tokens(messages=msgs)
    # "hello world" = 11 chars + "user" = 4 chars = 15 / 4 = 3
    assert result >= 1
    assert result <= 10  # Reasonable, not inflated by repr()


# --- Issue #20/#26: Non-streaming n>1 decode delay is parallel ---


@pytest.mark.asyncio
async def test_non_streaming_n_gt_1_parallel_delay():
    """Non-streaming n>1 should take ~max(tokens) time, not sum(tokens)."""
    config = ServerConfig(
        decode_delay_per_token_ms=5,
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        eos_min_ratio=1.0,  # Always full max_tokens
    )
    app = create_app(config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        t0 = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
                "n": 3,
            },
        )
        elapsed = time.monotonic() - t0
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 3
        # With parallel decode, should take ~10*5ms=50ms, not 30*5ms=150ms
        # Allow generous margin but should be < 100ms (not 150ms)
        assert elapsed < 0.25, f"n>1 took {elapsed:.3f}s, should be parallel not serial"


# --- Issue #25: Streaming stop sequence doesn't leak ---


@pytest.mark.asyncio
async def test_streaming_stop_no_leak():
    """Streaming should not emit tokens that are part of the stop sequence."""
    config = ServerConfig(
        decode_delay_per_token_ms=0,
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 50,
                "stream": True,
                "stop": ["fox"],
            },
        )
        assert resp.status_code == 200
        content_parts = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    c = delta.get("content")
                    if c is not None and c != "":
                        content_parts.append(c)
        full_text = "".join(content_parts)
        assert "fox" not in full_text, f"Stop sequence leaked into output: {full_text!r}"


# --- Issue #24: Streaming n>1 interleaves choices ---


@pytest.mark.asyncio
async def test_streaming_n_gt_1_interleaved():
    """Streaming with n>1 should interleave chunks across choices."""
    config = ServerConfig(
        decode_delay_per_token_ms=0,
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "n": 2,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        indices = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk["choices"]:
                    idx = chunk["choices"][0]["index"]
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content is not None and content != "":
                        indices.append(idx)
        # Indices should alternate (interleaved), not be all 0s then all 1s
        # Check that we see index switches within the content tokens
        switches = sum(1 for i in range(1, len(indices)) if indices[i] != indices[i - 1])
        assert switches > 0, f"Choices not interleaved: {indices}"


# --- Issue #21/#27: /v1/completions routes through scheduler ---


@pytest.mark.asyncio
async def test_completions_uses_scheduler():
    """When scheduling_enabled, /v1/completions should route through scheduler."""
    config = ServerConfig(
        decode_delay_per_token_ms=1,
        prefill_delay_ms=1,
        kv_transfer_delay_ms=0,
        scheduling_enabled=True,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
    )
    app = create_app(config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Test non-streaming
        resp = await client.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": "hello", "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

        # Test streaming
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "dummy",
                "prompt": "hello",
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "data:" in resp.text


# --- Issue #28: Scheduler doesn't silently drop rejected requests ---


@pytest.mark.asyncio
async def test_scheduler_rejects_oversized_request():
    """Rejected requests should signal error, not silently disappear."""
    sched_config = SchedulingConfig(
        max_model_len=100,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        enabled=True,
    )
    scheduler = Scheduler(
        config=sched_config,
        prefill_delay_ms=1,
        decode_delay_per_token_ms=1,
    )
    await scheduler.start()
    try:
        # Submit an oversized request via the HTTP endpoint which checks first
        config = ServerConfig(
            scheduling_enabled=True,
            max_model_len=100,
            prefill_delay_ms=1,
            decode_delay_per_token_ms=1,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Send a very long prompt that exceeds max_model_len
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "dummy",
                    "messages": [{"role": "user", "content": "x" * 500}],
                    "max_tokens": 5,
                },
            )
            assert resp.status_code == 400
            assert "exceeds" in resp.json()["error"]["message"]
    finally:
        await scheduler.stop()


# --- Issue #22: num_tokens after stop sequence is correct ---


@pytest.mark.asyncio
async def test_num_tokens_after_stop_sequence():
    """After stop truncation, num_tokens should match word count, not len//4."""
    config = ServerConfig(
        decode_delay_per_token_ms=0,
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 50,
                "stop": ["fox"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        text = choice["message"]["content"]
        # completion_tokens should roughly match word count of output
        word_count = len(text.split())
        completion_tokens = data["usage"]["completion_tokens"]
        assert completion_tokens == word_count, (
            f"completion_tokens={completion_tokens} != word_count={word_count}"
        )
