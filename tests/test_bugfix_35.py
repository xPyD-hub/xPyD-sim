"""Tests for bugfix #35: /v1/completions logprobs uses word-level tokenization."""

from __future__ import annotations

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


@pytest.mark.anyio
async def test_completions_logprobs_word_tokens(app):
    """logprobs tokens should be words, not individual characters."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/completions",
            json={"model": "dummy", "prompt": "Hello", "max_tokens": 5, "logprobs": 3},
        )
    assert resp.status_code == 200
    data = resp.json()
    lp = data["choices"][0]["logprobs"]
    assert lp is not None
    # Tokens should be words (possibly with leading space) or spaces, not
    # individual characters from the text.  The key indicator: at least some
    # tokens are multi-char words like "The", "quick", etc.
    word_tokens = [t for t in lp["tokens"] if t.strip()]
    assert any(len(t.strip()) > 1 for t in word_tokens), (
        f"Expected word-level tokens, got: {lp['tokens']}"
    )
