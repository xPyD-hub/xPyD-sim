"""Regression tests for issues #48-#56.

These tests ensure CI catches the specific bugs that were discovered
during the independent testing audit. Each test is tagged with the
issue number(s) it guards.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.common.helpers import DUMMY_TOKENS, render_dummy_text
from xpyd_sim.server import ServerConfig, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _scheduled_client(config: ServerConfig):
    """Client that manually starts/stops the scheduler (ASGI transport
    does not trigger lifespan events)."""
    app = create_app(config)
    if config._scheduler:
        await config._scheduler.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        if config._scheduler:
            await config._scheduler.stop()


# ===================================================================
# Issue #48 / #55 — render_dummy_text & usage.completion_tokens
# ===================================================================


class TestDummyTextCycling:
    """Guard against render_dummy_text silently truncating output."""

    def test_render_dummy_text_cycles_beyond_vocab(self):
        """#48: render_dummy_text(n) must produce exactly n words even
        when n > len(DUMMY_TOKENS)."""
        n = len(DUMMY_TOKENS) + 50
        text = render_dummy_text(n)
        assert len(text.split()) == n

    def test_render_dummy_text_large(self):
        """#48: Very large token count should still work."""
        text = render_dummy_text(1000)
        assert len(text.split()) == 1000

    @pytest.mark.asyncio
    async def test_usage_matches_content_length(self):
        """#48/#55: usage.completion_tokens must match actual word count
        in the response content."""
        config = ServerConfig(
            prefill_delay_ms=0,
            decode_delay_per_token_ms=0,
            max_model_len=131072,
        )
        app = create_app(config)
        max_tok = len(DUMMY_TOKENS) + 100  # Exceeds vocab
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": max_tok,
                    "ignore_eos": True,
                },
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            actual_words = len(content.split())
            reported = data["usage"]["completion_tokens"]
            assert actual_words == reported, (
                f"Content has {actual_words} words but usage reports "
                f"{reported} completion_tokens"
            )

    @pytest.mark.asyncio
    async def test_completions_usage_matches_content(self):
        """#48/#55: Same check for /v1/completions."""
        config = ServerConfig(
            prefill_delay_ms=0,
            decode_delay_per_token_ms=0,
            max_model_len=131072,
        )
        app = create_app(config)
        max_tok = len(DUMMY_TOKENS) + 50
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": max_tok, "ignore_eos": True},
            )
            data = resp.json()
            text = data["choices"][0]["text"]
            actual_words = len(text.split())
            reported = data["usage"]["completion_tokens"]
            assert actual_words == reported


# ===================================================================
# Issue #49 — Streaming requests must be logged to JSONL
# ===================================================================


class TestStreamingRequestLogging:
    """Guard against streaming requests silently skipping the JSONL log."""

    @pytest.mark.asyncio
    async def test_chat_streaming_logged(self):
        """#49: /v1/chat/completions stream=true must write to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "requests.jsonl")
            config = ServerConfig(
                prefill_delay_ms=0, decode_delay_per_token_ms=0,
                log_requests=log_path,
            )
            app = create_app(config)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 3,
                        "stream": True,
                    },
                )
                _ = resp.text  # consume

            assert os.path.exists(log_path), "Log file not created"
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) >= 1, "Streaming chat request not logged"
            record = json.loads(lines[0])
            assert "timestamp" in record
            assert "mode" in record

    @pytest.mark.asyncio
    async def test_completion_streaming_logged(self):
        """#49: /v1/completions stream=true must write to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "requests.jsonl")
            config = ServerConfig(
                prefill_delay_ms=0, decode_delay_per_token_ms=0,
                log_requests=log_path,
            )
            app = create_app(config)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post(
                    "/v1/completions",
                    json={"prompt": "Hello", "max_tokens": 3, "stream": True},
                )
                _ = resp.text

            assert os.path.exists(log_path), "Log file not created"
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) >= 1, "Streaming completion request not logged"

    @pytest.mark.asyncio
    async def test_scheduled_streaming_logged(self):
        """#49: Streaming via scheduler path must also be logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "requests.jsonl")
            config = ServerConfig(
                prefill_delay_ms=5, decode_delay_per_token_ms=2,
                scheduling_enabled=True, max_model_len=4096,
                max_num_batched_tokens=2048, max_num_seqs=4,
                log_requests=log_path,
            )
            async with _scheduled_client(config) as c:
                resp = await c.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 3,
                        "stream": True,
                    },
                )
                _ = resp.text

            assert os.path.exists(log_path)
            with open(log_path) as f:
                lines = f.readlines()
            # Should have batch events + the streaming log
            assert len(lines) >= 1, "Scheduled streaming request not logged"


# ===================================================================
# Issue #50 / #53 — max_model_len enforcement in ALL paths
# ===================================================================


class TestMaxModelLenEnforcement:
    """Guard against max_model_len being checked only in the scheduler path."""

    @pytest.mark.asyncio
    async def test_chat_legacy_rejects_oversized(self):
        """#50: /v1/chat/completions WITHOUT scheduling must reject
        input exceeding max_model_len."""
        config = ServerConfig(
            prefill_delay_ms=0,
            decode_delay_per_token_ms=0,
            max_model_len=50,
            scheduling_enabled=False,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "word " * 1000}],
                    "max_tokens": 3,
                },
            )
            assert resp.status_code == 400
            assert "exceeds" in resp.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_completions_legacy_rejects_oversized(self):
        """#53: /v1/completions WITHOUT scheduling must also reject."""
        config = ServerConfig(
            prefill_delay_ms=0,
            decode_delay_per_token_ms=0,
            max_model_len=50,
            scheduling_enabled=False,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/completions",
                json={"prompt": "word " * 1000, "max_tokens": 3},
            )
            assert resp.status_code == 400
            assert "exceeds" in resp.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_completions_scheduled_rejects_oversized(self):
        """#53: /v1/completions WITH scheduling must also reject."""
        config = ServerConfig(
            prefill_delay_ms=5,
            decode_delay_per_token_ms=2,
            max_model_len=50,
            scheduling_enabled=True,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
        )
        async with _scheduled_client(config) as c:
            resp = await c.post(
                "/v1/completions",
                json={"prompt": "word " * 1000, "max_tokens": 3},
            )
            assert resp.status_code == 400
            assert "exceeds" in resp.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_short_input_still_works(self):
        """Sanity: short input under max_model_len must still succeed."""
        config = ServerConfig(
            prefill_delay_ms=0,
            decode_delay_per_token_ms=0,
            max_model_len=200,
            scheduling_enabled=False,
        )
        app = create_app(config)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                },
            )
            assert resp.status_code == 200


# ===================================================================
# Issue #51 — prefill/app.py must have /ping and /metrics
# ===================================================================


class TestPrefillAppEndpoints:
    """Guard against prefill standalone app missing required endpoints."""

    @pytest.mark.asyncio
    async def test_prefill_app_get_ping(self):
        """#51: GET /ping must return 'pong'."""
        from xpyd_sim.prefill.app import create_prefill_app

        app = create_prefill_app(model_name="test")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/ping")
            assert resp.status_code == 200
            assert resp.text == "pong"

    @pytest.mark.asyncio
    async def test_prefill_app_post_ping(self):
        """#51: POST /ping must return 'pong'."""
        from xpyd_sim.prefill.app import create_prefill_app

        app = create_prefill_app(model_name="test")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post("/ping")
            assert resp.status_code == 200
            assert resp.text == "pong"

    @pytest.mark.asyncio
    async def test_prefill_app_metrics(self):
        """#51: GET /metrics must return Prometheus-format metrics."""
        from xpyd_sim.prefill.app import create_prefill_app

        app = create_prefill_app(model_name="test")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/metrics")
            assert resp.status_code == 200
            text = resp.text
            assert "xpyd_sim_requests_total" in text
            assert "xpyd_sim_active_requests" in text


# ===================================================================
# Issue #52 / #56 — Warmup penalty in scheduling mode
# ===================================================================


class TestWarmupWithScheduling:
    """Guard against warmup being silently skipped in scheduler path."""

    @pytest.mark.asyncio
    async def test_warmup_applies_to_scheduled_requests(self):
        """#52/#56: First N requests must have warmup penalty even when
        scheduling is enabled."""
        config = ServerConfig(
            mode="dual",
            prefill_delay_ms=1,
            decode_delay_per_token_ms=1,
            warmup_requests=2,
            warmup_penalty_ms=150,
            scheduling_enabled=True,
            max_model_len=4096,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
        )
        async with _scheduled_client(config) as c:
            body = {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 2,
            }
            # First request (warmup)
            start = time.monotonic()
            await c.post("/v1/chat/completions", json=body)
            first_ms = (time.monotonic() - start) * 1000

            # Second request (warmup)
            start = time.monotonic()
            await c.post("/v1/chat/completions", json=body)
            second_ms = (time.monotonic() - start) * 1000

            # Third request (no warmup)
            start = time.monotonic()
            await c.post("/v1/chat/completions", json=body)
            third_ms = (time.monotonic() - start) * 1000

            # First two should have ~150ms penalty
            assert first_ms > 100, (
                f"Warmup request 1 too fast: {first_ms:.0f}ms (expected >100ms)"
            )
            assert second_ms > 100, (
                f"Warmup request 2 too fast: {second_ms:.0f}ms (expected >100ms)"
            )
            # Third should be significantly faster
            assert third_ms < first_ms, (
                f"Post-warmup request not faster: {third_ms:.0f}ms vs {first_ms:.0f}ms"
            )


# ===================================================================
# Issue #54 — /v1/completions must route through scheduler
# ===================================================================


class TestCompletionsSchedulerRouting:
    """Guard against /v1/completions bypassing the scheduler."""

    @pytest.mark.asyncio
    async def test_completions_non_streaming_scheduled(self):
        """#54: /v1/completions non-streaming must work with scheduler."""
        config = ServerConfig(
            prefill_delay_ms=5,
            decode_delay_per_token_ms=2,
            scheduling_enabled=True,
            max_model_len=4096,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
        )
        async with _scheduled_client(config) as c:
            resp = await c.post(
                "/v1/completions",
                json={"prompt": "Hello world", "max_tokens": 5},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert data["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_completions_streaming_scheduled(self):
        """#54: /v1/completions streaming must work with scheduler."""
        config = ServerConfig(
            prefill_delay_ms=5,
            decode_delay_per_token_ms=2,
            scheduling_enabled=True,
            max_model_len=4096,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
        )
        async with _scheduled_client(config) as c:
            resp = await c.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 5, "stream": True},
            )
            assert resp.status_code == 200
            assert "data:" in resp.text
            assert "data: [DONE]" in resp.text

    @pytest.mark.asyncio
    async def test_completions_scheduled_logs_batch_events(self):
        """#54: Scheduled completions must produce batch events in log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "requests.jsonl")
            config = ServerConfig(
                prefill_delay_ms=5,
                decode_delay_per_token_ms=2,
                scheduling_enabled=True,
                max_model_len=4096,
                max_num_batched_tokens=2048,
                max_num_seqs=4,
                log_requests=log_path,
            )
            async with _scheduled_client(config) as c:
                resp = await c.post(
                    "/v1/completions",
                    json={"prompt": "Hello", "max_tokens": 3},
                )
                assert resp.status_code == 200

            assert os.path.exists(log_path)
            with open(log_path) as f:
                events = [json.loads(line) for line in f if line.strip()]
            event_types = [e.get("event") for e in events]
            assert "prefill_start" in event_types, (
                f"Missing prefill_start, got: {event_types}"
            )
            assert "decode_done" in event_types, (
                f"Missing decode_done, got: {event_types}"
            )


# ===================================================================
# Issue #58 — _form_prefill_batch crashes outside async context
# ===================================================================


class TestSchedulerSyncSafety:
    """Guard against _form_prefill_batch requiring an active event loop."""

    def test_form_prefill_batch_sync_context(self):
        """#58: _form_prefill_batch must work without a running event loop.

        The old code used asyncio.get_running_loop().call_soon() which
        raises RuntimeError in sync contexts. put_nowait() and set()
        are sync-safe and should be used directly.
        """
        from xpyd_sim.scheduler import InferenceRequest, Scheduler, SchedulingConfig

        sched_config = SchedulingConfig(
            max_model_len=100,
            max_num_batched_tokens=500,
            max_num_seqs=10,
            enabled=True,
        )
        scheduler = Scheduler(config=sched_config)

        big_req = InferenceRequest(
            request_id="big", input_tokens=200, max_tokens=5
        )
        small_req = InferenceRequest(
            request_id="small", input_tokens=50, max_tokens=5
        )

        # Must NOT raise RuntimeError: no running event loop
        batch, remaining = scheduler._form_prefill_batch([big_req, small_req])

        # big_req rejected, small_req in batch
        assert len(batch) == 1
        assert batch[0].request_id == "small"
        assert big_req.done_event.is_set()
        msg_type, msg = big_req.token_queue.get_nowait()
        assert msg_type == "error"
        assert "exceeds" in msg
