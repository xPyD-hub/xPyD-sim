"""Unified FastAPI server for xPyD-sim with mode-based latency."""

from __future__ import annotations

import asyncio
import math
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError

from xpyd_sim.common.helpers import (
    count_prompt_tokens,
    generate_id,
    get_effective_max_tokens,
    now_ts,
    render_dummy_text,
)
from xpyd_sim.common.logprobs import (
    generate_chat_logprobs,
    generate_completion_logprobs,
    tokenize_text,
)
from xpyd_sim.common.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    DeltaMessage,
    ModelCard,
    ModelListResponse,
    StreamChoice,
    UsageInfo,
)
from xpyd_sim.observability import Metrics, RequestLogger, WarmupTracker
from xpyd_sim.profile import LatencyProfile
from xpyd_sim.scheduler import InferenceRequest, Scheduler, SchedulingConfig

SYSTEM_FINGERPRINT = "fp_xpyd_sim"


@dataclass
class ServerConfig:
    """Server configuration."""

    mode: str = "dual"  # dual, prefill, decode
    model_name: str = "dummy"
    prefill_delay_ms: float = 50.0
    kv_transfer_delay_ms: float = 5.0
    decode_delay_per_token_ms: float = 10.0
    eos_min_ratio: float = 0.5
    max_model_len: int = 131072
    default_max_tokens: int = 16
    warmup_requests: int = 0
    warmup_penalty_ms: float = 0.0
    log_requests: str | None = None
    profile: str | None = None
    # Scheduling config
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    scheduling_enabled: bool = False
    _latency_profile: LatencyProfile | None = None
    _metrics: Metrics = field(default_factory=Metrics)
    _request_logger: RequestLogger | None = None
    _warmup_tracker: WarmupTracker | None = None
    _scheduler: Scheduler | None = None
    require_api_key: str | None = None

    def load_profile(self) -> None:
        """Load latency profile if configured."""
        if self.profile:
            self._latency_profile = LatencyProfile(self.profile)

    def init_observability(self) -> None:
        """Initialize metrics, request logger, and warmup tracker."""
        self._metrics = Metrics()
        self._request_logger = RequestLogger(self.log_requests)
        self._warmup_tracker = WarmupTracker(self.warmup_requests, self.warmup_penalty_ms)

    def init_scheduler(self) -> None:
        """Initialize the scheduler if scheduling is enabled."""
        if not self.scheduling_enabled:
            return
        sched_config = SchedulingConfig(
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            enabled=True,
        )
        log_cb = None
        if self._request_logger:
            log_cb = self._request_logger.log
        self._scheduler = Scheduler(
            config=sched_config,
            prefill_delay_ms=self.prefill_delay_ms,
            kv_transfer_delay_ms=self.kv_transfer_delay_ms,
            decode_delay_per_token_ms=self.decode_delay_per_token_ms,
            mode=self.mode,
            latency_profile=self._latency_profile,
            log_callback=log_cb,
        )


def _compute_output_length(
    max_tokens: int,
    eos_min_ratio: float,
    ignore_eos: bool,
) -> tuple[int, str]:
    """Return (num_tokens, finish_reason)."""
    if ignore_eos:
        return max_tokens, "length"
    if max_tokens <= 1:
        return max_tokens, "stop" if random.random() < eos_min_ratio else "length"
    min_len = max(1, math.ceil(max_tokens * eos_min_ratio))
    actual = random.randint(min_len, max_tokens)
    if actual < max_tokens:
        return actual, "stop"
    # Even at max_tokens, there's a chance EOS lands on the last token
    if random.random() < (1.0 - eos_min_ratio):
        return max_tokens, "stop"
    return max_tokens, "length"


def _check_stop_sequences(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    """Check text for stop sequences. Returns (possibly_truncated_text, was_stopped)."""
    if stop is None:
        return text, False
    if isinstance(stop, str):
        stop = [stop]
    earliest_pos = len(text)
    found = False
    for seq in stop:
        pos = text.find(seq)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            found = True
    if found:
        return text[:earliest_pos], True
    return text, False


def _compute_prefill_delay(config: ServerConfig, prompt_tokens: int) -> float:
    """Prefill delay in seconds based on mode."""
    if config.mode == "decode":
        return 0.0
    if config._latency_profile and config._latency_profile.has_prefill:
        return config._latency_profile.prefill_delay_ms(prompt_tokens) / 1000.0
    return config.prefill_delay_ms / 1000.0


def _compute_kv_delay(config: ServerConfig, prompt_tokens: int = 0) -> float:
    """KV transfer delay in seconds based on mode."""
    if config.mode == "dual":
        return 0.0  # local, no transfer
    if config.mode == "prefill":
        return 0.0  # prefill doesn't wait for KV
    # decode mode: wait for KV
    if config._latency_profile and config._latency_profile.has_kv_transfer:
        return config._latency_profile.kv_transfer_delay_ms(prompt_tokens) / 1000.0
    return config.kv_transfer_delay_ms / 1000.0


def _compute_decode_delay(
    config: ServerConfig, batch_size: int = 1, context_length: int = 512,
) -> float:
    """Per-token decode delay in seconds."""
    if config._latency_profile and config._latency_profile.has_decode:
        delay_ms = config._latency_profile.decode_delay_per_token_ms(
            batch_size, context_length,
        )
        return delay_ms / 1000.0
    return config.decode_delay_per_token_ms / 1000.0


def _log_request(
    config: ServerConfig,
    prompt_tokens: int,
    output_tokens: int,
    prefill_ms: float,
    kv_transfer_ms: float,
    decode_ms: float,
    total_ms: float,
) -> None:
    """Log a request to JSONL if logger is configured."""
    if config._request_logger is None:
        return
    config._request_logger.log({
        "timestamp": int(time.time()),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_ms": round(prefill_ms, 2),
        "kv_transfer_ms": round(kv_transfer_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "mode": config.mode,
    })


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create the unified xPyD-sim FastAPI application."""
    if config is None:
        config = ServerConfig()
    config.load_profile()
    config.init_observability()
    config.init_scheduler()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config._scheduler:
            await config._scheduler.start()
        yield
        if config._scheduler:
            await config._scheduler.stop()

    app = FastAPI(title="xPyD-sim", lifespan=lifespan)
    app.state.config = config

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if config.require_api_key and request.url.path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {config.require_api_key}":
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "type": "auth_error"}},
                )
        return await call_next(request)

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    @app.get("/ping")
    @app.post("/ping")
    async def ping():
        return PlainTextResponse("pong")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "mode": config.mode,
            "model": config.model_name,
            "config": {
                "prefill_delay_ms": config.prefill_delay_ms,
                "kv_transfer_delay_ms": config.kv_transfer_delay_ms,
                "decode_delay_per_token_ms": config.decode_delay_per_token_ms,
            },
        }

    @app.get("/metrics")
    async def metrics():
        if config._scheduler:
            state = config._scheduler.get_batch_state()
        else:
            state = {
                "prefill_queue_depth": 0,
                "prefill_batch_size": 0,
                "decode_batch_size": 0,
                "decode_avg_context_length": 0.0,
            }
        batch_metrics = (
            f"\n# HELP xpyd_sim_prefill_queue_depth Current prefill queue depth.\n"
            f"# TYPE xpyd_sim_prefill_queue_depth gauge\n"
            f"xpyd_sim_prefill_queue_depth {state['prefill_queue_depth']}\n"
            f"# HELP xpyd_sim_prefill_batch_size Current prefill batch size.\n"
            f"# TYPE xpyd_sim_prefill_batch_size gauge\n"
            f"xpyd_sim_prefill_batch_size {state['prefill_batch_size']}\n"
            f"# HELP xpyd_sim_decode_batch_size Current decode batch size.\n"
            f"# TYPE xpyd_sim_decode_batch_size gauge\n"
            f"xpyd_sim_decode_batch_size {state['decode_batch_size']}\n"
            f"# HELP xpyd_sim_decode_avg_context_length "
            f"Average context length in decode batch.\n"
            f"# TYPE xpyd_sim_decode_avg_context_length gauge\n"
            f"xpyd_sim_decode_avg_context_length {state['decode_avg_context_length']}\n"
        )
        return PlainTextResponse(
            config._metrics.render_prometheus() + batch_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/debug/batch")
    async def debug_batch():
        if not config._scheduler:
            return {
                "prefill_queue_depth": 0,
                "prefill_batch_size": 0,
                "prefill_batch_tokens": 0,
                "decode_batch_size": 0,
                "decode_avg_context_length": 0.0,
                "decode_requests": [],
            }
        return config._scheduler.get_batch_state()

    @app.get("/v1/models")
    async def list_models():
        card = ModelCard(
            id=config.model_name,
            created=now_ts(),
            max_model_len=config.max_model_len,
        )
        return ModelListResponse(data=[card])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        request_start = time.monotonic()
        config._metrics.inc_requests()
        config._metrics.inc_active()
        try:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "Invalid JSON body",
                            "type": "invalid_request_error",
                        }
                    },
                )
            if "messages" not in body:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "Missing required field: messages",
                            "type": "invalid_request_error",
                        }
                    },
                )

            try:
                req = ChatCompletionRequest(**body)
            except ValidationError as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": str(e), "type": "invalid_request_error"}},
                )

            prompt_tokens = count_prompt_tokens(messages=req.messages)
            max_tokens = get_effective_max_tokens(req.max_completion_tokens, req.max_tokens)
            n = req.n or 1
            ignore_eos = req.ignore_eos or False

            # Enforce max_model_len for all paths
            if prompt_tokens > config.max_model_len:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": (
                                f"Input length {prompt_tokens} exceeds "
                                f"max_model_len {config.max_model_len}"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            # Cap max_tokens so total (input + output) doesn't exceed max_model_len
            available = config.max_model_len - prompt_tokens
            if max_tokens > available:
                max_tokens = max(1, available)

            # Warm-up penalty (applies to all paths)
            warmup_penalty = config._warmup_tracker.get_penalty() if config._warmup_tracker else 0
            if warmup_penalty > 0:
                await asyncio.sleep(warmup_penalty)

            # If scheduling enabled, route through scheduler
            if config._scheduler:
                if req.stream:
                    return StreamingResponse(
                        _stream_chat_scheduled(
                            config, req, prompt_tokens, max_tokens, n, ignore_eos,
                        ),
                        media_type="text/event-stream",
                    )
                return await _non_stream_chat_scheduled(
                    config, req, prompt_tokens, max_tokens, n, ignore_eos, request_start,
                )

            # Simulate prefill + KV transfer
            prefill_delay = _compute_prefill_delay(config, prompt_tokens)
            kv_delay = _compute_kv_delay(config, prompt_tokens)
            await asyncio.sleep(prefill_delay + kv_delay)

            if req.stream:
                return StreamingResponse(
                    _stream_chat(config, req, prompt_tokens, max_tokens, n, ignore_eos),
                    media_type="text/event-stream",
                )

            # Non-streaming
            choices = []
            total_completion = 0
            max_choice_tokens = 0
            for i in range(n):
                num_tokens, finish_reason = _compute_output_length(
                    max_tokens, config.eos_min_ratio, ignore_eos
                )
                text = render_dummy_text(num_tokens)
                text, stopped = _check_stop_sequences(text, req.stop)
                if stopped:
                    finish_reason = "stop"
                    num_tokens = max(1, len(text.split()))
                total_completion += num_tokens
                max_choice_tokens = max(max_choice_tokens, num_tokens)
                lp_data = None
                if req.logprobs:
                    top_n = req.top_logprobs or 5
                    lp_data = generate_chat_logprobs(tokenize_text(text), top_n)
                choices.append(
                    Choice(
                        index=i,
                        message=ChoiceMessage(role="assistant", content=text),
                        finish_reason=finish_reason,
                        logprobs=lp_data,
                    )
                )

            # Simulate decode delay (parallel across n choices)
            decode_delay = _compute_decode_delay(config)
            await asyncio.sleep(decode_delay * max_choice_tokens)

            # Update metrics
            config._metrics.inc_tokens(total_completion)

            # Log request
            total_ms = (time.monotonic() - request_start) * 1000
            _log_request(
                config,
                prompt_tokens=prompt_tokens,
                output_tokens=total_completion,
                prefill_ms=prefill_delay * 1000,
                kv_transfer_ms=kv_delay * 1000,
                decode_ms=decode_delay * max_choice_tokens * 1000,
                total_ms=total_ms,
            )

            return ChatCompletionResponse(
                id=generate_id("chatcmpl"),
                created=now_ts(),
                model=config.model_name,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=total_completion,
                    total_tokens=prompt_tokens + total_completion,
                ),
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
        finally:
            config._metrics.dec_active()
            duration = time.monotonic() - request_start
            config._metrics.observe_duration(duration)

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_start = time.monotonic()
        config._metrics.inc_requests()
        config._metrics.inc_active()
        try:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "Invalid JSON body",
                            "type": "invalid_request_error",
                        }
                    },
                )
            try:
                req = CompletionRequest(**body)
            except ValidationError as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": str(e), "type": "invalid_request_error"}},
                )

            prompt_tokens = count_prompt_tokens(prompt=req.prompt)
            max_tokens = get_effective_max_tokens(req.max_tokens)
            n = req.n or 1
            ignore_eos = req.ignore_eos or False

            # Enforce max_model_len for all paths
            if prompt_tokens > config.max_model_len:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": (
                                f"Input length {prompt_tokens} exceeds "
                                f"max_model_len {config.max_model_len}"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            # Cap max_tokens so total (input + output) doesn't exceed max_model_len
            available = config.max_model_len - prompt_tokens
            if max_tokens > available:
                max_tokens = max(1, available)

            # Warm-up penalty
            warmup_penalty = (
                config._warmup_tracker.get_penalty() if config._warmup_tracker else 0
            )
            if warmup_penalty > 0:
                await asyncio.sleep(warmup_penalty)

            # If scheduling enabled, route through scheduler
            if config._scheduler:
                if req.stream:
                    return StreamingResponse(
                        _stream_completion_scheduled(
                            config, req, prompt_tokens, max_tokens, n, ignore_eos,
                        ),
                        media_type="text/event-stream",
                    )
                return await _non_stream_completion_scheduled(
                    config, req, prompt_tokens, max_tokens, n, ignore_eos, request_start,
                )

            # Simulate prefill + KV transfer
            prefill_delay = _compute_prefill_delay(config, prompt_tokens)
            kv_delay = _compute_kv_delay(config, prompt_tokens)
            await asyncio.sleep(prefill_delay + kv_delay)

            if req.stream:
                return StreamingResponse(
                    _stream_completion(config, req, prompt_tokens, max_tokens, n, ignore_eos),
                    media_type="text/event-stream",
                )

            # Non-streaming
            choices = []
            total_completion = 0
            max_choice_tokens = 0
            for i in range(n):
                num_tokens, finish_reason = _compute_output_length(
                    max_tokens, config.eos_min_ratio, ignore_eos
                )
                text = render_dummy_text(num_tokens)
                text, stopped = _check_stop_sequences(text, req.stop)
                if stopped:
                    finish_reason = "stop"
                    num_tokens = max(1, len(text.split()))
                total_completion += num_tokens
                max_choice_tokens = max(max_choice_tokens, num_tokens)

                # echo=True: prepend prompt text to output
                output_text = text
                if req.echo:
                    prompt_str = req.prompt if isinstance(req.prompt, str) else str(req.prompt)
                    output_text = prompt_str + text

                lp = None
                if req.logprobs is not None and req.logprobs > 0:
                    tokens = tokenize_text(output_text) if output_text else [""]
                    lp = generate_completion_logprobs(tokens, req.logprobs)
                choices.append(
                    CompletionChoice(
                        index=i,
                        text=output_text,
                        finish_reason=finish_reason,
                        logprobs=lp,
                    )
                )

            # Simulate decode delay (parallel across n choices)
            decode_delay = _compute_decode_delay(config)
            await asyncio.sleep(decode_delay * max_choice_tokens)

            # Update metrics
            config._metrics.inc_tokens(total_completion)

            # Log request
            total_ms = (time.monotonic() - request_start) * 1000
            _log_request(
                config,
                prompt_tokens=prompt_tokens,
                output_tokens=total_completion,
                prefill_ms=prefill_delay * 1000,
                kv_transfer_ms=kv_delay * 1000,
                decode_ms=decode_delay * max_choice_tokens * 1000,
                total_ms=total_ms,
            )

            return CompletionResponse(
                id=generate_id("cmpl"),
                created=now_ts(),
                model=config.model_name,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=total_completion,
                    total_tokens=prompt_tokens + total_completion,
                ),
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
        finally:
            config._metrics.dec_active()
            duration = time.monotonic() - request_start
            config._metrics.observe_duration(duration)

    return app


async def _non_stream_chat_scheduled(
    config: ServerConfig,
    req: ChatCompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
    request_start: float,
) -> JSONResponse:
    """Handle non-streaming chat completion with scheduler."""
    scheduler = config._scheduler
    assert scheduler is not None

    choices = []
    total_completion = 0

    for i in range(n):
        inf_req = InferenceRequest(
            request_id=generate_id("req"),
            input_tokens=prompt_tokens,
            max_tokens=max_tokens,
            eos_min_ratio=config.eos_min_ratio,
            ignore_eos=ignore_eos,
        )
        await scheduler.submit(inf_req)
        # Wait for completion
        await inf_req.done_event.wait()

        # Drain token queue
        tokens_received = 0
        while not inf_req.token_queue.empty():
            msg_type, _ = inf_req.token_queue.get_nowait()
            if msg_type == "token":
                tokens_received += 1
            elif msg_type == "error":
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": _, "type": "invalid_request_error"}},
                )

        text = render_dummy_text(inf_req.generated_tokens)
        text, stopped = _check_stop_sequences(text, req.stop)
        finish_reason = inf_req.finish_reason
        num_tokens = inf_req.generated_tokens
        if stopped:
            finish_reason = "stop"
            num_tokens = max(1, len(text.split()))
        total_completion += num_tokens

        lp_data = None
        if req.logprobs:
            top_n = req.top_logprobs or 5
            lp_data = generate_chat_logprobs(tokenize_text(text), top_n)

        choices.append(
            Choice(
                index=i,
                message=ChoiceMessage(role="assistant", content=text),
                finish_reason=finish_reason,
                logprobs=lp_data,
            )
        )

    config._metrics.inc_tokens(total_completion)

    resp = ChatCompletionResponse(
        id=generate_id("chatcmpl"),
        created=now_ts(),
        model=config.model_name,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion,
            total_tokens=prompt_tokens + total_completion,
        ),
        system_fingerprint=SYSTEM_FINGERPRINT,
    )
    return JSONResponse(content=resp.model_dump())


async def _stream_chat_scheduled(
    config: ServerConfig,
    req: ChatCompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
):
    """Stream chat completion using scheduler for token timing."""
    request_start = time.monotonic()
    scheduler = config._scheduler
    assert scheduler is not None
    req_id = generate_id("chatcmpl")
    created = now_ts()
    include_usage = req.stream_options.get("include_usage", False) if req.stream_options else False
    total_completion = 0

    for idx in range(n):
        inf_req = InferenceRequest(
            request_id=generate_id("req"),
            input_tokens=prompt_tokens,
            max_tokens=max_tokens,
            eos_min_ratio=config.eos_min_ratio,
            ignore_eos=ignore_eos,
        )
        await scheduler.submit(inf_req)

        # First chunk: role
        chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[
                StreamChoice(
                    index=idx,
                    delta=DeltaMessage(role="assistant", content=""),
                    logprobs=None,
                )
            ],
            system_fingerprint=SYSTEM_FINGERPRINT,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Collect all tokens from scheduler first, then apply stop
        # truncation and stream the safe output.
        token_count = 0
        while True:
            msg_type, value = await inf_req.token_queue.get()
            if msg_type == "error":
                break
            if msg_type == "done":
                break
            token_count += 1

        # Build full text, apply stop truncation, then stream
        text = render_dummy_text(token_count)
        finish_reason_override = None
        if req.stop:
            text, was_stopped = _check_stop_sequences(text, req.stop)
            if was_stopped:
                finish_reason_override = "stop"

        tokens = text.split(" ") if text else []
        for i, token in enumerate(tokens):
            token_text = (" " + token) if i > 0 else token
            chunk_lp = None
            if req.logprobs and req.top_logprobs and req.top_logprobs > 0:
                chunk_lp = generate_chat_logprobs([token_text], req.top_logprobs)
            chunk = ChatCompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    StreamChoice(
                        index=idx,
                        delta=DeltaMessage(content=token_text),
                        logprobs=chunk_lp,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            total_completion += 1

        # Finish chunk
        if finish_reason_override:
            finish_reason = finish_reason_override
        else:
            finish_reason = inf_req.finish_reason if inf_req.is_done() else "stop"
        chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[
                StreamChoice(
                    index=idx,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
            system_fingerprint=SYSTEM_FINGERPRINT,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    config._metrics.inc_tokens(total_completion)

    if include_usage:
        usage_chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[],
            system_fingerprint=SYSTEM_FINGERPRINT,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
                total_tokens=prompt_tokens + total_completion,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    # Log streaming request
    total_ms = (time.monotonic() - request_start) * 1000
    _log_request(
        config,
        prompt_tokens=prompt_tokens,
        output_tokens=total_completion,
        prefill_ms=0,
        kv_transfer_ms=0,
        decode_ms=total_ms,
        total_ms=total_ms,
    )

    yield "data: [DONE]\n\n"


async def _non_stream_completion_scheduled(
    config: ServerConfig,
    req: CompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
    request_start: float,
) -> JSONResponse:
    """Handle non-streaming completion with scheduler."""
    scheduler = config._scheduler
    assert scheduler is not None

    choices = []
    total_completion = 0

    for i in range(n):
        inf_req = InferenceRequest(
            request_id=generate_id("req"),
            input_tokens=prompt_tokens,
            max_tokens=max_tokens,
            eos_min_ratio=config.eos_min_ratio,
            ignore_eos=ignore_eos,
        )
        await scheduler.submit(inf_req)
        await inf_req.done_event.wait()

        # Drain token queue
        while not inf_req.token_queue.empty():
            msg_type, value = inf_req.token_queue.get_nowait()
            if msg_type == "error":
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": value, "type": "invalid_request_error"}},
                )

        text = render_dummy_text(inf_req.generated_tokens)
        text, stopped = _check_stop_sequences(text, req.stop)
        finish_reason = inf_req.finish_reason
        num_tokens = inf_req.generated_tokens
        if stopped:
            finish_reason = "stop"
            num_tokens = max(1, len(text.split()))
        total_completion += num_tokens

        output_text = text
        if req.echo:
            prompt_str = req.prompt if isinstance(req.prompt, str) else str(req.prompt)
            output_text = prompt_str + text

        lp = None
        if req.logprobs is not None and req.logprobs > 0:
            tokens = tokenize_text(output_text) if output_text else [""]
            lp = generate_completion_logprobs(tokens, req.logprobs)

        choices.append(
            CompletionChoice(
                index=i,
                text=output_text,
                finish_reason=finish_reason,
                logprobs=lp,
            )
        )

    config._metrics.inc_tokens(total_completion)

    resp = CompletionResponse(
        id=generate_id("cmpl"),
        created=now_ts(),
        model=config.model_name,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion,
            total_tokens=prompt_tokens + total_completion,
        ),
        system_fingerprint=SYSTEM_FINGERPRINT,
    )
    return JSONResponse(content=resp.model_dump())


async def _stream_completion_scheduled(
    config: ServerConfig,
    req: CompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
):
    """Stream completion using scheduler for token timing."""
    request_start = time.monotonic()
    scheduler = config._scheduler
    assert scheduler is not None
    req_id = generate_id("cmpl")
    created = now_ts()
    include_usage = req.stream_options.get("include_usage", False) if req.stream_options else False
    total_completion = 0

    for idx in range(n):
        inf_req = InferenceRequest(
            request_id=generate_id("req"),
            input_tokens=prompt_tokens,
            max_tokens=max_tokens,
            eos_min_ratio=config.eos_min_ratio,
            ignore_eos=ignore_eos,
        )
        await scheduler.submit(inf_req)

        # echo=True: emit prompt text first
        if req.echo:
            prompt_str = req.prompt if isinstance(req.prompt, str) else str(req.prompt)
            echo_chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    CompletionStreamChoice(index=idx, text=prompt_str, logprobs=None)
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {echo_chunk.model_dump_json()}\n\n"

        # Collect all tokens
        token_count = 0
        while True:
            msg_type, value = await inf_req.token_queue.get()
            if msg_type == "error":
                break
            if msg_type == "done":
                break
            token_count += 1

        text = render_dummy_text(token_count)
        finish_reason_override = None
        if req.stop:
            text, was_stopped = _check_stop_sequences(text, req.stop)
            if was_stopped:
                finish_reason_override = "stop"

        tokens = text.split(" ") if text else []
        for i, token in enumerate(tokens):
            token_text = (" " + token) if i > 0 else token
            chunk_lp = None
            if req.logprobs is not None and req.logprobs > 0:
                chunk_lp = generate_completion_logprobs([token_text], req.logprobs)
            chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    CompletionStreamChoice(index=idx, text=token_text, logprobs=chunk_lp)
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            total_completion += 1

        # Finish chunk
        if finish_reason_override:
            finish_reason = finish_reason_override
        else:
            finish_reason = inf_req.finish_reason if inf_req.is_done() else "stop"
        chunk = CompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[
                CompletionStreamChoice(
                    index=idx, text="", finish_reason=finish_reason, logprobs=None,
                )
            ],
            system_fingerprint=SYSTEM_FINGERPRINT,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    config._metrics.inc_tokens(total_completion)

    if include_usage:
        usage_chunk = CompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[],
            system_fingerprint=SYSTEM_FINGERPRINT,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
                total_tokens=prompt_tokens + total_completion,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    # Log streaming request
    total_ms = (time.monotonic() - request_start) * 1000
    _log_request(
        config,
        prompt_tokens=prompt_tokens,
        output_tokens=total_completion,
        prefill_ms=0,
        kv_transfer_ms=0,
        decode_ms=total_ms,
        total_ms=total_ms,
    )

    yield "data: [DONE]\n\n"


async def _stream_chat(
    config: ServerConfig,
    req: ChatCompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
):
    """Stream chat completion chunks with per-token decode delay."""
    request_start = time.monotonic()
    req_id = generate_id("chatcmpl")
    created = now_ts()
    include_usage = req.stream_options.get("include_usage", False) if req.stream_options else False
    decode_delay = _compute_decode_delay(config)
    total_completion = 0

    for idx in range(n):
        num_tokens, finish_reason = _compute_output_length(
            max_tokens, config.eos_min_ratio, ignore_eos
        )
        text = render_dummy_text(num_tokens)

        # First chunk: role
        chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[
                StreamChoice(
                    index=idx,
                    delta=DeltaMessage(role="assistant", content=""),
                    logprobs=None,
                )
            ],
            system_fingerprint=SYSTEM_FINGERPRINT,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Apply stop sequence truncation before streaming (ensures identical
        # output to non-streaming mode).
        if req.stop:
            text, stopped = _check_stop_sequences(text, req.stop)
            if stopped:
                finish_reason = "stop"

        # Content chunks (per token)
        tokens = text.split(" ") if text else []
        for i, token in enumerate(tokens):
            await asyncio.sleep(decode_delay)
            # Generate per-token logprobs for chat streaming
            token_text = (" " + token) if i > 0 else token
            chunk_lp = None
            if req.logprobs and req.top_logprobs and req.top_logprobs > 0:
                chunk_lp = generate_chat_logprobs([token_text], req.top_logprobs)
            chunk = ChatCompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    StreamChoice(
                        index=idx,
                        delta=DeltaMessage(content=token_text),
                        logprobs=chunk_lp,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        total_completion += len(tokens)

        # Finish chunk
        chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[
                StreamChoice(
                    index=idx,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
            system_fingerprint=SYSTEM_FINGERPRINT,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Update token metrics for streaming
    config._metrics.inc_tokens(total_completion)

    # Usage chunk
    if include_usage:
        usage_chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[],
            system_fingerprint=SYSTEM_FINGERPRINT,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
                total_tokens=prompt_tokens + total_completion,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    # Log streaming request
    prefill_delay = _compute_prefill_delay(config, prompt_tokens)
    kv_delay = _compute_kv_delay(config, prompt_tokens)
    total_ms = (time.monotonic() - request_start) * 1000
    _log_request(
        config,
        prompt_tokens=prompt_tokens,
        output_tokens=total_completion,
        prefill_ms=prefill_delay * 1000,
        kv_transfer_ms=kv_delay * 1000,
        decode_ms=decode_delay * total_completion * 1000,
        total_ms=total_ms,
    )

    yield "data: [DONE]\n\n"


async def _stream_completion(
    config: ServerConfig,
    req: CompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
):
    """Stream completion chunks with per-token decode delay."""
    request_start = time.monotonic()
    req_id = generate_id("cmpl")
    created = now_ts()
    include_usage = req.stream_options.get("include_usage", False) if req.stream_options else False
    decode_delay = _compute_decode_delay(config)
    total_completion = 0

    for idx in range(n):
        num_tokens, finish_reason = _compute_output_length(
            max_tokens, config.eos_min_ratio, ignore_eos
        )
        text = render_dummy_text(num_tokens)

        # echo=True: emit prompt text first
        if req.echo:
            prompt_str = req.prompt if isinstance(req.prompt, str) else str(req.prompt)
            echo_chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    CompletionStreamChoice(
                        index=idx,
                        text=prompt_str,
                        logprobs=None,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {echo_chunk.model_dump_json()}\n\n"

        # Apply stop sequence truncation before streaming
        if req.stop:
            text, stopped = _check_stop_sequences(text, req.stop)
            if stopped:
                finish_reason = "stop"

        tokens = text.split(" ") if text else []
        for i, token in enumerate(tokens):
            await asyncio.sleep(decode_delay)
            token_text = (" " + token) if i > 0 else token
            chunk_lp = None
            if req.logprobs is not None and req.logprobs > 0:
                chunk_lp = generate_completion_logprobs([token_text], req.logprobs)
            chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    CompletionStreamChoice(
                        index=idx,
                        text=token_text,
                        logprobs=chunk_lp,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        total_completion += len(tokens)

        # Finish chunk
        chunk = CompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[
                CompletionStreamChoice(
                    index=idx,
                    text="",
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
            system_fingerprint=SYSTEM_FINGERPRINT,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Update token metrics for streaming
    config._metrics.inc_tokens(total_completion)

    if include_usage:
        usage_chunk = CompletionChunk(
            id=req_id,
            created=created,
            model=config.model_name,
            choices=[],
            system_fingerprint=SYSTEM_FINGERPRINT,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
                total_tokens=prompt_tokens + total_completion,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    # Log streaming request
    prefill_delay = _compute_prefill_delay(config, prompt_tokens)
    kv_delay = _compute_kv_delay(config, prompt_tokens)
    total_ms = (time.monotonic() - request_start) * 1000
    _log_request(
        config,
        prompt_tokens=prompt_tokens,
        output_tokens=total_completion,
        prefill_ms=prefill_delay * 1000,
        kv_transfer_ms=kv_delay * 1000,
        decode_ms=decode_delay * total_completion * 1000,
        total_ms=total_ms,
    )

    yield "data: [DONE]\n\n"
