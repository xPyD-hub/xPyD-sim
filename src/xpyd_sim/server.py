"""Unified FastAPI server for xPyD-sim with mode-based latency."""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass

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


def _compute_output_length(
    max_tokens: int,
    eos_min_ratio: float,
    ignore_eos: bool,
) -> tuple[int, str]:
    """Return (num_tokens, finish_reason)."""
    if ignore_eos or max_tokens <= 1:
        return max_tokens, "length"
    min_len = max(1, math.ceil(max_tokens * eos_min_ratio))
    actual = random.randint(min_len, max_tokens)
    if actual < max_tokens:
        return actual, "stop"
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
    return config.prefill_delay_ms / 1000.0


def _compute_kv_delay(config: ServerConfig) -> float:
    """KV transfer delay in seconds based on mode."""
    if config.mode == "dual":
        return 0.0  # local, no transfer
    if config.mode == "prefill":
        return 0.0  # prefill doesn't wait for KV
    # decode mode: wait for KV
    return config.kv_transfer_delay_ms / 1000.0


def _compute_decode_delay(config: ServerConfig) -> float:
    """Per-token decode delay in seconds."""
    return config.decode_delay_per_token_ms / 1000.0


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create the unified xPyD-sim FastAPI application."""
    if config is None:
        config = ServerConfig()

    app = FastAPI(title="xPyD-sim")
    app.state.config = config

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
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {"message": "Invalid JSON body", "type": "invalid_request_error"}
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

        # Simulate prefill + KV transfer
        prefill_delay = _compute_prefill_delay(config, prompt_tokens)
        kv_delay = _compute_kv_delay(config)
        await asyncio.sleep(prefill_delay + kv_delay)

        if req.stream:
            return StreamingResponse(
                _stream_chat(config, req, prompt_tokens, max_tokens, n, ignore_eos),
                media_type="text/event-stream",
            )

        # Non-streaming
        choices = []
        total_completion = 0
        for i in range(n):
            num_tokens, finish_reason = _compute_output_length(
                max_tokens, config.eos_min_ratio, ignore_eos
            )
            text = render_dummy_text(num_tokens)
            text, stopped = _check_stop_sequences(text, req.stop)
            if stopped:
                finish_reason = "stop"
                num_tokens = max(1, len(text) // 4)
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

        # Simulate decode delay
        decode_delay = _compute_decode_delay(config)
        await asyncio.sleep(decode_delay * total_completion)

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

    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {"message": "Invalid JSON body", "type": "invalid_request_error"}
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

        # Simulate prefill + KV transfer
        prefill_delay = _compute_prefill_delay(config, prompt_tokens)
        kv_delay = _compute_kv_delay(config)
        await asyncio.sleep(prefill_delay + kv_delay)

        if req.stream:
            return StreamingResponse(
                _stream_completion(config, req, prompt_tokens, max_tokens, n, ignore_eos),
                media_type="text/event-stream",
            )

        # Non-streaming
        choices = []
        total_completion = 0
        for i in range(n):
            num_tokens, finish_reason = _compute_output_length(
                max_tokens, config.eos_min_ratio, ignore_eos
            )
            text = render_dummy_text(num_tokens)
            text, stopped = _check_stop_sequences(text, req.stop)
            if stopped:
                finish_reason = "stop"
                num_tokens = max(1, len(text) // 4)
            total_completion += num_tokens
            lp = None
            if req.logprobs is not None and req.logprobs > 0:
                tokens = list(text) if text else [""]
                lp = generate_completion_logprobs(tokens, req.logprobs)
            choices.append(
                CompletionChoice(
                    index=i,
                    text=text,
                    finish_reason=finish_reason,
                    logprobs=lp,
                )
            )

        # Simulate decode delay
        decode_delay = _compute_decode_delay(config)
        await asyncio.sleep(decode_delay * total_completion)

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

    return app


async def _stream_chat(
    config: ServerConfig,
    req: ChatCompletionRequest,
    prompt_tokens: int,
    max_tokens: int,
    n: int,
    ignore_eos: bool,
):
    """Stream chat completion chunks with per-token decode delay."""
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

        # Content chunks (per character as token proxy)
        emitted = ""
        for char in text:
            emitted += char
            # Check stop sequences
            if req.stop:
                _, was_stopped = _check_stop_sequences(emitted, req.stop)
                if was_stopped:
                    finish_reason = "stop"
                    break

            await asyncio.sleep(decode_delay)
            # Generate per-token logprobs for chat streaming
            chunk_lp = None
            if req.logprobs and req.top_logprobs and req.top_logprobs > 0:
                chunk_lp = generate_chat_logprobs([char], req.top_logprobs)
            chunk = ChatCompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    StreamChoice(
                        index=idx,
                        delta=DeltaMessage(content=char),
                        logprobs=chunk_lp,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        total_completion += len(emitted)

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

        emitted = ""
        for char in text:
            emitted += char
            if req.stop:
                _, was_stopped = _check_stop_sequences(emitted, req.stop)
                if was_stopped:
                    finish_reason = "stop"
                    break

            await asyncio.sleep(decode_delay)
            chunk_lp = None
            if req.logprobs is not None and req.logprobs > 0:
                chunk_lp = generate_completion_logprobs([char], req.logprobs)
            chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=config.model_name,
                choices=[
                    CompletionStreamChoice(
                        index=idx,
                        text=char,
                        logprobs=chunk_lp,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        total_completion += len(emitted)

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

    yield "data: [DONE]\n\n"
