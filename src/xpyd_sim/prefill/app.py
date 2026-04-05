"""FastAPI application for the prefill node simulator."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse

from xpyd_sim.common.helpers import (
    count_prompt_tokens,
    generate_id,
    get_effective_max_tokens,
    now_ts,
    render_dummy_text,
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
from xpyd_sim.observability import Metrics

SYSTEM_FINGERPRINT = "fp_xpyd_sim"

# Alias for prefill app metrics
_PrefillMetrics = Metrics


def create_prefill_app(
    model_name: str = "dummy-prefill",
    delay_per_token: Optional[float] = None,
    delay_fixed: Optional[float] = None,
) -> FastAPI:
    """Create a prefill node FastAPI app.

    Args:
        model_name: Model name to report in responses.
        delay_per_token: Per-token delay in seconds (applied to prompt tokens).
        delay_fixed: Fixed delay in seconds (used if delay_per_token is None).
    """
    app = FastAPI(title="xPyD-sim Prefill Node")
    metrics = _PrefillMetrics()

    async def _simulate_delay(prompt_tokens: int) -> None:
        if delay_per_token is not None:
            await asyncio.sleep(delay_per_token * prompt_tokens)
        elif delay_fixed is not None:
            await asyncio.sleep(delay_fixed)

    @app.get("/ping")
    @app.post("/ping")
    async def ping():
        return PlainTextResponse("pong")

    @app.get("/metrics")
    async def metrics_endpoint():
        return PlainTextResponse(
            metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelListResponse:
        card = ModelCard(id=model_name, created=now_ts())
        return ModelListResponse(data=[card])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        prompt_tokens = count_prompt_tokens(messages=request.messages)
        max_tokens = get_effective_max_tokens(
            request.max_completion_tokens, request.max_tokens
        )
        n = request.n or 1

        await _simulate_delay(prompt_tokens)

        if request.stream:
            return StreamingResponse(
                _stream_chat(request, prompt_tokens, max_tokens, n),
                media_type="text/event-stream",
            )

        text = render_dummy_text(max_tokens)
        completion_tokens = max_tokens * n
        choices = [
            Choice(
                index=i,
                message=ChoiceMessage(role="assistant", content=text),
                finish_reason="stop",
                logprobs=None,
            )
            for i in range(n)
        ]
        return ChatCompletionResponse(
            id=generate_id("chatcmpl"),
            created=now_ts(),
            model=model_name,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            system_fingerprint=SYSTEM_FINGERPRINT,
        )

    async def _stream_chat(
        request: ChatCompletionRequest,
        prompt_tokens: int,
        max_tokens: int,
        n: int,
    ):
        req_id = generate_id("chatcmpl")
        created = now_ts()
        include_usage = (
            request.stream_options.get("include_usage", False)
            if request.stream_options
            else False
        )
        text = render_dummy_text(max_tokens)

        for idx in range(n):
            # First chunk: role only
            chunk = ChatCompletionChunk(
                id=req_id,
                created=created,
                model=model_name,
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

            # Content chunks
            for char in text:
                chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            index=idx,
                            delta=DeltaMessage(content=char),
                            logprobs=None,
                        )
                    ],
                    system_fingerprint=SYSTEM_FINGERPRINT,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk: finish_reason
            chunk = ChatCompletionChunk(
                id=req_id,
                created=created,
                model=model_name,
                choices=[
                    StreamChoice(
                        index=idx,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Usage chunk
        if include_usage:
            completion_tokens = max_tokens * n
            usage_chunk = ChatCompletionChunk(
                id=req_id,
                created=created,
                model=model_name,
                choices=[],
                system_fingerprint=SYSTEM_FINGERPRINT,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt_tokens = count_prompt_tokens(prompt=request.prompt)
        max_tokens = get_effective_max_tokens(request.max_tokens)
        n = request.n or 1

        await _simulate_delay(prompt_tokens)

        if request.stream:
            return StreamingResponse(
                _stream_completion(request, prompt_tokens, max_tokens, n),
                media_type="text/event-stream",
            )

        text = render_dummy_text(max_tokens)
        completion_tokens = max_tokens * n
        choices = [
            CompletionChoice(
                index=i,
                text=text,
                finish_reason="stop",
                logprobs=None,
            )
            for i in range(n)
        ]
        return CompletionResponse(
            id=generate_id("cmpl"),
            created=now_ts(),
            model=model_name,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            system_fingerprint=SYSTEM_FINGERPRINT,
        )

    async def _stream_completion(
        request: CompletionRequest,
        prompt_tokens: int,
        max_tokens: int,
        n: int,
    ):
        req_id = generate_id("cmpl")
        created = now_ts()
        include_usage = (
            request.stream_options.get("include_usage", False)
            if request.stream_options
            else False
        )
        text = render_dummy_text(max_tokens)

        for idx in range(n):
            # Content chunks
            for char in text:
                chunk = CompletionChunk(
                    id=req_id,
                    created=created,
                    model=model_name,
                    choices=[
                        CompletionStreamChoice(
                            index=idx,
                            text=char,
                            logprobs=None,
                        )
                    ],
                    system_fingerprint=SYSTEM_FINGERPRINT,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk: finish_reason
            chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=model_name,
                choices=[
                    CompletionStreamChoice(
                        index=idx,
                        text="",
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                system_fingerprint=SYSTEM_FINGERPRINT,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Usage chunk
        if include_usage:
            completion_tokens = max_tokens * n
            usage_chunk = CompletionChunk(
                id=req_id,
                created=created,
                model=model_name,
                choices=[],
                system_fingerprint=SYSTEM_FINGERPRINT,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    return app
