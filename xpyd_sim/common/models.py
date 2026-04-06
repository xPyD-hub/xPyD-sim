"""Pydantic models for OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "dummy"
    messages: list[ChatMessage] = []
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    user: Optional[str] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    stream_options: Optional[dict] = None
    ignore_eos: Optional[bool] = None
    best_of: Optional[int] = None
    echo: Optional[bool] = False

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    model: str = "dummy"
    prompt: Any = ""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    suffix: Optional[str] = None
    best_of: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    stream_options: Optional[dict] = None
    ignore_eos: Optional[bool] = None

    model_config = {"extra": "allow"}


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallFunction(BaseModel):
    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function: ToolCallFunction = Field(default_factory=ToolCallFunction)


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage = Field(default_factory=ChoiceMessage)
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Any] = None
    stop_reason: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = "dummy"
    choices: list[Choice] = []
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    kv_transfer_params: Optional[dict] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Any] = None
    stop_reason: Optional[Any] = None


class CompletionResponse(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = "dummy"
    choices: list[CompletionChoice] = []
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    kv_transfer_params: Optional[dict] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    stop_reason: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "dummy"
    choices: list[StreamChoice] = []
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None
    service_tier: Optional[str] = None


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    stop_reason: Optional[Any] = None


class CompletionChunk(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = "dummy"
    choices: list[CompletionStreamChoice] = []
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None
    service_tier: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float] | str = []


class EmbeddingRequest(BaseModel):
    model: str = "dummy"
    input: Any = ""
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

    model_config = {"extra": "allow"}


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData] = []
    model: str = "dummy"
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "xpyd-sim"
    root: Optional[str] = None
    max_model_len: Optional[int] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []
