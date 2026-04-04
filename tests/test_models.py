"""Test request/response models."""

from xpyd_sim.common.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    ModelCard,
    ModelListResponse,
    StreamChoice,
    UsageInfo,
)


def test_chat_request_minimal():
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    assert req.model == "dummy"
    assert req.stream is False


def test_chat_request_all_params():
    req = ChatCompletionRequest(
        model="test", messages=[ChatMessage(role="user", content="Hi")],
        max_tokens=100, temperature=0.5, top_p=0.9, n=2, stream=True,
        stop=["end"], presence_penalty=0.1, frequency_penalty=0.2,
        seed=42, logprobs=True, top_logprobs=5, user="u1",
        response_format={"type": "json_object"}, ignore_eos=True,
    )
    assert req.n == 2
    assert req.ignore_eos is True


def test_completion_request_string_prompt():
    req = CompletionRequest(prompt="Hello")
    assert req.prompt == "Hello"


def test_completion_request_token_array():
    req = CompletionRequest(prompt=[1, 2, 3])
    assert req.prompt == [1, 2, 3]


def test_completion_request_all_params():
    req = CompletionRequest(
        prompt="Hi", max_tokens=50, echo=True, suffix=" end", best_of=3, seed=42,
    )
    assert req.echo is True
    assert req.suffix == " end"


def test_chat_response_structure():
    resp = ChatCompletionResponse(
        id="chatcmpl-abc", created=1000, model="test",
        choices=[Choice(message=ChoiceMessage(content="Hi"), finish_reason="stop")],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        system_fingerprint="fp_abc",
    )
    d = resp.model_dump()
    assert d["object"] == "chat.completion"
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert d["system_fingerprint"] == "fp_abc"
    assert "logprobs" in d["choices"][0]


def test_completion_response_structure():
    resp = CompletionResponse(
        id="cmpl-abc", created=1000, model="test",
        choices=[CompletionChoice(text="hello")],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )
    d = resp.model_dump()
    assert d["object"] == "text_completion"


def test_chat_chunk_with_role():
    chunk = ChatCompletionChunk(
        id="c", created=1, model="m",
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        system_fingerprint="fp",
    )
    d = chunk.model_dump()
    assert d["choices"][0]["delta"]["role"] == "assistant"
    assert d["system_fingerprint"] == "fp"


def test_chat_chunk_with_usage():
    chunk = ChatCompletionChunk(
        id="c", created=1, model="m",
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )
    assert chunk.usage.total_tokens == 15


def test_model_list():
    resp = ModelListResponse(data=[ModelCard(id="test", created=1)])
    d = resp.model_dump()
    assert d["object"] == "list"
    assert d["data"][0]["object"] == "model"
