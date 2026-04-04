"""Test helper functions."""

from xpyd_sim.common.helpers import (
    count_prompt_tokens,
    generate_id,
    get_effective_max_tokens,
    now_ts,
    render_dummy_text,
)
from xpyd_sim.common.models import ChatMessage


def test_generate_id_prefix():
    assert generate_id().startswith("chatcmpl-")
    assert generate_id("cmpl").startswith("cmpl-")


def test_generate_id_unique():
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100


def test_now_ts():
    assert isinstance(now_ts(), int)
    assert now_ts() > 0


def test_effective_max_tokens():
    assert get_effective_max_tokens(None, 50, 100) == 50
    assert get_effective_max_tokens(None, None) == 16
    assert get_effective_max_tokens(42) == 42


def test_count_tokens_string():
    assert count_prompt_tokens(prompt="Hello world test") >= 1


def test_count_tokens_array():
    assert count_prompt_tokens(prompt=[1, 2, 3, 4, 5]) == 5


def test_count_tokens_messages():
    msgs = [ChatMessage(role="user", content="Hello world")]
    assert count_prompt_tokens(messages=msgs) >= 1


def test_count_tokens_none():
    assert count_prompt_tokens() == 1


def test_render_dummy_text():
    assert isinstance(render_dummy_text(10), str)
    assert len(render_dummy_text(10)) > 0
    assert render_dummy_text(0) == ""
