"""Test tool calling support for vLLM compatibility."""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        },
    },
]

def _cfg():
    return ServerConfig(mode="dual", prefill_delay_ms=0, decode_delay_per_token_ms=0)

class TestToolCalling:
    def test_tool_calls_in_response(self):
        """When tools provided with tool_choice=required, response has tool_calls."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "weather in tokyo"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "max_tokens": 50,
            })
            assert r.status_code == 200
            d = r.json()
            choice = d["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["content"] is None
            assert len(choice["message"]["tool_calls"]) >= 1
            tc = choice["message"]["tool_calls"][0]
            assert tc["type"] == "function"
            assert tc["function"]["name"] == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert "location" in args
            assert tc["id"].startswith("call_")

    def test_tool_choice_none_returns_text(self):
        """tool_choice=none should return normal text, no tool_calls."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
                "tools": TOOLS,
                "tool_choice": "none",
                "max_tokens": 10,
            })
            d = r.json()
            assert d["choices"][0]["finish_reason"] in ("stop", "length")
            assert d["choices"][0]["message"]["content"] is not None
            assert d["choices"][0]["message"].get("tool_calls") is None

    def test_tool_choice_specific_function(self):
        """tool_choice with specific function name."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "find info"}],
                "tools": TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "search"}},
                "max_tokens": 50,
            })
            d = r.json()
            tc = d["choices"][0]["message"]["tool_calls"][0]
            assert tc["function"]["name"] == "search"

    def test_no_tools_returns_text(self):
        """Without tools, normal text response."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
            })
            d = r.json()
            assert d["choices"][0]["message"]["content"] is not None
            assert d["choices"][0]["message"].get("tool_calls") is None

    def test_streaming_tool_calls(self):
        """Streaming with tools and tool_choice=required should include tool_calls in delta."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "weather"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "max_tokens": 50,
                "stream": True,
            })
            assert r.status_code == 200
            chunks = [json.loads(line[6:]) for line in r.text.split("\n")
                      if line.startswith("data: ") and line != "data: [DONE]"]
            finish_reasons = [c["choices"][0].get("finish_reason") for c in chunks if c["choices"]]
            assert "tool_calls" in finish_reasons
            has_tool_delta = any(
                c["choices"][0].get("delta", {}).get("tool_calls") is not None
                for c in chunks if c["choices"]
            )
            assert has_tool_delta

    def test_usage_with_tools(self):
        """Usage should report completion_tokens for tool calls."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "weather"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "max_tokens": 50,
            })
            u = r.json()["usage"]
            assert u["completion_tokens"] > 0
            assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_tool_call_arguments_valid_json(self):
        """Tool call arguments should be valid JSON."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "weather"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "max_tokens": 50,
            })
            tc = r.json()["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            assert isinstance(args, dict)

    def test_tool_choice_auto_can_return_text(self):
        """tool_choice=auto (or None) should sometimes return text."""
        with patch("xpyd_sim.common.tools.random.random", return_value=0.9):
            with TestClient(create_app(_cfg())) as c:
                r = c.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": TOOLS,
                    "max_tokens": 10,
                })
                d = r.json()
                choice = d["choices"][0]
                assert choice["message"]["content"] is not None
                assert choice["message"].get("tool_calls") is None
                assert choice["finish_reason"] in ("stop", "length")

    def test_tool_choice_required_always_returns_tool_calls(self):
        """tool_choice=required should always generate tool_calls."""
        with TestClient(create_app(_cfg())) as c:
            for _ in range(5):
                r = c.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": TOOLS,
                    "tool_choice": "required",
                    "max_tokens": 50,
                })
                d = r.json()
                assert d["choices"][0]["finish_reason"] == "tool_calls"
                assert d["choices"][0]["message"]["tool_calls"] is not None

    def test_n_greater_than_one_consistent_choices(self):
        """With n>1, all choices should be the same type (all tool_calls or all text)."""
        with TestClient(create_app(_cfg())) as c:
            # required → all choices should have tool_calls
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "weather"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "n": 3,
                "max_tokens": 50,
            })
            d = r.json()
            assert len(d["choices"]) == 3
            for choice in d["choices"]:
                assert choice["finish_reason"] == "tool_calls"
                assert choice["message"]["tool_calls"] is not None
                assert choice["message"]["content"] is None

    def test_n_greater_than_one_auto_consistent(self):
        """With n>1 and auto, all choices must be same type."""
        # Force tool_calls path
        with patch("xpyd_sim.server.should_generate_tool_calls", return_value=True):
            with TestClient(create_app(_cfg())) as c:
                r = c.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": TOOLS,
                    "n": 3,
                    "max_tokens": 50,
                })
                d = r.json()
                assert len(d["choices"]) == 3
                types = set()
                for choice in d["choices"]:
                    if choice["message"].get("tool_calls"):
                        types.add("tool_calls")
                    else:
                        types.add("text")
                assert len(types) == 1, f"Mixed choice types: {types}"

        # Force text path
        with patch("xpyd_sim.server.should_generate_tool_calls", return_value=False):
            with TestClient(create_app(_cfg())) as c:
                r = c.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": TOOLS,
                    "n": 3,
                    "max_tokens": 50,
                })
                d = r.json()
                assert len(d["choices"]) == 3
                for choice in d["choices"]:
                    assert choice["message"]["content"] is not None
                    assert choice["message"].get("tool_calls") is None

    def test_n_greater_than_one_unique_call_ids(self):
        """With n>1, each choice should have unique tool call IDs."""
        with TestClient(create_app(_cfg())) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "weather"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "n": 3,
                "max_tokens": 50,
            })
            d = r.json()
            all_ids = []
            for choice in d["choices"]:
                for tc in choice["message"]["tool_calls"]:
                    all_ids.append(tc["id"])
            assert len(all_ids) == len(set(all_ids)), f"Duplicate IDs: {all_ids}"
