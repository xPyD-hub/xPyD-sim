"""Synthetic tool call generation."""

import json
import uuid
from typing import Any


def generate_dummy_args(schema: dict) -> dict:
    """Generate dummy arguments matching a JSON schema."""
    if not schema or schema.get("type") != "object":
        return {}
    props = schema.get("properties", {})
    result = {}
    for key, prop in props.items():
        ptype = prop.get("type", "string")
        if ptype == "string":
            enum = prop.get("enum")
            result[key] = enum[0] if enum else "dummy_value"
        elif ptype == "integer":
            result[key] = 42
        elif ptype == "number":
            result[key] = 3.14
        elif ptype == "boolean":
            result[key] = True
        elif ptype == "array":
            result[key] = []
        elif ptype == "object":
            result[key] = generate_dummy_args(prop)
    return result


def build_tool_calls(
    tools: list[dict],
    tool_choice: str | dict | None = None,
    parallel: bool | None = None,
) -> list[dict]:
    """Build synthetic tool call response from tool definitions."""
    selected = []
    if isinstance(tool_choice, dict):
        fname = tool_choice.get("function", {}).get("name", "")
        for t in tools:
            if t.get("type") == "function" and t.get("function", {}).get("name") == fname:
                selected.append(t)
                break
        if not selected and tools:
            selected.append(tools[0])
    elif tool_choice in ("required", "auto", None):
        if parallel and len(tools) > 1:
            selected = tools[:2]
        else:
            selected = tools[:1]

    result = []
    for t in selected:
        func = t.get("function", {})
        fname = func.get("name", "unknown")
        params = func.get("parameters", {})
        args = generate_dummy_args(params)
        result.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fname,
                "arguments": json.dumps(args),
            },
        })
    return result


def should_generate_tool_calls(
    tools: list | None, tool_choice: Any,
) -> bool:
    """Determine if we should generate tool calls instead of text."""
    return bool(tools) and tool_choice != "none"
