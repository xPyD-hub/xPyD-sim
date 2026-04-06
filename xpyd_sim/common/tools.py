"""Synthetic tool call generation."""

import json
import random
import uuid
from typing import Any


def generate_dummy_from_schema(schema: dict):
    """Generate dummy values matching a JSON schema.

    Handles: string, integer, number, boolean, array, object.
    Supports enum (picks first value) and items (array elements).
    Does not handle $ref / anyOf / oneOf.
    """
    if not schema:
        return {}
    schema_type = schema.get("type", "object")
    enum = schema.get("enum")
    if enum:
        return enum[0]
    if schema_type == "string":
        return "dummy_value"
    if schema_type == "integer":
        return 42
    if schema_type == "number":
        return 3.14
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        items = schema.get("items")
        if items:
            return [generate_dummy_from_schema(items)]
        return []
    if schema_type == "object":
        props = schema.get("properties", {})
        result = {}
        for key, prop in props.items():
            result[key] = generate_dummy_from_schema(prop)
        return result
    return None


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
        args = generate_dummy_from_schema(params)
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
    if not tools or tool_choice == "none":
        return False
    # Specific function dict or "required" → always generate
    if isinstance(tool_choice, dict) or tool_choice == "required":
        return True
    # "auto" or None → 50% chance (realistic model behavior)
    return random.random() < 0.5
