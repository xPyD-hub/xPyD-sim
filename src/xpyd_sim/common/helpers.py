"""Shared helper functions."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

DUMMY_TOKENS = ("The quick brown fox jumps over the lazy dog. " * 20).split()
DEFAULT_MAX_TOKENS = 16


def generate_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def now_ts() -> int:
    return int(time.time())


def get_effective_max_tokens(*values: Optional[int]) -> int:
    for v in values:
        if v is not None:
            return v
    return DEFAULT_MAX_TOKENS


def count_prompt_tokens(prompt: Any = None, messages: list | None = None) -> int:
    if messages is not None:
        total = 0
        for m in messages:
            content = getattr(m, "content", None)
            role = getattr(m, "role", "") or ""
            if content is None:
                pass
            elif isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += len(part.get("text", ""))
                    elif isinstance(part, dict):
                        total += 10
            else:
                total += len(str(content))
            total += len(role)
        return max(1, total // 4)
    if prompt is None:
        return 1
    if isinstance(prompt, str):
        return max(1, len(prompt) // 4)
    if isinstance(prompt, list):
        if all(isinstance(i, int) for i in prompt):
            return len(prompt)
        return max(1, sum(len(str(i)) for i in prompt) // 4)
    return max(1, len(str(prompt)) // 4)


def render_dummy_text(n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    if n_tokens <= len(DUMMY_TOKENS):
        return " ".join(DUMMY_TOKENS[:n_tokens])
    # Cycle through DUMMY_TOKENS to produce enough tokens
    tokens: list[str] = []
    pool_len = len(DUMMY_TOKENS)
    for i in range(n_tokens):
        tokens.append(DUMMY_TOKENS[i % pool_len])
    return " ".join(tokens)
