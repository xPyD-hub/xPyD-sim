"""Shared helper functions."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

DUMMY_TOKENS = list("The quick brown fox jumps over the lazy dog. " * 20)
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
        total = sum(
            len(str(getattr(m, "content", ""))) + len(getattr(m, "role", ""))
            for m in messages
        )
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
    return "".join(DUMMY_TOKENS[: min(n_tokens, len(DUMMY_TOKENS))])
