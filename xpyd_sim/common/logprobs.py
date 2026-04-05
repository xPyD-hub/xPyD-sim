"""Fake logprobs data generation for OpenAI API compliance."""

from __future__ import annotations

import random

_VOCAB = [
    "The", "A", "In", "It", "On", "He", "She", "We", "They", "I",
    "is", "was", "are", "were", "be", "been", "have", "has", "had",
    "and", "but", "or", "so", "yet", "for", "nor", "not", "no", "yes",
    "the", "a", "an", "this", "that", "these", "those", "my", "your",
    "of", "to", "in", "on", "at", "by", "with", "from", "up", "about",
    "quick", "brown", "fox", "lazy", "dog", "over", "jumps", "big",
]


def _random_logprob() -> float:
    return -random.uniform(0.01, 5.0)


def generate_completion_logprobs(tokens: list[str], num_top: int) -> dict:
    """Generate fake logprobs for /v1/completions format."""
    token_logprobs = []
    top_logprobs_list = []
    text_offset = []
    offset = 0

    for token in tokens:
        main_lp = round(_random_logprob(), 4)
        token_logprobs.append(main_lp)
        top = {token: main_lp}
        candidates = [t for t in _VOCAB if t != token]
        random.shuffle(candidates)
        for alt in candidates[: num_top - 1]:
            top[alt] = round(main_lp - random.uniform(0.1, 3.0), 4)
        top_logprobs_list.append(top)
        text_offset.append(offset)
        offset += len(token)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs_list,
        "text_offset": text_offset,
    }


def generate_chat_logprobs(tokens: list[str], num_top: int) -> dict:
    """Generate fake logprobs for /v1/chat/completions format."""
    content = []
    for token in tokens:
        main_lp = round(_random_logprob(), 4)
        top = [
            {"token": token, "logprob": main_lp, "bytes": list(token.encode("utf-8"))}
        ]
        candidates = [t for t in _VOCAB if t != token]
        random.shuffle(candidates)
        for alt in candidates[: num_top - 1]:
            alt_lp = round(main_lp - random.uniform(0.1, 3.0), 4)
            top.append(
                {"token": alt, "logprob": alt_lp, "bytes": list(alt.encode("utf-8"))}
            )
        content.append({
            "token": token,
            "logprob": main_lp,
            "bytes": list(token.encode("utf-8")),
            "top_logprobs": top,
        })
    return {"content": content}


def tokenize_text(text: str) -> list[str]:
    """Simple word-based tokenization."""
    if not text:
        return []
    tokens = []
    for word in text.split(" "):
        if word:
            tokens.append(word)
            tokens.append(" ")
    if tokens and tokens[-1] == " ":
        tokens.pop()
    return tokens if tokens else [text]
