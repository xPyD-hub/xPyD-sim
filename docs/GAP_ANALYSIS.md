# xPyD-sim Gap Analysis

Generated: 2026-04-06

## OpenAI Spec Gaps (Must Fix)

| # | Feature | Current State | Required | Difficulty |
|---|---|---|---|---|
| 1 | Parameter range validation (temperature, top_p, frequency_penalty, presence_penalty) | No validation — any value accepted silently | Return HTTP 400 for out-of-range values (temperature [0,2], top_p (0,1], frequency_penalty [-2,2], presence_penalty [-2,2]) | 简单 |
| 2 | `n` validation (n≤0) | No validation | Return HTTP 400 for n≤0 | 简单 |
| 3 | `response_format: json_object` | Field accepted but ignored — content is plain dummy text | Return valid JSON string as content | 中等 |
| 4 | `response_format: json_schema` | Field accepted but ignored | Return JSON conforming to provided schema | 复杂 |
| 5 | `response_format` in streaming | Not handled | Streamed content must assemble into valid JSON | 中等 |
| 6 | `encoding_format: base64` for embeddings | Field accepted but always returns float array | Return base64-encoded float vector when `encoding_format=base64` | 简单 |
| 7 | `best_of < n` validation | `best_of` exists on CompletionRequest but no cross-field validation | Return HTTP 400 when best_of < n | 简单 |

## vLLM Backend Gaps (Must Add)

| # | Feature | Current State | Required | Difficulty |
|---|---|---|---|---|
| 1 | Accept vLLM sampling params on ChatCompletionRequest | `ChatCompletionRequest` has no `extra="allow"` — unknown fields cause 422 | Add `model_config = {"extra": "allow"}` or explicit Optional fields for all vLLM sampling params (top_k, min_p, repetition_penalty, use_beam_search, etc.) | 简单 |
| 2 | Accept vLLM sampling params on CompletionRequest | `CompletionRequest` has no `extra="allow"` — unknown fields cause 422 | Add `model_config = {"extra": "allow"}` or explicit Optional fields | 简单 |
| 3 | Accept vLLM extra params (chat_template, documents, add_generation_prompt, priority, request_id, etc.) | Not accepted — 422 error | Accept without error on all request models | 简单 |
| 4 | `best_of` on ChatCompletionRequest | Only defined on CompletionRequest | Add `best_of` field to ChatCompletionRequest | 简单 |
| 5 | `echo` on ChatCompletionRequest | Only defined on CompletionRequest | Accept on chat endpoint too | 简单 |
| 6 | `stop_reason` in response choices | Not present in Choice/CompletionChoice models | Add `stop_reason: Optional[str] = None` to Choice, CompletionChoice, StreamChoice, CompletionStreamChoice | 简单 |
| 7 | `service_tier` in response objects | Not present in response models | Add `service_tier: Optional[str] = None` to ChatCompletionResponse, CompletionResponse, ChatCompletionChunk, CompletionChunk | 简单 |
| 8 | `kv_transfer_params` in response objects | Not present | Add `kv_transfer_params: Optional[dict] = None` to response models | 简单 |
| 9 | `prompt_logprobs` support | Not present — would 422 | Accept and return null in response | 简单 |

## Summary

- **OpenAI Spec Gaps**: 7 items (4 简单, 2 中等, 1 复杂)
- **vLLM Backend Gaps**: 9 items (all 简单)
- **Highest risk**: `response_format: json_schema` — requires parsing JSON Schema and generating conforming dummy data
- **Quick wins**: Parameter validation, `extra="allow"`, response field additions — can all be done in one PR
