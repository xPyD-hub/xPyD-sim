# xPyD-sim Roadmap

## M1: Project Skeleton + Common Models ✅
- Pydantic models for all OpenAI API request/response types
- Helper functions (token counting, ID generation, dummy text)
- Basic test suite for models and helpers

## M2: Prefill Node Simulator ⬜
- FastAPI app for prefill node
- /v1/completions, /v1/chat/completions, /v1/models, /health
- Configurable prefill delay (per-token or fixed)
- Streaming support with correct SSE format
- All OpenAI parameters accepted without error
- Tests with httpx TestClient

## M3: Decode Node Simulator ⬜
- FastAPI app for decode node
- Configurable decode delay per token
- EOS simulation (random output length, configurable min ratio)
- ignore_eos support
- Streaming with correct format (role in first chunk, usage in final)
- Tests

## M4: CLI ⬜
- xpyd-sim prefill / xpyd-sim decode commands
- Configuration via CLI args and/or YAML
- xpyd-sim all to run both in one process

## M5: Full OpenAI Spec Compliance ⬜
- Validate response format against OpenAI spec
- system_fingerprint in all responses
- logprobs field in choices (null when not requested)
- stream_options.include_usage support
- n > 1 in streaming mode
- Integration tests with openai Python client

## M6: Consolidate from xPyD-bench and xPyD-proxy ⬜
- Merge best features from bench dummy and proxy dummy_nodes
- Ensure backward compatibility with both projects
- Migration guide
