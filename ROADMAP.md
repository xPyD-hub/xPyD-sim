# xPyD-sim Roadmap

Based on [DESIGN.md](docs/DESIGN.md). Finite loop — complete when all 56 test cases pass.

## M1: Project Skeleton + Common Models ✅
- Pydantic models for all OpenAI API request/response types
- Helper functions (token counting, ID generation, dummy text)
- Basic test suite for models and helpers

## M2: Core Server — Unified Engine ✅
- Single FastAPI server with mode config (dual/prefill/decode)
- Three latency parameters: prefill delay, KV transfer delay, decode delay
- Simple mode: fixed latency values
- /v1/completions, /v1/chat/completions, /v1/models, /health, /ping
- Streaming support with correct SSE format (role in first chunk, system_fingerprint)
- EOS simulation (random output length, eos_min_ratio, ignore_eos)
- TC1.x, TC2.x, TC3.x, TC4.x, TC5.x, TC6.x

## M3: Parameter Handling + Response Compliance ✅
- Parameter type validation (400 on wrong types)
- Logprobs data generation (fake but spec-compliant)
- Stop sequence truncation
- n > 1 support in streaming
- stream_options.include_usage
- max_model_len in model card
- TC6.x, TC11.x

## M4: CLI + YAML Configuration ✅
- CLI args for all settings
- YAML config file support (--config)
- CLI overrides YAML
- Environment variable fallbacks
- TC11.9, TC11.10, TC11.11

## M5: Calibrate Tool + Profile Mode ✅
- xpyd-sim calibrate command
- Load sample points, fit curves (scipy)
- Generate profile.yaml + visualization PNG
- Profile mode: load fitted curves at startup
- TC7.x

## M6: Observability ✅
- /metrics endpoint (Prometheus format)
- Request logging to JSONL
- /health returns mode + config
- Warm-up simulation
- TC10.x

## M7: End-to-End + Concurrency ✅
- Test with xPyD-proxy (PD disaggregation flow)
- TTFT/TPOT validation
- Concurrent request handling
- TC8.x, TC9.x

## M8: Backward Compatibility ✅
- Replace proxy dummy_nodes with xpyd-sim, all proxy tests pass
- Replace bench dummy server with xpyd-sim, all bench tests pass
- TC12.1, TC12.2

## Done Criteria
All 56 test cases pass → project complete.

## M9: Scheduling & Batching ✅
- Prefill scheduling: blocking batch formation with max_num_batched_tokens and max_num_seqs
- Decode scheduling: iteration-granularity batching with batch-in/batch-out
- Engine loop: unified prefill→decode pipeline with realistic scheduling
- /debug/batch endpoint for real-time batch state inspection
- Batch-aware metrics and request logging
- Scheduling config: max_model_len, max_num_batched_tokens, max_num_seqs
- TC13.1-TC13.12
