# xPyD-sim Design Principles

## Core Positioning
OpenAI-compatible LLM inference simulator. Provides dummy prefill and decode nodes for testing xPyD-bench and xPyD-proxy without real GPU hardware.

## Architecture
- Separate prefill and decode nodes — each runs as an independent service
- Shared common models — Pydantic models for all OpenAI API types
- Full OpenAI API spec compliance — every parameter accepted, response format exactly matches spec
- Content can be dummy/fake, but format must be spec-compliant

## Response Format Rules
- All response JSON structures must exactly match OpenAI API spec
- Streaming: first chat chunk delta must include role: "assistant"
- Streaming: final chunk should include usage when stream_options.include_usage is set
- All chunks must include system_fingerprint
- Logprobs field must be present (null if not requested)

## Parameter Handling
- Accept ALL OpenAI API parameters without errors
- Validation: reject wrong types
- Parameters like temperature, top_p don't need to affect output
- ignore_eos and other vLLM extensions should be supported

## Rules
- Committer must be hlin99 <tony.lin@intel.com>
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format
