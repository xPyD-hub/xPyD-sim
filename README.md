# xPyD-sim

OpenAI-compatible LLM inference simulator for [xPyD](https://github.com/xPyD-hub).

Simulates prefill and decode nodes with realistic behavior for testing xPyD-bench and xPyD-proxy without real GPU hardware.

## Features

- Separate prefill and decode node simulators
- Full OpenAI API compatibility (/v1/completions, /v1/chat/completions, /v1/models)
- Configurable latency (prefill delay, decode delay per token)
- Streaming support with realistic token-by-token delivery
- EOS simulation with configurable output length distribution
- All OpenAI API parameters accepted
- Spec-compliant response formats

## Install

```bash
pip install xpyd-sim
```

## Quick Start

```bash
# Start prefill node
xpyd-sim prefill --port 8001

# Start decode node
xpyd-sim decode --port 8002
```

## License

TBD
