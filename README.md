# xPyD-sim

**OpenAI-compatible LLM inference simulator for testing and benchmarking.**

xPyD-sim simulates prefill and decode nodes with realistic latency behavior, enabling testing of xPyD-proxy and xPyD-bench without real GPU hardware.

## Key Features

- **Prefill/Decode simulation** — separate modes with configurable latency
- **Full OpenAI API** — /v1/completions, /v1/chat/completions, /v1/embeddings, /v1/models
- **vLLM compatible** — accepts all vLLM-specific parameters
- **Scheduling simulation** — batch formation, decode iteration, queue depth
- **Calibration tool** — fit latency curves from real hardware measurements
- **Prometheus metrics** — /metrics endpoint for monitoring

## Install

```bash
pip install xpyd-sim
```

Or as part of the full xPyD toolkit:

```bash
pip install xpyd
```

## Quick Start

```bash
# Start dual mode (prefill + decode)
xpyd-sim --mode dual --port 8000

# Start PD disaggregated
xpyd-sim --mode prefill --port 8001
xpyd-sim --mode decode --port 8002
```

## Part of xPyD

| Component | Description |
|-----------|-------------|
| [xpyd-proxy](https://github.com/xPyD-hub/xPyD-proxy) | PD-disaggregated proxy |
| **xpyd-sim** | OpenAI-compatible inference simulator |
| [xpyd-bench](https://github.com/xPyD-hub/xPyD-bench) | Benchmarking & planning tool |

📖 **[Full Guide →](docs/guide.md)** | 💡 **[Examples →](examples/)** | 🏗️ **[Contributing →](CONTRIBUTING.md)**

## License

Apache 2.0 — see [LICENSE](LICENSE)
