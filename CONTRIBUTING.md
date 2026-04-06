# Contributing to xPyD-sim

## Development Setup

```bash
git clone https://github.com/xPyD-hub/xPyD-sim
cd xPyD-sim
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -q
```

## Code Style

- Python 3.10+
- Ruff: `ruff check xpyd_sim tests`
- All PRs must pass CI (lint + tests + integration trigger)

## Bot Development

See [bot/](bot/) for automated development policies.
