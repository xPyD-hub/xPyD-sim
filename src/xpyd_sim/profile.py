"""Profile loader and runtime latency lookup from calibrated curves."""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml


class LatencyProfile:
    """Loads a calibrated profile.yaml and provides latency lookup methods."""

    def __init__(self, profile_path: str) -> None:
        with open(profile_path) as f:
            self._data: dict[str, Any] = yaml.safe_load(f)

    @property
    def has_prefill(self) -> bool:
        return "prefill" in self._data

    @property
    def has_kv_transfer(self) -> bool:
        return "kv_transfer" in self._data

    @property
    def has_decode(self) -> bool:
        return "decode" in self._data

    def prefill_delay_ms(self, batch_tokens: int) -> float:
        """Look up prefill delay from fitted 1D curve."""
        if not self.has_prefill:
            return 0.0
        coeffs = self._data["prefill"]["coefficients"]
        val = float(np.polyval(coeffs, batch_tokens))
        return max(0.0, val)

    def kv_transfer_delay_ms(self, batch_tokens: int) -> float:
        """Look up KV transfer delay from fitted 1D curve."""
        if not self.has_kv_transfer:
            return 0.0
        coeffs = self._data["kv_transfer"]["coefficients"]
        val = float(np.polyval(coeffs, batch_tokens))
        return max(0.0, val)

    def decode_delay_per_token_ms(self, batch_size: int, context_length: int) -> float:
        """Look up decode delay from fitted 2D surface."""
        if not self.has_decode:
            return 0.0
        coeffs = self._data["decode"]["coefficients"]
        bs = float(batch_size)
        ctx = float(context_length)
        val = (
            coeffs[0]
            + coeffs[1] * bs
            + coeffs[2] * ctx
            + coeffs[3] * bs * ctx
            + coeffs[4] * bs**2
            + coeffs[5] * ctx**2
        )
        return max(0.0, float(val))
