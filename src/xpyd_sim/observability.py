"""Observability: Prometheus metrics, JSONL request logging, warm-up tracking."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Metrics:
    """Thread-safe Prometheus-style metrics collector."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    requests_total: int = 0
    tokens_generated_total: int = 0
    active_requests: int = 0
    # Histogram buckets for request duration (seconds)
    _durations: list[float] = field(default_factory=list, repr=False)

    _BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def inc_requests(self) -> None:
        with self._lock:
            self.requests_total += 1

    def inc_tokens(self, n: int) -> None:
        with self._lock:
            self.tokens_generated_total += n

    def inc_active(self) -> None:
        with self._lock:
            self.active_requests += 1

    def dec_active(self) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    def observe_duration(self, seconds: float) -> None:
        with self._lock:
            self._durations.append(seconds)

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP xpyd_sim_requests_total Total requests received.")
            lines.append("# TYPE xpyd_sim_requests_total counter")
            lines.append(f"xpyd_sim_requests_total {self.requests_total}")

            lines.append("# HELP xpyd_sim_tokens_generated_total Total tokens generated.")
            lines.append("# TYPE xpyd_sim_tokens_generated_total counter")
            lines.append(f"xpyd_sim_tokens_generated_total {self.tokens_generated_total}")

            lines.append("# HELP xpyd_sim_active_requests Current concurrent requests.")
            lines.append("# TYPE xpyd_sim_active_requests gauge")
            lines.append(f"xpyd_sim_active_requests {self.active_requests}")

            lines.append(
                "# HELP xpyd_sim_request_duration_seconds Request latency histogram."
            )
            lines.append("# TYPE xpyd_sim_request_duration_seconds histogram")
            durations = list(self._durations)
            count = len(durations)
            total = sum(durations) if durations else 0.0
            for b in self._BUCKETS:
                bucket_count = sum(1 for d in durations if d <= b)
                lines.append(
                    f'xpyd_sim_request_duration_seconds_bucket{{le="{b}"}} {bucket_count}'
                )
            lines.append(
                f'xpyd_sim_request_duration_seconds_bucket{{le="+Inf"}} {count}'
            )
            lines.append(f"xpyd_sim_request_duration_seconds_sum {total:.6f}")
            lines.append(f"xpyd_sim_request_duration_seconds_count {count}")

            return "\n".join(lines) + "\n"


class RequestLogger:
    """Append-only JSONL request logger."""

    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

    def log(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            with open(self._path, "a") as f:
                f.write(line + "\n")


class WarmupTracker:
    """Track warm-up state and apply penalty."""

    def __init__(self, warmup_requests: int, warmup_penalty_ms: float) -> None:
        self._total = warmup_requests
        self._penalty_s = warmup_penalty_ms / 1000.0
        self._count = 0
        self._lock = threading.Lock()

    def get_penalty(self) -> float:
        """Return penalty in seconds if still warming up, else 0."""
        with self._lock:
            if self._count < self._total:
                self._count += 1
                return self._penalty_s
            return 0.0

    @property
    def is_warm(self) -> bool:
        with self._lock:
            return self._count >= self._total

    @property
    def requests_served(self) -> int:
        with self._lock:
            return self._count
