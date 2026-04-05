"""Scheduling & batching engine for realistic LLM inference simulation.

Implements prefill scheduling (blocking batch formation) and decode scheduling
(iteration-granularity batching with batch-in/batch-out).
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from xpyd_sim.profile import LatencyProfile


@dataclass
class SchedulingConfig:
    """Scheduling parameters."""

    max_model_len: int = 131072
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enabled: bool = False  # When False, use legacy per-request simulation


@dataclass
class InferenceRequest:
    """A request tracked by the scheduler."""

    request_id: str
    input_tokens: int
    max_tokens: int
    eos_min_ratio: float = 0.5
    ignore_eos: bool = False
    # Computed at creation
    target_output_tokens: int = 0
    finish_reason: str = "length"
    # Runtime state
    generated_tokens: int = 0
    context_length: int = 0  # input_tokens + generated_tokens
    # Streaming output queue
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Completion event
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Timestamps for logging
    created_at: float = 0.0
    prefill_start_at: float = 0.0
    prefill_done_at: float = 0.0
    decode_join_at: float = 0.0
    decode_done_at: float = 0.0

    def __post_init__(self) -> None:
        if self.target_output_tokens == 0:
            self._compute_output_length()
        if self.context_length == 0:
            self.context_length = self.input_tokens
        if self.created_at == 0.0:
            self.created_at = time.monotonic()

    def _compute_output_length(self) -> None:
        mt = self.max_tokens
        if self.ignore_eos:
            self.target_output_tokens = mt
            self.finish_reason = "length"
            return
        if mt <= 1:
            self.target_output_tokens = mt
            self.finish_reason = (
                "stop" if random.random() < self.eos_min_ratio else "length"
            )
            return
        min_len = max(1, math.ceil(mt * self.eos_min_ratio))
        actual = random.randint(min_len, mt)
        if actual < mt:
            self.target_output_tokens = actual
            self.finish_reason = "stop"
        elif random.random() < (1.0 - self.eos_min_ratio):
            self.target_output_tokens = mt
            self.finish_reason = "stop"
        else:
            self.target_output_tokens = mt
            self.finish_reason = "length"

    def is_done(self) -> bool:
        return self.generated_tokens >= self.target_output_tokens


class Scheduler:
    """Manages prefill and decode scheduling with realistic batching behavior.

    The engine runs as a background asyncio task. Requests are submitted via
    submit() and tokens are consumed from InferenceRequest.token_queue.
    """

    def __init__(
        self,
        config: SchedulingConfig,
        prefill_delay_ms: float = 50.0,
        kv_transfer_delay_ms: float = 5.0,
        decode_delay_per_token_ms: float = 10.0,
        mode: str = "dual",
        latency_profile: LatencyProfile | None = None,
        log_callback: Any = None,
        metrics_callback: Any = None,
    ) -> None:
        self.config = config
        self.prefill_delay_ms = prefill_delay_ms
        self.kv_transfer_delay_ms = kv_transfer_delay_ms
        self.decode_delay_per_token_ms = decode_delay_per_token_ms
        self.mode = mode
        self.latency_profile = latency_profile
        self.log_callback = log_callback
        self.metrics_callback = metrics_callback

        # Queues
        self._prefill_queue: list[InferenceRequest] = []
        self._decode_waiting: list[InferenceRequest] = []
        self._decode_batch: list[InferenceRequest] = []
        self._prefill_batch: list[InferenceRequest] = []

        # Lock for queue manipulation
        self._lock = asyncio.Lock()
        self._new_request_event = asyncio.Event()

        # Engine task
        self._engine_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the engine loop."""
        self._running = True
        self._engine_task = asyncio.create_task(self._engine_loop())

    async def stop(self) -> None:
        """Stop the engine loop."""
        self._running = False
        self._new_request_event.set()
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass

    async def submit(self, req: InferenceRequest) -> None:
        """Submit a request to the prefill queue.

        Raises ValueError if input_tokens > max_model_len.
        """
        if req.input_tokens > self.config.max_model_len:
            raise ValueError(
                f"Input length {req.input_tokens} exceeds max_model_len "
                f"{self.config.max_model_len}"
            )
        async with self._lock:
            self._prefill_queue.append(req)
        self._new_request_event.set()

    def get_batch_state(self) -> dict[str, Any]:
        """Return current batch state for /debug/batch."""
        decode_requests = []
        for r in self._decode_batch:
            decode_requests.append({
                "id": r.request_id,
                "input_tokens": r.input_tokens,
                "generated_tokens": r.generated_tokens,
                "context_length": r.context_length,
            })

        avg_ctx = 0.0
        if self._decode_batch:
            avg_ctx = sum(r.context_length for r in self._decode_batch) / len(
                self._decode_batch
            )

        return {
            "prefill_queue_depth": len(self._prefill_queue),
            "prefill_batch_size": len(self._prefill_batch),
            "prefill_batch_tokens": sum(r.input_tokens for r in self._prefill_batch),
            "decode_batch_size": len(self._decode_batch),
            "decode_avg_context_length": round(avg_ctx, 1),
            "decode_requests": decode_requests,
        }

    def _compute_prefill_delay_s(self, batch_tokens: int) -> float:
        """Compute prefill delay in seconds."""
        if self.mode == "decode":
            return 0.0
        if self.latency_profile and self.latency_profile.has_prefill:
            return self.latency_profile.prefill_delay_ms(batch_tokens) / 1000.0
        return self.prefill_delay_ms / 1000.0

    def _compute_kv_delay_s(self, tokens: int) -> float:
        """Compute KV transfer delay in seconds."""
        if self.mode == "dual":
            return 0.0
        if self.mode == "prefill":
            return 0.0
        if self.latency_profile and self.latency_profile.has_kv_transfer:
            return self.latency_profile.kv_transfer_delay_ms(tokens) / 1000.0
        return self.kv_transfer_delay_ms / 1000.0

    def _compute_decode_delay_s(self, batch_size: int, avg_context_length: int) -> float:
        """Compute per-token decode delay in seconds."""
        if self.latency_profile and self.latency_profile.has_decode:
            return (
                self.latency_profile.decode_delay_per_token_ms(batch_size, avg_context_length)
                / 1000.0
            )
        return self.decode_delay_per_token_ms / 1000.0

    def _log_event(self, event: dict) -> None:
        if self.log_callback:
            self.log_callback(event)

    async def _engine_loop(self) -> None:
        """Main engine loop: prefill then decode, repeat."""
        while self._running:
            did_work = False

            # --- Prefill phase ---
            async with self._lock:
                if self._prefill_queue and not self._prefill_batch:
                    batch, remaining = self._form_prefill_batch(self._prefill_queue)
                    self._prefill_queue = remaining
                    self._prefill_batch = batch

            if self._prefill_batch:
                did_work = True
                batch_tokens = sum(r.input_tokens for r in self._prefill_batch)
                for r in self._prefill_batch:
                    r.prefill_start_at = time.monotonic()
                    self._log_event({
                        "event": "prefill_start",
                        "request_id": r.request_id,
                        "batch_size": len(self._prefill_batch),
                        "batch_tokens": batch_tokens,
                        "queue_depth": len(self._prefill_queue),
                    })

                delay = self._compute_prefill_delay_s(batch_tokens)
                await asyncio.sleep(delay)

                for r in self._prefill_batch:
                    r.prefill_done_at = time.monotonic()
                    prefill_ms = (r.prefill_done_at - r.prefill_start_at) * 1000
                    self._log_event({
                        "event": "prefill_done",
                        "request_id": r.request_id,
                        "prefill_ms": round(prefill_ms, 2),
                    })

                # Move to decode waiting (with KV transfer delay)
                kv_delay = self._compute_kv_delay_s(batch_tokens)
                if kv_delay > 0:
                    await asyncio.sleep(kv_delay)

                async with self._lock:
                    self._decode_waiting.extend(self._prefill_batch)
                    self._prefill_batch = []

            # --- Decode phase ---
            # Batch in: add waiting requests
            async with self._lock:
                while self._decode_waiting and len(self._decode_batch) < self.config.max_num_seqs:
                    r = self._decode_waiting.pop(0)
                    self._decode_batch.append(r)
                    r.decode_join_at = time.monotonic()
                    self._log_event({
                        "event": "decode_join",
                        "request_id": r.request_id,
                        "decode_batch_size": len(self._decode_batch),
                    })

            if self._decode_batch:
                did_work = True
                batch_size = len(self._decode_batch)
                avg_ctx = int(
                    sum(r.context_length for r in self._decode_batch) / batch_size
                )
                delay = self._compute_decode_delay_s(batch_size, avg_ctx)
                await asyncio.sleep(delay)

                # Generate 1 token for all
                completed = []
                for r in self._decode_batch:
                    r.generated_tokens += 1
                    r.context_length += 1
                    # Put a token signal on the queue
                    await r.token_queue.put(("token", r.generated_tokens))
                    self._log_event({
                        "event": "decode_token",
                        "request_id": r.request_id,
                        "token_idx": r.generated_tokens,
                        "batch_size": batch_size,
                        "context_len": r.context_length,
                        "delay_ms": round(delay * 1000, 2),
                    })
                    if r.is_done():
                        completed.append(r)

                # Batch out
                async with self._lock:
                    for r in completed:
                        self._decode_batch.remove(r)
                        r.decode_done_at = time.monotonic()
                        await r.token_queue.put(("done", r.finish_reason))
                        r.done_event.set()
                        self._log_event({
                            "event": "decode_done",
                            "request_id": r.request_id,
                            "reason": r.finish_reason,
                            "total_tokens": r.generated_tokens,
                        })

            if not did_work:
                # Wait for new requests or a short timeout
                self._new_request_event.clear()
                try:
                    await asyncio.wait_for(self._new_request_event.wait(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass

    def _form_prefill_batch(
        self, queue: list[InferenceRequest]
    ) -> tuple[list[InferenceRequest], list[InferenceRequest]]:
        """Form a prefill batch from the queue. Returns (batch, remaining)."""
        batch: list[InferenceRequest] = []
        remaining: list[InferenceRequest] = []
        current_tokens = 0

        for i, req in enumerate(queue):
            if req.input_tokens > self.config.max_model_len:
                # Reject: signal error directly (put_nowait and set are sync-safe)
                req.token_queue.put_nowait(
                    (
                        "error",
                        f"Input length {req.input_tokens} exceeds "
                        f"max_model_len {self.config.max_model_len}",
                    )
                )
                req.done_event.set()
                continue
            if current_tokens + req.input_tokens > self.config.max_num_batched_tokens:
                # FIFO: stop here, all subsequent requests stay in queue
                remaining.extend(queue[i:])
                break
            if len(batch) >= self.config.max_num_seqs:
                remaining.extend(queue[i:])
                break
            batch.append(req)
            current_tokens += req.input_tokens

        return batch, remaining
