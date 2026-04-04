"""Test fix for issue #29: _compute_output_length with max_tokens=1."""

from __future__ import annotations

import random

from xpyd_sim.scheduler import InferenceRequest
from xpyd_sim.server import _compute_output_length


class TestComputeOutputLengthMaxTokens1:
    """Both server and scheduler should allow finish_reason='stop' when max_tokens=1."""

    def test_server_max_tokens_1_can_stop(self) -> None:
        """Server _compute_output_length with max_tokens=1 can return 'stop'."""
        random.seed(42)
        reasons = set()
        for _ in range(200):
            _, reason = _compute_output_length(
                max_tokens=1, eos_min_ratio=0.5, ignore_eos=False
            )
            reasons.add(reason)
        assert "stop" in reasons, "max_tokens=1 should sometimes produce 'stop'"
        assert "length" in reasons, "max_tokens=1 should sometimes produce 'length'"

    def test_scheduler_max_tokens_1_can_stop(self) -> None:
        """Scheduler InferenceRequest with max_tokens=1 can return 'stop'."""
        random.seed(42)
        reasons = set()
        for _ in range(200):
            req = InferenceRequest(
                request_id=f"test-{_}",
                input_tokens=10,
                max_tokens=1,
                eos_min_ratio=0.5,
                ignore_eos=False,
            )
            reasons.add(req.finish_reason)
        assert "stop" in reasons, "scheduler max_tokens=1 should sometimes produce 'stop'"
        assert "length" in reasons, "scheduler max_tokens=1 should sometimes produce 'length'"

    def test_ignore_eos_still_returns_length(self) -> None:
        """ignore_eos=True should always return 'length' regardless of max_tokens."""
        for _ in range(50):
            _, reason = _compute_output_length(
                max_tokens=1, eos_min_ratio=0.5, ignore_eos=True
            )
            assert reason == "length"

    def test_scheduler_ignore_eos_still_returns_length(self) -> None:
        """Scheduler ignore_eos=True should always return 'length'."""
        for _ in range(50):
            req = InferenceRequest(
                request_id=f"test-{_}",
                input_tokens=10,
                max_tokens=1,
                eos_min_ratio=0.5,
                ignore_eos=True,
            )
            assert req.finish_reason == "length"

    def test_scheduler_eos_on_last_token(self) -> None:
        """Scheduler should sometimes return 'stop' even when actual==max_tokens."""
        random.seed(0)
        reasons = set()
        for _ in range(500):
            req = InferenceRequest(
                request_id=f"test-{_}",
                input_tokens=10,
                max_tokens=10,
                eos_min_ratio=0.1,
                ignore_eos=False,
            )
            if req.target_output_tokens == req.max_tokens:
                reasons.add(req.finish_reason)
        assert "stop" in reasons, (
            "scheduler should sometimes return 'stop' even at max_tokens"
        )
