"""Calibrate tool: fit latency curves from sample points."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import yaml


def _fit_1d(points: list[dict[str, float]], x_key: str, y_key: str) -> dict[str, Any]:
    """Fit a 1D polynomial (degree 2) to sample points.

    Returns coefficients for: y = a*x^2 + b*x + c
    """
    xs = np.array([p[x_key] for p in points], dtype=np.float64)
    ys = np.array([p[y_key] for p in points], dtype=np.float64)
    degree = min(2, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, degree)
    # Pad to always have 3 coefficients (a, b, c)
    padded = [0.0] * (3 - len(coeffs)) + list(coeffs)
    return {
        "type": "poly1d",
        "coefficients": [float(c) for c in padded],
        "x_key": x_key,
        "y_key": y_key,
        "x_range": [float(xs.min()), float(xs.max())],
    }


def _fit_2d(points: list[dict[str, float]]) -> dict[str, Any]:
    """Fit a 2D surface: delay = a + b*bs + c*ctx + d*bs*ctx + e*bs^2 + f*ctx^2.

    Uses least-squares on polynomial features.
    """
    bs = np.array([p["batch_size"] for p in points], dtype=np.float64)
    ctx = np.array([p["context_length"] for p in points], dtype=np.float64)
    y = np.array([p["delay_per_token_ms"] for p in points], dtype=np.float64)

    # Build design matrix: [1, bs, ctx, bs*ctx, bs^2, ctx^2]
    X = np.column_stack([  # noqa: N806
        np.ones_like(bs),
        bs,
        ctx,
        bs * ctx,
        bs**2,
        ctx**2,
    ])

    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "type": "poly2d",
        "coefficients": [float(c) for c in coeffs],
        "bs_range": [float(bs.min()), float(bs.max())],
        "ctx_range": [float(ctx.min()), float(ctx.max())],
    }


def _validate_points(points: list[dict], min_count: int, label: str) -> None:
    """Validate minimum number of sample points."""
    if len(points) < min_count:
        print(
            f"Error: {label} requires at least {min_count} sample points, got {len(points)}",
            file=sys.stderr,
        )
        sys.exit(1)


def calibrate(input_path: str, output_path: str, plot_path: str | None = None) -> None:
    """Run calibration: load samples, fit curves, write profile + optional plot."""
    with open(input_path) as f:
        data = yaml.safe_load(f)

    profile: dict[str, Any] = {}

    # Prefill: 1D fit (batch_size → delay_ms)
    if "prefill" in data:
        points = data["prefill"]
        _validate_points(points, 3, "prefill")
        profile["prefill"] = _fit_1d(points, "batch_size", "delay_ms")

    # KV transfer: 1D fit (batch_size → delay_ms)
    if "kv_transfer" in data:
        points = data["kv_transfer"]
        _validate_points(points, 3, "kv_transfer")
        profile["kv_transfer"] = _fit_1d(points, "batch_size", "delay_ms")

    # Decode: 2D fit (batch_size, context_length → delay_per_token_ms)
    if "decode" in data:
        points = data["decode"]
        _validate_points(points, 9, "decode (2D)")
        profile["decode"] = _fit_2d(points)

    # Write profile
    with open(output_path, "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)

    print(f"Profile written to {output_path}")

    # Optional visualization
    if plot_path:
        _plot(data, profile, plot_path)
        print(f"Plot written to {plot_path}")


def _plot(
    data: dict[str, Any], profile: dict[str, Any], plot_path: str
) -> None:
    """Generate visualization PNG with sample points and fitted curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_plots = sum(1 for k in ("prefill", "kv_transfer", "decode") if k in data)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    idx = 0

    for key, label in [("prefill", "Prefill"), ("kv_transfer", "KV Transfer")]:
        if key not in data:
            continue
        ax = axes[idx]
        idx += 1
        points = data[key]
        xs = [p["batch_size"] for p in points]
        ys = [p["delay_ms"] for p in points]
        ax.scatter(xs, ys, color="red", zorder=5, label="Sample points")

        # Plot fitted curve
        coeffs = profile[key]["coefficients"]
        x_fit = np.linspace(min(xs) * 0.8, max(xs) * 1.2, 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color="blue", label="Fitted curve")
        ax.set_xlabel("Batch size (tokens)")
        ax.set_ylabel("Delay (ms)")
        ax.set_title(f"{label} Latency")
        ax.legend()

    if "decode" in data:
        ax = axes[idx]
        points = data["decode"]
        bs_vals = sorted(set(p["batch_size"] for p in points))
        for b in bs_vals:
            pts = [p for p in points if p["batch_size"] == b]
            ctxs = [p["context_length"] for p in pts]
            delays = [p["delay_per_token_ms"] for p in pts]
            ax.scatter(ctxs, delays, zorder=5, label=f"bs={b} (data)")

        # Fitted curves per batch_size
        coeffs = profile["decode"]["coefficients"]
        ctx_fit = np.linspace(
            min(p["context_length"] for p in points) * 0.8,
            max(p["context_length"] for p in points) * 1.2,
            100,
        )
        for b in bs_vals:
            y_fit = (
                coeffs[0]
                + coeffs[1] * b
                + coeffs[2] * ctx_fit
                + coeffs[3] * b * ctx_fit
                + coeffs[4] * b**2
                + coeffs[5] * ctx_fit**2
            )
            ax.plot(ctx_fit, y_fit, linestyle="--", label=f"bs={b} (fit)")

        ax.set_xlabel("Context length")
        ax.set_ylabel("Delay per token (ms)")
        ax.set_title("Decode Latency (2D)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
