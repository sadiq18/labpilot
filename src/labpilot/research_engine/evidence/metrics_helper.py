"""Helpers for rich metrics.json emission (used by train scaffolds / docs)."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def timed_section() -> Iterator[dict[str, float]]:
    """Yield a dict that receives ``elapsed_s`` when the block exits."""
    box: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["elapsed_s"] = time.perf_counter() - start


def fold_stats(fold_scores: list[float]) -> dict[str, float]:
    if not fold_scores:
        return {}
    mean = float(statistics.fmean(fold_scores))
    std = float(statistics.pstdev(fold_scores)) if len(fold_scores) > 1 else 0.0
    return {"cv_mean": mean, "cv_std": std}


def enrich_metrics(
    metrics: dict[str, Any],
    *,
    fold_scores: list[float] | None = None,
    train_time_s: float | None = None,
    inference_time_s: float | None = None,
    peak_memory_mb: float | None = None,
) -> dict[str, Any]:
    """Return a copy of metrics with stability / resource fields filled."""
    out = dict(metrics)
    if fold_scores:
        out["cv_fold_scores"] = [float(x) for x in fold_scores]
        out.update(fold_stats(fold_scores))
    if train_time_s is not None:
        out["train_time_s"] = float(train_time_s)
    if inference_time_s is not None:
        out["inference_time_s"] = float(inference_time_s)
    if peak_memory_mb is not None:
        out["peak_memory_mb"] = float(peak_memory_mb)
    return out


def peak_memory_mb_best_effort() -> float | None:
    """Best-effort RSS in MiB; None if unavailable."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB; macOS: bytes.
        import sys

        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0
    except Exception:
        return None
