"""Opt-in emission of internal retrieve/campaign metrics.

Production CLIs must not print BM25/graph/campaign internals. By default metrics
go to ``logger.debug`` only. Set ``LABPILOT_DEBUG_METRICS=1`` (or true/yes) to
also ``logger.info`` and print to stdout for local debugging.
"""

from __future__ import annotations

import logging
import os
from typing import Any


def debug_metrics_enabled() -> bool:
    raw = (os.environ.get("LABPILOT_DEBUG_METRICS") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    try:
        from labpilot.config import Settings

        return bool(Settings().labpilot_debug_metrics)
    except Exception:  # noqa: BLE001 — never block callers on settings load
        return False


def emit_debug_metrics(logger: logging.Logger, line: str, **extra: Any) -> None:
    """Log ``line`` at DEBUG; when enabled, also INFO + stdout."""
    if extra:
        logger.debug("%s | %s", line, extra)
    else:
        logger.debug(line)
    if not debug_metrics_enabled():
        return
    logger.info(line)
    print(line, flush=True)
