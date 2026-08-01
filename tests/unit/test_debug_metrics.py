"""Tests for opt-in debug metrics emission."""

from __future__ import annotations

import logging

from labpilot.research_engine.debug_metrics import debug_metrics_enabled, emit_debug_metrics


def test_debug_metrics_env_on(monkeypatch) -> None:
    monkeypatch.setenv("LABPILOT_DEBUG_METRICS", "1")
    assert debug_metrics_enabled() is True


def test_debug_metrics_env_off(monkeypatch) -> None:
    monkeypatch.setenv("LABPILOT_DEBUG_METRICS", "0")
    assert debug_metrics_enabled() is False


def test_emit_debug_metrics_silent_when_disabled(monkeypatch, capsys, caplog) -> None:
    log = logging.getLogger("test.debug_metrics.silent")
    monkeypatch.setenv("LABPILOT_DEBUG_METRICS", "0")
    with caplog.at_level(logging.DEBUG, logger="test.debug_metrics.silent"):
        emit_debug_metrics(log, "[context] secret internals")
    assert "[context] secret internals" in caplog.text
    assert "[context] secret internals" not in capsys.readouterr().out


def test_emit_debug_metrics_prints_when_enabled(monkeypatch, capsys, caplog) -> None:
    log = logging.getLogger("test.debug_metrics.loud")
    monkeypatch.setenv("LABPILOT_DEBUG_METRICS", "1")
    with caplog.at_level(logging.DEBUG, logger="test.debug_metrics.loud"):
        emit_debug_metrics(log, "[context] secret internals")
    assert "[context] secret internals" in capsys.readouterr().out
    assert "[context] secret internals" in caplog.text
