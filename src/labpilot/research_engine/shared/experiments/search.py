"""Composable experiment search filters (Milestone 2, Plan 7)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.comparator import load_comparison
from labpilot.research_engine.shared.experiments.graph import ExperimentGraph
from labpilot.research_engine.shared.experiments.models import Experiment, ExperimentComparison, Verdict

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([hms]?)\s*$", re.IGNORECASE)


@dataclass
class SearchFilters:
    config_equals: list[tuple[str, Any]] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)
    metric_gt: list[tuple[str, float]] = field(default_factory=list)
    metric_lt: list[tuple[str, float]] = field(default_factory=list)
    metric_delta_gt: list[tuple[str, float]] = field(default_factory=list)
    metric_delta_lt: list[tuple[str, float]] = field(default_factory=list)
    runtime_max_seconds: float | None = None
    runtime_min_seconds: float | None = None
    verdict: Verdict | None = None
    status: str | None = None
    template: str | None = None


def parse_duration(text: str) -> float:
    """Parse `4h` / `90m` / `30s` / bare seconds into float seconds."""
    match = _DURATION_RE.match(text)
    if not match:
        raise ValueError(
            f"Invalid duration '{text}'. Use forms like 4h, 90m, 30s, or bare seconds."
        )
    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    if unit == "h":
        return value * 3600.0
    if unit == "m":
        return value * 60.0
    return value


def parse_key_value(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Expected key=value, got '{text}'")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Empty key in '{text}'")
    return key, _coerce_scalar(raw.strip())


def parse_metric_threshold(text: str) -> tuple[str, float]:
    if ":" not in text:
        raise ValueError(f"Expected key:value, got '{text}'")
    key, raw = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Empty metric key in '{text}'")
    try:
        return key, float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Metric threshold must be numeric: '{text}'") from exc


def load_comparisons(
    runs_dir: Path, graph: ExperimentGraph
) -> dict[str, ExperimentComparison]:
    result: dict[str, ExperimentComparison] = {}
    for run_id in graph.nodes:
        comparison = load_comparison(runs_dir / run_id)
        if comparison is not None:
            result[run_id] = comparison
    return result


def search(
    graph: ExperimentGraph,
    comparisons: dict[str, ExperimentComparison],
    filters: SearchFilters,
) -> list[Experiment]:
    matched = [
        exp
        for exp in graph.nodes.values()
        if _matches(exp, comparisons.get(exp.id), filters)
    ]
    return sorted(matched, key=lambda exp: exp.created_at, reverse=True)


def _matches(
    exp: Experiment,
    comparison: ExperimentComparison | None,
    filters: SearchFilters,
) -> bool:
    for key, expected in filters.config_equals:
        actual = _resolve_config_value(exp, key)
        if actual != expected and str(actual) != str(expected):
            return False

    for recipe in filters.recipes:
        if recipe not in exp.feature_recipes:
            return False

    for key, threshold in filters.metric_gt:
        value = exp.metrics.get(key)
        if value is None or float(value) <= threshold:
            return False

    for key, threshold in filters.metric_lt:
        value = exp.metrics.get(key)
        if value is None or float(value) >= threshold:
            return False

    if filters.metric_delta_gt or filters.metric_delta_lt or filters.verdict is not None:
        if comparison is None:
            return False

    for key, threshold in filters.metric_delta_gt:
        assert comparison is not None
        delta = comparison.metric_deltas.get(key)
        if delta is None or float(delta) <= threshold:
            return False

    for key, threshold in filters.metric_delta_lt:
        assert comparison is not None
        delta = comparison.metric_deltas.get(key)
        if delta is None or float(delta) >= threshold:
            return False

    if filters.runtime_max_seconds is not None:
        if exp.runtime_seconds is None or exp.runtime_seconds > filters.runtime_max_seconds:
            return False

    if filters.runtime_min_seconds is not None:
        if exp.runtime_seconds is None or exp.runtime_seconds < filters.runtime_min_seconds:
            return False

    if filters.verdict is not None:
        assert comparison is not None
        if comparison.verdict != filters.verdict:
            return False

    if filters.status is not None and exp.status != filters.status:
        return False

    if filters.template is not None and exp.template_name != filters.template:
        return False

    return True


def _resolve_config_value(exp: Experiment, dotted_key: str) -> Any:
    if dotted_key.startswith("model_params."):
        return _lookup_path(exp.model_params, dotted_key.removeprefix("model_params."))
    if dotted_key == "model_params":
        return exp.model_params
    # Prefer config_snapshot (AppConfig), then model_params for short keys.
    from_snapshot = _lookup_path(exp.config_snapshot, dotted_key)
    if from_snapshot is not None:
        return from_snapshot
    if dotted_key in exp.model_params:
        return exp.model_params[dotted_key]
    return _lookup_path(exp.model_params, dotted_key)


def _lookup_path(data: Any, dotted_key: str) -> Any:
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
