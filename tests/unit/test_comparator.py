"""Unit tests for Milestone 2 Plan 3 — Automatic Comparator."""

from datetime import datetime
from pathlib import Path

import pytest

from labpilot.research_engine.shared.experiments.comparator import (
    compare,
    load_comparison,
    render_markdown,
    write_comparison,
)
from labpilot.research_engine.shared.experiments.models import (
    ChangeCategory,
    Experiment,
    Verdict,
)


def _exp(
    run_id: str,
    *,
    metrics: dict[str, float] | None = None,
    model_params: dict | None = None,
    feature_recipes: list[str] | None = None,
    runtime_seconds: float | None = None,
    template_name: str | None = "tabular_classification",
    problem_type: str | None = "tabular_classification",
) -> Experiment:
    return Experiment(
        id=run_id,
        competition="titanic",
        status="completed",
        progress="14/14 stages",
        description="test",
        template_name=template_name,
        problem_type=problem_type,
        model_params=model_params or {},
        feature_recipes=feature_recipes or [],
        metrics=metrics or {},
        runtime_seconds=runtime_seconds,
        created_at=datetime(2026, 1, 1),
    )


def test_feature_recipe_categories_and_worth_keeping():
    base = _exp(
        "base",
        metrics={"cv_accuracy": 0.75},
        feature_recipes=["log_numeric"],
        runtime_seconds=100.0,
    )
    child = _exp(
        "child",
        metrics={"cv_accuracy": 0.78},
        feature_recipes=["log_numeric", "target_encoding", "mixup"],
        runtime_seconds=110.0,
    )

    result = compare(
        base,
        child,
        primary_metric_key="cv_accuracy",
        maximize=True,
        noise_epsilon=0.001,
        max_runtime_increase_pct=50.0,
    )

    by_label = {change.label: change for change in result.changes}
    assert "+ target_encoding" in by_label
    assert by_label["+ target_encoding"].category == ChangeCategory.FEATURE_ENGINEERING
    assert "+ mixup" in by_label
    assert by_label["+ mixup"].category == ChangeCategory.AUGMENTATION
    assert result.verdict == Verdict.WORTH_KEEPING
    assert result.metric_deltas["cv_accuracy"] == pytest.approx(0.03)


def test_noise_band_is_inconclusive():
    base = _exp("base", metrics={"cv_accuracy": 0.7500}, runtime_seconds=100.0)
    child = _exp("child", metrics={"cv_accuracy": 0.7505}, runtime_seconds=100.0)

    result = compare(
        base,
        child,
        primary_metric_key="cv_accuracy",
        noise_epsilon=0.001,
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert "noise band" in result.verdict_reason


def test_runtime_cost_not_worth_keeping():
    base = _exp("base", metrics={"cv_accuracy": 0.75}, runtime_seconds=100.0)
    child = _exp("child", metrics={"cv_accuracy": 0.78}, runtime_seconds=200.0)

    result = compare(
        base,
        child,
        primary_metric_key="cv_accuracy",
        max_runtime_increase_pct=50.0,
    )
    assert result.verdict == Verdict.NOT_WORTH_KEEPING
    assert result.runtime_delta_pct == pytest.approx(100.0)


def test_regression_when_metric_worsens():
    base = _exp("base", metrics={"cv_accuracy": 0.80})
    child = _exp("child", metrics={"cv_accuracy": 0.70})

    result = compare(base, child, primary_metric_key="cv_accuracy")
    assert result.verdict == Verdict.REGRESSION


def test_minimize_metric_normalizes_improvement():
    base = _exp("base", metrics={"cv_logloss": 0.50}, runtime_seconds=100.0)
    child = _exp("child", metrics={"cv_logloss": 0.40}, runtime_seconds=100.0)

    result = compare(
        base,
        child,
        primary_metric_key="cv_logloss",
        maximize=False,
    )
    # Raw delta is -0.10; after normalize for minimize, improvement is +0.10.
    assert result.metric_deltas["cv_logloss"] == pytest.approx(-0.10)
    assert result.verdict == Verdict.WORTH_KEEPING


def test_minimize_metric_detects_regression():
    base = _exp("base", metrics={"cv_logloss": 0.40})
    child = _exp("child", metrics={"cv_logloss": 0.55})

    result = compare(
        base,
        child,
        primary_metric_key="cv_logloss",
        maximize=False,
    )
    assert result.verdict == Verdict.REGRESSION


def test_model_param_and_template_changes():
    base = _exp(
        "base",
        metrics={"cv_accuracy": 0.7},
        model_params={"learning_rate": 0.05, "num_leaves": 31},
        template_name="tabular_classification",
    )
    child = _exp(
        "child",
        metrics={"cv_accuracy": 0.72},
        model_params={"learning_rate": 0.03, "num_leaves": 63},
        template_name="deep_tabular",
    )

    result = compare(base, child, primary_metric_key="cv_accuracy")
    fields = {change.field: change for change in result.changes}
    assert fields["model_params.learning_rate"].category == ChangeCategory.TRAINING_STRATEGY
    assert fields["model_params.num_leaves"].category == ChangeCategory.MODEL
    assert fields["template_name"].category == ChangeCategory.MODEL


def test_render_markdown_sections_and_persistence(tmp_path: Path):
    base = _exp(
        "base-run",
        metrics={"cv_accuracy": 0.75},
        feature_recipes=[],
        runtime_seconds=50.0,
    )
    child = _exp(
        "child-run",
        metrics={"cv_accuracy": 0.80},
        feature_recipes=["target_encoding"],
        runtime_seconds=55.0,
    )
    comparison = compare(base, child, primary_metric_key="cv_accuracy")
    md = render_markdown(comparison)

    assert "## Changes" in md
    assert "## Metrics" in md
    assert "## Conclusion" in md
    assert "Inference: not tracked" in md
    assert comparison.verdict.value in md
    assert comparison.verdict_reason in md

    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    write_comparison(run_dir, comparison)
    assert (run_dir / "comparison.json").is_file()
    assert (run_dir / "comparison.md").read_text() == md
    loaded = load_comparison(run_dir)
    assert loaded is not None
    assert loaded.verdict == comparison.verdict
    assert render_markdown(loaded) == md


def test_missing_primary_metric_is_inconclusive():
    base = _exp("base", metrics={"cv_accuracy": 0.75})
    child = _exp("child", metrics={"cv_auc": 0.80})

    result = compare(base, child, primary_metric_key="cv_accuracy")
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.primary_metric_key is None
