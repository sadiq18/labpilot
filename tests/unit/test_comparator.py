"""Unit tests for Milestone 2 Plan 3 — Automatic Comparator."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from labpilot.research_engine.shared.experiments.comparator import (
    compare,
    load_comparison,
    render_markdown,
    resolve_primary_metric_key_and_direction,
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
    # The renderer must stay unstamped: `experiments compare --format markdown`
    # calls it live and recomputes whenever the stored JSON records a different
    # pair, so a stamp here tells that reader to open a `comparison.json` that
    # may not exist or may describe something else. The write site owns the
    # stamp. Guarded because re-adding it to the renderer left the whole suite
    # green — the file-level assertions below are satisfied by a doubled stamp.
    assert "not authoritative" not in md.lower()
    assert not md.lstrip().startswith(">")


    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    write_comparison(run_dir, comparison)
    assert (run_dir / "comparison.json").is_file()
    # The file carries a provenance stamp the renderer does not: `render_markdown`
    # stays a pure function of the comparison (asserted below), and
    # `write_comparison` prepends the note because the stamp is a fact about the
    # file rather than about the rendering.
    written = (run_dir / "comparison.md").read_text()
    assert written.endswith(md)
    assert written.startswith("> **Derived view")
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


# --- direction comes from the metric, not from the contract -----------------


def _competition_dir(tmp_path: Path, metric: dict) -> tuple[Path, ...]:
    (tmp_path / "competition.json").write_text(
        json.dumps({"slug": "demo", "evaluation_metric": metric}), encoding="utf-8"
    )
    return (tmp_path,)


def test_a_contract_that_states_the_wrong_direction_does_not_invert_the_verdict(
    tmp_path: Path,
) -> None:
    """The rogii failure, at the comparator. A contract claiming RMSE is
    maximised made a *worse* score read as an improvement, and fifteen evidence
    cards were built that way. The registry knows RMSE from its key, and the key
    is what the run was actually scored on.
    """
    dirs = _competition_dir(tmp_path, {"name": "rmse", "key": "rmse", "direction": "maximize"})

    comparison = compare(
        _exp("base", metrics={"cv_rmse": 10.0}, problem_type="tabular_regression"),
        _exp("child", metrics={"cv_rmse": 20.0}, problem_type="tabular_regression"),
        competition_dirs=dirs,
    )

    assert comparison.primary_metric_key == "cv_rmse"
    assert comparison.verdict is Verdict.REGRESSION, "a doubled RMSE read as an improvement"


def test_a_contract_that_states_no_direction_is_not_read_as_maximize(
    tmp_path: Path,
) -> None:
    """`direction != "minimize"` read *everything* that was not that literal
    string as maximize — including a contract that never said."""
    dirs = _competition_dir(tmp_path, {"name": "rmse", "key": "rmse"})

    comparison = compare(
        _exp("base", metrics={"cv_rmse": 10.0}, problem_type="tabular_regression"),
        _exp("child", metrics={"cv_rmse": 5.0}, problem_type="tabular_regression"),
        competition_dirs=dirs,
    )

    assert comparison.verdict is Verdict.WORTH_KEEPING, "halving RMSE is an improvement"


def test_a_stated_direction_still_decides_a_metric_the_registry_cannot_know(
    tmp_path: Path,
) -> None:
    """The registry answers for catalogued keys only. For anything else the
    contract is the best evidence available, and must still be used."""
    dirs = _competition_dir(
        tmp_path, {"name": "wellbore misfit", "key": "wellbore_misfit", "direction": "minimize"}
    )

    comparison = compare(
        _exp("base", metrics={"cv_wellbore_misfit": 10.0}, problem_type="tabular_regression"),
        _exp("child", metrics={"cv_wellbore_misfit": 5.0}, problem_type="tabular_regression"),
        competition_dirs=dirs,
    )

    assert comparison.verdict is Verdict.WORTH_KEEPING


def test_the_graph_and_the_comparator_never_disagree(tmp_path: Path) -> None:
    """Review finding. Both carried their own `direction != "minimize"`, and
    rewiring only the comparator was worse than rewiring neither: for a contract
    claiming RMSE is maximised, `compare()` reported a regression while
    `build_graph`'s flag fed `_pick_best`, which selected that same worse run as
    the best node on the path. One resolver, so a third caller cannot diverge.
    """
    from labpilot.research_engine.shared.experiments.graph import _resolve_metric_direction

    contracts = [
        {"name": "rmse", "key": "rmse", "direction": "maximize"},
        {"name": "rmse", "key": "rmse"},
        {"name": "auc", "key": "auc", "direction": "minimize"},
        {"name": "misfit", "key": "wellbore_misfit", "direction": "minimize"},
        {"name": "misfit", "key": "wellbore_misfit"},
    ]
    for index, metric in enumerate(contracts):
        run = tmp_path / f"r{index}"
        run.mkdir()
        (run / "competition.json").write_text(
            json.dumps({"slug": "demo", "evaluation_metric": metric}), encoding="utf-8"
        )
        _key, comparator_maximize = resolve_primary_metric_key_and_direction(
            _exp("a", metrics={"cv_rmse": 10.0}),
            _exp("b", metrics={"cv_rmse": 5.0}),
            competition_dirs=(run,),
        )

        assert _resolve_metric_direction(tmp_path, [f"r{index}"]) is comparator_maximize, metric
