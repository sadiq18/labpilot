"""Unit tests for Milestone 2 Plan 7 — Experiment Search."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.experiments.comparator import write_comparison
from labpilot.experiments.graph import build_graph
from labpilot.experiments.models import (
    ChangeCategory,
    ConfigChange,
    ExperimentComparison,
    Verdict,
)
from labpilot.experiments.search import (
    SearchFilters,
    load_comparisons,
    parse_duration,
    search,
)
from labpilot.experiments.legacy_run_overrides import TrainingOverrides, save_training_overrides
from labpilot.experiments.manifest import RunManifest, StageStatus, save_manifest


def _seed_run(
    runs_dir: Path,
    run_id: str,
    *,
    metrics: dict[str, float],
    recipes: list[str] | None = None,
    model_params: dict | None = None,
    runtime_seconds: float | None = None,
    status: StageStatus = StageStatus.COMPLETED,
    parent_id: str | None = None,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    metadata = {}
    if parent_id:
        metadata["parent_run_id"] = parent_id
        metadata["iteration"] = 1
    stages = []
    if runtime_seconds is not None:
        from labpilot.experiments.manifest import StageRecord
        from datetime import timedelta

        start = datetime(2026, 1, 1, 12, 0, 0)
        stages = [
            StageRecord(
                name="train_model",
                status=StageStatus.COMPLETED,
                started_at=start,
                finished_at=start + timedelta(seconds=runtime_seconds),
            )
        ]
    manifest = RunManifest(
        run_id=run_id,
        competition="titanic",
        status=status,
        stages=stages,
        metadata=metadata,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    save_manifest(run_dir, manifest)
    (run_dir / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_classification",
                "template_name": "tabular_classification",
                "rationale": "test",
                "target_column": "Survived",
            }
        )
    )
    save_training_overrides(
        run_dir,
        TrainingOverrides(
            model_params=model_params or {},
            feature_recipes=recipes or [],
        ),
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    from labpilot.experiments.store import ExperimentRecord, ExperimentStore

    ExperimentStore(run_dir).save(
        ExperimentRecord(
            run_id=run_id,
            competition="titanic",
            metrics=metrics,
            params={"model_params": model_params or {}},
        )
    )
    return run_dir


def test_parse_duration():
    assert parse_duration("4h") == 14400.0
    assert parse_duration("90m") == 5400.0
    assert parse_duration("30s") == 30.0
    assert parse_duration("15") == 15.0
    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration("nope")


def test_search_brief_style_filters(tmp_path: Path):
    runs = tmp_path / "runs"
    _seed_run(
        runs,
        "ema-good",
        metrics={"cv_macro_f1": 0.85},
        model_params={"ema": True},
        runtime_seconds=3600,
        parent_id="root",
    )
    _seed_run(
        runs,
        "ema-slow",
        metrics={"cv_macro_f1": 0.86},
        model_params={"ema": True},
        runtime_seconds=20000,
        parent_id="root",
    )
    _seed_run(
        runs,
        "no-ema",
        metrics={"cv_macro_f1": 0.90},
        model_params={"ema": False},
        runtime_seconds=1000,
        parent_id="root",
    )
    _seed_run(runs, "root", metrics={"cv_macro_f1": 0.80}, model_params={})

    for run_id, delta in [("ema-good", 0.05), ("ema-slow", 0.06), ("no-ema", 0.1)]:
        write_comparison(
            runs / run_id,
            ExperimentComparison(
                base_id="root",
                compare_id=run_id,
                primary_metric_key="cv_macro_f1",
                metric_deltas={"cv_macro_f1": delta},
                changes=[
                    ConfigChange(
                        category=ChangeCategory.AUGMENTATION,
                        field="model_params.ema",
                        base_value=False,
                        compare_value=True,
                        label="+ ema",
                    )
                ],
                runtime_delta_seconds=None,
                runtime_delta_pct=None,
                verdict=Verdict.WORTH_KEEPING,
                verdict_reason="test",
            ),
        )

    graph = build_graph(runs, "titanic")
    comparisons = load_comparisons(runs, graph)
    filters = SearchFilters(
        config_equals=[("model_params.ema", True)],
        metric_delta_gt=[("cv_macro_f1", 0.0)],
        runtime_max_seconds=parse_duration("4h"),
    )
    matches = search(graph, comparisons, filters)
    ids = {exp.id for exp in matches}
    assert ids == {"ema-good"}


def test_search_recipe_and_verdict(tmp_path: Path):
    runs = tmp_path / "runs"
    _seed_run(
        runs,
        "child",
        metrics={"cv_accuracy": 0.5},
        recipes=["focal_loss"],
        parent_id="root",
        runtime_seconds=100,
    )
    _seed_run(runs, "root", metrics={"cv_accuracy": 0.7}, recipes=[])
    write_comparison(
        runs / "child",
        ExperimentComparison(
            base_id="root",
            compare_id="child",
            primary_metric_key="cv_accuracy",
            metric_deltas={"cv_accuracy": -0.2},
            changes=[],
            runtime_delta_seconds=None,
            runtime_delta_pct=None,
            verdict=Verdict.REGRESSION,
            verdict_reason="down",
        ),
    )
    graph = build_graph(runs, "titanic")
    matches = search(
        graph,
        load_comparisons(runs, graph),
        SearchFilters(recipes=["focal_loss"], verdict=Verdict.REGRESSION),
    )
    assert [exp.id for exp in matches] == ["child"]


def test_search_no_filters_returns_all(tmp_path: Path):
    runs = tmp_path / "runs"
    _seed_run(runs, "a", metrics={"cv_accuracy": 0.7})
    _seed_run(runs, "b", metrics={"cv_accuracy": 0.8})
    graph = build_graph(runs, "titanic")
    matches = search(graph, {}, SearchFilters())
    assert {exp.id for exp in matches} == {"a", "b"}


def test_search_cli_bad_runtime(tmp_path: Path):
    runner = CliRunner()
    (tmp_path / "runs").mkdir()
    result = runner.invoke(
        app,
        [
            "experiments",
            "search",
            "--competition",
            "titanic",
            "--runtime-max",
            "zzz",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--config-file",
            "configs/default.yaml",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid duration" in result.output or "duration" in result.output.lower()
