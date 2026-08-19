"""Unit tests for Milestone 2 Plan 8 — Experiment Dashboard & Report."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from helpers.cli import cli_runner

from labpilot.cli.main import app
from labpilot.research_engine.shared.experiments.graph import build_graph
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.knowledge import KnowledgeBase
from labpilot.research_engine.shared.experiments.models import (
    ExperimentReport,
    KnowledgeEffect,
    KnowledgeEntry,
)
from labpilot.research_engine.shared.experiments.report import (
    NoExperimentsError,
    build_report,
    write_dashboard,
)
from labpilot.research_engine.shared.experiments.manifest import RunManifest, StageStatus, save_manifest


def _write_run(
    runs_dir: Path,
    run_id: str,
    *,
    competition: str = "titanic",
    parent_id: str | None = None,
    status: StageStatus = StageStatus.COMPLETED,
    metric: float | None = 0.8,
    created_offset_hours: int = 0,
    with_report: bool = True,
    with_comparison: bool = False,
    description_seed: str | None = None,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    created = datetime(2026, 1, 1, 12, 0, 0) + timedelta(hours=created_offset_hours)
    metadata: dict = {}
    if parent_id:
        metadata["parent_run_id"] = parent_id
        metadata["iteration"] = 1
    manifest = RunManifest(
        run_id=run_id,
        competition=competition,
        status=status,
        stages=[],
        metadata=metadata,
        created_at=created,
        updated_at=created,
    )
    save_manifest(run_dir, manifest)
    (run_dir / "competition.json").write_text(
        json.dumps(
            {
                "slug": competition,
                "title": "Titanic",
                "evaluation_metric": {
                    "name": "Accuracy",
                    "key": "accuracy",
                    "direction": "maximize",
                },
            }
        )
    )
    if metric is not None:
        (run_dir / "metrics.json").write_text(json.dumps({"cv_accuracy": metric}))
    if description_seed:
        (run_dir / "baseline_choice.json").write_text(
            json.dumps(
                {
                    "template_name": description_seed,
                    "problem_type": "binary_classification",
                    "rationale": "test",
                }
            )
        )
    if with_report:
        (run_dir / "report.html").write_text("<html>ok</html>")
    if with_comparison and parent_id:
        (run_dir / "comparison.md").write_text("# comparison\n")
        (run_dir / "comparison.json").write_text("{}")
    return run_dir


def _seed_kb(knowledge: Path, competition: str = "titanic") -> None:
    kb = KnowledgeBase(knowledge, competition)
    kb._entries[("ema", "cv_accuracy")] = KnowledgeEntry(
        technique="ema",
        metric_key="cv_accuracy",
        effect=KnowledgeEffect.IMPROVES,
        delta_estimate=0.02,
        confidence=0.8,
        sample_size=2,
        evidence_run_ids=["child"],
        updated_at=datetime(2026, 1, 2),
    )
    kb._entries[("cutmix", "cv_accuracy")] = KnowledgeEntry(
        technique="cutmix",
        metric_key="cv_accuracy",
        effect=KnowledgeEffect.HURTS,
        delta_estimate=-0.03,
        confidence=0.7,
        sample_size=1,
        evidence_run_ids=["bad"],
        updated_at=datetime(2026, 1, 2),
    )
    kb._save()


def test_build_report_composes_plans(tmp_path: Path):
    runs = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    _write_run(runs, "root", metric=0.75, description_seed="lightgbm", created_offset_hours=0)
    _write_run(
        runs,
        "child",
        parent_id="root",
        metric=0.82,
        description_seed="lightgbm",
        created_offset_hours=1,
        with_comparison=True,
    )
    _write_run(
        runs,
        "running",
        metric=None,
        status=StageStatus.RUNNING,
        created_offset_hours=2,
        with_report=False,
    )
    _seed_kb(knowledge)
    store = HypothesisStore(knowledge, "titanic")
    store.create(
        observation="Try focal",
        reason="imbalance",
        prediction="Self-Distillation",
        confidence=0.78,
        tags=["loss"],
    )

    report = build_report("titanic", runs, knowledge)
    assert report.experiment_count == 3
    assert report.best_experiment_id == "child"
    assert report.best_score == 0.82
    assert report.primary_metric_key == "cv_accuracy"
    assert [e.technique for e in report.top_discoveries] == ["ema"]
    assert [e.technique for e in report.known_failures] == ["cutmix"]
    assert [e.id for e in report.best_pipeline] == ["root", "child"]
    assert report.recommended_next is not None
    assert report.recommended_next.hypothesis.prediction == "Self-Distillation"
    assert {e.id for e in report.experiments} == {"root", "child", "running"}
    # newest first
    assert report.experiments[0].id == "running"
    running = next(e for e in report.experiments if e.id == "running")
    assert running.status == StageStatus.RUNNING.value


def test_report_json_round_trip(tmp_path: Path):
    runs = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    _write_run(runs, "only", metric=0.5)
    report = build_report("titanic", runs, knowledge)
    restored = ExperimentReport.model_validate_json(report.model_dump_json())
    assert restored.experiment_count == 1
    assert restored.experiments[0].id == "only"


def test_dashboard_writes_links(tmp_path: Path):
    runs = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    _write_run(runs, "root", metric=0.7)
    _write_run(
        runs,
        "child",
        parent_id="root",
        metric=0.75,
        with_comparison=True,
        created_offset_hours=1,
    )
    report = build_report("titanic", runs, knowledge)
    graph = build_graph(runs, "titanic", knowledge_dir=knowledge)
    path = write_dashboard(report, graph, knowledge_dir=knowledge, runs_dir=runs)
    assert path == knowledge / "titanic" / "dashboard.html"
    html = path.read_text()
    assert "All experiments" in html
    assert 'href="../../runs/root/report.html"' in html
    assert 'href="../../runs/child/comparison.md"' in html
    assert (runs / "root" / "report.html").is_file()
    assert (runs / "child" / "comparison.md").is_file()


def test_empty_competition_fails_loud(tmp_path: Path):
    try:
        build_report("missing", tmp_path / "runs", tmp_path / "knowledge")
        raise AssertionError("expected NoExperimentsError")
    except NoExperimentsError as exc:
        assert "No experiments yet" in str(exc)


def test_cli_report_and_dashboard(tmp_path: Path):
    runs = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    _write_run(runs, "r1", metric=0.66)
    runner = cli_runner()

    empty = runner.invoke(
        app,
        [
            "experiments",
            "report",
            "--competition",
            "nope",
            "--runs-dir",
            str(runs),
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert empty.exit_code == 0
    assert "No experiments yet" in empty.stdout

    text = runner.invoke(
        app,
        [
            "experiments",
            "report",
            "--competition",
            "titanic",
            "--runs-dir",
            str(runs),
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert text.exit_code == 0, text.output
    assert "titanic" in text.stdout
    assert "1 Experiments" in text.stdout

    js = runner.invoke(
        app,
        [
            "experiments",
            "report",
            "--competition",
            "titanic",
            "--format",
            "json",
            "--runs-dir",
            str(runs),
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert js.exit_code == 0, js.output
    payload = ExperimentReport.model_validate_json(js.stdout)
    assert payload.experiment_count == 1

    dash = runner.invoke(
        app,
        [
            "experiments",
            "dashboard",
            "--competition",
            "titanic",
            "--runs-dir",
            str(runs),
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert dash.exit_code == 0, dash.output
    assert "Dashboard written" in dash.stdout
    assert (knowledge / "titanic" / "dashboard.html").is_file()
