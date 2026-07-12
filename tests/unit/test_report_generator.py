from pathlib import Path

import pytest

from labpilot.baseline.selector import BaselineChoice
from labpilot.competition.models import CompetitionSpec, MetricSpec
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import RunManifest, StageStatus, save_manifest
from labpilot.profiler.tabular import DatasetProfile
from labpilot.report.generator import ReportGenerator, markdown_to_html


def test_markdown_to_html_renders_headings():
    html = markdown_to_html("# Title\n\nParagraph with **bold**.")
    assert "<h1" in html or "<p>" in html
    assert "Title" in html
    assert "Paragraph" in html


def test_report_generator_writes_self_contained_html(tmp_path: Path):
    run_dir = tmp_path / "20260712-test-titanic"
    run_dir.mkdir()

    competition = CompetitionSpec(
        slug="titanic",
        title="Titanic",
        evaluation_metric=MetricSpec(name="accuracy", direction="maximize", key="accuracy"),
    )
    profile = DatasetProfile(
        competition="titanic",
        row_count=891,
        test_row_count=418,
        target_column="Survived",
        id_column="PassengerId",
        train_file="train.csv",
    )
    baseline = BaselineChoice(
        problem_type="tabular_classification",
        template_name="tabular_classification",
        rationale="test",
        target_column="Survived",
        id_column="PassengerId",
        metric_name="accuracy",
    )
    (run_dir / "competition.json").write_text(competition.model_dump_json(indent=2))
    (run_dir / "profile.json").write_text(profile.model_dump_json(indent=2))
    (run_dir / "baseline_choice.json").write_text(baseline.model_dump_json(indent=2))
    (run_dir / "metrics.json").write_text('{"cv_accuracy": 0.76, "cv_folds": 5}')
    (run_dir / "brief.md").write_text("# Brief\n\nFocus on family structure.\n")
    (run_dir / "reflection.md").write_text("# Reflection\n\nTry target encoding next.\n")
    (run_dir / "profile.md").write_text("## Profile\n\n891 training rows.\n")
    (run_dir / "submission_result.json").write_text(
        SubmissionResult(
            competition="titanic",
            submission_path=str(run_dir / "submission.csv"),
            status="scored",
            public_score=0.77,
            submissions_url="https://www.kaggle.com/competitions/titanic/submissions",
        ).model_dump_json(indent=2)
    )

    manifest = RunManifest(run_id=run_dir.name, competition="titanic", status=StageStatus.COMPLETED)
    manifest.mark_completed("write_reflection", [str(run_dir / "reflection.md")])
    save_manifest(run_dir, manifest)

    output = ReportGenerator().generate(run_dir, manifest)
    html = output.read_text(encoding="utf-8")

    assert output.name == "report.html"
    assert "<!DOCTYPE html>" in html
    assert "Titanic" in html
    assert "cv_accuracy" in html
    assert "Focus on family structure" in html
    assert "Try target encoding next" in html
    assert "LabPilot Report" in html or "LabPilot" in html
