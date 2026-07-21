"""Plan 4 — ExperimentAnalyzer + DatasetAnalyzer.

All local-only: no network, no LLM. Fixtures seed M2-style run directories, a
knowledge base, and a dataset profile, then assert the analyzers turn them into
``ResearchArtifact`` batches (and soft-fail cleanly when data is missing).
"""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.intelligence.analyzers.dataset import DatasetAnalyzer
from labpilot.research_engine.intelligence.analyzers.experiments import ExperimentAnalyzer
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifactType,
)

_ALL_STAGES = [
    "parse_competition",
    "download_data",
    "profile_dataset",
    "generate_brief",
    "select_baseline",
    "generate_code",
    "train_model",
    "evaluate_cv",
    "generate_submission",
    "export_kernel",
    "upload_submission",
    "log_experiment",
    "write_reflection",
    "write_report",
]


def _stage(name: str) -> dict:
    return {
        "name": name,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:05:00",
        "error": None,
        "artifacts": [],
    }


def _seed_run(
    runs_dir: Path,
    run_id: str,
    *,
    competition: str = "birdclef-2026",
    parent_id: str | None = None,
    iteration: int = 0,
    metrics: dict[str, float] | None = None,
    feature_recipes: list[str] | None = None,
    created_at: str = "2026-01-01T00:00:00",
    profile: dict | None = None,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    metadata: dict = {"iteration": iteration}
    if parent_id is not None:
        metadata["parent_run_id"] = parent_id
        metadata["improvement_strategy"] = "features"
    manifest = {
        "run_id": run_id,
        "competition": competition,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "completed",
        "stages": [_stage(name) for name in _ALL_STAGES],
        "metadata": metadata,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "audio_classification",
                "template_name": "convnext_audio",
                "rationale": "baseline",
            }
        )
    )
    if feature_recipes is not None:
        (run_dir / "training_overrides.json").write_text(
            json.dumps({"model_params": {}, "feature_recipes": feature_recipes})
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics or {}))
    if profile is not None:
        (run_dir / "profile.json").write_text(json.dumps(profile))
    return run_dir


def _context(tmp_path: Path, competition: str = "birdclef-2026") -> AnalyzeContext:
    return AnalyzeContext(
        competition=competition,
        runs_dir=tmp_path / "runs",
        knowledge_dir=tmp_path / "knowledge",
    )


def _seed_knowledge_failure(knowledge_dir: Path, competition: str) -> None:
    kb_path = knowledge_dir / competition / "knowledge_base.json"
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    kb_path.write_text(
        json.dumps(
            {
                "competition": competition,
                "entries": [
                    {
                        "technique": "focal loss",
                        "metric_key": "cv_macro_f1",
                        "effect": "hurts",
                        "delta_estimate": -0.012,
                        "confidence": 0.7,
                        "sample_size": 2,
                        "evidence_run_ids": ["child-a"],
                        "updated_at": "2026-01-01T02:00:00",
                    }
                ],
            }
        )
    )


# --- ExperimentAnalyzer -----------------------------------------------------


def test_experiment_analyzer_assembles_artifacts(tmp_path: Path):
    ctx = _context(tmp_path)
    _seed_run(ctx.runs_dir, "root-1", metrics={"cv_macro_f1": 0.80})
    _seed_run(
        ctx.runs_dir,
        "child-a",
        parent_id="root-1",
        iteration=1,
        metrics={"cv_macro_f1": 0.78},
        feature_recipes=["focal loss", "mixup"],
        created_at="2026-01-01T01:00:00",
    )
    _seed_knowledge_failure(ctx.knowledge_dir, ctx.competition)

    result = ExperimentAnalyzer().analyze(ctx)

    assert result.analyzer == "experiments"
    assert [a.id for a in result.items] == ["exp:root-1", "exp:child-a"]
    assert all(a.type is ResearchArtifactType.EXPERIMENT for a in result.items)
    assert all(a.source == "m2" for a in result.items)

    child = next(a for a in result.items if a.id == "exp:child-a")
    assert "focal loss" in child.techniques
    assert "convnext_audio" in child.techniques  # template contributes a tag
    # Failure is flagged on the run that produced it.
    assert child.metadata["regressions"] == ["focal loss"]
    assert any("focal loss" in claim for claim in child.claims)

    assert "focal loss" in result.techniques
    assert any("focal loss" in note and "hurts" in note for note in result.notes)


def test_experiment_analyzer_soft_fails_without_runs(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.runs_dir.mkdir(parents=True)

    result = ExperimentAnalyzer().analyze(ctx)

    assert result.items == []
    assert any("No runs found" in note for note in result.notes)


# --- DatasetAnalyzer --------------------------------------------------------


def test_dataset_analyzer_reads_latest_profile(tmp_path: Path):
    ctx = _context(tmp_path)
    profile = {
        "competition": ctx.competition,
        "row_count": 5000,
        "test_row_count": 1200,
        "column_count": 3,
        "modality": "audio",
        "target_column": "primary_label",
        "id_column": "row_id",
        "columns": [
            {"name": "row_id", "dtype": "int64", "null_pct": 0.0},
            {"name": "primary_label", "dtype": "object", "null_pct": 0.0},
            {"name": "secondary_labels", "dtype": "object", "null_pct": 45.0},
        ],
        "warnings": ["Using sample submission as test reference."],
    }
    # Older run has no profile; newest run carries it.
    _seed_run(ctx.runs_dir, "root-1", created_at="2026-01-01T00:00:00")
    _seed_run(
        ctx.runs_dir,
        "child-a",
        parent_id="root-1",
        created_at="2026-01-01T01:00:00",
        profile=profile,
    )

    result = DatasetAnalyzer().analyze(ctx)

    assert len(result.items) == 1
    artifact = result.items[0]
    assert artifact.id == "dataset:birdclef-2026"
    assert artifact.type is ResearchArtifactType.DATASET
    assert artifact.metadata["run_id"] == "child-a"
    assert artifact.metadata["modality"] == "audio"
    assert artifact.metadata["null_heavy_columns"] == ["secondary_labels"]
    assert any("secondary_labels" in note for note in result.notes)
    assert any("test reference" in note for note in result.notes)


def test_dataset_analyzer_soft_fails_without_profile(tmp_path: Path):
    ctx = _context(tmp_path)
    _seed_run(ctx.runs_dir, "root-1")  # run exists but no profile.json

    result = DatasetAnalyzer().analyze(ctx)

    assert result.items == []
    assert any("No dataset profile" in note for note in result.notes)


# --- Registry wiring --------------------------------------------------------


def test_local_analyzers_are_default_enabled():
    assert ExperimentAnalyzer().default_enabled is True
    assert DatasetAnalyzer().default_enabled is True
