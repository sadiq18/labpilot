"""Unit tests for ExperienceExtractor."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.agents.git_evolution import write_experiment_git_record
from labpilot.research_engine.memory import ExperienceExtractor
from labpilot.research_engine.shared.experiments.models import (
    Experiment,
    StructuredReflection,
)
from labpilot.workspace import competition_workspace_path


def _experiment(**overrides: object) -> Experiment:
    base = dict(
        id="run-001",
        competition="birdclef-2026",
        status="completed",
        progress="done",
        description="Added SpecAugment + EMA",
        problem_type="audio",
        metrics={"lb_score": 0.706, "delta": 0.006},
        git_commit="deadbeefcafebabe",
        created_at=datetime.now(UTC),
    )
    base.update(overrides)
    return Experiment.model_validate(base)


def _facet_map(record):  # noqa: ANN001
    return {f.facet: f for f in record.facets}


def test_extract_from_experiment_model(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    extractor = ExperienceExtractor(knowledge)
    try:
        record = extractor.extract(
            competition="birdclef-2026",
            experiment=_experiment(),
            comparison={
                "delta": 0.006,
                "verdict": "worth_keeping",
                "maximize": True,
            },
            reflection=StructuredReflection(
                run_id="run-001",
                observation="SpecAugment helps minority classes",
                likely_cause="better coverage of rare calls",
                confidence=0.7,
                suggested_next=["try mixup"],
                generated_by="template_fallback",
            ),
        )
        assert record.id == "XR-001"
        assert record.idempotency_key == "run-001"
        assert record.outcome == "success"
        assert record.artifacts.git_commit == "deadbeefcafebabe"
        assert record.artifacts.experiment_id == "run-001"

        facets = _facet_map(record)
        assert "audio" in facets
        assert facets["audio"].source == "metadata"
        assert facets["audio"].confidence >= 0.8
        assert "audio" in [e.lower() for e in facets["audio"].evidence]
        assert "augmentation" in facets
        assert facets["augmentation"].source == "rules"
        assert facets["augmentation"].evidence
        assert facets["augmentation"].confidence < 1.0
        assert "imbalance" in facets
        assert "SpecAugment" in record.action or "SpecAugment" in record.hypothesis
        assert "0.006" in record.result or "delta=" in record.result
    finally:
        extractor.close()


def test_reextract_idempotent(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    extractor = ExperienceExtractor(knowledge)
    try:
        first = extractor.extract(
            competition="birdclef-2026",
            experiment=_experiment(),
            comparison={"delta": 0.006, "verdict": "worth_keeping", "maximize": True},
        )
        second = extractor.extract(
            competition="birdclef-2026",
            experiment=_experiment(description="Added SpecAugment + EMA (v2)"),
            comparison={"delta": 0.007, "verdict": "worth_keeping", "maximize": True},
        )
        assert second.id == first.id
        assert len(extractor.store.list()) == 1
        assert "(v2)" in second.action
    finally:
        extractor.close()


def test_extract_copies_git_commit_from_workspace_record(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "birdclef-2026"
    workspace = competition_workspace_path(knowledge, competition)
    workspace.mkdir(parents=True, exist_ok=True)
    write_experiment_git_record(
        workspace,
        {
            "experiment_id": "exp-9",
            "execution_id": "E-009",
            "competition": competition,
            "status": "completed",
            "metrics": {"cv_score": 0.81},
            "git_commit": "1111222233334444",
            "files_changed": ["src/augment.py"],
        },
    )

    extractor = ExperienceExtractor(knowledge)
    try:
        record = extractor.extract(
            competition=competition,
            experiment_id="exp-9",
            comparison={"delta": 0.02, "verdict": "worth_keeping", "maximize": True},
        )
        assert record.idempotency_key == "exp-9"
        assert record.artifacts.git_commit == "1111222233334444"
        assert record.artifacts.execution_id == "E-009"
        assert "augment" in record.action.lower() or "files:" in record.action
        assert record.outcome == "success"
    finally:
        extractor.close()


def test_extract_fail_on_regression(tmp_path: Path) -> None:
    extractor = ExperienceExtractor(tmp_path / "knowledge")
    try:
        record = extractor.extract(
            competition="titanic",
            experiment=_experiment(
                id="run-bad",
                competition="titanic",
                description="Dropped features",
                problem_type="tabular",
                metrics={"cv_score": 0.70},
                git_commit=None,
            ),
            comparison={"delta": -0.05, "verdict": "regression", "maximize": True},
        )
        assert record.outcome == "fail"
        facets = _facet_map(record)
        assert "tabular" in facets
        assert facets["tabular"].source == "metadata"
        assert record.artifacts.git_commit is None
    finally:
        extractor.close()


def test_low_signal_rules_have_lower_confidence(tmp_path: Path) -> None:
    extractor = ExperienceExtractor(tmp_path / "knowledge")
    try:
        record = extractor.extract(
            competition="misc-comp",
            experiment=_experiment(
                id="weak",
                competition="misc-comp",
                description="tried ema once",
                problem_type=None,
                metrics={"cv_score": 0.5},
                git_commit=None,
            ),
            comparison={"delta": 0.0, "verdict": "inconclusive", "maximize": True},
        )
        facets = _facet_map(record)
        assert "augmentation" in facets
        assert facets["augmentation"].confidence == 0.45
        assert facets["augmentation"].evidence == ["ema"]
        assert facets["augmentation"].source == "rules"
    finally:
        extractor.close()
