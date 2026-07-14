"""Unit tests for Milestone 2 Plan 5 — Knowledge Base."""

from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.experiments.knowledge import (
    KnowledgeBase,
    technique_tags_from_changes,
)
from labpilot.experiments.models import (
    ChangeCategory,
    ConfigChange,
    ExperimentComparison,
    KnowledgeEffect,
    StructuredReflection,
    Verdict,
)


def _comparison(
    *,
    compare_id: str,
    delta: float,
    changes: list[ConfigChange],
    metric_key: str = "cv_accuracy",
) -> ExperimentComparison:
    return ExperimentComparison(
        base_id="base",
        compare_id=compare_id,
        primary_metric_key=metric_key,
        metric_deltas={metric_key: delta},
        changes=changes,
        runtime_delta_seconds=None,
        runtime_delta_pct=None,
        verdict=Verdict.WORTH_KEEPING if delta > 0 else Verdict.REGRESSION,
        verdict_reason="test",
    )


def _ema_change() -> ConfigChange:
    return ConfigChange(
        category=ChangeCategory.AUGMENTATION,
        field="feature_recipes",
        base_value=None,
        compare_value="EMA",
        label="+ EMA",
    )


def test_technique_tags_prefer_field_and_recipe_names():
    changes = [
        ConfigChange(
            category=ChangeCategory.TRAINING_STRATEGY,
            field="model_params.learning_rate",
            base_value=0.05,
            compare_value=0.03,
            label="learning_rate: 0.05 → 0.03",
        ),
        ConfigChange(
            category=ChangeCategory.FEATURE_ENGINEERING,
            field="feature_recipes",
            base_value=None,
            compare_value="target_encoding",
            label="+ target_encoding",
        ),
    ]
    assert technique_tags_from_changes(changes) == ["learning_rate", "target_encoding"]


def test_two_consistent_positive_comparisons_raise_confidence(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    change = _ema_change()

    first = kb.update_from_comparison(
        _comparison(compare_id="run-1", delta=0.02, changes=[change]),
        maximize=True,
    )
    assert len(first) == 1
    assert first[0].technique == "ema"
    assert first[0].sample_size == 1
    assert first[0].effect == KnowledgeEffect.IMPROVES
    conf1 = first[0].confidence

    second = kb.update_from_comparison(
        _comparison(compare_id="run-2", delta=0.01, changes=[change]),
        maximize=True,
    )
    assert second[0].sample_size == 2
    assert second[0].effect == KnowledgeEffect.IMPROVES
    assert second[0].confidence > conf1
    assert (tmp_path / "knowledge/titanic/knowledge_base.json").is_file()


def test_opposite_sign_penalizes_confidence(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    change = _ema_change()
    kb.update_from_comparison(
        _comparison(compare_id="run-1", delta=0.02, changes=[change]),
    )
    before = kb.get("ema", "cv_accuracy")
    assert before is not None
    conf_before = before.confidence

    kb.update_from_comparison(
        _comparison(compare_id="run-2", delta=-0.03, changes=[change]),
    )
    after = kb.get("ema", "cv_accuracy")
    assert after is not None
    assert after.confidence < conf_before
    assert after.confidence == pytest.approx(max(0.3, conf_before - 0.15))


def test_idempotent_same_run_id(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    change = _ema_change()
    cmp = _comparison(compare_id="run-1", delta=0.02, changes=[change])
    kb.update_from_comparison(cmp)
    kb.update_from_comparison(cmp)
    entry = kb.get("ema", "cv_accuracy")
    assert entry is not None
    assert entry.sample_size == 1


def test_minimize_metric_signs_delta(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    change = _ema_change()
    # Raw logloss dropped (negative delta) but minimize → signed positive improvement.
    kb.update_from_comparison(
        _comparison(
            compare_id="run-1",
            delta=-0.05,
            changes=[change],
            metric_key="cv_logloss",
        ),
        maximize=False,
    )
    entry = kb.get("ema", "cv_logloss")
    assert entry is not None
    assert entry.effect == KnowledgeEffect.IMPROVES
    assert entry.delta_estimate == pytest.approx(0.05)


def test_top_discoveries_and_known_failures(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    kb.update_from_comparison(
        _comparison(
            compare_id="a",
            delta=0.05,
            changes=[
                ConfigChange(
                    category=ChangeCategory.AUGMENTATION,
                    field="feature_recipes",
                    base_value=None,
                    compare_value="mixup",
                    label="+ mixup",
                )
            ],
        )
    )
    kb.update_from_comparison(
        _comparison(
            compare_id="b",
            delta=-0.04,
            changes=[
                ConfigChange(
                    category=ChangeCategory.FEATURE_ENGINEERING,
                    field="feature_recipes",
                    base_value=None,
                    compare_value="target_encoding",
                    label="+ target_encoding",
                )
            ],
        )
    )
    top = kb.top_discoveries(3)
    assert top and top[0].technique == "mixup"
    fails = kb.known_failures(3)
    assert fails and fails[0].technique == "target_encoding"


def test_update_from_reflection_adds_unknown_tags(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    reflection = StructuredReflection(
        run_id="run-r1",
        observation="o",
        evidence=[],
        likely_cause="c",
        confidence=0.7,
        suggested_next=[],
        new_hypotheses=[
            {
                "observation": "a",
                "reason": "b",
                "prediction": "p",
                "confidence": 0.6,
                "tags": ["TitleExtraction", "leakage"],
            }
        ],
        generated_by="llm",
    )
    updated = kb.update_from_reflection(reflection, metric_key="cv_accuracy")
    assert len(updated) == 2
    assert all(e.effect == KnowledgeEffect.UNKNOWN for e in updated)
    assert all(e.confidence <= 0.4 for e in updated)


def test_reflection_skips_tags_already_corroborated(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    kb.update_from_comparison(
        _comparison(compare_id="run-1", delta=0.02, changes=[_ema_change()])
    )
    reflection = StructuredReflection(
        run_id="run-r1",
        observation="o",
        evidence=[],
        likely_cause="c",
        confidence=0.8,
        suggested_next=[],
        new_hypotheses=[
            {
                "observation": "a",
                "reason": "b",
                "prediction": "p",
                "confidence": 0.5,
                "tags": ["EMA", "new-idea"],
            }
        ],
        generated_by="llm",
    )
    updated = kb.update_from_reflection(reflection, metric_key="cv_accuracy")
    assert len(updated) == 1
    assert updated[0].technique == "new-idea"
    assert kb.get("ema", "cv_accuracy") is not None
    assert kb.get("ema", "cv_accuracy").effect == KnowledgeEffect.IMPROVES


def test_knowledge_list_cli_filters_hurts(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    kb.update_from_comparison(
        _comparison(
            compare_id="b",
            delta=-0.04,
            changes=[
                ConfigChange(
                    category=ChangeCategory.FEATURE_ENGINEERING,
                    field="feature_recipes",
                    base_value=None,
                    compare_value="target_encoding",
                    label="+ target_encoding",
                )
            ],
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "knowledge",
            "list",
            "--competition",
            "titanic",
            "--effect",
            "hurts",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "Knowledge: titanic" in result.output
    assert "hurts" in result.output
    # Technique column may be ellipsized by Rich in CliRunner; verify filter via API.
    entries = kb.list_entries(effect=KnowledgeEffect.HURTS)
    assert len(entries) == 1
    assert entries[0].technique == "target_encoding"
