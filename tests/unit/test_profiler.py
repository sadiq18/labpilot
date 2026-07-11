from pathlib import Path

import pandas as pd
import pytest

from labpilot.baseline.selector import BaselineSelector
from labpilot.competition.models import CompetitionSpec
from labpilot.config import ProfilerConfig
from labpilot.profiler.tabular import DatasetProfile, TabularProfiler


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "feature_a": [1.0, 2.0, None, 4.0, 5.0],
            "feature_b": ["a", "b", "a", "c", "b"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_tabular_profiler(sample_csv: Path):
    profiler = TabularProfiler.__new__(TabularProfiler)
    profiler.config = type("C", (), {"max_rows_sample": 1000})()

    profile = TabularProfiler.profile_file(profiler, sample_csv)

    assert profile.row_count == 5
    assert profile.column_count == 4
    assert len(profile.columns) == 4


def test_baseline_selector_defaults():
    competition = CompetitionSpec(slug="titanic")
    profile = DatasetProfile(competition="titanic", row_count=891, column_count=12)

    choice = BaselineSelector().select(competition, profile)

    assert choice.problem_type == "tabular_classification"
    assert choice.template_name == "tabular_classification"


def test_profile_directory_infers_titanic_contract(titanic_data_dir: Path):
    profile = TabularProfiler(ProfilerConfig()).profile_directory(
        titanic_data_dir,
        "titanic",
    )

    assert profile.train_file == "train.csv"
    assert profile.test_file == "test.csv"
    assert profile.sample_submission_file == "gender_submission.csv"
    assert profile.target_column == "Survived"
    assert profile.id_column == "PassengerId"
    assert profile.submission_columns == ["PassengerId", "Survived"]
    assert profile.test_row_count == 4


def test_profile_directory_honors_custom_file_patterns(tmp_path: Path):
    data_dir = tmp_path / "custom-fixture"
    data_dir.mkdir()

    # File names don't start with "train"/"test", so the default patterns
    # would fail to find them; a competition-specific override should still
    # let the profiler resolve the correct roles.
    training = pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 0]})
    scoring = pd.DataFrame({"id": [4, 5]})
    submission = pd.DataFrame({"id": [4, 5], "label": [0, 0]})

    training.to_csv(data_dir / "learn_data.csv", index=False)
    scoring.to_csv(data_dir / "score_data.csv", index=False)
    submission.to_csv(data_dir / "answer_key.csv", index=False)

    profile = TabularProfiler(ProfilerConfig()).profile_directory(
        data_dir,
        "generic-competition",
        train_pattern="learn",
        test_pattern="score",
        submission_pattern="answer",
    )

    assert profile.train_file == "learn_data.csv"
    assert profile.test_file == "score_data.csv"
    assert profile.sample_submission_file == "answer_key.csv"
    assert profile.target_column == "label"
    assert profile.id_column == "id"
