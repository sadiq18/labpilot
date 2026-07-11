from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def titanic_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "titanic-fixture"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "PassengerId": range(1, 13),
            "Survived": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1],
            "Name": [f"Passenger {index}" for index in range(1, 13)],
            "Sex": ["male", "female"] * 6,
            "Age": [22, 38, 26, 35, 28, None, 42, 19, 31, 24, 45, 30],
            "Fare": [7.25, 71.28, 7.92, 53.1, 13.0, 8.05, 26.0, 91.08, 7.75, 12.35, 8.46, 83.16],
            "Embarked": ["S", "C", "S", "S", "Q", "S", "S", "C", "Q", "S", "S", "C"],
        }
    )
    test = pd.DataFrame(
        {
            "PassengerId": range(13, 17),
            "Pclass": [3, 1, 2, 3],
            "Name": [f"Test Passenger {index}" for index in range(13, 17)],
            "Sex": ["male", "female", "female", "male"],
            "Age": [34, 27, None, 40],
            "Fare": [8.05, 82.17, 15.5, 7.9],
            "Embarked": ["S", "C", "Q", "S"],
        }
    )
    sample_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [0, 0, 0, 0]})

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "gender_submission.csv", index=False)
    return data_dir


@pytest.fixture
def generic_regression_data_dir(tmp_path: Path) -> Path:
    """A synthetic, competition-agnostic tabular regression dataset."""
    data_dir = tmp_path / "regression-fixture"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "id": range(1, 21),
            "size": [i * 10.5 for i in range(1, 21)],
            "category": ["a", "b", "c"] * 6 + ["a", "b"],
            "target": [i * 1000.0 + 500.33 for i in range(1, 21)],
        }
    )
    test = pd.DataFrame(
        {
            "id": range(21, 26),
            "size": [i * 10.5 for i in range(21, 26)],
            "category": ["a", "b", "c", "a", "b"],
        }
    )
    sample_submission = pd.DataFrame({"id": test["id"], "target": [0.0] * len(test)})

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


@pytest.fixture
def competition_configs_dir(tmp_path: Path) -> Path:
    """A local, non-committed directory of competition contracts for tests.

    Competition contracts are never committed to the repo (see
    configs/competitions/README.md), so tests create their own here instead
    of depending on a file in the working tree.
    """
    configs_dir = tmp_path / "competition-configs"
    configs_dir.mkdir()
    (configs_dir / "titanic.yaml").write_text(
        "title: Titanic - Machine Learning from Disaster\n"
        "description: Predict which passengers survived the Titanic shipwreck.\n"
        "problem_type: tabular_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - PassengerId\n"
        "  - Survived\n"
    )
    (configs_dir / "generic-regression-competition.yaml").write_text(
        "title: Generic Regression Competition\n"
        "description: A synthetic, competition-agnostic regression fixture.\n"
        "problem_type: tabular_regression\n"
        "evaluation_metric:\n"
        "  name: rmse\n"
        "  direction: minimize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - target\n"
    )
    return configs_dir
