from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_real_dotenv_in_tests(monkeypatch):
    """Tests must never read a developer's real `.env` at the repo root
    (a normal, README-documented setup for real Kaggle/LLM credentials).
    Without this, `labpilot.config.Settings()` — used internally by
    `resolve_llm_client()` and friends — would silently pick up real API
    keys and make real, slow, rate-limited network calls from otherwise
    hermetic unit/integration tests. Tests that want specific env vars
    can still set them explicitly via `monkeypatch.setenv(...)`.

    Two layers to neutralize, both required: (1) `uv run` itself loads
    `.env` and injects its values directly into the process's real
    `os.environ` *before* pytest even starts, so `monkeypatch.delenv` on
    every var `Settings` reads is what actually matters; (2) disabling
    `env_file` on `Settings.model_config` additionally guards against
    `Settings()` re-reading `.env` on its own (e.g. if `uv run` isn't
    what invoked pytest).
    """
    from labpilot.config import Settings

    for var in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "LABPILOT_RUNS_DIR",
        "LABPILOT_KNOWLEDGE_DIR",
        "LABPILOT_LLM_PROVIDER",
        "LABPILOT_LLM_MODEL",
        "LABPILOT_RUNTIMES_DIR",
        "LABPILOT_DEFAULT_RUNTIME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


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
def multiclass_data_dir(tmp_path: Path) -> Path:
    """A synthetic 3-class dataset with *string* labels (unlike Titanic's
    numeric binary target), to exercise multi-class support end-to-end.
    """
    data_dir = tmp_path / "multiclass-fixture"
    data_dir.mkdir()

    species = ["setosa", "versicolor", "virginica"]
    rows_per_class = 6
    train_species = species * rows_per_class
    train = pd.DataFrame(
        {
            "id": range(1, len(train_species) + 1),
            "petal_length": [1.4 + 0.1 * i for i in range(len(train_species))],
            "petal_width": [0.2 + 0.05 * i for i in range(len(train_species))],
            "color": (["pale", "bright"] * (len(train_species) // 2 + 1))[: len(train_species)],
            "species": train_species,
        }
    )
    test_species = species * 2
    test = pd.DataFrame(
        {
            "id": range(len(train_species) + 1, len(train_species) + len(test_species) + 1),
            "petal_length": [1.5 + 0.2 * i for i in range(len(test_species))],
            "petal_width": [0.3 + 0.04 * i for i in range(len(test_species))],
            "color": (["pale", "bright"] * (len(test_species) // 2 + 1))[: len(test_species)],
        }
    )
    sample_submission = pd.DataFrame({"id": test["id"], "species": ["setosa"] * len(test)})

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
    (configs_dir / "generic-multiclass-competition.yaml").write_text(
        "title: Generic Multi-Class Competition\n"
        "description: A synthetic, competition-agnostic multi-class classification fixture.\n"
        "problem_type: tabular_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - species\n"
    )
    (configs_dir / "text-sentiment.yaml").write_text(
        "title: Text Sentiment Fixture\n"
        "description: Small text classification fixture for integration tests.\n"
        "problem_type: text_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - label\n"
    )
    (configs_dir / "image-pets.yaml").write_text(
        "title: Image Pets Fixture\n"
        "description: Tiny image classification fixture for integration tests.\n"
        "problem_type: image_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - label\n"
    )
    (configs_dir / "text-deep.yaml").write_text(
        "title: Text Deep Fixture\n"
        "description: Deep text classification fixture.\n"
        "problem_type: text_classification\n"
        "baseline_strategy: deep\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - label\n"
    )
    (configs_dir / "image-deep.yaml").write_text(
        "title: Image Deep Fixture\n"
        "description: Deep image classification fixture.\n"
        "problem_type: image_classification\n"
        "baseline_strategy: deep\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - label\n"
    )
    return configs_dir


@pytest.fixture
def text_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "text-fixture"
    data_dir.mkdir()
    train = pd.DataFrame(
        {
            "id": range(1, 13),
            "text": [
                "This product is absolutely wonderful and exceeded expectations.",
                "Terrible experience, would not recommend to anyone at all.",
                "Average quality, nothing special but acceptable for the price.",
                "Outstanding service and fast delivery, very happy customer.",
                "Worst purchase ever, complete waste of money and time.",
                "Pretty good overall, minor issues but generally satisfied.",
            ]
            * 2,
            "label": ["positive", "negative", "neutral"] * 4,
        }
    )
    test = pd.DataFrame(
        {
            "id": range(13, 17),
            "text": [
                "Great value and excellent build quality throughout.",
                "Disappointed with the results and poor support.",
                "Neutral feelings about this item, neither good nor bad.",
                "Amazing features and intuitive design, love it.",
            ],
        }
    )
    sample = pd.DataFrame({"id": test["id"], "label": ["positive"] * len(test)})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


@pytest.fixture
def image_data_dir(tmp_path: Path) -> Path:
    pytest.importorskip("PIL")
    from PIL import Image

    data_dir = tmp_path / "image-fixture"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)

    labels = ["cat", "dog", "cat", "dog", "cat", "dog"] * 2
    train_rows = []
    test_rows = []
    for index, label in enumerate(labels, start=1):
        filename = f"{index}.jpg"
        Image.new("RGB", (32, 32), color=(index * 20 % 255, 50, 100)).save(
            images_dir / filename
        )
        train_rows.append({"id": index, "file": filename, "label": label})
    for index in range(len(labels) + 1, len(labels) + 5):
        filename = f"{index}.jpg"
        Image.new("RGB", (32, 32), color=(100, index * 15 % 255, 50)).save(
            images_dir / filename
        )
        test_rows.append({"id": index, "file": filename})

    pd.DataFrame(train_rows).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [row["id"] for row in test_rows], "label": ["cat"] * len(test_rows)}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir
