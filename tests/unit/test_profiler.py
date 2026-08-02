from pathlib import Path

import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.selector import BaselineSelector
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec, MetricSpec, ProblemType
from labpilot.config import ProfilerConfig
from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler


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


def test_column_profile_is_numeric_flag(sample_csv: Path):
    """`is_numeric` must reflect actual numeric-ness rather than a fixed set
    of dtype-string spellings — pandas has changed how it spells "this is a
    plain string column" across versions (e.g. "object" vs "str"), so
    anything downstream that needs "is this categorical?" should read this
    flag instead of re-deriving it from `ColumnProfile.dtype`.
    """
    profiler = TabularProfiler.__new__(TabularProfiler)
    profiler.config = type("C", (), {"max_rows_sample": 1000})()

    profile = TabularProfiler.profile_file(profiler, sample_csv)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["id"].is_numeric is True
    assert by_name["feature_a"].is_numeric is True
    assert by_name["feature_b"].is_numeric is False
    assert by_name["target"].is_numeric is True


def test_column_profile_treats_bool_as_non_numeric(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2, 3, 4], "flag": [True, False, True, False]})
    path = tmp_path / "bool.csv"
    df.to_csv(path, index=False)

    profiler = TabularProfiler.__new__(TabularProfiler)
    profiler.config = type("C", (), {"max_rows_sample": 1000})()
    profile = TabularProfiler.profile_file(profiler, path)

    flag = next(column for column in profile.columns if column.name == "flag")
    assert flag.is_numeric is False


def test_baseline_selector_defaults():
    competition = CompetitionSpec(slug="titanic")
    profile = DatasetProfile(competition="titanic", row_count=891, column_count=12)

    choice = BaselineSelector().select(competition, profile)

    assert choice.problem_type == "tabular_classification"
    assert choice.template_name == "tabular_classification"
    assert choice.metric_name == "accuracy"


def test_baseline_selector_metric_name_ignores_mismatched_competition_metric():
    """The P0 regression template only ever writes `cv_rmse`, so the metric
    key used for evaluation must come from the fixed per-problem-type
    default, not from whatever a competition's (possibly auto-resolved)
    metadata says — otherwise a real Kaggle metric like RMSLE would make an
    otherwise-correct run fail at the evaluate_cv stage.
    """
    competition = CompetitionSpec(
        slug="house-prices",
        problem_type=ProblemType.TABULAR_REGRESSION,
        evaluation_metric=MetricSpec(name="rmsle", direction="minimize"),
    )
    profile = DatasetProfile(competition="house-prices", row_count=100, column_count=5)

    choice = BaselineSelector().select(competition, profile)

    assert choice.metric_name == "rmse"


def test_baseline_selector_infers_classification_for_string_multiclass_target():
    """A string-labeled target with many classes must still be read as
    classification, not regression — this is the scenario that first
    surfaced the pandas 3.0 "str" dtype vs "object" mismatch (see
    `test_column_profile_is_numeric_flag`).
    """
    from labpilot.accessor.profiler.tabular import ColumnProfile

    competition = CompetitionSpec(slug="species-competition")
    profile = DatasetProfile(
        competition="species-competition",
        row_count=18,
        column_count=3,
        target_column="species",
        columns=[
            ColumnProfile(name="species", dtype="str", unique_count=3, is_numeric=False),
        ],
    )

    choice = BaselineSelector().select(competition, profile)

    assert choice.problem_type == "tabular_classification"


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


def test_profile_directory_infers_image_layout_without_test_csv(tmp_path: Path):
    data_dir = tmp_path / "image-fixture"
    data_dir.mkdir()
    train_dir = data_dir / "train"
    train_dir.mkdir()
    (train_dir / "img_a.jpg").write_bytes(b"fake")

    training = pd.DataFrame(
        {
            "id": [f"img_{i}.jpg" for i in range(10)],
            "has_cactus": [1] * 10,
        }
    )
    for i in range(10):
        (train_dir / f"img_{i}.jpg").write_bytes(b"fake")
    submission = pd.DataFrame({"id": ["img_b.jpg", "img_c.jpg"], "has_cactus": [0.5, 0.5]})

    training.to_csv(data_dir / "train.csv", index=False)
    submission.to_csv(data_dir / "sample_submission.csv", index=False)

    profile = TabularProfiler(ProfilerConfig()).profile_directory(data_dir, "aerial-cactus")

    assert profile.train_file == "train.csv"
    assert profile.test_file == "sample_submission.csv"
    assert profile.target_column == "has_cactus"
    assert profile.id_column == "id"
    assert profile.modality == "image"
    assert profile.image_column == "id"


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


# --- partitioned (one file per entity) datasets -----------------------------


@pytest.fixture
def partitioned_data_dir(tmp_path):
    """train/<entity>__<kind>.csv layout, with a train-only label column."""
    root = tmp_path / "partitioned-fixture"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()

    def rows(seed: int, n: int, with_label: bool):
        frame = {
            "MD": [seed + i for i in range(n)],
            "GR": [float(i % 7) for i in range(n)],
            "feat_in": [float(i) for i in range(n)],
        }
        if with_label:
            frame["LABEL"] = [float(i) * 2 for i in range(n)]
            frame["train_only_marker"] = [float(i) for i in range(n)]
        return pd.DataFrame(frame)

    for i in range(6):
        entity = f"e{i:03d}"
        rows(i * 100, 10, True).to_csv(root / "train" / f"{entity}__main.csv", index=False)
        rows(i * 100, 4, True).to_csv(root / "train" / f"{entity}__ref.csv", index=False)
    for i in range(2):
        entity = f"t{i:03d}"
        rows(i * 100, 10, False).to_csv(root / "test" / f"{entity}__main.csv", index=False)
        rows(i * 100, 4, False).to_csv(root / "test" / f"{entity}__ref.csv", index=False)

    pd.DataFrame({"id": ["t000_1", "t000_2"], "label": [0.0, 0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return root


def _profiler():
    from labpilot.config import ProfilerConfig
    from labpilot.accessor.profiler.tabular import TabularProfiler

    return TabularProfiler(ProfilerConfig())


def test_partitioned_dataset_is_detected_instead_of_raising(partitioned_data_dir):
    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    assert profile.partitioned is True
    assert profile.train_partition_count == 6
    assert profile.test_partition_count == 2
    assert profile.partition_kinds == {"main": 6, "ref": 6}


def test_partitioned_target_inferred_from_train_test_schema_diff(partitioned_data_dir):
    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    # LABEL is train-only AND named in the submission header -> the target.
    # train_only_marker is train-only but not in the submission -> not target.
    assert profile.target_column == "LABEL"
    assert profile.submission_columns == ["id", "label"]


def test_partitioned_warns_that_rows_are_not_iid(partitioned_data_dir):
    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    joined = " ".join(profile.warnings)
    assert "NOT iid" in joined
    assert "train_only_marker" in joined  # leakage columns surfaced


def test_partitioned_row_count_is_estimated_not_zero(partitioned_data_dir):
    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    assert profile.row_count == 60  # 6 partitions x 10 rows
    assert profile.row_count_estimated is True
    assert profile.column_count == 5


def test_single_train_file_dataset_is_unaffected(titanic_data_dir):
    profile = _profiler().profile_directory(titanic_data_dir, "titanic")
    assert profile.partitioned is False
    assert profile.train_file == "train.csv"
    assert profile.row_count == 12


# --- validation plan derivation --------------------------------------------


def _suffix_scored_dir(tmp_path):
    """Partitioned dataset whose submission ids cover only each partition's tail."""
    root = tmp_path / "suffix-fixture"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()

    def frame(n, with_label):
        data = {"MD": list(range(n)), "GR": [float(i % 5) for i in range(n)]}
        if with_label:
            data["LABEL"] = [float(i) for i in range(n)]
            data["leak_marker"] = [1.0] * n
        return pd.DataFrame(data)

    for i in range(4):
        frame(10, True).to_csv(root / "train" / f"e{i}__main.csv", index=False)
    for i in range(2):
        frame(10, False).to_csv(root / "test" / f"t{i}__main.csv", index=False)

    ids = [f"t{w}_{idx}" for w in range(2) for idx in range(6, 10)]  # tail 6..9
    pd.DataFrame({"id": ids, "label": [0.0] * len(ids)}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return root


def test_suffix_scoring_detected(tmp_path):
    profile = _profiler().profile_directory(_suffix_scored_dir(tmp_path), "suffix-comp")
    assert profile.scored_is_partition_suffix is True
    assert profile.scored_fraction == pytest.approx(0.4)


def test_validation_plan_uses_suffix_holdout_for_forecast_tasks(tmp_path):
    from labpilot.research_engine.execution.baseline.selector import derive_validation_plan

    profile = _profiler().profile_directory(_suffix_scored_dir(tmp_path), "suffix-comp")
    plan = derive_validation_plan(profile)
    assert plan.scheme == "partition_suffix_holdout"
    assert plan.holdout_fraction == pytest.approx(0.4)
    # leakage column excluded, target kept
    assert "leak_marker" in plan.exclude_features
    assert profile.target_column not in plan.exclude_features


def test_validation_plan_groups_for_partitioned_non_forecast(partitioned_data_dir):
    from labpilot.research_engine.execution.baseline.selector import derive_validation_plan

    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    profile.scored_is_partition_suffix = False
    plan = derive_validation_plan(profile)
    assert plan.scheme == "group_kfold"
    assert plan.group_key == "file_stem_entity"


def test_validation_plan_plain_kfold_for_iid_dataset(titanic_data_dir):
    from labpilot.research_engine.execution.baseline.selector import derive_validation_plan

    profile = _profiler().profile_directory(titanic_data_dir, "titanic")
    plan = derive_validation_plan(profile)
    assert plan.scheme == "kfold"
    assert plan.exclude_features == []


def test_partitioned_dataset_selects_partition_aware_template(partitioned_data_dir):
    from labpilot.research_engine.execution.baseline.selector import BaselineSelector
    from labpilot.research_engine.intelligence.competition.models import (
        CompetitionSpec,
        MetricSpec,
        ProblemType,
    )

    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    spec = CompetitionSpec(
        slug="part-comp",
        problem_type=ProblemType.TABULAR_REGRESSION,
        evaluation_metric=MetricSpec(name="mse", direction="minimize", key="mse"),
    )
    choice = BaselineSelector().select(spec, profile)
    assert choice.template_name == "tabular_regression_partitioned"
    assert choice.partitioned is True


def test_iid_dataset_keeps_plain_regression_template(generic_regression_data_dir):
    from labpilot.research_engine.execution.baseline.selector import BaselineSelector
    from labpilot.research_engine.intelligence.competition.models import (
        CompetitionSpec,
        MetricSpec,
        ProblemType,
    )

    profile = _profiler().profile_directory(generic_regression_data_dir, "reg-comp")
    spec = CompetitionSpec(
        slug="reg-comp",
        problem_type=ProblemType.TABULAR_REGRESSION,
        evaluation_metric=MetricSpec(name="rmse", direction="minimize", key="rmse"),
    )
    choice = BaselineSelector().select(spec, profile)
    assert choice.template_name == "tabular_regression"


def test_partitioned_template_is_registered():
    from labpilot.research_engine.execution.baseline.registry import get_template

    template = get_template("tabular_regression", template_name="tabular_regression_partitioned")
    assert template is not None
    assert (template.template_dir / "train.py.j2").is_file()
    # default for the problem type must stay the plain template
    assert get_template("tabular_regression").name == "tabular_regression"


def test_flat_multi_file_dataset_is_not_treated_as_partitioned(tmp_path):
    """Regression: `len(train_files) > 1` matched train.csv + train_extra.csv by
    filename prefix, so an ordinary multi-file dataset took the partitioned
    path and would have been given group splits and a partition-aware template."""
    root = tmp_path / "flat"
    root.mkdir()
    for name in ("train.csv", "train_extra.csv", "test.csv"):
        pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [1, 0, 1]}).to_csv(
            root / name, index=False
        )
    pd.DataFrame({"id": [1], "y": [0]}).to_csv(root / "sample_submission.csv", index=False)

    # Two ambiguous train CSVs is a genuine error, not a partitioned dataset.
    with pytest.raises(ValueError):
        _profiler().profile_directory(root, "flat")


def test_two_partitions_are_too_few_to_infer_a_partitioned_layout(tmp_path):
    """A directory with a couple of files is not evidence of one-file-per-entity."""
    root = tmp_path / "thin"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i in range(2):
        pd.DataFrame({"MD": [1, 2], "GR": [1.0, 2.0], "LABEL": [1.0, 2.0]}).to_csv(
            root / "train" / f"e{i}__main.csv", index=False
        )
    pd.DataFrame({"MD": [1, 2], "GR": [1.0, 2.0]}).to_csv(
        root / "test" / "t0__main.csv", index=False
    )
    pd.DataFrame({"id": ["t0_1"], "label": [0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )
    profile = _profiler()._try_profile_partitioned(
        root,
        "thin",
        sorted(root.rglob("*.csv")),
        train_pattern="train",
        test_pattern="test",
        submission_pattern="submission",
    )
    assert profile is None


def test_real_partitioned_layout_still_detected(partitioned_data_dir):
    """The strengthened gate must not break the case it was built for."""
    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    assert profile.partitioned is True
    assert profile.train_partition_count == 6
