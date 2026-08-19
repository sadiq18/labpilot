from pathlib import Path

import pandas as pd
import pytest

from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig
from labpilot.research_engine.execution.baseline.selector import BaselineSelector
from labpilot.research_engine.intelligence.competition.models import (
    CompetitionSpec,
    MetricSpec,
    ProblemType,
)


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
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

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
    """Every kind counts, because `load_data` concatenates every CSV.

    This asserted 60 — the primary kind alone, 6 files x 10 rows — while the
    fixture also holds 6 `__ref` files of 4 rows. The frame the pipeline builds
    has 84. Counting one kind's columns and another kind's rows was the same
    asymmetry, one field apart.
    """
    profile = _profiler().profile_directory(partitioned_data_dir, "part-comp")
    assert profile.row_count == 84  # 6x10 main + 6x4 ref
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
    # No `.j2` to check any more: M19 §2 deleted the pack, and the registry is
    # a declared catalogue of what a baseline looks like per problem type —
    # model family and validation plan, which codegen reads whatever writes
    # the code.
    assert template.model_family == "lightgbm"
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


# --- a partitioned profile must describe the frame the pipeline builds -------


def _partitioned_dataset(root, *, kinds: dict[str, dict[str, list]], n: int = 4):
    """`train/<entity>__<kind>.csv` for several entities, one schema per kind."""
    import pandas as pd

    train = root / "train"
    test = root / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    for index in range(n):
        for kind, data in kinds.items():
            pd.DataFrame(data).to_csv(train / f"e{index}__{kind}.csv", index=False)
            pd.DataFrame({k: v for k, v in data.items() if k != "TVT"}).to_csv(
                test / f"e{index}__{kind}.csv", index=False
            )
    pd.DataFrame({"id": [0], "TVT": [0.0]}).to_csv(root / "sample_submission.csv", index=False)
    return root


def test_a_column_that_only_exists_in_a_second_kind_is_profiled(tmp_path):
    """Measured on rogii 2026-08-09, twice, two days apart.

    The dataset has two kinds of file per well. `max()` over the kind counts
    picked `horizontal_well`, and the profile was built from one file of that
    kind — so `Geology`, which lives only in `typewell`, never appeared. The
    generated `load_data` concatenates *all* the CSVs, so the training frame
    had it anyway, and codegen — told the data was thirteen columns and every
    one numeric — wrote "use every column except this exclusion list".

    LightGBM: `pandas dtypes must be int, float or bool. Fields with bad pandas
    dtypes: Geology: object`.
    """
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    root = _partitioned_dataset(
        tmp_path,
        kinds={
            "horizontal_well": {"MD": [1.0, 2.0], "GR": [3.0, 4.0], "TVT": [5.0, 6.0]},
            "typewell": {"GR": [7.0, 8.0], "Geology": ["shale", "sand"], "TVT": [9.0, 1.0]},
        },
    )

    profile = TabularProfiler(ProfilerConfig()).profile_directory(root, "demo")

    names = {c.name for c in profile.columns}
    assert "Geology" in names, f"only saw {sorted(names)}"


def test_that_column_is_marked_non_numeric(tmp_path):
    """Presence is not enough — codegen decides what to feed the model from
    `is_numeric`, and a string column reported as numeric is the same crash."""
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    root = _partitioned_dataset(
        tmp_path,
        kinds={
            "horizontal_well": {"MD": [1.0, 2.0], "GR": [3.0, 4.0], "TVT": [5.0, 6.0]},
            "typewell": {"GR": [7.0, 8.0], "Geology": ["shale", "sand"], "TVT": [9.0, 1.0]},
        },
    )

    profile = TabularProfiler(ProfilerConfig()).profile_directory(root, "demo")
    geology = next(c for c in profile.columns if c.name == "Geology")

    assert geology.is_numeric is False


def test_columns_from_the_primary_kind_survive(tmp_path):
    """Widening the profile must not lose what it already reported."""
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    root = _partitioned_dataset(
        tmp_path,
        kinds={
            "horizontal_well": {"MD": [1.0, 2.0], "GR": [3.0, 4.0], "TVT": [5.0, 6.0]},
            "typewell": {"GR": [7.0, 8.0], "Geology": ["shale", "sand"], "TVT": [9.0, 1.0]},
        },
    )

    profile = TabularProfiler(ProfilerConfig()).profile_directory(root, "demo")
    names = {c.name for c in profile.columns}

    assert {"MD", "GR", "TVT"} <= names


def test_a_single_kind_dataset_is_unchanged(tmp_path):
    """No second kind, nothing to union — the common case must not shift."""
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    root = _partitioned_dataset(
        tmp_path,
        kinds={"well": {"MD": [1.0, 2.0], "GR": [3.0, 4.0], "TVT": [5.0, 6.0]}},
    )

    profile = TabularProfiler(ProfilerConfig()).profile_directory(root, "demo")

    assert {c.name for c in profile.columns} == {"MD", "GR", "TVT"}
    assert all(c.is_numeric for c in profile.columns)


def test_a_column_in_a_non_primary_kinds_test_files_is_not_train_only(tmp_path):
    """Reported on PR #117 and reproduced.

    The sample frame spans every kind, but `test_columns` was read from the
    primary kind's test files alone — so a column present in *another* kind's
    train **and** test looked train-only. `train_only[-1]` is the target
    fallback, so a categorical feature could be inferred as the label while the
    real target was passed over, and codegen would train against the wrong
    column entirely.
    """
    import pandas as pd

    root = tmp_path / "two-kinds"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i in range(4):
        pd.DataFrame({"MD": [1.0, 2.0], "TVT": [3.0, 4.0]}).to_csv(
            root / "train" / f"e{i}__main.csv", index=False
        )
        pd.DataFrame({"MD": [1.0, 2.0]}).to_csv(root / "test" / f"e{i}__main.csv", index=False)
        # `Geology` lives only in this kind, on *both* sides — a feature, not a label.
        pd.DataFrame({"GR": [5.0, 6.0], "Geology": ["shale", "sand"], "TVT": [7.0, 8.0]}).to_csv(
            root / "train" / f"e{i}__ref.csv", index=False
        )
        pd.DataFrame({"GR": [5.0, 6.0], "Geology": ["shale", "sand"]}).to_csv(
            root / "test" / f"e{i}__ref.csv", index=False
        )
    pd.DataFrame({"id": [0], "TVT": [0.0]}).to_csv(root / "sample_submission.csv", index=False)

    profile = _profiler().profile_directory(root, "demo")

    assert "Geology" not in profile.train_only_columns
    assert profile.target_column == "TVT"


def test_the_target_fallback_is_not_decided_by_union_ordering(tmp_path):
    """Reported on PR #117 — a regression from that PR's own union fix.

    Widening `sample_df` to every kind changed what "last column" means: the
    union appends each other kind's novel columns after the primary's, so
    `train_only[-1]` became whichever secondary kind contributed last. A `main`
    kind carrying the real target and an `aux` kind carrying an unrelated
    train-only column inferred the wrong label — silently, with no crash.

    No `sample_submission.csv` here on purpose: with one, the submission-match
    branch answers first and hides this entirely, which is why the tests added
    with the union fix missed it.
    """
    import pandas as pd

    root = tmp_path / "ordering"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i in range(5):
        pd.DataFrame({"MD": [1.0, 2.0], "GR": [3.0, 4.0], "TVT": [5.0, 6.0]}).to_csv(
            root / "train" / f"e{i}__main.csv", index=False
        )
        pd.DataFrame({"MD": [1.0, 2.0], "GR": [3.0, 4.0]}).to_csv(
            root / "test" / f"e{i}__main.csv", index=False
        )
    for i in range(3):
        pd.DataFrame({"GR": [3.0, 4.0], "AuxNote": [7.0, 8.0]}).to_csv(
            root / "train" / f"a{i}__aux.csv", index=False
        )
        pd.DataFrame({"GR": [3.0, 4.0]}).to_csv(root / "test" / f"a{i}__aux.csv", index=False)

    profile = _profiler().profile_directory(root, "demo")

    assert profile.target_column == "TVT"
    # The union is still what `train_only_columns` reports — both kinds' labels
    # must be excluded from features, which is what that field is read for.
    assert set(profile.train_only_columns) == {"TVT", "AuxNote"}


def test_the_target_fallback_reads_every_primary_kind_file(tmp_path):
    """Reported on PR #117: the cross-kind fix read `frames[0]` alone, so
    schema drift *within* the primary kind reproduced the same bug one layer
    down — a `QCFlag` column in the first file and the real target only in
    later ones inferred `QCFlag` as the label."""
    import pandas as pd

    root = tmp_path / "drift"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    pd.DataFrame({"MD": [1.0, 2.0], "GR": [3.0, 4.0], "QCFlag": [1.0, 0.0]}).to_csv(
        root / "train" / "e0__main.csv", index=False
    )
    for i in (1, 2, 3):
        pd.DataFrame({"MD": [1.0, 2.0], "GR": [3.0, 4.0], "TVT": [5.0, 6.0]}).to_csv(
            root / "train" / f"e{i}__main.csv", index=False
        )
    for i in range(4):
        pd.DataFrame({"MD": [1.0, 2.0], "GR": [3.0, 4.0]}).to_csv(
            root / "test" / f"e{i}__main.csv", index=False
        )

    profile = _profiler().profile_directory(root, "demo")

    assert profile.target_column == "TVT"


def test_a_quirk_column_in_a_later_file_does_not_win_the_fallback(tmp_path):
    """Reported on PR #117, the mirror image of the previous round's fix.

    Reading only `frames[0]` missed a target absent from the first file;
    reading the union in order then let a column appearing only in a *later*
    file win, because the fallback takes the last. A label is in every
    partition of its kind and a stray note column is not, so presence
    everywhere separates them without relying on position at all.
    """
    import pandas as pd

    root = tmp_path / "quirk"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i in (0, 1):
        pd.DataFrame({"MD": [1.0, 2.0], "TARGET": [5.0, 6.0]}).to_csv(
            root / "train" / f"e{i}__main.csv", index=False
        )
    pd.DataFrame({"MD": [1.0, 2.0], "TARGET": [5.0, 6.0], "ExtraNote": [9.0, 9.0]}).to_csv(
        root / "train" / "e2__main.csv", index=False
    )
    for i in range(3):
        pd.DataFrame({"MD": [1.0, 2.0]}).to_csv(root / "test" / f"e{i}__main.csv", index=False)

    assert _profiler().profile_directory(root, "demo").target_column == "TARGET"


def test_one_file_missing_the_target_does_not_lose_it(tmp_path):
    """Reported on PR #117. Requiring the label in *every* sampled file meant a
    single schema quirk dropped it and fell back to the order-dependent union —
    with `max_files_sample` at 25, some file having a quirk is likely rather
    than remote. Counting degrades; an intersection fails outright."""
    import pandas as pd

    root = tmp_path / "quirky"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i in range(5):
        cols = {"MD": [1.0, 2.0], f"Note{i}": [1.0, 2.0]}
        if i != 4:
            cols["TVT"] = [5.0, 6.0]
        pd.DataFrame(cols).to_csv(root / "train" / f"e{i}__main.csv", index=False)
        pd.DataFrame({"MD": [1.0, 2.0]}).to_csv(root / "test" / f"e{i}__main.csv", index=False)

    assert _profiler().profile_directory(root, "demo").target_column == "TVT"


def test_a_genuine_tie_does_not_let_column_order_decide(tmp_path):
    """Reported on PR #117, the last of five rounds where position stood in for
    evidence. When two columns are equally supported we do not know which is
    the label — so the answer is order-independent and the ambiguity is
    warned, rather than resolved by whichever happens to come last."""
    import pandas as pd

    def build(order):
        root = tmp_path / ("-".join(order))
        (root / "train").mkdir(parents=True)
        (root / "test").mkdir()
        for i in range(4):
            pd.DataFrame({c: [1.0, 2.0] for c in order}).to_csv(
                root / "train" / f"e{i}__main.csv", index=False
            )
            pd.DataFrame({"MD": [1.0, 2.0]}).to_csv(root / "test" / f"e{i}__main.csv", index=False)
        return _profiler().profile_directory(root, "demo")

    first = build(["MD", "QCFlag", "TVT"])
    reversed_order = build(["MD", "TVT", "QCFlag"])

    assert first.target_column == reversed_order.target_column
    assert any("ambiguous" in w for w in first.warnings)


# --- the target's known prefix ----------------------------------------------
#
# Measured on rogii 2026-08-13. `TVT_input` is a contiguous, byte-exact prefix
# of `TVT` in every well and NaN over exactly the scored rows. The profile
# listed it as an ordinary numeric column with 164k nulls, so codegen built
# KMeans clusters and a kriging feature from it and never anchored to it.
# Carrying it forward scores RMSE 15.1; the pipeline built without knowing what
# it was scored 1380 — worse than predicting a constant.


def _anchor_dir(tmp_path, *, anchor_values=None, name="TVT_input", train_only=False):
    """Partitioned dataset whose `name` column holds a prefix of the target."""
    root = tmp_path / f"anchor-{name}-{train_only}"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    n, known = 10, 4

    def frame(with_label):
        target = [float(i) * 3 for i in range(n)]
        col = anchor_values(target, known) if anchor_values else (
            target[:known] + [None] * (n - known)
        )
        data = {"MD": list(range(n)), "Z": [t + 1000 for t in target], name: col}
        if with_label:
            data["TVT"] = target
        return pd.DataFrame(data)

    for i in range(4):
        frame(True).to_csv(root / "train" / f"e{i}__main.csv", index=False)
    for i in range(2):
        test = frame(False)
        if train_only:
            test = test.drop(columns=[name])
        test.to_csv(root / "test" / f"t{i}__main.csv", index=False)
    pd.DataFrame({"id": [f"t{w}_{i}" for w in range(2) for i in range(known, n)],
                  "tvt": [0.0] * (2 * (n - known))}).to_csv(
        root / "sample_submission.csv", index=False)
    return root


def test_the_targets_known_prefix_is_named(tmp_path):
    """The gap: without this the anchor is just a sparse numeric column."""
    profile = _profiler().profile_directory(_anchor_dir(tmp_path), "anchor-comp")

    assert profile.target_column == "TVT"
    assert profile.anchor_column == "TVT_input"
    assert any("known prefix" in w for w in profile.warnings)


def test_the_warning_says_what_to_do_with_it(tmp_path):
    """A name codegen cannot act on is the finding-that-gates-nothing failure."""
    profile = _profiler().profile_directory(_anchor_dir(tmp_path), "anchor-comp")
    note = next(w for w in profile.warnings if "known prefix" in w)

    assert "residual" in note
    assert "copy" in note, "the leak — identical to the target in training — must be stated"


def test_a_complete_correlated_column_is_not_an_anchor(tmp_path):
    """`Z` tracks the target closely and is never missing. Correlation is not a
    prefix, and anchoring to it would predict the wrong series."""
    profile = _profiler().profile_directory(
        _anchor_dir(tmp_path, anchor_values=lambda t, k: [v + 1000 for v in t]),
        "anchor-comp",
    )

    assert profile.anchor_column is None


def test_scattered_nulls_are_missing_data_not_a_masked_future(tmp_path):
    profile = _profiler().profile_directory(
        _anchor_dir(
            tmp_path,
            anchor_values=lambda t, k: [v if i % 2 else None for i, v in enumerate(t)],
        ),
        "anchor-comp",
    )

    assert profile.anchor_column is None


def test_a_prefix_that_disagrees_with_the_target_is_not_an_anchor(tmp_path):
    """Equality is the whole test — a shifted or smoothed column is a feature."""
    profile = _profiler().profile_directory(
        _anchor_dir(
            tmp_path,
            anchor_values=lambda t, k: [v + 0.5 for v in t[:k]] + [None] * (len(t) - k),
        ),
        "anchor-comp",
    )

    assert profile.anchor_column is None


def test_a_train_only_column_cannot_anchor_anything(tmp_path):
    """It is absent exactly when a prediction needs it."""
    profile = _profiler().profile_directory(
        _anchor_dir(tmp_path, train_only=True), "anchor-comp"
    )

    assert profile.anchor_column is None


def test_a_secondary_table_sharing_the_targets_name_does_not_hide_it(tmp_path):
    """Per-kind column roles, both directions.

    Measured on rogii 2026-08-13. `typewell.csv` carries its own `TVT` and ships
    in test, so against the *union* of every kind's test columns the horizontal
    well's `TVT` — the real label, absent from horizontal test files — stopped
    looking withheld and target inference fell through to `EGFDU`, a horizon
    depth. Against the primary kind alone, PR #117's `Geology` bug returns.
    """
    root = tmp_path / "shared-name"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i in range(4):
        pd.DataFrame({"MD": [0, 1], "GR": [1.0, 2.0], "TVT": [5.0, 6.0]}).to_csv(
            root / "train" / f"e{i}__main.csv", index=False
        )
        # The secondary table has a TVT of its own, and keeps it at test.
        pd.DataFrame({"TVT": [1.0, 2.0], "GR": [3.0, 4.0], "Note": ["a", "b"]}).to_csv(
            root / "train" / f"e{i}__ref.csv", index=False
        )
    for i in range(2):
        pd.DataFrame({"MD": [0, 1], "GR": [1.0, 2.0]}).to_csv(
            root / "test" / f"t{i}__main.csv", index=False
        )
        pd.DataFrame({"TVT": [1.0, 2.0], "GR": [3.0, 4.0]}).to_csv(
            root / "test" / f"t{i}__ref.csv", index=False
        )
    pd.DataFrame({"id": ["t0_1"], "tvt": [0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )

    profile = _profiler().profile_directory(root, "shared-comp")

    assert profile.target_column == "TVT"
    assert "TVT" in profile.train_only_columns
    # `Note` is withheld by its own kind; `GR` reaches test in both kinds.
    assert "Note" in profile.train_only_columns
    assert "GR" not in profile.train_only_columns


# --- the per-kind rule's blind spots ----------------------------------------
#
# `_is_withheld_at_test` reads a column's availability from the kind that
# carries it. Kinds are parsed out of filenames, and every one of these is a
# shape where that parse does not line up with reality. All three inferred the
# wrong target — or none — while `train_only_columns` said something else.


def _partitioned(root, train, test, submission):
    """Write a partitioned dataset: {filename: frame} for train and test."""
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for name, frame in train.items():
        frame.to_csv(root / "train" / name, index=False)
    for name, frame in test.items():
        frame.to_csv(root / "test" / name, index=False)
    pd.DataFrame(submission).to_csv(root / "sample_submission.csv", index=False)
    return root


def test_train_and_test_need_not_spell_the_kind_the_same_way(tmp_path):
    """`train/well_001.csv` against `test/well_051.csv`: no kind matches.

    `_split_entity_kind` partitions on the first separator, so each partition
    lands in a kind of its own and no train kind has a test counterpart. Read as
    "a kind with no test files withholds everything", every column became a
    label candidate and the first one named in the submission won — the id
    column. Base inferred `TVT` here; the per-kind rule had to not lose that.
    """
    labelled = pd.DataFrame({"id": [1, 2], "MD": [0, 1], "GR": [1.0, 2.0], "TVT": [5.0, 6.0]})
    unlabelled = pd.DataFrame({"id": [3, 4], "MD": [0, 1], "GR": [1.0, 2.0]})
    root = _partitioned(
        tmp_path / "unmatched",
        {f"well_00{i}.csv": labelled for i in range(1, 5)},
        {f"well_0{i}.csv": unlabelled for i in (51, 52)},
        {"id": [3], "TVT": [0.0]},
    )

    profile = _profiler().profile_directory(root, "c")

    assert profile.target_column == "TVT"
    assert profile.target_column != profile.id_column
    assert profile.train_only_columns == ["TVT"]


def test_a_column_missing_from_one_partition_is_still_the_target(tmp_path):
    """The kind's columns are every sampled file's, not `frames[0]`'s.

    With `max_files_sample` at 25 a single file with a schema quirk is likely
    rather than remote. Reading only the first made the label resolve to no kind
    at all, so it was declared available at test, dropped out of `train_only`,
    and `target_column` came back None — the `frames[0]` mistake PR #117 removed
    from the fallback, re-made one layer up.
    """
    root = _partitioned(
        tmp_path / "quirk",
        {
            "e0__main.csv": pd.DataFrame({"MD": [0, 1], "GR": [1.0, 2.0]}),
            **{
                f"e{i}__main.csv": pd.DataFrame(
                    {"MD": [0, 1], "GR": [1.0, 2.0], "TVT": [5.0, 6.0]}
                )
                for i in range(1, 4)
            },
        },
        {f"t{i}__main.csv": pd.DataFrame({"MD": [0, 1], "GR": [1.0, 2.0]}) for i in range(2)},
        {"id": [3], "tvt": [0.0]},
    )

    profile = _profiler().profile_directory(root, "c")

    assert profile.target_column == "TVT"
    assert profile.train_only_columns == ["TVT"]


def test_the_target_fallback_asks_the_same_question_as_train_only(tmp_path):
    """The submission-header path was fixed; the fallback beside it was not.

    Same fixture as `test_a_secondary_table_sharing_the_targets_name_does_not_hide_it`
    with a submission that does not name the label, which is the only difference
    between the two. The fallback filtered against the cross-kind union, so
    `typewell`'s own `TVT` still hid the real one and the answer fell through to
    a text column — a profile asserting `TVT` is withheld and `Note` is the
    label, in the same breath.
    """
    root = _partitioned(
        tmp_path / "fallback",
        {
            **{
                f"e{i}__main.csv": pd.DataFrame(
                    {"MD": [0, 1], "GR": [1.0, 2.0], "TVT": [5.0, 6.0]}
                )
                for i in range(4)
            },
            **{
                f"e{i}__ref.csv": pd.DataFrame(
                    {"TVT": [1.0, 2.0], "GR": [3.0, 4.0], "Note": ["a", "b"]}
                )
                for i in range(4)
            },
        },
        {
            **{f"t{i}__main.csv": pd.DataFrame({"MD": [0, 1], "GR": [1.0, 2.0]}) for i in range(2)},
            **{
                f"t{i}__ref.csv": pd.DataFrame({"TVT": [1.0, 2.0], "GR": [3.0, 4.0]})
                for i in range(2)
            },
        },
        {"id": ["t0_1"], "prediction": [0.0]},
    )

    profile = _profiler().profile_directory(root, "c")

    assert profile.target_column == "TVT"


def test_the_profile_never_names_an_anchor_it_calls_withheld(tmp_path):
    """One predicate for both, so the two answers cannot disagree.

    The anchor took a column set and the call site defaulted to the cross-kind
    union whenever the primary kind had no test files of its own — reachable,
    since the only entry guard counts *train* files. The profile then told
    codegen to carry forward a column it also listed as unavailable at test.
    """
    masked = pd.DataFrame(
        {
            "MD": range(6),
            "TVT_input": [1.0, 2.0, 3.0, None, None, None],
            "TVT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    complete = pd.DataFrame({"MD": range(6), "TVT_input": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    root = _partitioned(
        tmp_path / "contradiction",
        {
            **{f"e{i}__horizontal.csv": masked for i in range(4)},
            **{f"e{i}__typewell.csv": complete for i in range(2)},
        },
        {f"t{i}__typewell.csv": complete for i in range(2)},
        {"id": ["t0_1"], "tvt": [0.0]},
    )

    profile = _profiler().profile_directory(root, "c")

    assert profile.anchor_column not in profile.train_only_columns


def test_one_fully_observed_partition_does_not_veto_the_anchor(tmp_path):
    """A well with no masked tail has no opinion; it is not a contradiction.

    Requiring every partition to show the prefix meant one complete well — or
    one merely longer than `max_rows_sample`, whose sample holds only the known
    part — discarded the anchor for the whole dataset, with no warning to say
    so. Four wells here carry the prefix and the fifth is complete.
    """
    masked = pd.DataFrame(
        {
            "MD": range(6),
            "TVT_input": [1.0, 2.0, 3.0, None, None, None],
            "TVT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    complete = pd.DataFrame(
        {
            "MD": range(6),
            "TVT_input": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "TVT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    root = _partitioned(
        tmp_path / "unanimity",
        {"e0__main.csv": complete, **{f"e{i}__main.csv": masked for i in range(1, 5)}},
        {
            f"t{i}__main.csv": pd.DataFrame(
                {"MD": range(6), "TVT_input": [1.0, 2.0, 3.0, None, None, None]}
            )
            for i in range(2)
        },
        {"id": ["t0_3"], "tvt": [0.0]},
    )

    profile = _profiler().profile_directory(root, "c")

    assert profile.anchor_column == "TVT_input"


def test_a_partition_that_disagrees_still_vetoes_the_anchor(tmp_path):
    """The other half: "no opinion" must not become "anything goes"."""
    masked = pd.DataFrame(
        {
            "MD": range(6),
            "TVT_input": [1.0, 2.0, 3.0, None, None, None],
            "TVT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    disagrees = pd.DataFrame(
        {
            "MD": range(6),
            "TVT_input": [9.0, 9.0, 9.0, None, None, None],
            "TVT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    root = _partitioned(
        tmp_path / "veto",
        {"e0__main.csv": disagrees, **{f"e{i}__main.csv": masked for i in range(1, 5)}},
        {
            f"t{i}__main.csv": pd.DataFrame(
                {"MD": range(6), "TVT_input": [1.0, 2.0, 3.0, None, None, None]}
            )
            for i in range(2)
        },
        {"id": ["t0_3"], "tvt": [0.0]},
    )

    profile = _profiler().profile_directory(root, "c")

    assert profile.anchor_column is None


def test_the_anchor_reaches_the_validation_plan(tmp_path):
    """A field nothing reads gates nothing.

    `anchor_column` was written by the profiler and read nowhere in `src/`:
    `BaselineSelector.select`, `derive_validation_plan` and every prompt builder
    ignored it, so the profiler's finding reached the pipeline only as one
    sentence in `profile.warnings`. The validation plan is where it belongs —
    it already decides `exclude_features`, and the anchor is precisely the
    column that must be neither a plain feature nor excluded.
    """
    from labpilot.research_engine.execution.baseline.selector import derive_validation_plan

    profile = _profiler().profile_directory(_anchor_dir(tmp_path), "anchor-comp")
    plan = derive_validation_plan(profile)

    assert profile.anchor_column == "TVT_input"
    assert plan.anchor_column == "TVT_input"
    assert "residual" in plan.rationale
    # Not excluded: dropping it discards the strongest signal in the dataset.
    assert "TVT_input" not in plan.exclude_features


def test_a_dataset_with_no_anchor_says_nothing_about_one(tmp_path):
    """The note must not fire on every plan, or readers learn to skip it."""
    from labpilot.research_engine.execution.baseline.selector import derive_validation_plan

    profile = _profiler().profile_directory(
        _anchor_dir(tmp_path, anchor_values=lambda t, k: [v + 1000 for v in t]), "anchor-comp"
    )
    plan = derive_validation_plan(profile)

    assert plan.anchor_column is None
    assert "residual" not in plan.rationale
