from labpilot.profiler.modality import ModalityDetector
from labpilot.profiler.tabular import ColumnProfile, DatasetProfile


def _tabular_profile(**kwargs) -> DatasetProfile:
    defaults = dict(
        competition="test",
        files=["train.csv"],
        train_file="train.csv",
        row_count=100,
        column_count=3,
        target_column="label",
        id_column="id",
        columns=[
            ColumnProfile(name="id", dtype="int64", is_numeric=True, unique_count=100),
            ColumnProfile(name="feature", dtype="float64", is_numeric=True, unique_count=50),
            ColumnProfile(name="label", dtype="int64", is_numeric=True, unique_count=2),
        ],
    )
    defaults.update(kwargs)
    return DatasetProfile(**defaults)


def test_modality_defaults_to_tabular(tmp_path):
    result = ModalityDetector().detect(tmp_path, _tabular_profile())
    assert result.modality == "tabular"


def test_modality_detects_long_text_column(tmp_path):
    profile = _tabular_profile(
        columns=[
            ColumnProfile(name="id", dtype="int64", is_numeric=True, unique_count=100),
            ColumnProfile(
                name="review",
                dtype="str",
                is_numeric=False,
                unique_count=95,
                stats={"avg_length": 120.0},
            ),
            ColumnProfile(name="label", dtype="int64", is_numeric=True, unique_count=2),
        ]
    )
    result = ModalityDetector().detect(tmp_path, profile)
    assert result.modality == "text"
    assert result.text_column == "review"
