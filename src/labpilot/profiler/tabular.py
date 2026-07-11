import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from labpilot.config import ProfilerConfig

logger = logging.getLogger(__name__)


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    is_target_candidate: bool = False
    # Computed once here via `pd.api.types`, rather than re-derived downstream
    # by matching against `dtype` strings — pandas' own dtype names aren't
    # stable across versions (e.g. pandas 3.0 reports plain string columns as
    # dtype "str", not "object"), so string-matching `dtype` for "is this
    # categorical?" silently breaks when that changes.
    is_numeric: bool = False
    stats: dict[str, Any] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    competition: str
    files: list[str] = Field(default_factory=list)
    train_file: str | None = None
    test_file: str | None = None
    sample_submission_file: str | None = None
    row_count: int = 0
    test_row_count: int = 0
    column_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    target_column: str | None = None
    id_column: str | None = None
    submission_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TabularProfiler:
    """Profile tabular competition datasets."""

    def __init__(self, config: ProfilerConfig) -> None:
        self.config = config

    def profile_file(self, path: Path) -> DatasetProfile:
        df = pd.read_csv(path, nrows=self.config.max_rows_sample)
        columns: list[ColumnProfile] = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            # Bool is numeric by pandas' own `is_numeric_dtype`, but a 0/1 (or
            # True/False) column reads as a class label, not a quantity — so
            # it's treated as categorical here, matching how it's always been
            # handled for classification-target inference.
            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
                series
            )
            columns.append(
                ColumnProfile(
                    name=col,
                    dtype=str(series.dtype),
                    null_count=null_count,
                    null_pct=round(null_count / max(len(df), 1) * 100, 2),
                    unique_count=int(series.nunique(dropna=True)),
                    is_numeric=is_numeric,
                    stats=self._numeric_stats(series) if is_numeric else {},
                )
            )

        return DatasetProfile(
            competition="",
            files=[str(path)],
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
        )

    def profile_directory(
        self,
        data_dir: Path,
        competition: str,
        train_pattern: str = "train",
        test_pattern: str = "test",
        submission_pattern: str = "submission",
    ) -> DatasetProfile:
        # File-role detection is a naming-convention heuristic. `train_pattern`,
        # `test_pattern`, and `submission_pattern` let a competition's local
        # config (`configs/competitions/<slug>.yaml`) override the defaults
        # when a dataset doesn't follow the "train*/test*/*submission*"
        # convention.
        # TODO: fetch the real file roles from the Kaggle competition
        # portal/API automatically instead of relying on name matching.
        logger.info("Profiling dataset directory %s for '%s'", data_dir, competition)
        csv_files = sorted(data_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}.")

        train_path = self._single_file(
            [path for path in csv_files if path.name.lower().startswith(train_pattern.lower())],
            "training",
        )
        test_path = self._single_file(
            [path for path in csv_files if path.name.lower().startswith(test_pattern.lower())],
            "test",
        )
        sample_path = self._single_file(
            [path for path in csv_files if submission_pattern.lower() in path.name.lower()],
            "sample submission",
        )

        train_columns = list(pd.read_csv(train_path, nrows=0).columns)
        test_columns = list(pd.read_csv(test_path, nrows=0).columns)
        submission = pd.read_csv(sample_path, nrows=self.config.max_rows_sample)
        submission_columns = list(submission.columns)

        target_candidates = [column for column in train_columns if column not in test_columns]
        if len(target_candidates) != 1:
            raise ValueError(
                "Unable to infer one target column from train/test schemas; "
                f"found {target_candidates or 'none'}."
            )
        target_column = target_candidates[0]

        id_candidates = [
            column
            for column in submission_columns
            if column in train_columns and column in test_columns and column != target_column
        ]
        if not id_candidates:
            raise ValueError("Unable to infer an ID column from the sample submission.")
        id_column = id_candidates[0]

        expected_submission_columns = [id_column, target_column]
        if submission_columns != expected_submission_columns:
            raise ValueError(
                "Sample submission schema does not match the inferred ID and target columns: "
                f"expected {expected_submission_columns}, got {submission_columns}."
            )

        profile = self.profile_file(train_path)
        profile.competition = competition
        profile.files = [str(p.relative_to(data_dir)) for p in csv_files]
        profile.train_file = str(train_path.relative_to(data_dir))
        profile.test_file = str(test_path.relative_to(data_dir))
        profile.sample_submission_file = str(sample_path.relative_to(data_dir))
        profile.test_row_count = sum(
            len(chunk) for chunk in pd.read_csv(test_path, usecols=[id_column], chunksize=10_000)
        )
        profile.target_column = target_column
        profile.id_column = id_column
        profile.submission_columns = submission_columns
        for column in profile.columns:
            column.is_target_candidate = column.name == target_column
        logger.info(
            "Profiled '%s': target=%s, id=%s, train_rows=%d, test_rows=%d",
            competition,
            target_column,
            id_column,
            profile.row_count,
            profile.test_row_count,
        )
        return profile

    def _single_file(self, matches: list[Path], role: str) -> Path:
        if len(matches) != 1:
            names = [path.name for path in matches]
            raise ValueError(f"Expected one {role} CSV, found {len(matches)}: {names}")
        return matches[0]

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        return {
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
        }
