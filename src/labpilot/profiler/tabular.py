from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from labpilot.config import ProfilerConfig


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    is_target_candidate: bool = False
    stats: dict[str, Any] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    competition: str
    files: list[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    target_column: str | None = None
    id_column: str | None = None
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
            columns.append(
                ColumnProfile(
                    name=col,
                    dtype=str(series.dtype),
                    null_count=null_count,
                    null_pct=round(null_count / max(len(df), 1) * 100, 2),
                    unique_count=int(series.nunique(dropna=True)),
                    stats=self._numeric_stats(series) if pd.api.types.is_numeric_dtype(series) else {},
                )
            )

        return DatasetProfile(
            competition="",
            files=[str(path)],
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
        )

    def profile_directory(self, data_dir: Path, competition: str) -> DatasetProfile:
        csv_files = sorted(data_dir.rglob("*.csv"))
        if not csv_files:
            return DatasetProfile(
                competition=competition,
                warnings=["No CSV files found in data directory."],
            )

        # Profile the largest CSV as the primary training file
        primary = max(csv_files, key=lambda p: p.stat().st_size)
        profile = self.profile_file(primary)
        profile.competition = competition
        profile.files = [str(p.relative_to(data_dir)) for p in csv_files]
        return profile

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        return {
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
        }
