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
    modality: str = "tabular"
    image_dir: str | None = None
    image_column: str | None = None
    text_column: str | None = None
    # Partitioned layouts: one file *per entity* (per well / patient / store)
    # under train/ and test/ dirs, rather than a single train.csv. Rows are not
    # IID across partitions, so downstream CV must group by `partition_key`.
    partitioned: bool = False
    partition_key: str | None = None
    partition_kinds: dict[str, int] = Field(default_factory=dict)
    train_partition_count: int = 0
    test_partition_count: int = 0
    row_count_estimated: bool = False
    # True when the scored rows form a contiguous *suffix* of each test
    # partition (predict-forward / forecast tasks). Validation must then hold
    # out the tail of each training partition, not random rows.
    scored_is_partition_suffix: bool = False
    scored_fraction: float = 0.0
    train_only_columns: list[str] = Field(default_factory=list)


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
        llm_client: Any | None = None,
        competition_title: str = "",
        competition_description: str = "",
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

        # Partitioned layouts (train/<entity>.csv) match no filename prefix, so
        # try them before the single-file heuristic reports "found 0".
        partitioned = self._try_profile_partitioned(
            data_dir,
            competition,
            csv_files,
            train_pattern=train_pattern,
            test_pattern=test_pattern,
            submission_pattern=submission_pattern,
        )
        if partitioned is not None:
            return partitioned

        train_path = self._single_file(
            [path for path in csv_files if path.name.lower().startswith(train_pattern.lower())],
            "training",
        )
        sample_path = self._single_file(
            [path for path in csv_files if submission_pattern.lower() in path.name.lower()],
            "sample submission",
        )
        test_matches = [
            path for path in csv_files if path.name.lower().startswith(test_pattern.lower())
        ]
        test_warnings: list[str] = []
        if len(test_matches) == 1:
            test_path = test_matches[0]
        elif len(test_matches) == 0:
            test_path = sample_path
            test_warnings.append(
                "No test CSV found; using sample submission as the test reference file."
            )
        else:
            names = [path.name for path in test_matches]
            raise ValueError(f"Expected one test CSV, found {len(test_matches)}: {names}")

        train_columns = list(pd.read_csv(train_path, nrows=0).columns)
        test_columns = list(pd.read_csv(test_path, nrows=0).columns)
        submission = pd.read_csv(sample_path, nrows=self.config.max_rows_sample)
        submission_columns = list(submission.columns)

        target_candidates = [column for column in train_columns if column not in test_columns]
        if len(target_candidates) == 1:
            target_column = target_candidates[0]
        elif test_path == sample_path:
            overlap = [column for column in submission_columns if column in train_columns]
            if len(overlap) >= 2:
                target_column = overlap[1]
            else:
                raise ValueError(
                    "Unable to infer one target column from train/sample submission schemas; "
                    f"found {target_candidates or 'none'}."
                )
        else:
            raise ValueError(
                "Unable to infer one target column from train/test schemas; "
                f"found {target_candidates or 'none'}."
            )

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
        profile.warnings.extend(test_warnings)
        train_sample = pd.read_csv(train_path, nrows=self.config.max_rows_sample)
        from labpilot.accessor.profiler.modality import ModalityDetector

        detector = ModalityDetector()
        detector.enrich_column_stats(train_sample, profile.columns)
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

        modality = detector.detect(
            data_dir,
            profile,
            llm_client=llm_client,
            competition_title=competition_title,
            competition_description=competition_description,
        )
        profile.modality = modality.modality
        profile.image_dir = modality.image_dir
        profile.image_column = modality.image_column
        profile.text_column = modality.text_column
        if modality.signals:
            profile.warnings.extend(modality.signals)

        logger.info(
            "Profiled '%s': target=%s, id=%s, train_rows=%d, test_rows=%d",
            competition,
            target_column,
            id_column,
            profile.row_count,
            profile.test_row_count,
        )
        return profile

    def _role_of(self, path: Path, data_dir: Path, train_pattern: str, test_pattern: str) -> str:
        """Classify a CSV as train/test by directory, falling back to filename."""
        parts = [p.lower() for p in path.relative_to(data_dir).parts[:-1]]
        for part in parts:
            if part.startswith(train_pattern.lower()):
                return "train"
            if part.startswith(test_pattern.lower()):
                return "test"
        name = path.name.lower()
        if name.startswith(train_pattern.lower()):
            return "train"
        if name.startswith(test_pattern.lower()):
            return "test"
        return "other"

    @staticmethod
    def _split_entity_kind(stem: str) -> tuple[str, str]:
        """Split ``<entity>__<kind>`` into its parts; kind is "" when absent."""
        for sep in ("__", "-", "_"):
            if sep in stem:
                entity, _, kind = stem.partition(sep)
                return entity, kind
        return stem, ""

    def _try_profile_partitioned(
        self,
        data_dir: Path,
        competition: str,
        csv_files: list[Path],
        *,
        train_pattern: str,
        test_pattern: str,
        submission_pattern: str,
    ) -> DatasetProfile | None:
        """Profile one-file-per-entity datasets, or return None if not that shape."""
        by_role: dict[str, list[Path]] = {"train": [], "test": [], "other": []}
        for path in csv_files:
            by_role[self._role_of(path, data_dir, train_pattern, test_pattern)].append(path)
        train_files, test_files = by_role["train"], by_role["test"]
        if len(train_files) <= 1:
            return None  # ordinary single-train-file dataset

        sample_paths = [
            p for p in by_role["other"] if submission_pattern.lower() in p.name.lower()
        ]
        sample_path = sample_paths[0] if sample_paths else None

        # Group files by "kind" suffix (horizontal_well / typewell / …). A kind
        # shared by many entities is a real per-entity table, not a one-off.
        kinds: dict[str, list[Path]] = {}
        for path in train_files:
            _, kind = self._split_entity_kind(path.stem)
            kinds.setdefault(kind, []).append(path)
        primary_kind = max(kinds, key=lambda k: len(kinds[k]))

        limit = max(1, min(self.config.max_files_sample, len(kinds[primary_kind])))
        sampled = kinds[primary_kind][:limit]
        frames = [pd.read_csv(p, nrows=self.config.max_rows_sample) for p in sampled]
        sample_df = pd.concat(frames, ignore_index=True)

        mean_rows = sum(len(f) for f in frames) / len(frames)
        row_count = int(mean_rows * len(kinds[primary_kind]))

        test_columns: set[str] = set()
        test_kind_files = [
            p for p in test_files if self._split_entity_kind(p.stem)[1] == primary_kind
        ]
        for path in test_kind_files[:limit]:
            test_columns.update(pd.read_csv(path, nrows=0).columns)

        submission_columns: list[str] = []
        if sample_path is not None:
            submission_columns = list(pd.read_csv(sample_path, nrows=0).columns)

        # Target inference: a column present in train but absent from test is a
        # label candidate; the one also named in the submission header wins.
        train_only = [c for c in sample_df.columns if c not in test_columns]
        sub_lower = {c.lower() for c in submission_columns}
        target = next((c for c in train_only if c.lower() in sub_lower), None)
        if target is None and train_only:
            target = train_only[-1]

        profile = self.profile_file(sampled[0])
        profile.competition = competition
        profile.columns = [c for c in profile.columns if c.name in sample_df.columns]
        profile.files = [str(p.relative_to(data_dir)) for p in csv_files[:200]]
        profile.train_file = str(sampled[0].relative_to(data_dir))
        profile.test_file = (
            str(test_kind_files[0].relative_to(data_dir)) if test_kind_files else None
        )
        profile.sample_submission_file = (
            str(sample_path.relative_to(data_dir)) if sample_path else None
        )
        profile.submission_columns = submission_columns
        profile.target_column = target
        profile.id_column = submission_columns[0] if submission_columns else None
        profile.row_count = row_count
        profile.row_count_estimated = True
        profile.column_count = len(sample_df.columns)
        profile.partitioned = True
        profile.partition_key = "file_stem_entity"
        profile.partition_kinds = {k: len(v) for k, v in sorted(kinds.items())}
        profile.train_partition_count = len(kinds[primary_kind])
        profile.test_partition_count = len(test_kind_files)
        profile.train_only_columns = train_only
        self._detect_suffix_scoring(profile, sample_path, test_kind_files, data_dir)
        profile.warnings = [
            f"partitioned dataset: {len(train_files)} train / {len(test_files)} test CSVs",
            f"primary kind={primary_kind!r}; kinds={profile.partition_kinds}",
            f"row_count estimated from {len(sampled)} sampled files",
            "rows are NOT iid across partitions — validation must group by partition",
        ]
        if train_only:
            profile.warnings.append(f"train-only columns (unavailable at test): {train_only}")
        if profile.scored_is_partition_suffix:
            profile.warnings.append(
                f"scored rows are a contiguous suffix of each test partition "
                f"(~{profile.scored_fraction:.0%} of rows) — this is a forecast task; "
                "validate by holding out each partition's tail"
            )
        return profile

    def _detect_suffix_scoring(
        self,
        profile: DatasetProfile,
        sample_path: Path | None,
        test_kind_files: list[Path],
        data_dir: Path,
    ) -> None:
        """Detect ``<entity>_<row_index>`` submission ids covering only a tail.

        A random split is meaningless for these: at inference the model has the
        head of the partition and must predict forward, so validation has to
        reproduce that gap rather than sampling rows uniformly.
        """
        if sample_path is None or not test_kind_files:
            return
        try:
            submission = pd.read_csv(sample_path)
        except Exception:  # noqa: BLE001 — detection is best-effort
            return
        if submission.empty:
            return

        ids = submission[submission.columns[0]].astype(str)
        split = ids.str.rsplit("_", n=1)
        if not (split.str.len() == 2).all():
            return
        entities = split.str[0]
        try:
            indices = split.str[1].astype(int)
        except (TypeError, ValueError):
            return

        fractions: list[float] = []
        for path in test_kind_files:
            entity, _ = self._split_entity_kind(path.stem)
            scored = indices[entities == entity]
            if scored.empty:
                continue
            try:
                n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            except OSError:
                continue
            if n_rows <= 0:
                continue
            # Contiguous tail: every index from min(scored) to the last row.
            expected_tail = n_rows - int(scored.min())
            if len(scored) != expected_tail or int(scored.max()) != n_rows - 1:
                return
            fractions.append(len(scored) / n_rows)

        if fractions:
            profile.scored_is_partition_suffix = True
            profile.scored_fraction = sum(fractions) / len(fractions)

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
