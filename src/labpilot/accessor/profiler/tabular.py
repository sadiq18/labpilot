import logging
from collections import Counter
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


# Below this many per-entity train files, treat the dataset as ordinary
# multi-file rather than partitioned. Group-aware splits and the partitioned
# template are expensive to get wrong on a normal competition.
_MIN_PARTITIONS = 3


class TabularProfiler:
    """Profile tabular competition datasets."""

    def __init__(self, config: ProfilerConfig) -> None:
        self.config = config

    def profile_file(self, path: Path) -> DatasetProfile:
        df = pd.read_csv(path, nrows=self.config.max_rows_sample)
        return DatasetProfile(
            competition="",
            files=[str(path)],
            row_count=len(df),
            column_count=len(df.columns),
            columns=self.profile_columns(df),
        )

    def profile_columns(self, df: pd.DataFrame) -> list[ColumnProfile]:
        """Per-column facts for a frame that is already in memory.

        Split out from `profile_file` because a partitioned dataset's training
        frame is a concatenation of files rather than any one of them, and the
        profile has to describe the frame the pipeline will actually build.
        """
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

        return columns

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

    def _role_of(
        self,
        path: Path,
        data_dir: Path,
        train_pattern: str,
        test_pattern: str,
        *,
        by_directory_only: bool = False,
    ) -> str:
        """Classify a CSV as train/test by directory, falling back to filename.

        ``by_directory_only`` skips the filename fallback. Partitioned-layout
        detection uses it because a filename prefix is far too weak a signal
        there: ``train.csv`` + ``train_extra.csv`` both match "train" and would
        otherwise be read as two partitions of a partitioned dataset.
        """
        parts = [p.lower() for p in path.relative_to(data_dir).parts[:-1]]
        for part in parts:
            if part.startswith(train_pattern.lower()):
                return "train"
            if part.startswith(test_pattern.lower()):
                return "test"
        if by_directory_only:
            return "other"
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
            role = self._role_of(
                path, data_dir, train_pattern, test_pattern, by_directory_only=True
            )
            by_role[role].append(path)
        train_files, test_files = by_role["train"], by_role["test"]
        # Require a real per-entity layout: files grouped under a train/
        # directory, and enough of them that "one table per entity" is the only
        # sensible reading. A flat `train.csv` + `train_extra.csv` is an
        # ordinary multi-file dataset and must not take the partitioned path,
        # which would impose group splits and a partition-aware template on it.
        if len(train_files) < _MIN_PARTITIONS:
            return None

        sample_paths = [p for p in by_role["other"] if submission_pattern.lower() in p.name.lower()]
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

        # Every kind, not only the most common one. The generated `load_data`
        # concatenates *all* the CSVs under `train/`, so the frame it trains on
        # holds the union of the kinds' columns — and a profile that describes
        # one kind is not a description of that frame.
        #
        # Measured on rogii 2026-08-09. Two kinds of equal size, so `max()` on
        # the counts picked `horizontal_well` arbitrarily; `Geology` lives only
        # in `typewell` and never reached `profile.json`. Codegen, told the
        # dataset had thirteen columns and all of them numeric, wrote feature
        # selection as "every column except this exclusion list" — which is
        # correct given that profile and fatal given the data. Training died on
        # `pandas dtypes must be int, float or bool. Fields with bad pandas
        # dtypes: Geology: object`, twice, two days apart.
        #
        # Null counts rise here, because a column absent from one kind is NaN
        # for those rows. That is not noise: it is the true shape of the
        # concatenated frame, and it is what makes a column's sparsity visible
        # to whoever decides whether to use it.
        union_frames = list(frames)
        # Rows are estimated per kind and summed, because `load_data`
        # concatenates every CSV under `train/`. Scaling the primary kind's
        # mean by the primary kind's file count undercounts the training set by
        # whatever the other kinds contribute — the same mistake as profiling
        # one kind's columns, one field over.
        row_count = int(sum(len(f) for f in frames) / len(frames) * len(kinds[primary_kind]))
        for kind, paths in kinds.items():
            if kind == primary_kind:
                continue
            kind_limit = max(1, min(self.config.max_files_sample, len(paths)))
            kind_frames = [
                pd.read_csv(p, nrows=self.config.max_rows_sample) for p in paths[:kind_limit]
            ]
            union_frames.extend(kind_frames)
            row_count += int(sum(len(f) for f in kind_frames) / len(kind_frames) * len(paths))
        sample_df = pd.concat(union_frames, ignore_index=True)

        # Test columns from **every** kind, for the same reason the sample frame
        # spans every kind. Read from the primary kind alone, a column that
        # exists in another kind's train *and* test looked train-only — and
        # `train_only[-1]` is the target fallback, so `Geology` (a categorical
        # feature present on both sides of its own kind) was inferred as the
        # label while the real target `TVT` was passed over. Codegen then trains
        # against the wrong column entirely.
        test_columns: set[str] = set()
        test_by_kind: dict[str, list[Path]] = {}
        for path in test_files:
            test_by_kind.setdefault(self._split_entity_kind(path.stem)[1], []).append(path)
        for kind, paths in test_by_kind.items():
            kind_limit = max(1, min(self.config.max_files_sample, len(paths)))
            for path in paths[:kind_limit]:
                test_columns.update(pd.read_csv(path, nrows=0).columns)
        test_kind_files = test_by_kind.get(primary_kind, [])

        submission_columns: list[str] = []
        if sample_path is not None:
            submission_columns = list(pd.read_csv(sample_path, nrows=0).columns)

        # Target inference: a column present in train but absent from test is a
        # label candidate; the one also named in the submission header wins.
        ambiguous_target: list[str] = []
        train_only = [c for c in sample_df.columns if c not in test_columns]
        sub_lower = {c.lower() for c in submission_columns}
        target = next((c for c in train_only if c.lower() in sub_lower), None)
        if target is None and train_only:
            # The fallback reads the **primary kind's** order, not the union's.
            # Widening `sample_df` to every kind changed what "last column"
            # means: the union appends each other kind's novel columns after the
            # primary's, so `train_only[-1]` became whichever secondary kind
            # happened to contribute last. Reported on PR #117 and reproduced —
            # a `main` kind carrying the real target `TVT` and an `aux` kind
            # carrying an unrelated `AuxNote` inferred `AuxNote` as the label,
            # silently, with no crash to catch it. A regression from the union
            # fix itself, and invisible whenever a `sample_submission.csv`
            # names the target, which is why the tests added with that fix
            # missed it.
            # The primary kind, and within it the columns *every* sampled file
            # carries. Reading only `frames[0]` missed a target absent from the
            # first file; reading the union in order then let a quirk column
            # appearing only in a later file win instead, because the fallback
            # takes the last. Both reported on PR #117, one round apart, and
            # both are the same mistake: position standing in for evidence.
            #
            # A label is in every partition of its kind. A stray note column is
            # not, so requiring presence everywhere separates them without
            # relying on order at all. The union is the fallback's fallback,
            # for a kind whose files genuinely share nothing.
            # How *many* of the sampled files carry it, not whether all of
            # them do. Requiring every file was the previous answer and one
            # missing file collapsed it back to the order-dependent union it
            # replaced — with `max_files_sample` at 25, some file having a
            # schema quirk is likely rather than remote. Reported on PR #117.
            #
            # A label is in most partitions of its kind; a per-file note column
            # is in one. Counting separates them and degrades gracefully, where
            # an intersection fails outright on a single quirk.
            union: list[str] = []
            seen_in = Counter[str]()
            for frame in frames:
                seen_in.update(set(frame.columns))
                union.extend(c for c in frame.columns if c not in union)
            candidates = [c for c in union if c not in test_columns]
            if candidates:
                most = max(seen_in[c] for c in candidates)
                candidates = [c for c in candidates if seen_in[c] == most]
            # A genuine tie is a thing we do not know, and picking the last one
            # is position deciding again — the fragility four rounds on PR #117
            # kept coming back to. Sorted so the answer at least does not
            # depend on column order, and warned so it is visible rather than
            # silently wrong.
            if len(candidates) > 1:
                ambiguous_target.append(
                    "Target inference is ambiguous: "
                    f"{sorted(candidates)} are equally supported by the training "
                    "partitions and none is named in a sample submission. Set "
                    "`target_column` in the competition config to decide it."
                )
                candidates = sorted(candidates)
            primary_only = candidates
            target = (primary_only or train_only)[-1]

        profile = self.profile_file(sampled[0])
        profile.competition = competition
        # From the union frame, not from one file filtered down to it. The
        # filter could only ever remove columns, so a column that exists in
        # another kind had no way to appear.
        profile.columns = self.profile_columns(sample_df)
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
        profile.warnings.extend(ambiguous_target)
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
