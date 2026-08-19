import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from labpilot.accessor.profiler.source import DeclaredFacts, LocalFileSource, TableRef
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


#: Bumped whenever the profiler learns to describe something it could not
#: before. `prepare_workspace` reuses an existing `profile.json` rather than
#: paying to rebuild it, so without this a workspace keeps the description it
#: was first given and every later improvement is invisible to it. rogii's was
#: written 2026-08-02 and reused by every campaign since; the anchor column
#: added on 08-13 would never have reached it.
PROFILE_SCHEMA_VERSION = 2


class DatasetProfile(BaseModel):
    #: Zero, not `PROFILE_SCHEMA_VERSION`: the default is what an *unstamped*
    #: file validates to, and defaulting it to the current version made every
    #: legacy profile claim to be current the moment it went through the model.
    #: Only `_profile_is_current` was unaffected, because it reads the raw dict;
    #: `load_profile` and `DatasetProfile.model_validate_json` — which feed
    #: planning and baseline selection — could not tell a 2026-08-02 profile
    #: from today's. `write_profile` stamps the current version on the way out,
    #: so the value is a fact about the file rather than about the reader.
    schema_version: int = 0
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
    # A column carrying the target's *known prefix*: equal to the target
    # wherever it is present, absent exactly where the scored rows are. It is
    # the strongest predictor in the dataset and the only one that says where
    # the series actually was, so a forecast should be a residual from its last
    # known value rather than a fit over the other columns.
    anchor_column: str | None = None


# Below this many per-entity train files, treat the dataset as ordinary
# multi-file rather than partitioned. Group-aware splits and the partitioned
# template are expensive to get wrong on a normal competition.
_MIN_PARTITIONS = 3


def _name_of(table: TableRef) -> str:
    """The table's file name, lower-cased — the form every pattern test uses.

    Messages print `Path(table.uri).name` instead: an operator looking for
    `Train.csv` should not be shown `train.csv` because a matcher folded case.
    """
    return Path(table.uri).name.lower()


def _detect_anchor_column(
    frames: list["pd.DataFrame"],
    target: str | None,
    reaches_test: Callable[[str], bool],
) -> str | None:
    """The column holding the target's known prefix, or None.

    Three conditions, all mechanical:

    * it survives to test — a train-only column cannot anchor a prediction;
    * wherever it is present it **equals** the target, exactly;
    * its nulls are a contiguous suffix, which is the region being scored.

    Equality is what separates an anchor from a merely correlated column, and
    the suffix shape is what separates it from an ordinary sparse feature. Both
    are required: `Z` correlates with `TVT` and is complete, while a column with
    scattered nulls is missing data rather than a masked future.

    Measured on rogii 2026-08-13. `TVT_input` satisfies all three in every well
    and appeared in the profile as an ordinary numeric column with 164k nulls,
    so codegen built KMeans clusters and a kriging feature from it and never
    anchored to it. Carrying it forward scores RMSE 15.1; the pipeline built
    without knowing what it was scored 1380.

    Availability at test is asked through `reaches_test`, which is the same
    predicate that decides `train_only_columns`. This took a column set and the
    call site passed the cross-kind union whenever the primary kind had no test
    files of its own, so one profile could name an anchor and list it as
    withheld in the same breath.

    Evidence is per partition, and a partition may have none to give: one whose
    column is fully observed has no masked tail to judge, which is not the same
    as contradicting the prefix. Requiring *every* partition to show one meant a
    single complete well — or merely one longer than `max_rows_sample`, whose
    sample then holds only the known part — discarded the anchor for the whole
    dataset, silently. One partition showing the prefix and none contradicting
    it is the rule.
    """
    if not target or not frames:
        return None
    for name in frames[0].columns:
        if name == target or not reaches_test(str(name)):
            continue
        # Lazily, and stopping at the first refusal: a candidate ruled out by
        # partition one must not be compared against the other twenty-four.
        # Building the full list first cost that short-circuit, which the
        # `all(...)` generator this replaced had, on frames of up to
        # `max_rows_sample` rows read inline in the campaign's first step.
        verdicts = (_is_known_prefix_of(frame, str(name), target) for frame in frames)
        supported = False
        for verdict in verdicts:
            if verdict is False:
                supported = False
                break
            supported = supported or verdict is True
        if supported:
            return str(name)
    return None


def _is_withheld_at_test(
    column: str,
    primary_kind: str,
    train_cols_by_kind: dict[str, set[str]],
    test_cols_by_kind: dict[str, set[str]],
    any_test_columns: set[str],
) -> bool:
    """Whether `column` is absent from test **in the kind that carries it**.

    Compared against the union of every kind's test columns, a target shared by
    name with a secondary table stops looking withheld. Measured on rogii
    2026-08-13: `typewell.csv` carries its own `TVT`, and it ships in test, so
    the horizontal well's `TVT` — the actual label, absent from horizontal test
    files — dropped out of `train_only` and target inference fell through to
    `EGFDU`. Codegen would have trained against a horizon depth.

    Compared against the primary kind alone, the opposite happens: `Geology`
    lives only in `typewell`, is present on both sides of it, and looked
    train-only. That was PR #117. Per-kind is what both bugs were reaching for —
    ask the question of the table the column actually comes from.

    But the per-kind question is only answerable when the column's kind has test
    files of its own, and kinds are parsed out of filenames that train and test
    need not spell the same way. `train/well_001.csv` against `test/well_051.csv`
    puts every partition in a kind of its own, so no train kind has a test
    counterpart at all; reading that as "withheld" made every column a label
    candidate and target inference picked the submission's id column. An
    unmatched kind therefore falls back to the cross-kind union — the older,
    looser rule, conservative in the right direction, since it calls a column
    withheld only when no test file anywhere names it.
    """
    kind = primary_kind if column in train_cols_by_kind.get(primary_kind, set()) else None
    if kind is None:
        kind = next((k for k, cols in train_cols_by_kind.items() if column in cols), None)
    if kind is None:
        return False
    test_cols = test_cols_by_kind.get(kind)
    if test_cols is None:
        return column not in any_test_columns
    return column not in test_cols


def _is_known_prefix_of(frame: "pd.DataFrame", name: str, target: str) -> bool | None:
    """Whether `name` holds a contiguous, exact prefix of `target` in one partition.

    Tri-state, because "this partition cannot say" is a real answer and is not a
    refusal. `None` when the column is absent, entirely null, or fully observed
    — and that last case covers both a partition with no masked tail and one
    longer than `max_rows_sample`, whose sample holds only the known prefix.
    Returning False for those let a single such partition veto an anchor that
    every other partition supported.
    """
    if name not in frame or target not in frame:
        return None
    known = frame[name].notna().to_numpy()
    if not known.any() or known.all():
        return None
    # Contiguous prefix: the first False sits exactly at the count of Trues.
    if int(known.argmin()) != int(known.sum()):
        return False
    return bool((frame.loc[known, name] == frame.loc[known, target]).all())


class TabularProfiler:
    """Profile tabular competition datasets."""

    def __init__(self, config: ProfilerConfig) -> None:
        self.config = config

    def profile_file(self, path: Path) -> DatasetProfile:
        """Describe one file that is not part of a dataset layout.

        Reads through a source over the file's own directory, so this module
        holds no direct `read_csv` of its own: one owner for every read is what
        makes the seam real rather than decorative.
        """
        source = LocalFileSource(path.parent)
        df = source.sample(TableRef(id=path.name, uri=path.name), self.config.max_rows_sample)
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
        source = LocalFileSource(
            data_dir,
            DeclaredFacts(title=competition_title, description=competition_description),
        )
        tables = source.tables()
        if not tables:
            raise FileNotFoundError(f"No CSV files found in {data_dir}.")

        # Partitioned layouts (train/<entity>.csv) match no filename prefix, so
        # try them before the single-file heuristic reports "found 0".
        partitioned = self._try_profile_partitioned(
            source,
            competition,
            tables,
            train_pattern=train_pattern,
            test_pattern=test_pattern,
            submission_pattern=submission_pattern,
        )
        if partitioned is not None:
            return partitioned

        train_table = self._single_table(
            [table for table in tables if _name_of(table).startswith(train_pattern.lower())],
            "training",
        )
        sample_table = self._single_table(
            [table for table in tables if submission_pattern.lower() in _name_of(table)],
            "sample submission",
        )
        test_matches = [
            table for table in tables if _name_of(table).startswith(test_pattern.lower())
        ]
        test_warnings: list[str] = []
        if len(test_matches) == 1:
            test_table = test_matches[0]
        elif len(test_matches) == 0:
            test_table = sample_table
            test_warnings.append(
                "No test CSV found; using sample submission as the test reference file."
            )
        else:
            names = [Path(table.uri).name for table in test_matches]
            raise ValueError(f"Expected one test CSV, found {len(test_matches)}: {names}")

        train_columns = source.columns(train_table)
        test_columns = source.columns(test_table)
        submission_columns = source.columns(sample_table)

        target_candidates = [column for column in train_columns if column not in test_columns]
        if len(target_candidates) == 1:
            target_column = target_candidates[0]
        elif test_table == sample_table:
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

        # One read of the training table, where there were two: `profile_file`
        # sampled it and then `enrich_column_stats` sampled it again with the
        # same cap. Same bytes, same frame — the second read only cost time.
        train_sample = source.sample(train_table, self.config.max_rows_sample)
        profile = DatasetProfile(
            competition=competition,
            row_count=len(train_sample),
            column_count=len(train_sample.columns),
            columns=self.profile_columns(train_sample),
        )
        profile.warnings.extend(test_warnings)
        from labpilot.accessor.profiler.modality import ModalityDetector

        detector = ModalityDetector()
        detector.enrich_column_stats(train_sample, profile.columns)
        profile.files = [table.uri for table in tables]
        profile.train_file = train_table.uri
        profile.test_file = test_table.uri
        profile.sample_submission_file = sample_table.uri
        profile.test_row_count = source.exact_unit_count(test_table, id_column)
        profile.target_column = target_column
        profile.id_column = id_column
        profile.submission_columns = submission_columns
        for column in profile.columns:
            column.is_target_candidate = column.name == target_column

        declared = source.declared()
        modality = detector.detect(
            source.root,
            profile,
            llm_client=llm_client,
            competition_title=declared.title,
            competition_description=declared.description,
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
        table: TableRef,
        train_pattern: str,
        test_pattern: str,
        *,
        by_directory_only: bool = False,
    ) -> str:
        """Classify a table as train/test by directory, falling back to filename.

        ``by_directory_only`` skips the filename fallback. Partitioned-layout
        detection uses it because a filename prefix is far too weak a signal
        there: ``train.csv`` + ``train_extra.csv`` both match "train" and would
        otherwise be read as two partitions of a partitioned dataset.

        This is *layout* inference, not a fact the source states — which is why
        it lives here and why `TableRef` carries no role until step 3 gives that
        answer its evidence.
        """
        parts = [part.lower() for part in Path(table.uri).parts[:-1]]
        for part in parts:
            if part.startswith(train_pattern.lower()):
                return "train"
            if part.startswith(test_pattern.lower()):
                return "test"
        if by_directory_only:
            return "other"
        name = _name_of(table)
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
        source: LocalFileSource,
        competition: str,
        tables: list[TableRef],
        *,
        train_pattern: str,
        test_pattern: str,
        submission_pattern: str,
    ) -> DatasetProfile | None:
        """Profile one-file-per-entity datasets, or return None if not that shape.

        Typed to `LocalFileSource` rather than the protocol because
        `_detect_suffix_scoring` still counts lines on disk. That routine moves
        behind the protocol in step 5; every *read* here already goes through it.
        """
        by_role: dict[str, list[TableRef]] = {"train": [], "test": [], "other": []}
        for table in tables:
            role = self._role_of(table, train_pattern, test_pattern, by_directory_only=True)
            by_role[role].append(table)
        train_files, test_files = by_role["train"], by_role["test"]
        # Require a real per-entity layout: files grouped under a train/
        # directory, and enough of them that "one table per entity" is the only
        # sensible reading. A flat `train.csv` + `train_extra.csv` is an
        # ordinary multi-file dataset and must not take the partitioned path,
        # which would impose group splits and a partition-aware template on it.
        if len(train_files) < _MIN_PARTITIONS:
            return None

        sample_tables = [t for t in by_role["other"] if submission_pattern.lower() in _name_of(t)]
        sample_table = sample_tables[0] if sample_tables else None

        # Group files by "kind" suffix (horizontal_well / typewell / …). A kind
        # shared by many entities is a real per-entity table, not a one-off.
        kinds: dict[str, list[TableRef]] = {}
        for table in train_files:
            _, kind = self._split_entity_kind(Path(table.uri).stem)
            kinds.setdefault(kind, []).append(table)
        primary_kind = max(kinds, key=lambda k: len(kinds[k]))

        limit = max(1, min(self.config.max_files_sample, len(kinds[primary_kind])))
        sampled = kinds[primary_kind][:limit]
        frames = [source.sample(table, self.config.max_rows_sample) for table in sampled]

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
        # Every sampled file of the kind, not `frames[0]`. A column missing from
        # the first file alone resolved to "no kind" below and was declared
        # available at test, so it dropped out of `train_only` — and when the
        # label was the column that happened to be missing, `target_column` came
        # back None. That is the `frames[0]`-only mistake PR #117 spent four
        # rounds removing from the fallback fifty lines down, re-made one layer
        # up: with `max_files_sample` at 25, one file with a schema quirk is
        # likely rather than remote.
        train_cols_by_kind: dict[str, set[str]] = {
            primary_kind: {str(c) for f in frames for c in f.columns}
        }
        for kind, kind_tables in kinds.items():
            if kind == primary_kind:
                continue
            kind_limit = max(1, min(self.config.max_files_sample, len(kind_tables)))
            kind_frames = [
                source.sample(table, self.config.max_rows_sample)
                for table in kind_tables[:kind_limit]
            ]
            train_cols_by_kind[kind] = {str(c) for f in kind_frames for c in f.columns}
            union_frames.extend(kind_frames)
            row_count += int(sum(len(f) for f in kind_frames) / len(kind_frames) * len(kind_tables))
        sample_df = pd.concat(union_frames, ignore_index=True)

        # Test columns from **every** kind, for the same reason the sample frame
        # spans every kind. Read from the primary kind alone, a column that
        # exists in another kind's train *and* test looked train-only — and
        # `train_only[-1]` is the target fallback, so `Geology` (a categorical
        # feature present on both sides of its own kind) was inferred as the
        # label while the real target `TVT` was passed over. Codegen then trains
        # against the wrong column entirely.
        test_columns: set[str] = set()
        test_cols_by_kind: dict[str, set[str]] = {}
        test_by_kind: dict[str, list[TableRef]] = {}
        for table in test_files:
            kind = self._split_entity_kind(Path(table.uri).stem)[1]
            test_by_kind.setdefault(kind, []).append(table)
        for kind, kind_tables in test_by_kind.items():
            kind_limit = max(1, min(self.config.max_files_sample, len(kind_tables)))
            for table in kind_tables[:kind_limit]:
                found = set(source.columns(table))
                test_columns.update(found)
                test_cols_by_kind.setdefault(kind, set()).update(found)
        test_kind_tables = test_by_kind.get(primary_kind, [])

        # One predicate, asked the same way everywhere a column's availability
        # at test matters: `train_only`, the target fallback, and the anchor.
        # Each of the three used to spell it differently — the fallback against
        # the cross-kind union, the anchor against the primary kind with the
        # union as a default — so one profile could report a column withheld and
        # name it as the anchor at the same time.
        def withheld_at_test(column: str) -> bool:
            return _is_withheld_at_test(
                column, primary_kind, train_cols_by_kind, test_cols_by_kind, test_columns
            )

        submission_columns: list[str] = []
        if sample_table is not None:
            submission_columns = source.columns(sample_table)

        # Target inference: a column present in train but absent from test is a
        # label candidate; the one also named in the submission header wins.
        ambiguous_target: list[str] = []
        train_only = [c for c in sample_df.columns if withheld_at_test(str(c))]
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
            # The same per-kind question `train_only` asks. This filtered against
            # the cross-kind union, so the bug the per-kind rule was written for
            # survived here untouched: whenever no sample submission named the
            # label, a secondary table shipping a column of the same name still
            # removed the real target from the candidates and the answer fell
            # through to a note column.
            candidates = [c for c in union if withheld_at_test(str(c))]
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

        # From the union frame, not from one file filtered down to it. The
        # filter could only ever remove columns, so a column that exists in
        # another kind had no way to appear.
        #
        # Built directly rather than from `profile_file(sampled[0])`, which read
        # one partition to fill three fields that every line below overwrites.
        profile = DatasetProfile(
            competition=competition,
            columns=self.profile_columns(sample_df),
        )
        profile.files = [table.uri for table in tables[:200]]
        profile.train_file = sampled[0].uri
        profile.test_file = test_kind_tables[0].uri if test_kind_tables else None
        profile.sample_submission_file = sample_table.uri if sample_table else None
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
        profile.test_partition_count = len(test_kind_tables)
        profile.train_only_columns = train_only
        self._detect_suffix_scoring(profile, source, sample_table, test_kind_tables)
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
        profile.anchor_column = _detect_anchor_column(
            frames, profile.target_column, lambda name: not withheld_at_test(name)
        )
        if profile.anchor_column:
            profile.warnings.append(
                f"{profile.anchor_column!r} is the known prefix of {profile.target_column!r}: "
                f"equal to it wherever present, absent exactly on the scored rows. Carrying its "
                f"last known value forward is the baseline to beat — predict the residual from "
                f"it, not {profile.target_column!r} from the other columns. Note it is identical "
                f"to the target in training, so using it as a plain feature learns 'copy' and "
                f"then meets NaN on every scored row."
            )
        return profile

    def _detect_suffix_scoring(
        self,
        profile: DatasetProfile,
        source: LocalFileSource,
        sample_table: TableRef | None,
        test_kind_tables: list[TableRef],
    ) -> None:
        """Detect ``<entity>_<row_index>`` submission ids covering only a tail.

        A random split is meaningless for these: at inference the model has the
        head of the partition and must predict forward, so validation has to
        reproduce that gap rather than sampling rows uniformly.

        The row count below is a *line* count, not `exact_unit_count`: it counts
        physical lines, which is what makes it cheap and what makes it wrong on
        a quoted newline. Left as it is here — changing how rows are counted is
        step 5's job, and doing it inside a refactor that promises no behaviour
        change is how a "value-neutral" step stops being one.
        """
        if sample_table is None or not test_kind_tables:
            return
        try:
            submission = source.sample(sample_table)
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
        for table in test_kind_tables:
            entity, _ = self._split_entity_kind(Path(table.uri).stem)
            scored = indices[entities == entity]
            if scored.empty:
                continue
            try:
                n_rows = source.physical_line_count(table) - 1
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

    def _single_table(self, matches: list[TableRef], role: str) -> TableRef:
        if len(matches) != 1:
            names = [Path(table.uri).name for table in matches]
            raise ValueError(f"Expected one {role} CSV, found {len(matches)}: {names}")
        return matches[0]

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        return {
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
        }
