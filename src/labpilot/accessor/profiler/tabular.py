import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, computed_field, model_validator

from labpilot.accessor.profiler.evidence import (
    Alternative,
    Inference,
    Note,
    RejectedClaim,
    Signal,
    combine,
)
from labpilot.accessor.profiler.questions import answers_fingerprint
from labpilot.accessor.profiler.schema import (
    ExclusionReason,
    MetricRef,
    ModalityPresence,
    PredictionUnit,
    SplitRelationship,
)
from labpilot.accessor.profiler.source import (
    DatasetSource,
    DeclaredFacts,
    LocalFileSource,
    TableRef,
)
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
PROFILE_SCHEMA_VERSION = 3


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
    #: A list, because a composite key is ordinary outside Kaggle —
    #: `(store, date)`, `(patient, visit)`. `id_column` below is the first of
    #: them, kept for every existing reader.
    id_columns: list[str] = Field(default_factory=list)
    #: Why each non-feature column is not one, by reason code. A measurement:
    #: two people with the same bytes would agree, so no confidence attaches.
    excluded_columns: dict[str, ExclusionReason] = Field(default_factory=dict)
    #: How the scored units relate to the training units. What validation has to
    #: reproduce, and the first thing M23's floor needs to be computed on.
    train_test_relationship: SplitRelationship = "unknown"
    #: What the dataset is scored by, and how that was reached.
    metric: MetricRef | None = None
    submission_columns: list[str] = Field(default_factory=list)
    #: Structured reasons. `warnings` below is the prose view over these.
    notes: list[Note] = Field(default_factory=list)
    #: Which answers this profile was built from (`questions.answers_fingerprint`).
    #: Empty when there were none. A profile built before an answer was given
    #: describes a different question, so this is part of what makes it stale.
    answers_fingerprint: str = ""
    #: Why the value plane says what it says, keyed by field name. Absent for a
    #: field nothing has reasoned about yet, which `confidence_in` reports as
    #: 0.0 — "no evidence recorded", not "no evidence exists".
    inferences: dict[str, Inference] = Field(default_factory=dict)
    #: Every modality present, primary first. `modality` below is the mirror
    #: over the primary, so the six modules that read a string keep working.
    modalities: list[ModalityPresence] = Field(default_factory=list)
    #: What one prediction is about — a row, a row of a partition, an episode.
    prediction_unit: PredictionUnit = "unknown"
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
    #: How many rows the per-column statistics were computed over.
    #:
    #: `row_count` is the file's; these are the sample's, and once the cap binds
    #: the two differ. Without this a reader computing a null *fraction* as
    #: `null_count / row_count` is wrong by the sampling ratio — 6.9× on
    #: `playground-series-s6e7` — and nothing in the profile says so.
    column_stats_rows: int = 0
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modality(self) -> str:
        """The primary modality, as a string. A view over `modalities`.

        Six modules read this name and none of them should have to learn a list
        to keep working; computed rather than stored so the two cannot drift.
        Empty list means nothing was detected — `"tabular"` is the same default
        the field carried before, and the accompanying note says which happened.
        """
        return self.modalities[0].modality if self.modalities else "tabular"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """One number a gate can read: the **weakest** of the required answers.

        Not an average. A schema certain about four things and guessing at the
        target is a guessing schema, and a mean would let three confident
        answers hide the one that matters.
        """
        return min((self.confidence_in(field) for field in REQUIRED_FIELDS), default=0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id_column(self) -> str | None:
        """The first key column. A view over `id_columns`, so the two cannot drift."""
        return self.id_columns[0] if self.id_columns else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feature_columns(self) -> list[str]:
        """Everything a model may use: the columns, minus the exclusions.

        Derived rather than stored because this is the one answer where a wrong
        value is worse than no value — a leak makes a score look *better*, so
        nothing downstream detects it — and a stored copy that drifts from
        `excluded_columns` would be exactly that failure with no symptom.
        """
        return [c.name for c in self.columns if c.name not in self.excluded_columns]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warnings(self) -> list[str]:
        """The prose view over `notes`, in the order they were recorded.

        Kept because four things render it — one of them the codegen prompt —
        and computed rather than stored so the two can never disagree. A reader
        that wants to *act* on a reason reads `notes[].code` instead of matching
        substrings against this.
        """
        return [note.text for note in self.notes]

    @model_validator(mode="before")
    @classmethod
    def _adopt_legacy_modality(cls, data: Any) -> Any:
        """A pre-step-5 profile's `modality` string becomes its primary presence.

        The same shape as the `warnings` adoption below, and the same reason:
        `modality` is computed now, so pydantic would drop the stored value and
        every legacy profile would read `tabular` — birdclef's says `audio`, and
        the analyzers that key off it would silently start describing an audio
        competition as a tabular one.
        """
        # `not in`, not falsy. An *empty* list is a profile that recorded
        # "nothing detected" — the non-local-source branch writes exactly that,
        # beside a `modality_not_detected` note — and treating it as legacy made
        # every later read fabricate a `tabular` presence with a provenance line
        # claiming it came from an older profile. Both halves false, and the
        # file then contradicted its own note.
        if not isinstance(data, dict) or "modalities" in data:
            return data
        stored = data.get("modality")
        if isinstance(stored, str) and stored:
            data = dict(data)
            data["modalities"] = [
                {
                    "modality": stored,
                    "role": "primary",
                    "detail": "adopted from a profile written before modalities were a list",
                    "image_dir": data.get("image_dir"),
                    "image_column": data.get("image_column"),
                    "text_column": data.get("text_column"),
                }
            ]
        return data

    @model_validator(mode="before")
    @classmethod
    def _adopt_legacy_warnings(cls, data: Any) -> Any:
        """A pre-M22 profile's prose becomes notes, rather than being dropped.

        `warnings` used to be a stored field, so every `profile.json` on disk
        carries one and pydantic would ignore it now that the name is computed.
        Silently losing it would take the anchor-column advice — the one line
        that tells codegen not to fit the target from a column identical to it —
        out of every workspace still serving a stale profile.
        """
        if not isinstance(data, dict) or data.get("notes"):
            return data
        legacy = data.get("warnings")
        if isinstance(legacy, list) and legacy:
            data = dict(data)
            data["notes"] = [
                {"code": "legacy", "text": str(text), "severity": "info"} for text in legacy
            ]
        return data

    def confidence_in(self, field: str) -> float:
        """How sure the profiler is about one field, or 0.0 if it never reasoned about it."""
        inference = self.inferences.get(field)
        return inference.confidence if inference else 0.0


#: What the schema-level `confidence` summarises. Broader than the fields whose
#: uncertainty *stops* a campaign (`questions.BLOCKING_FIELDS`): a capped
#: `disjoint_units` and a missing metric both drag the number down without being
#: worth stopping for, which is the distinction between "how good is this
#: description" and "may I proceed on it".
REQUIRED_FIELDS = ("target_column", "id_columns", "train_test_relationship", "metric")


def _answered(
    candidates: dict[str, list[Signal]],
    answer: str | None,
    *,
    known: set[str],
    field: str = "target_column",
) -> tuple[dict[str, list[Signal]], list[RejectedClaim]]:
    """Fold an operator's answer into the candidate set, if it names a column.

    The answer becomes a candidate carrying `operator_answer` (1.00), so it wins
    by evidence rather than by bypassing the resolver — and its other signals are
    kept, so the profile still shows what the data said about the column a person
    chose. A column the profiler never *considered* is added, because not seeing
    something is why the question was asked.

    A column that does not **exist** is refused and recorded in `rejected`.
    `operator_answer` is the top of the scale, so an unchecked value would assert
    a target that is not in the dataset — and silently re-admit the
    `equals_target` leak, since nothing equals a column that is not there. The
    CLI checks too; this is the guard for every other way a `DeclaredFacts`
    reaches the profiler.
    """
    if not answer:
        return candidates, []
    from labpilot.accessor.profiler.questions import parse_answer

    try:
        named = parse_answer(field, answer)
    except ValueError as exc:
        return candidates, [RejectedClaim(claim=answer, source="operator", refuted_by=str(exc))]
    unknown = [name for name in named if known and name not in known]
    if unknown:
        return candidates, [
            RejectedClaim(
                claim=answer,
                source="operator",
                refuted_by=f"{unknown} names no column in this dataset",
            )
        ]
    settled = dict(candidates)
    for name in named:
        settled[name] = [
            Signal(id="operator_answer", detail=f"answered: {name!r}"),
            *candidates.get(name, []),
        ]
    return settled, []


def _key_columns(
    answer: str | None, refused: list[RejectedClaim], resolved: str | None
) -> list[str]:
    """The key, which an answer may state as several columns.

    `_resolve` picks one winner, which is right when the profiler is inferring
    and wrong when a person has answered `store_id,date`: taking the first of
    those would settle a composite key as half of itself and close the question.
    """
    from labpilot.accessor.profiler.questions import parse_answer

    if answer and not refused:
        try:
            return parse_answer("id_columns", answer)
        except ValueError:
            pass
    return [resolved] if resolved else []


def _resolve(candidates: dict[str, list[Signal]]) -> tuple[str | None, Inference]:
    """Pick the best-evidenced candidate, and record what every candidate had.

    One decision procedure for every question, replacing a chain of `if`s per
    path: score each candidate against the catalogue, take the highest, keep the
    rest as alternatives with their own evidence.

    A tie keeps today's answer — the last of the tied candidates in sort order —
    so this step changes no value that today's code gets right. That is not a
    defence of the rule: it is position deciding, the caller records a note
    saying so, and step 4 replaces it with a question. What has changed already
    is that the tie is *visible*, because the runners-up carry the same
    confidence in the profile.
    """
    if not candidates:
        return None, Inference.of([])
    scored = sorted(
        ((name, combine(signals), signals) for name, signals in candidates.items()),
        key=lambda row: (-row[1], row[0]),
    )
    tied = [row for row in scored if row[1] == scored[0][1]]
    winner, _, winning_signals = tied[-1] if len(tied) > 1 else scored[0]
    return winner, Inference.of(
        winning_signals,
        alternatives=[
            Alternative.of(name, signals) for name, _, signals in scored if name != winner
        ],
    )


def _exclusions(
    profile: DatasetProfile,
    *,
    target: str | None,
    ids: list[str],
    unavailable: set[str],
    equals_target: str | None = None,
) -> dict[str, ExclusionReason]:
    """Why each column is not a feature.

    Order matters only in that a column gets one reason; the first that applies
    is the one a reader most needs. `equals_target` is last because it is the
    subtlest and the most expensive to get wrong: rogii's `TVT_input` is the
    strongest predictor in the dataset *and* unusable as a plain feature, and a
    profile that lists it among the features is how a model learns to copy a
    column that is NaN on every scored row.
    """
    reasons: dict[str, ExclusionReason] = {}
    for column in profile.columns:
        name = column.name
        if target is not None and name == target:
            reasons[name] = "is_target"
        elif name in ids:
            reasons[name] = "is_id"
        elif name in unavailable:
            reasons[name] = "unavailable_at_scoring"
        elif equals_target is not None and name == equals_target:
            reasons[name] = "equals_target"
        elif column.unique_count <= 1:
            reasons[name] = "constant"
    return reasons


def _modality_signals(
    presences: list[ModalityPresence], *, tiebroken: bool = False
) -> list[Signal]:
    """What is known about which modality carries the signal.

    One modality present is the strong case: there is nothing to weigh it
    against. Several means a preference was applied, and a preference is worth
    less than an absence of alternatives. A model breaking the tie is capped, so
    a modality chosen that way can be acted on and never asserted.
    """
    if not presences:
        return []
    if tiebroken:
        return [
            Signal(
                id="llm_modality_tiebreak",
                detail=f"a model chose {presences[0].modality!r} between the candidates",
            )
        ]
    if len(presences) == 1:
        return [
            Signal(
                id="single_modality_present",
                detail=f"only {presences[0].modality!r} is present",
            )
        ]
    # One id for both cases. Two branches sharing `csv_majority` meant an
    # image-primary dataset fired a signal whose catalogue entry said "tables
    # outnumber the other modality's files" — the id reading as the opposite of
    # what happened, to anything keying off ids rather than prose.
    others = ", ".join(sorted({p.modality for p in presences[1:]}))
    return [
        Signal(
            id="modality_majority",
            detail=f"{presences[0].modality!r} ({presences[0].detail}) over {others}",
        )
    ]


def _unit_from(signals: list[Signal]) -> PredictionUnit:
    """The unit the evidence names, rather than the one the call site expected.

    Both call sites used to map "any signal" onto a fixed value — `row` on the
    flat path, `partition_row` on the partitioned one — so the day a flat-path
    suffix detector lands, the profile would read `prediction_unit: row` while
    its own evidence said the scored rows are a partition tail.
    """
    fired = {signal.id for signal in signals}
    if "scored_rows_are_a_partition_tail" in fired:
        return "partition_row"
    if "submission_row_per_scoring_row" in fired:
        return "row"
    return "unknown"


def _prediction_unit_signals(
    profile: DatasetProfile, *, template_matches_test: bool
) -> list[Signal]:
    """What is known about what one prediction is about."""
    signals: list[Signal] = []
    if profile.scored_is_partition_suffix:
        signals.append(
            Signal(
                id="scored_rows_are_a_partition_tail",
                detail="one row of one partition; rows are not exchangeable across them",
            )
        )
    elif template_matches_test:
        signals.append(
            Signal(
                id="submission_row_per_scoring_row",
                detail="the template has one row per row of the scoring input",
            )
        )
    return signals


def _split_signals(*, has_scoring_input: bool, scored_is_partition_suffix: bool) -> list[Signal]:
    """What is known about how the scored units relate to the training ones."""
    if not has_scoring_input:
        return [Signal(id="no_scoring_input", detail="no scoring input in this dataset")]
    if scored_is_partition_suffix:
        return [
            Signal(
                id="scored_rows_are_partition_tail",
                detail="scored rows are a contiguous tail of each test partition",
            )
        ]
    return [Signal(id="scoring_input_present", detail="a scoring input exists; no split signal")]


def _metric_signals(metric: MetricRef | None) -> list[Signal]:
    signals: list[Signal] = []
    if metric is None:
        return signals
    signals.append(Signal(id="declared_by_source", detail=f"declared metric {metric.name!r}"))
    if metric.direction is not None:
        signals.append(Signal(id="direction_declared", detail=f"direction={metric.direction}"))
    return signals


def _column_signals(profile: DatasetProfile, name: str | None) -> list[Signal]:
    """The distributional evidence a profiled column carries about being a label.

    Weak on purpose. "It is numeric" is true of most columns in most datasets;
    it earns 0.15 because it is consistent with the answer rather than because
    it points at one.
    """
    column = next((c for c in profile.columns if c.name == name), None)
    if column is None:
        return []
    signals: list[Signal] = []
    if column.null_count == 0:
        signals.append(Signal(id="non_null_in_train", detail=f"{column.name}: no nulls in train"))
    if column.is_numeric:
        signals.append(Signal(id="is_numeric", detail=f"{column.name}: {column.dtype}"))
    return signals


def _identity_signals(
    profile: DatasetProfile,
    name: str | None,
    *,
    unit_count: int,
    in_template: bool,
    first_in_template: bool,
    on_both_sides: bool,
) -> list[Signal]:
    """What is known about a column being the key.

    `unique_per_unit` is measured against the frame that was actually profiled,
    not against `row_count`: a partitioned profile estimates its row count from
    a sample, and comparing a sampled cardinality to an extrapolated total would
    make uniqueness look violated on every large dataset.
    """
    signals: list[Signal] = []
    if in_template:
        signals.append(Signal(id="named_in_prediction_template", detail=f"template names {name!r}"))
    if first_in_template:
        signals.append(
            Signal(id="first_template_column", detail=f"{name!r} is the template's first column")
        )
    if on_both_sides:
        signals.append(
            Signal(id="present_in_train_and_scoring", detail=f"{name!r} is on both sides")
        )
    column = next((c for c in profile.columns if c.name == name), None)
    if column is not None and unit_count > 0 and column.unique_count == unit_count:
        signals.append(
            Signal(id="unique_per_unit", detail=f"{column.unique_count} distinct in {unit_count}")
        )
    return signals


def _note(
    profile: DatasetProfile,
    code: str,
    text: str,
    *,
    field: str | None = None,
    severity: str = "info",
) -> None:
    """Record a reason. One writer, so `warnings` has one order and one source."""
    profile.notes.append(Note(code=code, text=text, field=field, severity=severity))  # type: ignore[arg-type]


# Below this many per-entity train files, treat the dataset as ordinary
# multi-file rather than partitioned. Group-aware splits and the partitioned
# template are expensive to get wrong on a normal competition.
_MIN_PARTITIONS = 3


def _local_root(source: DatasetSource) -> Path | None:
    """The directory behind a source, when it has one.

    Modality detection walks a tree and suffix-scoring counts lines; neither is
    expressible through the protocol yet, and both move behind it in step 5.
    Asking one question here — rather than `isinstance` at each call site —
    keeps the list of filesystem-only capabilities to one place, and countable.
    """
    return source.root if isinstance(source, LocalFileSource) else None


def _where(source: DatasetSource) -> str:
    """Something an operator can act on when a source turns up empty."""
    root = _local_root(source)
    return str(root) if root is not None else type(source).__name__


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
        df = source.sample(TableRef(uri=path.name), self.config.max_rows_sample)
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
        """Profile a directory of CSVs. A thin wrapper over :meth:`profile_dataset`.

        Kept as the entry point every existing caller uses, and reduced to the
        one thing it knows that the general path does not: how to build a source
        over a directory.
        """
        logger.info("Profiling dataset directory %s for '%s'", data_dir, competition)
        source = LocalFileSource(
            data_dir,
            DeclaredFacts(title=competition_title, description=competition_description),
        )
        return self.profile_dataset(
            source,
            competition,
            train_pattern=train_pattern,
            test_pattern=test_pattern,
            submission_pattern=submission_pattern,
            llm_client=llm_client,
        )

    def profile_dataset(
        self,
        source: DatasetSource,
        competition: str,
        *,
        train_pattern: str = "train",
        test_pattern: str = "test",
        submission_pattern: str = "submission",
        llm_client: Any | None = None,
    ) -> DatasetProfile:
        """Profile whatever a source exposes.

        The seam M12 needs: an adapter over a warehouse, an object store or an
        environment is profiled by passing it here, with no edit to this module.
        Title and description come from `source.declared()` rather than from
        parameters, because a source is what knows them.

        Two capabilities remain filesystem-only and are skipped, visibly, for a
        source that has no root: modality detection (which walks a directory for
        images) and suffix-scoring detection (which counts lines). Both move
        behind the protocol in step 5; until then a skipped one says so in
        `warnings` rather than leaving a default that reads as a finding.

        File-role detection is a naming-convention heuristic. `train_pattern`,
        `test_pattern`, and `submission_pattern` let a competition's local
        config (`configs/competitions/<slug>.yaml`) override the defaults when a
        dataset doesn't follow the "train*/test*/*submission*" convention.
        """
        # TODO: fetch the real file roles from the Kaggle competition
        # portal/API automatically instead of relying on name matching.
        tables = source.tables()
        if not tables:
            # Not an error. A dataset with no tables is an *environment* — a
            # ConnectX-shaped competition, an interactive harness — and refusing
            # to describe it sent the workspace to `_write_inventory_profile`,
            # which wrote a valid-looking profile with a null target and a
            # modality guessed from file extensions. Describing it honestly and
            # asking the questions it cannot answer is the whole point.
            return self._profile_environment(source, competition)

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

        train_matches = [
            table for table in tables if _name_of(table).startswith(train_pattern.lower())
        ]
        if not train_matches and len(tables) == 1:
            # One table is the training table. A dataset that is not a
            # competition has no `train` prefix to match — a warehouse extract,
            # a study export, a log dump — and refusing to describe it at all
            # was the profiler's answer to the entire world outside Kaggle.
            train_table = tables[0]
        else:
            train_table = self._single_table(train_matches, "training")

        sample_matches = [
            table for table in tables if submission_pattern.lower() in _name_of(table)
        ]
        # Absent is a fact about the dataset; two is an ambiguity nobody can
        # resolve from here, so only the second still refuses.
        sample_table = (
            self._single_table(sample_matches, "sample submission") if sample_matches else None
        )

        test_matches = [
            table for table in tables if _name_of(table).startswith(test_pattern.lower())
        ]
        test_warnings: list[str] = []
        if len(test_matches) == 1:
            test_table = test_matches[0]
        elif len(test_matches) == 0:
            test_table = sample_table
            if sample_table is not None:
                test_warnings.append(
                    "No test CSV found; using sample submission as the test reference file."
                )
        else:
            names = [Path(table.uri).name for table in test_matches]
            raise ValueError(f"Expected one test CSV, found {len(test_matches)}: {names}")

        train_columns = source.columns(train_table)
        test_columns = source.columns(test_table) if test_table is not None else []
        submission_columns = source.columns(sample_table) if sample_table is not None else []

        # The profile is built before the answers are resolved, because the
        # evidence for an answer includes what the columns look like — whether
        # the candidate is complete, whether it is numeric — and those are facts
        # about the frame rather than about the header.
        train_sample = source.sample(train_table, self.config.max_rows_sample)
        profile = DatasetProfile(
            competition=competition,
            row_count=len(train_sample),
            column_count=len(train_sample.columns),
            columns=self.profile_columns(train_sample),
            column_stats_rows=len(train_sample),
        )
        if len(train_sample) == self.config.max_rows_sample and train_columns:
            # The cap bound, so the sample's length is a floor rather than a
            # count. `playground-series-s6e7/profile.json` records 100,000 rows
            # for a file of 690,088 and does not say it is a sample. One pass
            # over one column is what the truth costs.
            try:
                profile.row_count = source.exact_unit_count(train_table, train_columns[0])
                _note(
                    profile,
                    "columns_sampled",
                    f"column statistics describe {profile.column_stats_rows:,} sampled rows of "
                    f"{profile.row_count:,} — read null and unique counts against "
                    "`column_stats_rows`, not `row_count`",
                    field="columns",
                    severity="caution",
                )
            except (OSError, ValueError):
                profile.row_count_estimated = True
        for text in test_warnings:
            _note(profile, "no_test_file", text, severity="caution")
        from labpilot.accessor.profiler.modality import ModalityDetector

        detector = ModalityDetector()
        detector.enrich_column_stats(train_sample, profile.columns)
        profile.files = [table.uri for table in tables]
        profile.train_file = train_table.uri
        profile.test_file = test_table.uri if test_table is not None else None
        profile.sample_submission_file = sample_table.uri if sample_table is not None else None
        profile.submission_columns = submission_columns

        # --- which columns could be the label ------------------------------
        withheld = [column for column in train_columns if column not in test_columns]
        target_candidates: dict[str, list[Signal]] = {}
        if test_table is not None:
            for candidate in withheld:
                signals: list[Signal] = []
                if candidate in submission_columns:
                    signals.append(
                        Signal(
                            id="named_in_prediction_template",
                            detail=f"submission header names {candidate!r}",
                        )
                    )
                if withheld == [candidate]:
                    signals.append(
                        Signal(
                            id="sole_withheld_column",
                            detail=f"{candidate!r} is the only column train has and test does not",
                        )
                    )
                target_candidates[candidate] = signals + _column_signals(profile, candidate)
        if not target_candidates and test_table is not None and test_table == sample_table:
            # The positional branch: no column is withheld, so the template's
            # own overlap with train is all there is. Capped at 0.50 by the
            # catalogue, which is what makes it ask rather than answer.
            overlap = [column for column in submission_columns if column in train_columns]
            if len(overlap) >= 2:
                target_candidates[overlap[1]] = [
                    Signal(
                        id="positional_template_overlap",
                        detail=f"second of {len(overlap)} columns the template shares with train",
                    )
                ] + _column_signals(profile, overlap[1])

        answers = source.declared().answers
        profile.answers_fingerprint = answers_fingerprint(answers)
        known = {column.name for column in profile.columns}
        settled, refused = _answered(target_candidates, answers.get("target_column"), known=known)
        target_column, target_inference = _resolve(settled)
        profile.target_column = target_column
        profile.inferences["target_column"] = target_inference.model_copy(
            update={"rejected": refused}
        )
        for claim in refused:
            _note(
                profile,
                "answer_refused",
                f"answer {claim.claim!r} refused: {claim.refuted_by}",
                field="target_column",
                severity="blocking",
            )

        # --- which columns could be the key --------------------------------
        id_candidates: dict[str, list[Signal]] = {}
        for candidate in submission_columns:
            if candidate == target_column:
                continue
            id_candidates[candidate] = _identity_signals(
                profile,
                candidate,
                unit_count=len(train_sample),
                in_template=True,
                first_in_template=candidate == submission_columns[0],
                on_both_sides=candidate in train_columns and candidate in test_columns,
            )
        id_settled, id_refused = _answered(
            id_candidates, answers.get("id_columns"), known=known, field="id_columns"
        )
        id_column, id_inference = _resolve(id_settled)
        profile.id_columns = _key_columns(answers.get("id_columns"), id_refused, id_column)
        profile.inferences["id_columns"] = id_inference.model_copy(update={"rejected": id_refused})
        for claim in id_refused:
            _note(
                profile,
                "answer_refused",
                f"answer {claim.claim!r} refused: {claim.refuted_by}",
                field="id_columns",
                severity="blocking",
            )

        # A contract, not an inference: a template whose columns are not the id
        # and the target describes a submission this profile cannot produce, and
        # failing here is closer to the cause than failing at submission time.
        if sample_table is not None:
            expected_submission_columns = [id_column, target_column]
            if submission_columns != expected_submission_columns:
                raise ValueError(
                    "Sample submission schema does not match the inferred ID and target columns: "
                    f"expected {expected_submission_columns}, got {submission_columns}."
                )

        for column in profile.columns:
            column.is_target_candidate = column.name == target_column

        if test_table is not None and id_column is not None:
            profile.test_row_count = source.exact_unit_count(test_table, id_column)

        # --- the remaining answers -----------------------------------------
        profile.train_only_columns = withheld if test_table is not None else []
        profile.excluded_columns = _exclusions(
            profile,
            target=target_column,
            ids=profile.id_columns,
            unavailable=set(profile.train_only_columns),
        )
        split_signals = _split_signals(
            has_scoring_input=test_table is not None,
            scored_is_partition_suffix=False,
        )
        profile.train_test_relationship = (
            "disjoint_units" if test_table is not None else "no_test_provided"
        )
        profile.inferences["train_test_relationship"] = Inference.of(split_signals)
        profile.metric = source.declared().metric
        profile.inferences["metric"] = Inference.of(_metric_signals(profile.metric))
        # Counted over one column rather than parsed whole: the template has a
        # row per scored row, so on a large competition this read 690,000 rows
        # into a frame and threw it away to learn one integer.
        template_rows = (
            source.exact_unit_count(sample_table, submission_columns[0])
            if sample_table is not None and submission_columns
            else 0
        )
        template_matches_test = (
            sample_table is not None
            and test_table is not None
            and sample_table != test_table
            and profile.test_row_count > 0
            and template_rows == profile.test_row_count
        )
        unit_signals = _prediction_unit_signals(
            profile, template_matches_test=template_matches_test
        )
        profile.prediction_unit = _unit_from(unit_signals)
        profile.inferences["prediction_unit"] = Inference.of(unit_signals)
        if profile.target_column is None:
            why = (
                "no scoring input to compare train against"
                if test_table is None
                else "no candidate column"
            )
            _note(
                profile,
                "no_target_identified",
                f"no column could be identified as the label: {why}",
                field="target_column",
                severity="blocking",
            )

        declared = source.declared()
        if self.config.llm_proposals and llm_client is not None:
            self._fold_in_proposal(profile, llm_client, declared)
        root = _local_root(source)
        if root is None:
            # Not a default that reads as a finding: `modality` stays "tabular"
            # either way, and the difference between "detected tabular" and
            # "never looked" has to be visible or it is the silent degrade M14
            # exists to remove.
            _note(
                profile,
                "modality_not_detected",
                "modality not detected: source exposes no directory",
                field="modality",
                severity="caution",
            )
        else:
            modality = detector.detect(
                root,
                profile,
                llm_client=llm_client,
                competition_title=declared.title,
                competition_description=declared.description,
            )
            profile.modalities = detector.presences(root, profile)
            profile.image_dir = modality.image_dir
            profile.image_column = modality.image_column
            profile.text_column = modality.text_column
            profile.inferences["modality"] = Inference.of(
                _modality_signals(profile.modalities, tiebroken=modality.tiebroken)
            )
            for text in modality.signals:
                _note(profile, "modality_signal", text, field="modality")
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

    def _fold_in_proposal(self, profile: DatasetProfile, llm_client: Any, declared: Any) -> None:
        """Ask a model what it thinks, and let the data answer back.

        Called after every deterministic answer is settled, and given none of
        them: the proposal is worth something only if it was reached
        independently. It can raise a confidence by 0.10 or add an alternative;
        it cannot change a value, which `test_the_value_plane_ignores_the_model`
        checks against a proposer that is wrong about everything.
        """
        from labpilot.accessor.profiler.proposer import apply_proposal, propose_schema

        proposal = propose_schema(
            profile,
            llm_client=llm_client,
            title=declared.title,
            description=declared.description,
        )
        if proposal is None:
            _note(
                profile,
                "llm_proposal_unavailable",
                "the schema proposer was enabled and produced nothing",
                severity="info",
            )
            return
        apply_proposal(profile, proposal)

    def _profile_environment(self, source: DatasetSource, competition: str) -> DatasetProfile:
        """A dataset with no tables: say so, and ask what cannot be inferred.

        No columns, so no target and no key — both `uncertain` at 0.0, which
        raises the questions that stop a campaign rather than letting it act on
        a description nobody produced. `action_space` is deliberately **not**
        inferred: no fixture exists and the output would be unfalsifiable.
        """
        profile = DatasetProfile(competition=competition)
        root = _local_root(source)
        if root is not None:
            profile.files = [
                str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()
            ][:200]
            from labpilot.accessor.profiler.modality import ModalityDetector

            profile.modalities = ModalityDetector().presences(root, profile)
        else:
            profile.modalities = [
                ModalityPresence(
                    modality="environment", role="primary", detail="source exposes no tables"
                )
            ]
        profile.train_test_relationship = "environment"
        profile.inferences["train_test_relationship"] = Inference.of(
            [Signal(id="no_tabular_data", detail="no tables in this dataset")]
        )
        profile.inferences["modality"] = Inference.of(_modality_signals(profile.modalities))
        profile.prediction_unit = "episode"
        profile.inferences["prediction_unit"] = Inference.of(
            [Signal(id="no_tabular_data", detail="no units to predict, only an environment")]
        )
        profile.metric = source.declared().metric
        profile.inferences["metric"] = Inference.of(_metric_signals(profile.metric))
        profile.inferences["target_column"] = Inference.of([])
        profile.inferences["id_columns"] = Inference.of([])
        profile.answers_fingerprint = answers_fingerprint(source.declared().answers)
        _note(
            profile,
            "environment_dataset",
            f"no tables found: {len(profile.files)} file(s), described as an environment. "
            "Nothing here can name a target or a key.",
            severity="blocking",
        )
        return profile

    def _try_profile_partitioned(
        self,
        source: DatasetSource,
        competition: str,
        tables: list[TableRef],
        *,
        train_pattern: str,
        test_pattern: str,
        submission_pattern: str,
    ) -> DatasetProfile | None:
        """Profile one-file-per-entity datasets, or return None if not that shape."""
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
        # How many of the primary kind's sampled tables carry each column. Read
        # here rather than inside the fallback below, because it is evidence
        # about every candidate — a label is in most partitions of its kind —
        # and the fallback is only one of the two paths that need it.
        seen_in = Counter[str]()
        for frame in frames:
            seen_in.update({str(c) for c in frame.columns})
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
            for frame in frames:
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
                    "partitions and none is named in a sample submission. Settle "
                    "it with `research schema answer target_column <column>`."
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
            column_stats_rows=len(sample_df),
        )
        profile.files = [table.uri for table in tables[:200]]
        profile.train_file = sampled[0].uri
        profile.test_file = test_kind_tables[0].uri if test_kind_tables else None
        profile.sample_submission_file = sample_table.uri if sample_table else None
        profile.submission_columns = submission_columns
        profile.target_column = target
        profile.row_count = row_count
        profile.row_count_estimated = True
        profile.column_count = len(sample_df.columns)
        profile.partitioned = True
        profile.partition_key = "file_stem_entity"
        profile.partition_kinds = {k: len(v) for k, v in sorted(kinds.items())}
        profile.train_partition_count = len(kinds[primary_kind])
        profile.test_partition_count = len(test_kind_tables)
        profile.train_only_columns = train_only

        def target_signals_for(candidate: str) -> list[Signal]:
            signals: list[Signal] = []
            if candidate.lower() in sub_lower:
                signals.append(
                    Signal(
                        id="named_in_prediction_template",
                        detail=f"submission header names {candidate!r}",
                    )
                )
            if train_only == [candidate]:
                signals.append(
                    Signal(
                        id="sole_withheld_column",
                        detail=f"{candidate!r} is the only column withheld at test",
                    )
                )
            elif seen_in and seen_in[candidate] == max(
                (seen_in[str(c)] for c in train_only), default=0
            ):
                # The modal count among the candidates, not "in every file": one
                # partition with a schema quirk should not retire a label, and
                # `max_files_sample` is 25, so some file having one is likely.
                signals.append(
                    Signal(
                        id="present_across_train_units",
                        detail=f"in {seen_in[candidate]}/{len(frames)} {primary_kind} tables",
                    )
                )
            return signals + _column_signals(profile, candidate)

        # The same resolver the flat path uses. `target` above is still the
        # value — this step does not move answers the code already gets right —
        # and the resolver is asserted to agree with it, so the two cannot drift
        # while the old procedure is still in place.
        answers = source.declared().answers
        profile.answers_fingerprint = answers_fingerprint(answers)
        known = {column.name for column in profile.columns}
        settled, refused = _answered(
            {str(c): target_signals_for(str(c)) for c in train_only},
            answers.get("target_column"),
            known=known,
        )
        scored_target, target_inference = _resolve(settled)
        target_inference = target_inference.model_copy(update={"rejected": refused})
        for claim in refused:
            _note(
                profile,
                "answer_refused",
                f"answer {claim.claim!r} refused: {claim.refuted_by}",
                field="target_column",
                severity="blocking",
            )
        if answers.get("target_column") and not refused:
            # An answer is not a hint. Where a person has settled the question,
            # the value follows the answer rather than the procedure that could
            # not settle it — and `profile.target_column` is assigned above, so
            # updating `target` alone would leave the answer in the evidence and
            # the guess in the value.
            target = scored_target
            profile.target_column = scored_target
        if target is not None:
            profile.inferences["target_column"] = target_inference
            if scored_target != str(target):
                # Evidence and procedure disagreeing is a finding, not something
                # to resolve silently in favour of either.
                _note(
                    profile,
                    "target_disagrees_with_evidence",
                    f"the inferred target is {target!r}; the best-evidenced candidate is "
                    f"{scored_target!r}",
                    field="target_column",
                    severity="blocking",
                )

        id_settled, id_refused = _answered(
            {
                candidate: _identity_signals(
                    profile,
                    candidate,
                    unit_count=len(sample_df),
                    # Named, but not *chosen* for being named: this path takes
                    # the template's first column without checking anything
                    # about it, so the evidence has to say position.
                    in_template=False,
                    first_in_template=candidate == submission_columns[0],
                    on_both_sides=candidate in set(sample_df.columns) and candidate in test_columns,
                )
                for candidate in submission_columns[:1]
            },
            answers.get("id_columns"),
            known=known,
            field="id_columns",
        )
        id_column, id_inference = _resolve(id_settled)
        profile.id_columns = _key_columns(answers.get("id_columns"), id_refused, id_column)
        if id_column is not None:
            profile.inferences["id_columns"] = id_inference.model_copy(
                update={"rejected": id_refused}
            )
        for claim in id_refused:
            _note(
                profile,
                "answer_refused",
                f"answer {claim.claim!r} refused: {claim.refuted_by}",
                field="id_columns",
                severity="blocking",
            )
        self._detect_suffix_scoring(profile, source, sample_table, test_kind_tables)
        # Once, not twice: it reads every sampled partition to answer, and both
        # the exclusion below and `anchor_column` further down want the same
        # column for the same reason.
        profile.anchor_column = _detect_anchor_column(
            frames, profile.target_column, lambda name: not withheld_at_test(name)
        )
        profile.excluded_columns = _exclusions(
            profile,
            target=profile.target_column,
            ids=profile.id_columns,
            unavailable=set(train_only),
            equals_target=profile.anchor_column,
        )
        profile.train_test_relationship = (
            "partition_suffix"
            if profile.scored_is_partition_suffix
            else ("disjoint_units" if test_kind_tables else "no_test_provided")
        )
        profile.inferences["train_test_relationship"] = Inference.of(
            _split_signals(
                has_scoring_input=bool(test_kind_tables),
                scored_is_partition_suffix=profile.scored_is_partition_suffix,
            )
        )
        profile.metric = source.declared().metric
        profile.inferences["metric"] = Inference.of(_metric_signals(profile.metric))
        # Modality was never detected on this path at all: it returns before the
        # block that does it, so every partitioned profile carried the field's
        # *default*. rogii reads `modality: tabular` and nothing ever looked —
        # which is why the PNG previews beside its 1,546 tables were invisible.
        modality_root = _local_root(source)
        if modality_root is None:
            _note(
                profile,
                "modality_not_detected",
                "modality not detected: source exposes no directory",
                field="modality",
                severity="caution",
            )
        else:
            from labpilot.accessor.profiler.modality import ModalityDetector

            profile.modalities = ModalityDetector().presences(modality_root, profile)
            primary = profile.modalities[0] if profile.modalities else None
            profile.image_dir = primary.image_dir if primary else None
            profile.inferences["modality"] = Inference.of(_modality_signals(profile.modalities))
        unit_signals = _prediction_unit_signals(profile, template_matches_test=False)
        profile.prediction_unit = _unit_from(unit_signals)
        profile.inferences["prediction_unit"] = Inference.of(unit_signals)
        # Appended, not assigned. `profile.warnings = [...]` here discarded
        # everything recorded earlier in this method — including the note
        # `_detect_suffix_scoring` writes one line above when a source has no
        # directory to count lines in, which was written and thrown away in the
        # same breath.
        _note(
            profile,
            "partitioned_layout",
            f"partitioned dataset: {len(train_files)} train / {len(test_files)} test CSVs",
        )
        _note(
            profile,
            "primary_kind",
            f"primary kind={primary_kind!r}; kinds={profile.partition_kinds}",
        )
        _note(
            profile,
            "row_count_estimated",
            f"row_count estimated from {len(sampled)} sampled files",
            field="row_count",
        )
        _note(
            profile,
            "rows_not_iid",
            "rows are NOT iid across partitions — validation must group by partition",
            severity="caution",
        )
        for text in ambiguous_target:
            _note(profile, "ambiguous_target", text, field="target_column", severity="caution")
        if train_only:
            _note(
                profile,
                "train_only_columns",
                f"train-only columns (unavailable at test): {train_only}",
            )
        if profile.scored_is_partition_suffix:
            _note(
                profile,
                "scored_is_partition_suffix",
                f"scored rows are a contiguous suffix of each test partition "
                f"(~{profile.scored_fraction:.0%} of rows) — this is a forecast task; "
                "validate by holding out each partition's tail",
                severity="caution",
            )
        if profile.anchor_column:
            _note(
                profile,
                "anchor_column",
                f"{profile.anchor_column!r} is the known prefix of {profile.target_column!r}: "
                f"equal to it wherever present, absent exactly on the scored rows. Carrying its "
                f"last known value forward is the baseline to beat — predict the residual from "
                f"it, not {profile.target_column!r} from the other columns. Note it is identical "
                f"to the target in training, so using it as a plain feature learns 'copy' and "
                f"then meets NaN on every scored row.",
                field="anchor_column",
                severity="caution",
            )
        return profile

    def _detect_suffix_scoring(
        self,
        profile: DatasetProfile,
        source: DatasetSource,
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
        change is how a "value-neutral" step stops being one. It is also why
        this needs a filesystem: a source without one skips detection and says
        so, rather than reporting `scored_is_partition_suffix=False`, which a
        reader cannot tell from "looked, and it is not a forecast".
        """
        if sample_table is None or not test_kind_tables:
            return
        if not isinstance(source, LocalFileSource):
            _note(
                profile,
                "suffix_scoring_not_detected",
                "suffix scoring not detected: source exposes no directory to count lines in",
                severity="caution",
            )
            return
        try:
            submission = source.sample(sample_table, None)
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
            except (OSError, UnicodeDecodeError):
                # `UnicodeDecodeError` is a `ValueError`, so the original catch
                # let a latin-1 partition escape a best-effort detector and take
                # the whole profile down with it — the workspace then falls back
                # to a filesystem inventory because one file had an accent in it.
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
