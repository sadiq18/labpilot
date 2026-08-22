"""M23 step 1: what shape the target is, measured rather than read off prose.

The design's line is the reason this lives beside the schema and not beside the
modality: *"the floor is determined by the shape of the prediction target, not
the modality of the input."* An image competition's label is still a class
column and its floor is still the class prior.

`target_type` is **derived**, not stored — every input is already in the profile,
so a stored copy could only be a second answer free to disagree with the first,
and every profile written before this field existed acquires it on load.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers.dataset_sources import DictSource

from labpilot.accessor.profiler.questions import pending_schema_questions
from labpilot.accessor.profiler.source import DeclaredFacts
from labpilot.accessor.profiler.tabular import (
    ColumnProfile,
    DatasetProfile,
    TabularProfiler,
    _target_distribution,
)
from labpilot.config import ProfilerConfig


def _profile(target: list, *, extra: dict | None = None) -> DatasetProfile:
    """A minimal three-table dataset whose target holds `target`."""
    n = len(target)
    frame = pd.DataFrame({"Id": range(1, n + 1), "x": [float(i) for i in range(n)], "y": target})
    for name, values in (extra or {}).items():
        frame[name] = values
    tables = {
        "train.csv": frame,
        "test.csv": frame[["Id", "x"]],
        "sample_submission.csv": frame[["Id", "y"]],
    }
    return TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "demo")


# --- what it decides ---------------------------------------------------------


def test_two_values_is_binary() -> None:
    assert _profile([0, 1] * 8).target_type == "binary"


def test_a_bool_target_is_binary() -> None:
    """`is_numeric` excludes bools, so the numeric branch never sees this one.

    `spaceship-titanic` is in the corpus for exactly this: `True`/`False`, not
    0/1, and a rule that only read numbers would call it multiclass.
    """
    profile = _profile([True, False] * 8)

    assert profile.target_type == "binary"
    assert set(profile.target_distribution.class_counts) == {"True", "False"}


def test_a_handful_of_labels_is_multiclass() -> None:
    profile = _profile(["cat", "dog", "bird"] * 6)

    assert profile.target_type == "multiclass"
    assert profile.target_distribution.class_counts == {"cat": 6, "dog": 6, "bird": 6}


def test_a_price_is_continuous_not_a_class_per_row() -> None:
    """The misreading this milestone's corpus names by name.

    `SalePrice` is an **integer** column: 663 distinct values, maximum 755,000.
    Reading it as classification makes the floor a class prior over hundreds of
    classes — wrong, and slow. Two guards catch it: the value ceiling, and the
    fact that a label set has labels that repeat.
    """
    prices = [34900 + 1000 * i for i in range(40)]

    profile = _profile(prices)

    assert profile.target_type == "continuous"
    assert profile.target_distribution.class_counts == {}, "40 distinct prices are not 40 classes"
    assert profile.target_distribution.median is not None


def test_distinct_values_below_the_ceiling_are_still_not_labels() -> None:
    """The same misreading through the small-fixture door.

    Twelve distinct prices over twelve rows clear the label ceiling on count
    alone. What tells them apart from twelve classes is that classes recur and
    these do not — which is why the rule is `unique < row_count` and not a
    threshold on `unique` by itself.
    """
    profile = _profile(
        [
            208500,
            181500,
            223500,
            140000,
            250000,
            143000,
            307000,
            200000,
            129900,
            118000,
            157000,
            232000,
        ]
    )

    assert profile.target_type == "continuous"
    assert profile.target_distribution.class_counts == {}


def test_dense_non_negative_integers_are_counts() -> None:
    """0-30 over 60 rows: the values crowd their own range, which a price does not."""
    profile = _profile([i % 31 for i in range(93)])

    assert profile.target_type == "count"


def test_a_wide_template_is_multilabel() -> None:
    """Several scored columns per unit, each of them a column of train. One of
    them being the target does not make the task single-output.

    Still asserted on the model as well as through the profiler below, because
    `DatasetProfile` is loaded from `profile.json` by readers that never ran the
    profiler — the rule has to hold for a profile that merely arrives.
    """
    profile = DatasetProfile(
        competition="wide",
        target_column="a",
        row_count=4,
        id_columns=["Id"],
        submission_columns=["Id", "a", "b"],
        columns=[
            ColumnProfile(name="a", dtype="int64", unique_count=2, is_numeric=True),
            ColumnProfile(name="b", dtype="int64", unique_count=2, is_numeric=True),
        ],
    )

    assert profile.target_type == "multilabel", "two scored columns, not one"


def test_the_profiler_refuses_a_wide_template_by_asking() -> None:
    """It used to raise, which made every multi-target competition unprofileable
    — and unmeasurable, because the exception escaped the benchmark rather than
    being scored.

    It still refuses, and the refusal is the point: `[Id, a, b]` has no single
    target column, so naming one would be picking `b` and never mentioning `a`.
    What changed is the *mechanism*. `Note.severity` is written and never read,
    so a note alone would have been a silent wrong run; leaving `target_column`
    unresolved routes it through `pending_schema_questions`, which is what
    actually stops a campaign, and is how rogii's unresolvable key already works.
    """
    frame = pd.DataFrame(
        {"Id": [1, 2, 3, 4], "x": [1.0, 2, 3, 4], "a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}
    )
    tables = {
        "train.csv": frame,
        "test.csv": frame[["Id", "x"]],
        "sample_submission.csv": frame[["Id", "a", "b"]],
    }

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "wide")

    assert profile.target_column is None, "naming one of two targets is the bug, not the fix"
    assert profile.target_type == "multilabel"
    assert [q.field for q in pending_schema_questions(profile)] == ["target_column"]
    assert {a.candidate for a in profile.inferences["target_column"].alternatives} == {"a", "b"}
    assert any(note.code == "multi_output_template" for note in profile.notes)


def _wide_tables() -> dict:
    """`[Id, a, b]` — two scored columns, both of them columns of train."""
    frame = pd.DataFrame(
        {"Id": [1, 2, 3, 4], "x": [1.0, 2, 3, 4], "a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}
    )
    return {
        "train.csv": frame,
        "test.csv": frame[["Id", "x"]],
        "sample_submission.csv": frame[["Id", "a", "b"]],
    }


def test_an_answered_target_survives_the_multi_output_refusal() -> None:
    """Review finding, and the one that made the refusal a dead end.

    The refusal used to run unconditionally, so an operator who answered the
    question it raises had their answer honoured by `_answered` and then thrown
    away. `pending_schema_questions` skips answered fields, so nothing asked
    again either: the profile carried no target, nothing blocked, and a campaign
    ran against it. Askable but unanswerable is worse than the `ValueError` this
    replaced, because that at least stopped.
    """
    answers = {"target_column": "a"}
    source = DictSource(_wide_tables(), DeclaredFacts(answers=answers))

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, "wide")

    assert profile.target_column == "a", "the operator answered; the answer is the answer"
    assert pending_schema_questions(profile, answers) == []
    # Still multi-output — a person naming a primary does not narrow the
    # template — and the note says which of the two they chose.
    assert profile.target_type == "multilabel"
    assert any(note.code == "multi_output_template_answered" for note in profile.notes)
    assert not any(note.code == "multi_output_template" for note in profile.notes)


def test_an_answer_naming_no_column_does_not_count_as_answered() -> None:
    """`_key_columns` uses `answer and not refused` for the same decision, and a
    refused answer must not buy its way past a refusal — otherwise a typo would
    resolve a target the dataset does not have.
    """
    source = DictSource(_wide_tables(), DeclaredFacts(answers={"target_column": "no_such_column"}))

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, "wide")

    assert profile.target_column is None
    assert [q.field for q in pending_schema_questions(profile)] == ["target_column"]


def test_a_template_column_that_is_not_the_target_is_recorded() -> None:
    """The other half of what the removed guard checked.

    `[Id, Prediction]` against a `SalePrice` target used to raise. Accepting it
    is right — competitions rename the scored column all the time — but the
    profile would otherwise record a target and a template that do not
    correspond, with nothing to say whether that was checked and allowed or
    never looked at.
    """
    n = 40
    train = pd.DataFrame(
        {
            "Id": range(n),
            "x": [i % 7 for i in range(n)],
            "SalePrice": [100000 + i * 97 for i in range(n)],
        }
    )
    tables = {
        "train.csv": train,
        "test.csv": train[["Id", "x"]],
        "sample_submission.csv": pd.DataFrame({"Id": range(n), "Prediction": [0.0] * n}),
    }

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "renamed")

    assert profile.target_column == "SalePrice"
    assert any(note.code == "template_column_is_not_the_target" for note in profile.notes)


def test_a_template_that_names_the_target_says_nothing() -> None:
    """The negative case. A note on every ordinary competition is a note nobody
    reads by the time it matters.
    """
    n = 40
    train = pd.DataFrame(
        {
            "Id": range(n),
            "x": [i % 7 for i in range(n)],
            "SalePrice": [100000 + i * 97 for i in range(n)],
        }
    )
    tables = {
        "train.csv": train,
        "test.csv": train[["Id", "x"]],
        "sample_submission.csv": pd.DataFrame({"Id": range(n), "SalePrice": [0.0] * n}),
    }

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "plain")

    template_notes = {
        "multi_output_template",
        "multi_output_template_answered",
        "encoded_target_template",
        "template_column_is_not_the_target",
    }
    assert not [note.code for note in profile.notes if note.code in template_notes]


def test_a_template_of_class_probabilities_is_not_multi_output() -> None:
    """`class_0, class_1, class_2` is one target written out, not three targets.

    The discriminator is whether train holds the scored columns. Without it,
    every multiclass competition scored on probabilities reads as `multilabel`
    and its single, perfectly resolvable target gets refused — which would trade
    one uncapturable class of competition for another.
    """
    n = 60
    train = pd.DataFrame(
        {"Id": range(n), "x": [i % 7 for i in range(n)], "y": [i % 3 for i in range(n)]}
    )
    tables = {
        "train.csv": train,
        "test.csv": train[["Id", "x"]],
        "sample_submission.csv": pd.DataFrame(
            {"Id": range(n), "class_0": [0.3] * n, "class_1": [0.3] * n, "class_2": [0.4] * n}
        ),
    }

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "probs")

    assert profile.target_column == "y"
    assert profile.target_type == "multiclass", "three columns of one answer, not three answers"
    assert pending_schema_questions(profile) == [], "nothing here is ambiguous"


def test_an_unresolved_id_does_not_make_a_target_multilabel() -> None:
    """Review finding, reproduced against the shipped rogii profile.

    An empty `id_columns` means the key was **not resolved**, not that there is
    no key — and it is empty exactly when the profiler decided to ask, which is
    rogii. Subtracting nothing left `['id', 'tvt']` as two scored columns, so the
    fixture this milestone is built around read as `multilabel`, its continuous
    depth target went down the classification path, and the
    `partitioned and TABULAR_REGRESSION` guard in `select` — the one that stops a
    partitioned dataset being validated on shuffled rows — was skipped with it.
    """
    profile = DatasetProfile(
        competition="rogii",
        row_count=1000,
        target_column="tvt",
        id_columns=[],
        submission_columns=["id", "tvt"],
        columns=[
            ColumnProfile(
                name="tvt",
                dtype="float64",
                unique_count=900,
                is_numeric=True,
                stats={"min": 0.0, "max": 999.5},
            )
        ],
    )

    assert profile.target_type == "continuous"


def test_an_unresolved_id_on_a_wide_template_is_unknown() -> None:
    """Two columns are unambiguous whichever the key is; four are not.

    "key plus three labels" and "four labels" are the same shape from here, and
    saying so beats guessing either way.
    """
    profile = DatasetProfile(
        competition="d",
        row_count=100,
        target_column="a",
        id_columns=[],
        submission_columns=["id", "a", "b", "c"],
        columns=[ColumnProfile(name="a", dtype="int64", unique_count=2, is_numeric=True)],
    )

    assert profile.target_type == "unknown"


@pytest.mark.parametrize(
    "stats",
    [{"min": "0", "max": "10"}, {"min": [1], "max": 5.0}, {"min": True, "max": 5.0}],
)
def test_unreadable_stats_do_not_break_the_profile(stats: dict) -> None:
    """Review finding. `target_type` is a `computed_field`.

    A raise here does not fail one field — it fails `model_dump_json()` for the
    whole profile with `PydanticSerializationError`, so one bad number in
    `stats` (typed `dict[str, Any]`, and read from disk) takes down the artifact.
    A bool is in the list because `isinstance(True, int)` is true and `True >= 0`
    would have sailed through a numeric check.
    """
    profile = DatasetProfile(
        competition="d",
        row_count=100,
        target_column="y",
        id_columns=["Id"],
        submission_columns=["Id", "y"],
        columns=[
            ColumnProfile(name="y", dtype="int64", unique_count=50, is_numeric=True, stats=stats)
        ],
    )

    assert profile.target_type == "unknown"
    assert profile.model_dump_json(), "the whole profile must still serialize"


def test_the_class_labels_carry_their_dtype() -> None:
    """JSON keys are strings; the floor needs the label back.

    A float64 target — which is any integer column pandas met a NaN in — gives
    keys "0.0"/"1.0". A floor predicting the argmax without `class_dtype` writes
    that string into a submission whose sample column is an integer.
    """
    import pandas as pd

    distribution = _target_distribution(pd.Series([0.0, 1.0, 1.0, 0.0], dtype="float64"))

    assert set(distribution.class_counts) == {"0.0", "1.0"}
    assert distribution.class_dtype == "float64"


# --- what it refuses to decide -----------------------------------------------


def test_no_target_is_none_not_unknown() -> None:
    """Different situations. `none` is "there is no target column here"; a
    dataset whose target could not be identified is not the same as one that
    has none, and the floor treats them differently."""
    assert DatasetProfile(competition="demo").target_type == "none"


def test_ordinal_is_never_derived() -> None:
    """Whether 1-5 are ranks or five unrelated labels is a fact about the world.

    A detector that answered it would be asserting semantics from arithmetic.
    The member exists so an operator's answer has somewhere to land; nothing
    computes it, and this test says so out loud.
    """
    profile = _profile([1, 2, 3, 4, 5] * 4)

    assert profile.target_type == "multiclass", "measured as labels, not ranked"


def test_a_constant_target_is_unknown_not_binary() -> None:
    """A floor built on a constant is perfect and meaningless."""
    profile = DatasetProfile(
        competition="demo",
        target_column="y",
        row_count=10,
        columns=[
            ColumnProfile(
                name="y",
                dtype="int64",
                unique_count=1,
                is_numeric=True,
                # With stats present the column otherwise reads as a one-label
                # multiclass problem, which is what makes the guard load-bearing
                # rather than shadowed by a missing-stats fallthrough.
                stats={"min": 5.0, "max": 5.0, "mean": 5.0, "std": 0.0},
            )
        ],
    )

    assert profile.target_type == "unknown"


def test_a_target_naming_no_column_is_unknown() -> None:
    """A name with nothing behind it. Reporting a shape for it would be
    describing a column that is not in the profile."""
    profile = DatasetProfile(competition="demo", target_column="ghost", row_count=10)

    assert profile.target_type == "unknown"


# --- the distribution --------------------------------------------------------


def test_the_distribution_is_what_the_floor_needs() -> None:
    """Median, not just mean: the optimal constant under absolute error.

    `stats` on the column already carries the mean. A floor that only knew the
    mean would be the wrong constant for every MAE competition, which is the
    whole reason this field is not just a pointer at `stats`.
    """
    distribution = _target_distribution(pd.Series([1.0, 2.0, 2.0, 10.0]))

    assert distribution.median == 2.0
    assert distribution.zero_fraction == 0.0
    assert distribution.skew is not None


def test_an_unmeasurable_target_is_empty_rather_than_zero() -> None:
    """Empty is a real answer, and a safe one: M23's floor refuses to build on
    it. Zeros would put a floor under a column nobody could read."""
    distribution = _target_distribution(pd.Series([None, None], dtype="float64"))

    assert distribution.class_counts == {}
    assert distribution.median is None
    assert distribution.null_count == 2


def test_a_partitioned_dataset_is_measured_too() -> None:
    """rogii takes the partitioned path, not the flat one.

    Both paths resolve the target differently and both must measure it; the
    partitioned one was missed on the first pass, which left the fixture M23's
    floor most needs with no distribution at all.
    """
    frame = pd.DataFrame(
        {"id": ["a", "b", "c"], "md": [1.0, 2.0, 3.0], "Depth": [10.0, 11.0, 12.0]}
    )
    tables = {f"train/w{i}__well.csv": frame for i in range(3)}
    tables["test/w0__well.csv"] = frame[["id", "md"]]
    tables["sample_submission.csv"] = pd.DataFrame({"id": ["w0_1"], "Depth": [0.0]})

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "parts")

    assert profile.partitioned
    assert profile.target_distribution.median is not None


# --- derived, therefore free on legacy profiles ------------------------------


def test_a_legacy_profile_acquires_the_field_on_load() -> None:
    """No migration, because nothing was stored.

    Every `profile.json` written before this milestone gets a `target_type` the
    moment it is read, computed from the columns it already carried.
    """
    legacy = {
        "competition": "old",
        "target_column": "y",
        "row_count": 100,
        "submission_columns": ["Id", "y"],
        "id_columns": ["Id"],
        "columns": [{"name": "y", "dtype": "int64", "unique_count": 2, "is_numeric": True}],
    }

    profile = DatasetProfile.model_validate(legacy)

    assert profile.target_type == "binary"


@pytest.mark.parametrize("field", ["target_type"])
def test_the_derived_field_serializes(field: str) -> None:
    """Readers outside the process see it, which is the point of `computed_field`."""
    import json

    payload = json.loads(_profile([0, 1] * 8).model_dump_json())

    assert payload[field] == "binary"
