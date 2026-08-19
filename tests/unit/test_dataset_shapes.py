"""What today's profiler says about each M22 dataset shape.

M22 step 0. Every test here asserts **current** behaviour, including the wrong
parts, and names the step that will flip it. Two reasons for writing them before
any fix:

* A fixture that cannot express a defect proves nothing about the fix that
  follows. These tests are how the shapes in `helpers/dataset_shapes.py` earn
  their place — each one *is* the defect, reproduced.
* The golden snapshots pin the profile byte for byte, so step 1 (routing the
  profiler through a source protocol) can be shown to change nothing, rather
  than asserted to.

Plan: `docs/research-os/autonomy-roadmap/17-dataset-understanding.md` ·
Design: `docs/research-os/autonomy-roadmap/design/17-dataset-understanding.md`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from helpers.dataset_shapes import (
    SAMPLED_BEYOND_CAP_ROWS,
    build_bool_target,
    build_partitioned_with_template,
    build_partitioned_without_template,
    build_sampled_beyond_cap,
    build_strong_signals,
)

from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "dataset_shapes"

#: The cap used by the sampling fixture. Small on purpose: the defect is that a
#: bound cap is reported as an exact count, which is true at any cap, and paying
#: 690,088 rows to show it would cost seconds per run.
SAMPLE_CAP = 10

#: shape -> (builder, config). Keyed by builder rather than by fixture so both
#: checks below can build the *same* shape twice in different directories, which
#: a `tmp_path` fixture cannot do. Shapes that raise are exercised separately.
PROFILEABLE = {
    "strong_signals": (build_strong_signals, ProfilerConfig()),
    "partitioned_with_template": (build_partitioned_with_template, ProfilerConfig()),
    "partitioned_without_template": (build_partitioned_without_template, ProfilerConfig()),
    "bool_target": (build_bool_target, ProfilerConfig()),
    "sampled_beyond_cap": (build_sampled_beyond_cap, ProfilerConfig(max_rows_sample=SAMPLE_CAP)),
}


def _profile(data_dir: Path, config: ProfilerConfig | None = None) -> DatasetProfile:
    return TabularProfiler(config or ProfilerConfig()).profile_directory(data_dir, data_dir.name)


def _normalized(profile: DatasetProfile) -> dict:
    """The profile as JSON, minus what is a fact about the library rather than
    about the dataset.

    `dtype` is dropped because pandas has changed how it spells "a plain string
    column" between versions (`ColumnProfile.dtype` says so), and a golden that
    pins it fails on an upgrade for a reason that has nothing to do with this
    code. `is_numeric` covers the part that carries meaning. Floats in `stats`
    are rounded because their last bits are a property of the BLAS underneath.
    """
    data = json.loads(profile.model_dump_json())
    for column in data.get("columns", []):
        column.pop("dtype", None)
        column["stats"] = {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in (column.get("stats") or {}).items()
        }
    return data


# --- golden snapshots -------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(PROFILEABLE))
def test_profile_matches_its_golden_snapshot(tmp_path: Path, shape: str) -> None:
    """Pin every profileable shape, so step 1 must change nothing.

    Delete a golden file to regenerate it; the diff is then the review.
    """
    builder, config = PROFILEABLE[shape]
    actual = _normalized(_profile(builder(tmp_path), config))

    # A golden comparison passes trivially against an empty description, which
    # is exactly what a broken profiler produces.
    assert actual["columns"], "profile has no columns; the snapshot would be vacuous"

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden = GOLDEN_DIR / f"{shape}.golden.json"
    if not golden.is_file():
        golden.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    assert actual == json.loads(golden.read_text(encoding="utf-8"))


@pytest.mark.parametrize("shape", sorted(PROFILEABLE))
def test_profiling_does_not_depend_on_where_it_ran(tmp_path: Path, shape: str) -> None:
    """Requirement 4 (determinism), and the half a golden cannot see.

    A snapshot compares one run against a file; this compares two runs against
    each other from different directories. `test_capstone`'s renderer bug was
    invisible to a comparison that shared a directory.

    Over every shape, not just the flat one: the partitioned path is where the
    path-derived values live — `files[:200]` ordering, `sampled[:limit]`, and a
    `primary_kind` picked by `max()` over a dict whose insertion order comes
    from `rglob`.
    """
    builder, config = PROFILEABLE[shape]
    first = _normalized(_profile(builder(tmp_path / "one"), config))
    second = _normalized(_profile(builder(tmp_path / "two"), config))

    assert first["columns"], "an empty profile would make this comparison vacuous"
    # `competition` is the directory name, and both runs build the same shape,
    # so it is the same string; everything else describes the bytes.
    assert first == second


# --- what the profile cannot say --------------------------------------------


def test_a_correct_answer_now_says_why(strong_signals_data_dir: Path) -> None:
    """Case A, flipped at step 2: right, and able to say why.

    Every strong signal fires here — the template names the label, it is the
    only column withheld from the scoring input, it is complete and numeric —
    and each one is now named in the profile rather than implied by a value
    appearing out of nowhere.
    """
    profile = _profile(strong_signals_data_dir)
    target = profile.inferences["target_column"]

    assert profile.target_column == "SalePrice"
    assert profile.id_column == "Id"
    assert [signal.id for signal in target.signals] == [
        "named_in_prediction_template",
        "sole_withheld_column",
        "non_null_in_train",
        "is_numeric",
    ]
    assert target.band == "asserted"
    assert profile.confidence_in("target_column") >= 0.85
    # One candidate, so nothing to be an alternative to.
    assert target.alternatives == []
    assert profile.warnings == ["default_tabular"]


def test_the_decoy_wins_when_no_template_names_the_label(
    partitioned_without_template_data_dir: Path,
) -> None:
    """Case B′: the answer is decided by a sort. Flips at step 3.

    `Depth` and `Zone_Depth` are withheld at test, complete, numeric, and in
    every partition of the primary kind. Nothing in the profile separates them —
    asserted below rather than claimed — and the tie is broken by taking the
    last of the sorted candidates.
    """
    profile = _profile(partitioned_without_template_data_dir)
    by_name = {column.name: column for column in profile.columns}

    assert set(profile.train_only_columns) == {"Depth", "Zone_Depth"}
    # Indistinguishable on every fact the profile records about them.
    assert by_name["Depth"].is_numeric == by_name["Zone_Depth"].is_numeric is True
    assert by_name["Depth"].null_pct == by_name["Zone_Depth"].null_pct
    assert by_name["Depth"].unique_count == by_name["Zone_Depth"].unique_count
    # And the winner is the last one alphabetically, not the better-evidenced one.
    assert profile.target_column == sorted(profile.train_only_columns)[-1] == "Zone_Depth"


def test_the_only_advertised_escape_does_not_exist(
    partitioned_without_template_data_dir: Path,
) -> None:
    """The ambiguity warning names a config field that was never built.

    Flips at step 4, where the escape becomes a question with an answer file.
    """
    profile = _profile(partitioned_without_template_data_dir)
    advice = [w for w in profile.warnings if "Target inference is ambiguous" in w]

    assert advice, "the ambiguous fixture produced no ambiguity warning"
    assert "`target_column` in the competition config" in advice[0]
    assert "target_column" not in CompetitionSpec.model_fields


def test_one_file_decides_the_label(
    partitioned_with_template_data_dir: Path,
    partitioned_without_template_data_dir: Path,
) -> None:
    """The two partitioned shapes differ by exactly one file, and disagree.

    Mutating the *input* rather than the code is what shows the template is
    load-bearing: same tables, same columns, same values, one file apart.
    """
    with_names = {p.name for p in partitioned_with_template_data_dir.rglob("*.csv")}
    without_names = {p.name for p in partitioned_without_template_data_dir.rglob("*.csv")}

    assert with_names - without_names == {"sample_submission.csv"}
    assert not without_names - with_names

    assert _profile(partitioned_with_template_data_dir).target_column == "Depth"
    assert _profile(partitioned_without_template_data_dir).target_column == "Zone_Depth"


def test_a_dataset_without_kaggle_inputs_is_profiled(no_kaggle_inputs_data_dir: Path) -> None:
    """Case C, flipped at step 3: the M12 shape produces a schema.

    One table, no split, no template, no declared metric. Every structural
    signal is unavailable by construction, so the answers that depend on them
    are `uncertain` — and the description a model needs to do anything at all,
    the columns and their statistics, is there regardless.

    It used to raise `ValueError: Expected one training CSV, found 0`, which is
    the profiler's answer to the entire world outside Kaggle.
    """
    profile = _profile(no_kaggle_inputs_data_dir)

    assert profile.train_test_relationship == "no_test_provided"
    assert profile.confidence_in("train_test_relationship") >= 0.85
    assert profile.feature_columns == [
        "event_id",
        "account",
        "amount",
        "occurred_at",
        "churned",
    ]
    # Uncertain on exactly the three answers that need something declared.
    uncertain = {f for f, i in profile.inferences.items() if i.band == "uncertain"}
    assert uncertain == {"target_column", "id_column", "metric"}
    assert any(note.code == "no_target_identified" for note in profile.notes)


def test_an_environment_has_no_description_at_all(environment_data_dir: Path) -> None:
    """No CSVs, so the profiler refuses. Flips at step 5.

    The workspace layer catches this and writes a filesystem inventory whose
    modality and null target have nothing behind either
    (`workspace/capability.py:503-580`).
    """
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        _profile(environment_data_dir)


def test_the_sample_cap_is_reported_as_an_exact_row_count(
    sampled_beyond_cap_data_dir: Path,
) -> None:
    """The row count is the cap, and the profile says it is exact.

    Flips at step 5. On disk today: `playground-series-s6e7/profile.json` says
    100,000 rows, unstamped, for a file of 690,088.
    """
    real_rows = len(pd.read_csv(sampled_beyond_cap_data_dir / "train.csv"))
    profile = _profile(sampled_beyond_cap_data_dir, ProfilerConfig(max_rows_sample=SAMPLE_CAP))

    # Both halves: what the builder says it wrote, and what is on disk. The
    # constant exists to express this relationship, so it has to be read here or
    # it is a declaration nothing reaches.
    assert SAMPLED_BEYOND_CAP_ROWS > SAMPLE_CAP
    assert real_rows == SAMPLED_BEYOND_CAP_ROWS > SAMPLE_CAP, (
        "the fixture must exceed the cap or it proves nothing"
    )
    assert profile.row_count == SAMPLE_CAP
    assert profile.row_count_estimated is False


def test_a_boolean_label_is_not_the_numeric_column(bool_target_data_dir: Path) -> None:
    """spaceship-titanic's shape: the label is the one column that is not numeric.

    A guard for later steps — "the target is numeric" is worth 0.15 in the
    catalogue precisely because it is wrong here.
    """
    profile = _profile(bool_target_data_dir)
    by_name = {column.name: column for column in profile.columns}

    assert profile.target_column == "Transported"
    assert by_name["Transported"].is_numeric is False
    assert by_name["Age"].is_numeric is True
