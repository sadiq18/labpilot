"""The evidence plane: confidences that are functions of what fired.

M22 step 2. The value plane does not move — the golden snapshots in
`test_dataset_shapes.py` prove that, and their diff for this step is two added
keys and nothing else. What is new is that the profile can now say *why*, and
that a tie between two candidates is visible as a tie instead of being settled
by a sort.

The load-bearing check is `test_every_confidence_is_its_signals`: goal 1 of the
design, enforced over every fixture rather than reviewed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from helpers.dataset_shapes import (
    build_environment,
    build_image_and_text,
    build_no_kaggle_inputs,
    build_partition_suffix,
    build_partitioned_with_template,
    build_partitioned_without_template,
    build_strong_signals,
    build_tables_with_previews,
    build_template_only,
)
from helpers.dataset_sources import DictSource
from pydantic import ValidationError

from labpilot.accessor.profiler import evidence as evidence_module
from labpilot.accessor.profiler.evidence import (
    CATALOGUE,
    Inference,
    Signal,
    band_of,
    combine,
)
from labpilot.accessor.profiler.schema import MetricRef
from labpilot.accessor.profiler.source import DeclaredFacts
from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig

SRC_ROOT = Path(evidence_module.__file__).resolve().parents[3]


class _SaysImage:
    """A tie-breaker that answers, so the capped signal has a producer."""

    def complete(self, system: str, user: str) -> str:
        return "image"


def _profile(data_dir: Path) -> DatasetProfile:
    return TabularProfiler(ProfilerConfig()).profile_directory(data_dir, data_dir.name)


# --- goal 1: no call site writes a float ------------------------------------


def test_every_confidence_is_its_signals(tmp_path: Path) -> None:
    """`confidence == combine(signals)` and `band == band_of(confidence)`.

    Over every inference in every fixture, including the ones inside
    `alternatives`, and asserted non-empty first so it cannot pass by finding
    nothing to check.
    """
    profiles = [
        _profile(build(tmp_path))
        for build in (
            build_strong_signals,
            build_partitioned_with_template,
            build_partitioned_without_template,
        )
    ]
    checked = 0
    for profile in profiles:
        assert profile.inferences, "a profile with no inferences would make this vacuous"
        for inference in profile.inferences.values():
            assert inference.confidence == combine(inference.signals)
            assert inference.band == band_of(inference.confidence)
            checked += 1
            for alternative in inference.alternatives:
                assert alternative.confidence == combine(alternative.signals)
                checked += 1
    assert checked >= 6


def test_a_hand_set_confidence_cannot_survive_the_model() -> None:
    """The rule is enforced on load, not only on construction.

    A profile written by a future version, or edited by hand, is re-checked when
    it is read — which is what makes 'no call site writes a float' a property of
    the data rather than of the code that happened to produce it.
    """
    honest = Inference.of([Signal(id="is_numeric")])

    with pytest.raises(ValidationError, match="is not combine"):
        Inference(signals=honest.signals, confidence=0.99, band="asserted")
    with pytest.raises(ValidationError, match="is not band_of"):
        Inference(signals=honest.signals, confidence=honest.confidence, band="asserted")


def test_an_unknown_signal_is_refused() -> None:
    """A signal with no catalogue entry has no weight, so it cannot be evidence."""
    with pytest.raises(ValidationError, match="unknown signal"):
        Signal(id="looks_like_a_label")


def test_a_capped_signal_stops_capping_once_real_evidence_arrives() -> None:
    """The cap is about deciding alone, not about being worthless.

    `positional_template_overlap` alone cannot exceed 0.50 however it is
    combined; beside a structural signal the cap lifts, because the objection
    was never that position is noise — it is that position should not decide.
    """
    alone = combine([Signal(id="positional_template_overlap")])
    with_structure = combine(
        [Signal(id="positional_template_overlap"), Signal(id="named_in_prediction_template")]
    )

    assert alone <= 0.50
    assert with_structure > 0.50
    assert band_of(alone) == "uncertain"


# --- the worked examples ----------------------------------------------------


def test_the_worked_examples_come_out_of_the_catalogue(tmp_path: Path) -> None:
    """§7.6 of the design, computed rather than asserted.

    If a weight changes, these move — which is the point: the numbers in the
    design are the catalogue's output, not a target it was tuned to hit.
    """
    strong = _profile(build_strong_signals(tmp_path))
    partitioned = _profile(build_partitioned_with_template(tmp_path))

    assert strong.inferences["target_column"].confidence == 0.9592
    assert strong.inferences["target_column"].band == "asserted"
    assert partitioned.inferences["target_column"].confidence == 0.9184
    assert partitioned.inferences["target_column"].band == "asserted"
    # The template is the whole difference: drop it and the same dataset falls
    # to the distributional evidence alone.
    assert partitioned.inferences["target_column"].alternatives[0].confidence == 0.592


def test_a_tie_is_visible_as_a_tie(tmp_path: Path) -> None:
    """Case B′: two candidates, identical evidence, and no asserted answer.

    Today's code still picks `sorted(candidates)[-1]` — the value plane does not
    move in this step. What changes is that the profile now says the winner and
    the loser fired exactly the same signals and scored exactly the same, which
    is what makes the pick visible as a coin flip rather than as an answer.
    """
    profile = _profile(build_partitioned_without_template(tmp_path))
    target = profile.inferences["target_column"]

    assert profile.target_column == "Zone_Depth"
    assert [alternative.candidate for alternative in target.alternatives] == ["Depth"]
    assert target.confidence == target.alternatives[0].confidence == 0.592
    assert target.band == "uncertain"
    assert [signal.id for signal in target.signals] == [
        signal.id for signal in target.alternatives[0].signals
    ]


# --- notes, and the prose view over them ------------------------------------


def test_warnings_is_the_prose_view_of_notes(tmp_path: Path) -> None:
    """Same strings, same order, one definition."""
    profile = _profile(build_partitioned_with_template(tmp_path))

    assert profile.notes, "the partitioned path always records something"
    assert profile.warnings == [note.text for note in profile.notes]
    assert {note.code for note in profile.notes} >= {"partitioned_layout", "rows_not_iid"}


def test_a_legacy_profile_keeps_its_warnings() -> None:
    """A pre-M22 `profile.json` still reads, and does not lose its prose.

    `warnings` used to be stored, so every profile on disk carries one; now that
    the name is computed, pydantic would drop the incoming value. That would
    take the anchor-column advice — the line telling codegen not to fit a target
    from a column identical to it — out of every workspace serving a stale
    profile.
    """
    legacy = {
        "competition": "rogii",
        "warnings": ["'TVT_input' is the known prefix of 'TVT'", "rows are NOT iid"],
    }

    profile = DatasetProfile.model_validate(legacy)

    assert profile.warnings == legacy["warnings"]
    assert [note.code for note in profile.notes] == ["legacy", "legacy"]


def test_nothing_writes_warnings_directly() -> None:
    """`warnings` is computed, so appending to it silently does nothing.

    That is the dangerous shape: `profile.warnings.append(...)` still *runs*, on
    a fresh list, and the note vanishes. A structural check is the only guard
    that catches it, since there is no exception to catch.
    """
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "extend"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "warnings"
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")

    assert not offenders, f"append a Note instead; `warnings` is a derived view: {offenders}"


def test_a_note_recorded_before_the_partition_block_survives() -> None:
    """The partitioned path used to *assign* `warnings`, discarding what came before.

    `_detect_suffix_scoring` runs one line earlier and records why it skipped;
    the assignment then threw that away in the same breath. Only a source with
    no directory reaches the skip, which is why no fixture caught it.
    """
    frame = pd.DataFrame(
        {"id": ["a", "b", "c"], "md": [1.0, 2.0, 3.0], "Depth": [10.0, 11.0, 12.0]}
    )
    tables = {f"train/w{i}__well.csv": frame for i in range(3)}
    tables["test/w0__well.csv"] = frame[["id", "md"]]
    tables["sample_submission.csv"] = pd.DataFrame({"id": ["w0_1"], "Depth": [0.0]})

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(DictSource(tables), "in-memory")

    assert profile.partitioned
    assert any(note.code == "suffix_scoring_not_detected" for note in profile.notes)
    assert any("suffix scoring not detected" in warning for warning in profile.warnings)


def test_the_catalogue_carries_no_entry_nothing_can_fire(tmp_path: Path) -> None:
    """Every signal in the catalogue is produced by some path.

    A weight nothing can reach is a declaration nothing reaches — the shape this
    milestone exists to remove — and it is easiest to add one by writing the
    catalogue ahead of the code.
    """
    profiles = [
        _profile(build(tmp_path))
        for build in (
            build_strong_signals,
            build_partitioned_with_template,
            build_partitioned_without_template,
            build_no_kaggle_inputs,
            build_partition_suffix,
            build_template_only,
            build_tables_with_previews,
            build_environment,
        )
    ]
    # A modality tie a model breaks: images *and* a text column, so the
    # rule-based detector cannot choose and asks. The only producer of
    # `llm_modality_tiebreak`.
    profiles.append(
        TabularProfiler(ProfilerConfig()).profile_directory(
            build_image_and_text(tmp_path), "tiebreak", llm_client=_SaysImage()
        )
    )
    frame = pd.DataFrame({"Id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    tables = {
        "train.csv": frame,
        "test.csv": frame[["Id", "x"]],
        "sample_submission.csv": frame[["Id", "y"]],
    }
    profiles.append(
        TabularProfiler(ProfilerConfig()).profile_dataset(
            DictSource(
                tables,
                DeclaredFacts(metric=MetricRef(name="RMSE", key="rmse", direction="minimize")),
            ),
            "declared",
        )
    )
    # And a dataset someone has answered, which is the only way `operator_answer`
    # ever fires.
    profiles.append(
        TabularProfiler(ProfilerConfig()).profile_dataset(
            DictSource(tables, DeclaredFacts(answers={"target_column": "y"})),
            "answered",
        )
    )

    fired: set[str] = set()
    for profile in profiles:
        for inference in profile.inferences.values():
            fired |= {signal.id for signal in inference.signals}
            for alternative in inference.alternatives:
                fired |= {signal.id for signal in alternative.signals}

    unreachable = set(CATALOGUE) - fired
    assert not unreachable, f"catalogue entries nothing fires: {sorted(unreachable)}"
