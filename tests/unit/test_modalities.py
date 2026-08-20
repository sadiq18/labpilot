"""A dataset is often more than one thing.

M22 step 5, and goal 6 of the design. The detector used to pick a winner and
throw the rest away: rogii is 1,553 tables *and* 773 well previews, and only the
first half ever reached a profile — the second was counted, used to prefer
tabular, and discarded in the same expression.

The other fixes here are the same shape — an answer stated more confidently than
it was reached: a modality "decided" by the absence of a tie-breaker, a zarr
branch no input could reach, and an environment reported as an error. The two
remaining step-5 flips live with the characterizations they replace, in
`test_dataset_shapes.py`: a sample cap reported as a row count, and an ambiguity
warning pointing at a config field that has never existed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from helpers.dataset_shapes import (
    build_environment,
    build_image_and_text,
    build_strong_signals,
    build_tables_with_previews,
)

from labpilot.accessor.profiler.modality import ModalityDetector
from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig


def _profile(data_dir: Path, llm_client: object | None = None) -> DatasetProfile:
    return TabularProfiler(ProfilerConfig()).profile_directory(
        data_dir, data_dir.name, llm_client=llm_client
    )


class _SaysImage:
    def complete(self, system: str, user: str) -> str:
        return "image"


# --- goal 6: a list, not a winner -------------------------------------------


def test_tables_with_previews_are_both_recorded(tmp_path: Path) -> None:
    """rogii's shape: prefer the tables, and still say the images are there.

    Preferring tabular is right — the well logs carry the signal and the PNGs do
    not. Returning `image_dir=None` while doing it is what made them invisible,
    and "prefer" never meant "pretend they do not exist".
    """
    profile = _profile(build_tables_with_previews(tmp_path))

    assert [(m.modality, m.role) for m in profile.modalities] == [
        ("tabular", "primary"),
        ("image", "auxiliary"),
    ]
    assert profile.modalities[1].image_dir == "previews"
    assert "3 image file(s)" in profile.modalities[1].detail
    # The mirror, so the six modules that read a string keep working.
    assert profile.modality == "tabular"


def test_the_modality_string_is_a_view_over_the_list(tmp_path: Path) -> None:
    """One fact, one place. A stored string could drift from the list."""
    profile = _profile(build_strong_signals(tmp_path))

    assert profile.modality == profile.modalities[0].modality
    assert profile.modalities[0].role == "primary"
    assert profile.confidence_in("modality") > 0.0, "one modality present is evidence"


def test_a_legacy_profile_keeps_the_modality_it_stated() -> None:
    """`modality` is computed now, so a stored value would otherwise be dropped.

    birdclef's profile says `audio` — a modality this profiler has no detector
    for. Losing it would make every analyzer that keys off the field describe an
    audio competition as a tabular one.
    """
    profile = DatasetProfile.model_validate(
        {"competition": "birdclef-2026", "modality": "audio", "image_dir": "audio/train"}
    )

    assert profile.modality == "audio"
    assert profile.modalities[0].role == "primary"
    assert profile.modalities[0].image_dir == "audio/train"


# --- a tie broken by a model is not a tie resolved by the data --------------


def test_a_model_broken_tie_is_capped(tmp_path: Path) -> None:
    """Images and text, and no rule that prefers either.

    Whatever the model answers is worth 0.30 with a 0.50 ceiling, so a modality
    reached this way can be acted on and never asserted.
    """
    profile = _profile(build_image_and_text(tmp_path), llm_client=_SaysImage())
    inference = profile.inferences["modality"]

    assert profile.modality == "image"
    assert [signal.id for signal in inference.signals] == ["llm_modality_tiebreak"]
    assert inference.band == "uncertain"


def test_no_tie_breaker_is_not_a_decision(tmp_path: Path) -> None:
    """The branch production always takes: `_ensure_profile` passes no client.

    It used to return `confidence="high"` — the *absence* of a tie-breaker
    reported as the presence of an answer.
    """
    detector = ModalityDetector()
    data_dir = build_image_and_text(tmp_path)
    profile = _profile(data_dir)

    result = detector.detect(data_dir, profile, llm_client=None)

    assert result.confidence == "ambiguous"
    assert "llm_unavailable" in result.signals
    assert result.tiebroken is False


# --- the zarr branch, reachable for the first time --------------------------


def test_a_zarr_store_is_found_beside_a_submission_file(tmp_path: Path) -> None:
    """Every zarr competition ships a `sample_submission.csv`.

    The CSV preference returned before the branch that looked for a store, so
    the branch could not fire on any real dataset — an unreachable declaration,
    which is the defect class this milestone removes.
    """
    data_dir = tmp_path / "zarr-comp"
    (data_dir / "cube.zarr").mkdir(parents=True)
    (data_dir / "cube.zarr" / "chunk").write_bytes(b"")
    frame = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.0]})
    frame.to_csv(data_dir / "train.csv", index=False)
    frame[["id", "x"]].to_csv(data_dir / "test.csv", index=False)
    frame[["id", "y"]].to_csv(data_dir / "sample_submission.csv", index=False)

    profile = _profile(data_dir)

    assert [m.modality for m in profile.modalities] == ["tabular", "image"]
    assert profile.modalities[1].image_dir == "cube.zarr"
    assert "zarr store" in profile.modalities[1].detail


# --- an environment is a shape, not an error --------------------------------


def test_an_environment_is_described_and_asks(tmp_path: Path) -> None:
    """No tables: name the shape, list the files, and ask what cannot be known.

    `action_space` is deliberately not inferred — no fixture exists and the
    output would be unfalsifiable.
    """
    profile = _profile(build_environment(tmp_path))

    assert profile.modality == "environment"
    assert profile.train_test_relationship == "environment"
    assert profile.confidence_in("train_test_relationship") >= 0.85
    assert profile.prediction_unit == "episode"
    assert profile.files, "the files are what there is to describe"
    assert profile.target_column is None
    assert profile.confidence == 0.0, "a campaign must ask before acting on this"
