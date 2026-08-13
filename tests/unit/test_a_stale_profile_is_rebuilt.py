"""A workspace must not keep the description it was first given.

`prepare_workspace` reuses an existing `profile.json` rather than paying to
re-profile every partition. Reuse is right; reuse *forever* is not. rogii's
profile was written 2026-08-02 and served every campaign since, so the partition
warnings and the anchor column added later never reached the codegen that needed
them — the profiler improved six times and the workspace consuming it did not
change once.

This is M20 criterion 4 in the wild: a derived artifact nothing re-derives. The
stamp is the version of the profiler that wrote it, and a mismatch rebuilds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labpilot.accessor.profiler.tabular import PROFILE_SCHEMA_VERSION
from labpilot.research_engine.execution.capabilities.workspace.capability import (
    _profile_is_current,
)


def _profile(tmp_path: Path, body: object) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_profile_from_this_profiler_is_reused(tmp_path: Path) -> None:
    """The behaviour being preserved: profiling every partition is not free."""
    path = _profile(tmp_path, {"competition": "demo", "schema_version": PROFILE_SCHEMA_VERSION})

    assert _profile_is_current(path) is True


def test_a_profile_from_an_older_profiler_is_rebuilt(tmp_path: Path) -> None:
    path = _profile(tmp_path, {"competition": "demo", "schema_version": PROFILE_SCHEMA_VERSION - 1})

    assert _profile_is_current(path) is False


def test_an_unstamped_profile_is_rebuilt(tmp_path: Path) -> None:
    """Every profile written before this change, rogii's included."""
    path = _profile(tmp_path, {"competition": "demo", "target_column": "TVT"})

    assert _profile_is_current(path) is False


@pytest.mark.parametrize("body", ["{ not json", '"a string"', "[1, 2]"])
def test_a_profile_that_cannot_be_read_is_rebuilt(tmp_path: Path, body: str) -> None:
    """Cheaper to re-derive than to run on a description nothing can vouch for."""
    path = tmp_path / "profile.json"
    path.write_text(body, encoding="utf-8")

    assert _profile_is_current(path) is False


def test_a_missing_profile_is_not_current(tmp_path: Path) -> None:
    assert _profile_is_current(tmp_path / "nothing.json") is False


def test_the_profiler_stamps_what_it_writes(tmp_path: Path) -> None:
    """Both halves must agree, or every profile rebuilds on every run."""
    import pandas as pd

    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    pd.DataFrame({"id": [1, 2], "f": [1.0, 2.0], "target": [0, 1]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"id": [3], "f": [3.0]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": [3], "target": [0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    profile = TabularProfiler(ProfilerConfig()).profile_directory(tmp_path, "demo")

    written = _profile(tmp_path, json.loads(profile.model_dump_json()))
    assert _profile_is_current(written) is True
