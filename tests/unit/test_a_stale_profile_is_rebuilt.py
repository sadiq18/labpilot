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


def _demo_dataset(tmp_path: Path) -> None:
    import pandas as pd

    pd.DataFrame({"id": [1, 2], "f": [1.0, 2.0], "target": [0, 1]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"id": [3], "f": [3.0]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": [3], "target": [0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )


def test_the_profiler_stamps_what_it_writes(tmp_path: Path) -> None:
    """Both halves must agree, or every profile rebuilds on every run.

    Through `write_profile`, which is the only thing that writes `profile.json`
    in production. This hand-rolled `json.loads(profile.model_dump_json())`
    instead, so it asserted a stamp the writer never had to apply: deleting the
    field from the writer's output left this green.
    """
    from labpilot.accessor.profiler.report import write_profile
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    _demo_dataset(tmp_path)
    profile = TabularProfiler(ProfilerConfig()).profile_directory(tmp_path, "demo")

    written, _ = write_profile(tmp_path, profile)

    assert json.loads(written.read_text())["schema_version"] == PROFILE_SCHEMA_VERSION
    assert _profile_is_current(written) is True


def test_a_profile_read_back_through_the_model_knows_it_is_stale(tmp_path: Path) -> None:
    """The stamp has to survive `DatasetProfile`, not just `json.loads`.

    `schema_version` defaulted to `PROFILE_SCHEMA_VERSION`, so an unstamped
    2026-08-02 profile validated as current and the two typed readers —
    `load_profile`, which feeds planning, and `model_validate_json` in
    `code_engineering`, which feeds baseline selection — had no way to tell it
    from today's. Only `_profile_is_current` escaped, by reading the raw dict.
    """
    from labpilot.accessor.profiler.report import load_profile
    from labpilot.accessor.profiler.tabular import DatasetProfile

    legacy = DatasetProfile.model_validate({"competition": "rogii", "target_column": "TVT"})
    assert legacy.schema_version == 0

    _profile(tmp_path, {"competition": "rogii", "target_column": "TVT"})
    assert load_profile(tmp_path).schema_version == 0


def _ensure_profile(root: Path, **constraints: object) -> tuple[object, dict, list[str]]:
    """Drive the real `_ensure_profile`, returning (result, metadata, checks)."""
    from types import SimpleNamespace

    from labpilot.research_engine.execution.capabilities.workspace.capability import (
        WorkspaceCapability,
    )

    context = SimpleNamespace(competition="demo", constraints=dict(constraints))
    metadata: dict = {}
    checks: list[str] = []
    result = WorkspaceCapability()._ensure_profile(context, root, metadata, [], checks)
    return result, metadata, checks


def test_the_reuse_gate_actually_consults_the_stamp(tmp_path: Path) -> None:
    """Driven through `_ensure_profile`, the gate's only production caller.

    Every other test here calls the predicate directly, so deleting
    `and _profile_is_current(profile_path)` — the whole point of the change —
    left the entire suite green.
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    _demo_dataset(raw)
    stale = _profile(tmp_path, {"competition": "demo", "target_column": "STALE"})

    result, metadata, checks = _ensure_profile(tmp_path)

    assert result is True
    assert "profile_reused" not in metadata, "a stale profile must not be reused"
    assert "profile_written" in checks
    rebuilt = json.loads(stale.read_text())
    assert rebuilt["target_column"] == "target"
    assert rebuilt["schema_version"] == PROFILE_SCHEMA_VERSION


def test_a_profile_that_cannot_be_refreshed_is_kept_not_replaced(tmp_path: Path) -> None:
    """A failed re-profile must not cost the description already in hand.

    Every pre-existing profile is unstamped, so all of them now take the rebuild
    path — where `_write_inventory_profile` used to overwrite `profile.json` in
    place with a bare file listing, return True, and pass the step. The listing
    was written stamped, so `_profile_is_current` then accepted it forever.
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    # *Two* files that match no train/test/submission pattern: the tabular
    # profiler raises, which is the documented reason the inventory path exists.
    # One unmatched file no longer does — M22 step 3 reads a lone table as the
    # training table, because a dataset that is not a competition has no `train`
    # prefix to match. Two is a genuine ambiguity nobody can resolve from here.
    (raw / "well_logs.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (raw / "core_samples.csv").write_text("a,b\n3,4\n", encoding="utf-8")
    kept = _profile(tmp_path, {"competition": "demo", "target_column": "DTC", "columns": [1, 2]})

    result, metadata, checks = _ensure_profile(tmp_path)

    assert result is True
    assert metadata["profile_stale"] == "reprofile_failed"
    assert "profile_stale" in checks
    assert json.loads(kept.read_text())["target_column"] == "DTC"


def test_a_profile_that_cannot_be_rebuilt_is_named_stale_not_absent(tmp_path: Path) -> None:
    """With no data to re-derive from, saying "no profile" changed only the books.

    `write_code` gates on `profile_path.is_file()` alone, so returning None left
    the stale description reaching codegen exactly as before while the evidence
    card stopped naming it. Reachable via skip_download, dry_run, or a resumed
    workspace whose raw tree was reclaimed.
    """
    kept = _profile(tmp_path, {"competition": "demo", "target_column": "DTC"})

    result, metadata, checks = _ensure_profile(tmp_path, skip_download=True)

    assert result is True
    assert metadata["profile_stale"] == "no_data"
    assert metadata["profile"] == str(kept)
    assert "profile_stale" in checks


def test_an_absent_profile_with_no_data_still_reports_nothing(tmp_path: Path) -> None:
    """The behaviour the two carve-outs must not swallow."""
    result, metadata, _ = _ensure_profile(tmp_path)

    assert result is None
    assert metadata["profile_skipped"] == "no_data"


def test_a_profile_nothing_can_parse_is_replaced_not_kept(tmp_path: Path) -> None:
    """"Stale" and "corrupt" are different answers, and only one is worth keeping.

    `stale = profile_path.is_file()` could not tell them apart, so a
    `profile.json` truncated by a crash took the keep-it branch: the step
    reported success with `metadata["profile"]` pointing at bytes no reader can
    use, and the inventory write that would have put a valid file there was
    skipped.
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    # Two unmatched files, so the tabular profiler still raises: see the note in
    # `test_a_profile_that_cannot_be_refreshed_is_kept_not_replaced`.
    (raw / "well_logs.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (raw / "core_samples.csv").write_text("a,b\n3,4\n", encoding="utf-8")
    corrupt = tmp_path / "profile.json"
    corrupt.write_text('{"competition": "demo", "target_col', encoding="utf-8")

    result, metadata, checks = _ensure_profile(tmp_path)

    assert result is True
    assert "profile_stale" not in metadata, "a corrupt profile is not worth keeping"
    assert "profile_inventory" in checks
    json.loads(corrupt.read_text())  # parses now


def test_the_three_profile_states_are_distinguished(tmp_path: Path) -> None:
    from labpilot.research_engine.execution.capabilities.workspace.capability import (
        _profile_state,
    )

    assert _profile_state(_profile(tmp_path, {"schema_version": PROFILE_SCHEMA_VERSION})) == "current"
    assert _profile_state(_profile(tmp_path, {"competition": "demo"})) == "stale"
    bad = tmp_path / "profile.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert _profile_state(bad) == "unusable"
