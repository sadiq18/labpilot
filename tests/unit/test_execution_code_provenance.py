"""M19 §6 — each execution's source is addressable by that execution's id.

The property under test is not "a file got copied". It is that a *branching*
experiment graph can name the code its parent ran, which the pre-existing
``artifacts/code_backups/train_<id>.py`` cannot do: that file is keyed by the
execution that overwrote it, so reading it back answers a different question.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.execution.delta import (
    TRAIN_RELPATH,
    execution_source,
    record_execution_source,
    snapshot_dir,
)


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_an_execution_can_be_asked_what_code_it_ran(tmp_path):
    train = _write(tmp_path, TRAIN_RELPATH, "print('E-001')\n")
    record_execution_source(tmp_path, "E-001", [train])

    assert execution_source(tmp_path, "E-001") == "print('E-001')\n"


def test_the_snapshot_survives_the_working_copy_being_overwritten(tmp_path):
    """The whole point: `pipeline/train.py` is a working copy, not a record."""
    train = _write(tmp_path, TRAIN_RELPATH, "print('E-001')\n")
    record_execution_source(tmp_path, "E-001", [train])

    _write(tmp_path, TRAIN_RELPATH, "print('E-002')\n")

    assert execution_source(tmp_path, "E-001") == "print('E-001')\n"


def test_two_children_of_one_parent_each_resolve_to_their_own_code(tmp_path):
    """The failure §6 names: the graph branches, the filesystem does not.

    H-1 and H-2 both fork from the baseline, so "the previous file" is
    ambiguous for at least one of them. Keyed by execution id, neither is.
    """
    parent = _write(tmp_path, TRAIN_RELPATH, "print('baseline')\n")
    record_execution_source(tmp_path, "E-BASE", [parent])

    child_a = _write(tmp_path, TRAIN_RELPATH, "print('branch A')\n")
    record_execution_source(tmp_path, "E-A", [child_a])

    child_b = _write(tmp_path, TRAIN_RELPATH, "print('branch B')\n")
    record_execution_source(tmp_path, "E-B", [child_b])

    # Both children still resolve the same parent, and each other's code has
    # not bled across. Write order cannot answer this; the key can.
    assert execution_source(tmp_path, "E-BASE") == "print('baseline')\n"
    assert execution_source(tmp_path, "E-A") == "print('branch A')\n"
    assert execution_source(tmp_path, "E-B") == "print('branch B')\n"


def test_an_unrecorded_execution_reads_as_unknown_not_as_empty(tmp_path):
    """None and "" are different claims.

    An empty parent means a baseline — nothing to preserve, no verdict owed. An
    *unknown* parent means the check never ran. Collapsing them is the
    fabricated-pass failure the delta card already keeps separate via
    `delta_unchecked`.
    """
    assert execution_source(tmp_path, "E-NEVER-RAN") is None


def test_a_retry_records_the_attempt_that_landed(tmp_path):
    """A snapshot claims "this is what ran", so the last applied attempt wins."""
    first = _write(tmp_path, TRAIN_RELPATH, "print('attempt 1')\n")
    record_execution_source(tmp_path, "E-001", [first])
    second = _write(tmp_path, TRAIN_RELPATH, "print('attempt 2')\n")
    record_execution_source(tmp_path, "E-001", [second])

    assert execution_source(tmp_path, "E-001") == "print('attempt 2')\n"


def test_layout_is_mirrored_so_a_multi_file_proposal_reads_back_by_key(tmp_path):
    """`CodeProposal.files` is a list — a delta may touch more than train.py."""
    train = _write(tmp_path, TRAIN_RELPATH, "import helpers\n")
    helper = _write(tmp_path, "pipeline/helpers.py", "def f():\n    return 1\n")
    record_execution_source(tmp_path, "E-001", [train, helper])

    assert execution_source(tmp_path, "E-001", "pipeline/helpers.py") == (
        "def f():\n    return 1\n"
    )
    assert (snapshot_dir(tmp_path, "E-001") / "pipeline" / "train.py").is_file()


def test_a_file_outside_the_workspace_is_not_recorded(tmp_path):
    """Snapshot paths are workspace-relative; an outside path has no key."""
    outside = tmp_path.parent / "elsewhere.py"
    outside.write_text("print('not ours')\n", encoding="utf-8")
    try:
        assert record_execution_source(tmp_path, "E-001", [outside]) == []
    finally:
        outside.unlink()


def test_no_execution_id_records_nothing(tmp_path):
    """A snapshot nobody can address costs disk and answers no question."""
    train = _write(tmp_path, TRAIN_RELPATH, "print('x')\n")
    assert record_execution_source(tmp_path, "", [train]) == []


def test_an_unreadable_source_does_not_break_the_run(tmp_path):
    """Provenance is metadata. Losing it must not cost a working training run."""
    train = _write(tmp_path, TRAIN_RELPATH, "print('x')\n")
    missing = tmp_path / "pipeline" / "gone.py"
    created = record_execution_source(tmp_path, "E-001", [missing, train])

    assert [p.name for p in created] == ["train.py"]
