"""`labpilot revert` must find the record where M11 now writes it.

`ExperimentSpecialist` writes experiment records under `effective_runs_dir`
(shared across branches), not `workspace.root` (private per branch since
task 7). `revert_command` looks a record up by experiment id to recover its
git commit — if it looked in the old place, every M11-produced experiment
would report "No experiment record for" and revert would be unusable for
exactly the runs this milestone exists to produce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from labpilot.cli.revert import revert_command
from labpilot.research_engine.agents.git_evolution import write_experiment_git_record
from labpilot.workspace import scaffold_workspace


def test_revert_finds_a_record_written_to_the_shared_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    ws_root = Path(client.root)
    runs_dir = ws_root / "runs"
    write_experiment_git_record(
        runs_dir,
        {
            "experiment_id": "exp_titanic_E-1",
            "execution_id": "E-1",
            "git_commit": "abc1234",
            "git_branch": "research/local/E-1",
            "aliases": [],
        },
    )

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "labpilot.cli.revert.revert_to_commit",
        lambda root, commit, **kw: calls.append((root, commit)),
    )

    revert_command("exp_titanic_E-1", workspace_path=ws_root)

    assert calls == [(ws_root, "abc1234")]


def test_revert_reports_a_clear_error_when_the_record_is_genuinely_absent(
    tmp_path: Path,
) -> None:
    import typer

    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    ws_root = Path(client.root)

    with pytest.raises(typer.Exit):
        revert_command("exp_titanic_does-not-exist", workspace_path=ws_root)
