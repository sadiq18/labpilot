"""The experience-memory hook must look up disk records where M11 writes
them (`runs_dir`, shared), not where the code lives (`workspace_root`,
private per branch since task 7) — the two diverge exactly when a payload's
own fields are incomplete and the disk-record fallback is what fills them.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.agents.git_evolution import write_experiment_git_record
from labpilot.research_engine.memory.hooks import persist_experience_from_completion
from labpilot.workspace import scaffold_workspace


def test_the_disk_fallback_finds_a_record_under_runs_dir_not_workspace_root(
    tmp_path: Path,
) -> None:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    ws_root = Path(client.root)
    runs_dir = ws_root / "runs"  # what ExperimentSpecialist now writes to
    write_experiment_git_record(
        runs_dir,
        {
            "experiment_id": "exp-runsdir-1",
            "execution_id": "exec-runsdir-1",
            "git_commit": "deadbeef",
            "status": "completed",
            "metrics": {"rmse": 0.5},
            "files_changed": ["pipeline/train.py"],
            "aliases": [],
        },
    )

    # A payload missing the fields only the disk record can supply, with
    # both workspace_root (code, private) and runs_dir (record, shared) set
    # the way ExperimentSpecialist's real event payload sets them.
    payload = {
        "competition": "titanic",
        "knowledge_dir": str(client.knowledge_dir),
        "workspace_root": str(ws_root),
        "runs_dir": str(runs_dir),
        "experiment_id": "exp-runsdir-1",
        "execution_id": "exec-runsdir-1",
        "plan_id": "P-001",
        "description": "baseline",
        # git_commit deliberately absent — only the disk record has it.
    }

    record = persist_experience_from_completion(payload)

    assert record is not None
    assert record.artifacts.git_commit == "deadbeef"


def test_without_runs_dir_the_disk_fallback_misses_it(tmp_path: Path) -> None:
    """Confirms the fallback genuinely depends on `runs_dir`, not luck: the
    same record, looked up with only `workspace_root` (pre-M11 shape), is not
    found — `workspace_root` and `runs_dir` point at different directories.
    """
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    ws_root = Path(client.root)
    runs_dir = ws_root / "runs"
    write_experiment_git_record(
        runs_dir,
        {
            "experiment_id": "exp-runsdir-2",
            "execution_id": "exec-runsdir-2",
            "git_commit": "deadbeef",
            "aliases": [],
        },
    )

    payload = {
        "competition": "titanic",
        "knowledge_dir": str(client.knowledge_dir),
        "workspace_root": str(ws_root),
        # no "runs_dir" key at all
        "experiment_id": "exp-runsdir-2",
        "execution_id": "exec-runsdir-2",
        "plan_id": "P-001",
        "description": "baseline",
    }

    record = persist_experience_from_completion(payload)

    assert record is not None
    assert record.artifacts.git_commit is None
