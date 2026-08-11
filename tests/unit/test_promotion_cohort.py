"""The cohort file has to survive K branches landing at once (M11).

Members arrive concurrently, in any order, each from its own worktree. The
verdict is recomputed on every arrival, so it must converge on one answer
however the arrivals interleave — and it must not lose a member to a
read-then-write race with a sibling.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from labpilot.research_engine.agents.git_evolution import write_experiment_git_record
from labpilot.research_engine.agents.promotion import (
    cohort_path,
    promote_within_cohort,
)


def _record(runs_dir: Path, execution_id: str, value: float) -> None:
    write_experiment_git_record(
        runs_dir,
        {
            "experiment_id": f"exp_{execution_id}",
            "execution_id": execution_id,
            "competition": "titanic",
            "status": "succeeded",
            "metrics": {"cv_rmse": value},
            "aliases": [],
        },
    )


def _land(runs_dir: Path, execution_id: str, completed_at: str | None = None) -> dict:
    return promote_within_cohort(
        runs_dir,
        "C-1",
        member_id=execution_id,
        completed_at=completed_at,
        competition="titanic",
        metric_key="cv_rmse",
        maximize=False,
    )


def test_the_best_scoring_branch_is_promoted(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    _record(runs, "E-2", 0.20)
    _record(runs, "E-3", 0.90)

    _land(runs, "E-1")
    _land(runs, "E-2")
    state = _land(runs, "E-3")

    assert state["promoted"] == "E-2"
    assert sorted(state["demoted"]) == ["E-1", "E-3"]


def test_the_verdict_does_not_depend_on_arrival_order(tmp_path: Path) -> None:
    verdicts = set()
    for order in (("E-1", "E-2", "E-3"), ("E-3", "E-1", "E-2"), ("E-2", "E-3", "E-1")):
        runs = tmp_path / f"runs-{'-'.join(order)}"
        _record(runs, "E-1", 0.50)
        _record(runs, "E-2", 0.20)
        _record(runs, "E-3", 0.90)
        for execution_id in order:
            state = _land(runs, execution_id)
        verdicts.add(state["promoted"])

    assert verdicts == {"E-2"}


def test_an_interim_verdict_is_revised_when_a_better_branch_lands(
    tmp_path: Path,
) -> None:
    """Ranking the members present so far is the point — the last arrival's
    computation is the complete one, and nothing waits for a count it has no
    way to know."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    _record(runs, "E-2", 0.20)

    assert _land(runs, "E-1")["promoted"] == "E-1"
    assert _land(runs, "E-2")["promoted"] == "E-2"


def test_a_member_landing_twice_is_recorded_once(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)

    _land(runs, "E-1")
    state = _land(runs, "E-1")

    assert [m["id"] for m in state["members"]] == ["E-1"]
    assert state["promoted"] == "E-1"
    assert state["demoted"] == []


def test_no_member_is_lost_when_eight_branches_land_at_once(tmp_path: Path) -> None:
    """The read-append-write race. Unlocked, a sibling's append is clobbered."""
    runs = tmp_path / "runs"
    ids = [f"E-{i}" for i in range(8)]
    for i, execution_id in enumerate(ids):
        _record(runs, execution_id, 1.0 - i / 100)

    start = threading.Barrier(len(ids))

    def land(execution_id: str) -> None:
        start.wait()
        _land(runs, execution_id)

    threads = [threading.Thread(target=land, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    state = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert sorted(m["id"] for m in state["members"]) == sorted(ids)
    assert state["promoted"] == "E-7"  # lowest cv_rmse


def test_a_member_with_no_record_yet_is_not_ranked(tmp_path: Path) -> None:
    """It is a branch that has not written one, not a branch that scored nothing."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)

    state = _land(runs, "E-1")
    state = _land(runs, "E-missing")

    assert sorted(m["id"] for m in state["members"]) == ["E-1", "E-missing"]
    assert state["promoted"] == "E-1"


def test_without_a_metric_the_members_are_recorded_but_nobody_wins(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)

    state = promote_within_cohort(
        runs,
        "C-1",
        member_id="E-1",
        completed_at=None,
        competition="titanic",
        metric_key=None,
        maximize=True,
    )

    assert [m["id"] for m in state["members"]] == ["E-1"]
    assert "promoted" not in state


def test_completed_at_is_stored_per_member(tmp_path: Path) -> None:
    """Ranking needs every member's timestamp long after its event is gone."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    _record(runs, "E-2", 0.50)

    _land(runs, "E-2", "2026-08-11T10:00:09+00:00")
    state = _land(runs, "E-1", "2026-08-11T10:00:01+00:00")

    stamps = {m["id"]: m["completed_at"] for m in state["members"]}
    assert stamps["E-1"] == "2026-08-11T10:00:01+00:00"
    assert stamps["E-2"] == "2026-08-11T10:00:09+00:00"
    assert state["promoted"] == "E-1"  # tied score, finished first


def test_a_corrupt_cohort_file_does_not_stop_the_next_branch(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    path = cohort_path(runs, "C-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    state = _land(runs, "E-1")

    assert state["promoted"] == "E-1"


@pytest.mark.parametrize("cohort_id", ["../escape", "a/b", "", "  ", "/abs"])
def test_a_cohort_id_cannot_escape_the_cohorts_directory(
    tmp_path: Path, cohort_id: str
) -> None:
    with pytest.raises(ValueError):
        cohort_path(tmp_path / "runs", cohort_id)
