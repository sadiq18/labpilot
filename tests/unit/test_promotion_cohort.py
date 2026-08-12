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


def test_a_candidate_looks_the_same_however_many_times_it_is_ranked(
    tmp_path: Path,
) -> None:
    """Ranking reads `experiment_id` to break ties, so a candidate must carry
    the record's own id every time. Caching a member's metrics into the cohort
    file and rebuilding a stand-in record around them supplied the execution
    id instead, making the tie-break depend on how often the cohort had been
    re-ranked.
    """
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.5)
    _record(runs, "E-2", 0.5)  # a genuine tie

    seen = set()
    for _ in range(3):
        _land(runs, "E-1")
        _land(runs, "E-2")
        state = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
        seen.add(state["promoted"])

    assert len(seen) == 1
    # And the cohort file stays a membership record, not a copy of the records.
    assert all(set(m) == {"id", "completed_at"} for m in state["members"])


def test_a_later_arrival_that_resolves_no_metric_still_gets_ranked(
    tmp_path: Path,
) -> None:
    """The cohort already agreed on a key; one branch's unreadable outcome
    file must not leave a better result sitting unpromoted."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    _record(runs, "E-2", 0.20)

    _land(runs, "E-1")
    state = promote_within_cohort(
        runs,
        "C-1",
        member_id="E-2",
        completed_at=None,
        competition="titanic",
        metric_key=None,  # this event could not resolve one
        maximize=True,  # and would have flipped the direction too
    )

    assert state["promoted"] == "E-2"
    assert state["metric_key"] == "cv_rmse"
    assert state["maximize"] is False


def test_a_member_that_was_never_ranked_is_not_reported_as_demoted(
    tmp_path: Path,
) -> None:
    """`demoted` means "compared and lost". A branch with no record yet has
    not lost anything."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)

    _land(runs, "E-1")
    state = _land(runs, "E-no-record-yet")

    assert state["promoted"] == "E-1"
    assert state["demoted"] == []
    assert [m["id"] for m in state["members"]] == ["E-1", "E-no-record-yet"]


def test_a_placeholder_branch_is_recorded_but_never_promoted(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_experiment_git_record(
        runs,
        {
            "experiment_id": "exp_E-stub",
            "execution_id": "E-stub",
            "metrics": {"status": "dry_run_stub", "cv_rmse": 0.5},
            "aliases": [],
        },
    )
    _record(runs, "E-real", 194.80)

    _land(runs, "E-stub")
    state = _land(runs, "E-real")

    assert state["promoted"] == "E-real"
    assert state["demoted"] == []  # the stub never entered the comparison
    assert sorted(m["id"] for m in state["members"]) == ["E-real", "E-stub"]


def test_a_cohort_file_whose_members_are_the_wrong_shape_recovers(
    tmp_path: Path,
) -> None:
    """Parses as JSON, wrong shape. Reading it straight raises inside the
    subscriber's own guard, which would silently retire the cohort."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    path = cohort_path(runs, "C-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"members": "not-a-list"}', encoding="utf-8")

    state = _land(runs, "E-1")

    assert state["promoted"] == "E-1"
    assert [m["id"] for m in state["members"]] == ["E-1"]


def test_malformed_member_entries_are_dropped_not_fatal(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    path = cohort_path(runs, "C-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"members": ["bare-string", {}, {"id": "E-9"}]}', encoding="utf-8")

    state = _land(runs, "E-1")

    assert [m["id"] for m in state["members"]] == ["E-9", "E-1"]


@pytest.mark.parametrize(
    "stored_key", ['{"not": "a string"}', '["cv_rmse"]', "42"]
)
def test_a_wrongly_typed_stored_metric_key_does_not_kill_the_cohort(
    tmp_path: Path, stored_key: str
) -> None:
    """Only read when *this* arrival resolved no key of its own — which is the
    fallback path added for stale verdicts, so it has to survive the file. A
    non-string key reaches `metrics.get(...)` as an unhashable value and
    raises behind the subscriber's guard, retiring the cohort silently.
    """
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    path = cohort_path(runs, "C-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{{"metric_key": {stored_key}, "members": []}}', encoding="utf-8")

    state = promote_within_cohort(
        runs,
        "C-1",
        member_id="E-1",
        completed_at=None,
        competition="titanic",
        metric_key=None,  # forces the stored key to be read
        maximize=True,
    )

    assert [m["id"] for m in state["members"]] == ["E-1"]
    assert state.get("promoted") is None


def test_a_wrongly_typed_stored_direction_is_not_trusted(tmp_path: Path) -> None:
    """A non-bool `maximize` is truthy, so an error metric would be maximised
    and the worst branch promoted."""
    runs = tmp_path / "runs"
    _record(runs, "E-1", 0.50)
    _record(runs, "E-2", 0.20)
    path = cohort_path(runs, "C-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"metric_key": "cv_rmse", "maximize": "yes", "members": []}', encoding="utf-8"
    )

    state = promote_within_cohort(
        runs,
        "C-1",
        member_id="E-1",
        completed_at=None,
        competition="titanic",
        metric_key=None,
        maximize=False,  # the caller's own direction, which must win
    )
    state = promote_within_cohort(
        runs,
        "C-1",
        member_id="E-2",
        completed_at=None,
        competition="titanic",
        metric_key=None,
        maximize=False,
    )

    assert state["maximize"] is False
    assert state["promoted"] == "E-2"


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
