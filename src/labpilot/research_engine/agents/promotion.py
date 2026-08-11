"""Pick the winner among the K branches of one fan-out step (M11).

A cohort is the set of experiments one campaign step dispatched in parallel.
Its members finish at different times and in any order, so there is no moment
at which some caller "has all K results" — the verdict is instead recomputed
from durable state each time a member lands, and the last arrival's
computation is the complete one.

What this does *not* do is file a reflection for the losers. Every successful
execution already reflects, winner and loser alike, inside
`record_successful_execution`. Filing a second one here would count the same
evidence into a belief's confidence twice, and it could not say anything the
first could not: reflection runs when a branch finishes, which is before its
siblings have finished and therefore before "did it lose?" has an answer.
Losing is recorded here, in the cohort file, for whoever needs to read it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.accessor.common.atomic_write import atomic_write_text
from labpilot.accessor.common.file_lock import locked
from labpilot.research_engine.agents.events import EXPERIMENT_COMPLETED, EventBus
from labpilot.research_engine.agents.git_evolution import find_experiment_record

logger = logging.getLogger(__name__)

#: Sorts after any ISO-8601 timestamp, so a member whose `completed_at` never
#: arrived loses a tie rather than winning one by being absent.
_NEVER = "￿"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def cohort_path(runs_dir: Path | str, cohort_id: str) -> Path:
    """Where a cohort's membership and verdict live.

    Under `runs_dir` for the same reason experiment records are (see
    `agents/experiment.py`): every branch of the step must agree on one
    location, and it has to outlive the worktrees the branches ran in.
    """
    return Path(runs_dir) / "experiment" / "cohorts" / f"{_safe_id(cohort_id)}.json"


def _safe_id(cohort_id: str) -> str:
    """Reject a cohort id that would not stay inside the cohorts directory.

    The id arrives as task metadata, and it is used as a filename — `..` or a
    leading `/` in one would put this file somewhere else entirely.
    """
    text = str(cohort_id).strip()
    if not _SAFE_ID.match(text):
        raise ValueError(f"unusable cohort id: {cohort_id!r}")
    return text


def rank_candidates(
    candidates: Sequence[dict[str, Any]], metric_key: str, *, maximize: bool
) -> dict[str, Any] | None:
    """The best candidate on `metric_key`, or None if none is comparable.

    Ties break on earliest `completed_at`, then on `experiment_id`: two
    branches can genuinely score identically, and a winner that depends on
    dict ordering would make the same cohort promote differently on a re-read.

    A candidate whose metric is missing, non-numeric, or non-finite is not
    ranked at all rather than sorted to the bottom — a diverged run's NaN
    compares False against everything, so admitting it would let it win by
    default under `min`.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict):
            continue
        raw = metrics.get(metric_key)
        # `bool` is an `int` in Python: a metric of `True` would otherwise
        # rank as 1.0 against real scores.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        scored.append((value, candidate))

    if not scored:
        return None

    def sort_key(item: tuple[float, dict[str, Any]]) -> tuple[float, str, str]:
        value, candidate = item
        return (
            -value if maximize else value,
            str(candidate.get("completed_at") or _NEVER),
            str(candidate.get("experiment_id") or ""),
        )

    return min(scored, key=sort_key)[1]


def promote_within_cohort(
    runs_dir: Path | str,
    cohort_id: str,
    *,
    member_id: str,
    completed_at: str | None,
    competition: str | None,
    metric_key: str | None,
    maximize: bool,
) -> dict[str, Any]:
    """Add a member to its cohort and recompute the verdict. Returns the state.

    The whole read-append-rank-write runs under one cross-process lock: K
    branches land concurrently, and an unlocked read-then-write here loses
    whichever member was appended by the racer that wrote second.

    Idempotent by construction — the verdict is derived from the full member
    list every time rather than adjusted incrementally, so a member arriving
    twice, or a cohort re-ranked after each of K arrivals, converges on the
    same answer instead of accumulating one.

    `completed_at` is stored here rather than on the experiment record because
    this is the only writer that has it: it reaches the subscriber on the live
    event, and ranking needs every member's, long after their events are gone.
    """
    path = cohort_path(runs_dir, cohort_id)
    with locked(path.with_suffix(".lock")):
        state = _read_state(path)
        members: list[dict[str, Any]] = state.get("members") or []
        if not any(m.get("id") == member_id for m in members):
            members.append({"id": member_id, "completed_at": completed_at})

        state["cohort_id"] = cohort_id
        state["members"] = members
        state["updated_at"] = datetime.now(UTC).isoformat()
        if competition:
            state["competition"] = competition

        if metric_key:
            state["metric_key"] = metric_key
            state["maximize"] = maximize
            best = rank_candidates(
                _candidates_for(runs_dir, members), metric_key, maximize=maximize
            )
            promoted = best.get("cohort_member_id") if best else None
            state["promoted"] = promoted
            state["demoted"] = [
                m["id"] for m in members if promoted and m["id"] != promoted
            ]
        else:
            # No resolvable metric means no comparison — record who ran, and
            # say nothing about who won rather than promoting arbitrarily.
            logger.info("cohort %s has no resolvable metric; recorded members only", cohort_id)

        atomic_write_text(path, json.dumps(state, indent=2) + "\n")
    return state


def _candidates_for(
    runs_dir: Path | str, members: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Each member's experiment record, carrying the id it was recorded under.

    A member whose record cannot be found is skipped: it is a branch that has
    not written one yet (or wrote it somewhere this cohort cannot see), and
    ranking it on absent metrics would be inventing a result.
    """
    candidates: list[dict[str, Any]] = []
    for member in members:
        member_id = str(member.get("id") or "")
        if not member_id:
            continue
        record = find_experiment_record(runs_dir, member_id)
        if record is None:
            continue
        candidates.append(
            {**record, "completed_at": member.get("completed_at"), "cohort_member_id": member_id}
        )
    return candidates


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Starting over loses membership recorded before the corruption, which
        # is worse than it sounds only if it is silent — hence the warning.
        logger.warning("unreadable cohort file %s; starting a fresh one", path)
        return {}
    return state if isinstance(state, dict) else {}


def _metric_for(payload: dict[str, Any]) -> tuple[str | None, bool]:
    """The metric key and direction the campaign is steering by.

    Asks the resolver the conductor's own scoring uses, rather than reading a
    key off the payload's metrics: a cohort ranked on a different key than the
    campaign's would promote a branch the campaign then treats as a
    regression. It lives under `shared/` and not in the conductor precisely so
    a specialist can ask it — see `test_agents_do_not_import_conductor`.
    """
    from labpilot.research_engine.shared.experiments.scoring import (
        resolve_metric_and_direction,
    )
    from labpilot.research_engine.workspace_facade import Workspace

    knowledge_dir = payload.get("knowledge_dir")
    execution_id = payload.get("execution_id")
    metrics = payload.get("metrics")
    if not knowledge_dir or not execution_id or not isinstance(metrics, dict):
        return None, True

    runs_dir = payload.get("runs_dir")
    workspace = Workspace(
        competition=str(payload.get("competition") or ""),
        knowledge_dir=Path(str(knowledge_dir)),
        root=Path(str(payload.get("workspace_root") or knowledge_dir)),
        runs_dir=Path(str(runs_dir)) if runs_dir else None,
    )
    return resolve_metric_and_direction(workspace, str(execution_id), metrics)


def promote_from_completion(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Fold one ExperimentCompleted payload into its cohort's verdict.

    Returns None when the event is not part of a comparison: no `cohort_id`
    is the ordinary K=1 campaign step, which has nothing to rank against.
    """
    cohort_id = payload.get("cohort_id")
    if not cohort_id:
        return None
    if str(payload.get("status") or "").lower() == "failed" or payload.get("error"):
        # A branch that failed produced no result to compare. The publisher
        # already routes these to ModelFailed, but promotion decides what the
        # campaign carries forward, so it guards its own input.
        return None

    # `runs_dir` is where the records actually are; `workspace_root` is the
    # pre-M11 shape and is only a fallback for payloads that predate the split.
    runs_dir = payload.get("runs_dir") or payload.get("workspace_root")
    member_id = payload.get("execution_id") or payload.get("experiment_id")
    if not runs_dir or not member_id:
        return None

    metric_key, maximize = _metric_for(payload)
    return promote_within_cohort(
        runs_dir,
        str(cohort_id),
        member_id=str(member_id),
        completed_at=payload.get("completed_at"),
        competition=payload.get("competition"),
        metric_key=metric_key,
        maximize=maximize,
    )


def install_promotion_subscriber(bus: EventBus) -> None:
    """On ExperimentCompleted, keep the branch's cohort verdict up to date."""

    def _on_experiment_completed(event: str, payload: dict[str, Any]) -> None:
        del event
        try:
            promote_from_completion(payload)
        except Exception:  # noqa: BLE001 — a lost verdict must not fail the run
            logger.exception("promotion subscriber failed")

    bus.subscribe(EXPERIMENT_COMPLETED, _on_experiment_completed)
