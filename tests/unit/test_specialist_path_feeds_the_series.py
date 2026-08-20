"""The `run_experiment` specialist path must reach the score series.

Measured 2026-08-20 across three campaigns of one workspace: every run that
reached `run_plan` recorded a score, and every run that took the specialist
path recorded nothing.

| run | run_plan | run_experiment | scores |
|-----|----------|----------------|--------|
| 1   | 2        | 0              | 1      |
| 2   | 0        | 6              | 0      |
| 3   | 0        | 6              | 0      |

Both objective stops read that series, so a campaign preferring this tool could
never fire `metric_target` or `plateau` however well it trained. Two causes: the
handler never surfaced the execution id it had already stamped on its own
artifacts, and the score writer read only `execution_outcome.json`, which this
path does not write.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from labpilot.research_engine.agents.git_evolution import write_experiment_git_record
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    evaluate_stops,
)
from labpilot.research_engine.conductor.scoring import score_event_for
from labpilot.research_engine.tools.handlers.specialists import _execution_id_from
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "specialist") -> Workspace:
    return Workspace.from_client(scaffold_workspace(tmp_path / slug, slug)).ensure_roots()


def _ref(kind: str, ref_id: str) -> ArtifactRef:
    return ArtifactRef(
        kind=kind, id=ref_id, schema_id=f"{kind}.v1", path="/tmp/x", competition="c"
    )


def _record(
    ws: Workspace, execution_id: str, metrics: dict, *, status: str = "completed"
) -> None:
    write_experiment_git_record(
        ws.effective_runs_dir,
        {
            "experiment_id": f"exp_{ws.competition}_{execution_id}",
            "execution_id": execution_id,
            "plan_id": "P-001",
            "competition": ws.competition,
            "status": status,
            "metrics": metrics,
        },
    )


def _competition(ws: Workspace, execution_id: str, key: str, direction: str) -> None:
    import json

    run_dir = ws.effective_runs_dir / execution_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "competition.json").write_text(
        json.dumps(
            {
                "slug": ws.competition,
                "title": ws.competition,
                "evaluation_metric": {"name": key, "key": key, "direction": direction},
            }
        ),
        encoding="utf-8",
    )



class _RegistryDouble:
    """`build_default_specialist_registry(...).candidates(capability=...)`."""

    def candidates(self, **_kw: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="double", agent=object())]


def _registry_double() -> _RegistryDouble:
    return _RegistryDouble()


# -- the id the handler already had ----------------------------------------


def test_the_execution_id_is_read_off_the_refs() -> None:
    assert _execution_id_from(_ref("experiment", "experiment:E-007"), None) == "E-007"
    assert _execution_id_from(None, _ref("metrics", "metrics:E-007")) == "E-007"


def test_a_ref_with_no_id_yields_nothing_rather_than_a_guess() -> None:
    assert _execution_id_from(None, None) is None
    assert _execution_id_from(_ref("metrics", "")) is None


# -- the series it can now feed --------------------------------------------


def test_a_specialist_run_records_a_score(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    _record(ws, "E-001", {"cv_rmse": 190.97})
    _competition(ws, "E-001", "rmse", "minimize")

    event = score_event_for(ws, "E-001", fallback_maximize=False)

    assert event is not None
    assert event.metric_name == "cv_rmse"
    assert event.value == pytest.approx(190.97)


def test_a_specialist_run_can_end_a_campaign_on_its_objective(tmp_path: Path) -> None:
    """The consequence: this path is no longer invisible to `metric_target`."""
    ws = _ws(tmp_path, "reached")
    _record(ws, "E-001", {"cv_score": 1.65, "metric": "rmse"})
    _competition(ws, "E-001", "rmse", "minimize")
    event = score_event_for(ws, "E-001", fallback_maximize=False)
    assert event is not None

    config = BudgetConfig(target_metric="rmse", target_value=2.0, maximize=False)
    state = BudgetState(score_events=[event], last_metric=event.value)

    assert evaluate_stops(config, state) == "metric_target"


def test_another_runs_record_is_never_scored_against_this_execution(tmp_path: Path) -> None:
    """`find_experiment_record` falls back to the *latest* record when an id is
    not indexed. Taking that would score one execution from another's numbers —
    the exact confusion keying on the execution id exists to prevent."""
    ws = _ws(tmp_path, "attribution")
    _record(ws, "E-001", {"cv_rmse": 190.97})
    _competition(ws, "E-002", "rmse", "minimize")

    assert score_event_for(ws, "E-002", fallback_maximize=False) is None


def test_no_record_and_no_outcome_is_still_no_score(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "empty")

    assert score_event_for(ws, "E-404", fallback_maximize=False) is None


# -- what the record does not prove ----------------------------------------


@pytest.mark.parametrize("status", ["failed", "error", "cancelled", "ERRORED"])
def test_a_run_that_reported_failure_is_not_scored(tmp_path: Path, status: str) -> None:
    """An `execution_outcome.json` exists only for an execution that completed,
    so the path this record stands in for could never be handed a failed run's
    numbers. This record is written either way, and its `metrics` are read from
    whatever `metrics.json` sits at the workspace root — so a failed run that
    left an earlier run's file in place would be credited with that score."""
    ws = _ws(tmp_path, f"failed-{status}")
    _record(ws, "E-001", {"cv_rmse": 1.65}, status=status)
    _competition(ws, "E-001", "rmse", "minimize")

    assert score_event_for(ws, "E-001", fallback_maximize=False) is None


def test_a_status_nobody_recognises_is_not_read_as_failure(tmp_path: Path) -> None:
    """Refusing known failures, not demanding a known success: the specialist
    reports several spellings for a run that did produce metrics, and requiring
    one exact word would put this path back where it started — invisible."""
    ws = _ws(tmp_path, "unknown-status")
    _record(ws, "E-001", {"cv_rmse": 1.65}, status="unknown")
    _competition(ws, "E-001", "rmse", "minimize")

    assert score_event_for(ws, "E-001", fallback_maximize=False) is not None


def test_a_dry_run_does_not_name_itself_to_the_score_writer(tmp_path: Path) -> None:
    """`run_experiment` defaults to `dry_run=True`, and the freshness check that
    would catch a stale `metrics.json` sits behind `if not dry_run`. Surfacing
    the execution id anyway let a dry run re-submit the previous run's number as
    a fresh reading — three of those make `plateau` fire on one measurement
    counted thrice, and reset the counter watching for exactly that stall.
    """
    from labpilot.research_engine.tools.handlers import specialists

    ws = _ws(tmp_path, "dry")
    refs = [_ref("experiment", "experiment:E-001"), _ref("metrics", "metrics:E-001")]

    with (
        patch.object(specialists, "execute_agent_sync", return_value=refs),
        patch.object(specialists, "_metrics_written_since", return_value=False),
        patch.object(
            specialists,
            "build_default_specialist_registry",
            return_value=_registry_double(),
        ),
    ):
        result = specialists.run_experiment(ws, plan_id="P-001", dry_run=True)

    assert result.data["execution_id"] is None, "a stale reading must not be scorable"


def test_a_run_that_wrote_its_own_metrics_does_name_itself(tmp_path: Path) -> None:
    from labpilot.research_engine.tools.handlers import specialists

    ws = _ws(tmp_path, "fresh")
    refs = [_ref("experiment", "experiment:E-001"), _ref("metrics", "metrics:E-001")]

    with (
        patch.object(specialists, "execute_agent_sync", return_value=refs),
        patch.object(specialists, "_metrics_written_since", return_value=True),
        patch.object(
            specialists,
            "build_default_specialist_registry",
            return_value=_registry_double(),
        ),
    ):
        result = specialists.run_experiment(ws, plan_id="P-001", dry_run=True)

    assert result.data["execution_id"] == "E-001"


def test_two_runs_of_one_plan_do_not_share_a_task_id(tmp_path: Path) -> None:
    """The agent falls back to `E-agent-{task.id}` when the inner result names
    no execution. A constant default made every such run share one id — and
    `ScoreEvent.experiment_id` is what exit criterion 1 and the stagnation mint
    cite."""
    from labpilot.research_engine.tools.handlers import specialists

    ws = _ws(tmp_path, "ids")
    seen: list[str] = []

    def _capture(agent, task, workspace, bundle):  # noqa: ANN001, ARG001
        seen.append(task.id)
        return []

    ws_ids = []
    for stamp in (1_000.0, 1_001.0):
        with (
            patch.object(specialists, "execute_agent_sync", _capture),
            patch.object(specialists, "_metrics_written_since", return_value=True),
            patch.object(specialists.time, "time", return_value=stamp),
            patch.object(
                specialists,
                "build_default_specialist_registry",
                return_value=_registry_double(),
            ),
        ):
            specialists.run_experiment(ws, plan_id="P-001", dry_run=True)
        ws_ids.append(seen[-1])

    assert ws_ids[0] != ws_ids[1]
