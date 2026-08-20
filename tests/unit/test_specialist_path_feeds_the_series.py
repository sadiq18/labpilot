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


def _record(ws: Workspace, execution_id: str, metrics: dict) -> None:
    write_experiment_git_record(
        ws.effective_runs_dir,
        {
            "experiment_id": f"exp_{ws.competition}_{execution_id}",
            "execution_id": execution_id,
            "plan_id": "P-001",
            "competition": ws.competition,
            "status": "completed",
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
