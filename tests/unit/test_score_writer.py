"""M8-2: a finished experiment leaves a comparable score behind.

`metric_history`/`last_metric` have been read by `evaluate_stops` since M3 and
written by nothing, so `plateau` and `metric_target` could never fire. This is
the writer.

The score is read from the execution's *own* `execution_outcome.json`, not the
`metrics.json` at the workspace root: the root file survives a failed run, so
"is there a file?" and "did this run write one?" are different questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    evaluate_stops,
)
from labpilot.research_engine.conductor.checkpoint import load_budget_pair
from labpilot.research_engine.conductor.loop import _record_experiment_outcome, _score_event_for
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _write_outcome(ws: Workspace, execution_id: str, **fields) -> Path:
    """Write an execution's own outcome artifact, where the engineer puts it."""
    paths = ResearchPaths(ws.knowledge_dir, ws.competition)
    out = paths.executions_dir / execution_id / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "competition": ws.competition,
        "execution_id": execution_id,
        "plan_id": "P-001",
        "metrics": {"cv_rmse": 194.80},
        **fields,
    }
    path = out / "execution_outcome.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _competition_json(ws: Workspace, execution_id: str, key: str, direction: str) -> None:
    """The competition profile the metric-key resolver reads."""
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


def test_a_finished_experiment_produces_a_score_event(tmp_path: Path):
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-001", hypothesis_id="H-001")
    _competition_json(ws, "E-001", "rmse", "minimize")

    event = _score_event_for(ws, "E-001")

    assert event is not None
    assert event.experiment_id == "E-001"
    assert event.hypothesis_id == "H-001"
    assert event.metric_name == "cv_rmse"
    assert event.value == 194.80
    assert event.maximize is False


def test_the_score_comes_from_this_executions_own_outcome(tmp_path: Path):
    """The workspace-root metrics.json survives a failed run. Reading it would
    credit one execution with another's score; the per-execution artifact
    cannot be confused that way."""
    ws = _ws(tmp_path)
    (ws.root / "metrics.json").write_text(json.dumps({"cv_rmse": 999.0}), encoding="utf-8")
    _write_outcome(ws, "E-002", metrics={"cv_rmse": 190.97})
    _competition_json(ws, "E-002", "rmse", "minimize")

    event = _score_event_for(ws, "E-002")

    assert event is not None
    assert event.value == 190.97


def test_an_execution_with_no_outcome_records_nothing(tmp_path: Path):
    """`run_experiment` writes no execution outcome. Skipping is the answer,
    not inventing a score."""
    ws = _ws(tmp_path)

    assert _score_event_for(ws, "E-404") is None


def test_a_placeholder_run_records_nothing(tmp_path: Path):
    """A run that never trained a model has no score, for the same reason it
    must not reach an evidence card."""
    ws = _ws(tmp_path)
    # The real marker a scaffolded run writes — a plausible-looking number is
    # exactly what fooled every check that only asked whether a score existed.
    _write_outcome(ws, "E-003", metrics={"status": "dry_run_stub", "cv_rmse": 0.5})

    assert _score_event_for(ws, "E-003") is None


def test_a_diverged_run_records_nothing(tmp_path: Path):
    """NaN is not a comparable score: every NaN comparison is False, so
    admitting one silently disables the stops that read this series."""
    ws = _ws(tmp_path)
    out = _write_outcome(ws, "E-004")
    out.write_text(
        json.dumps({"execution_id": "E-004", "metrics": {"cv_rmse": float("nan")}}),
        encoding="utf-8",
    )
    _competition_json(ws, "E-004", "rmse", "minimize")

    assert _score_event_for(ws, "E-004") is None


def test_an_unresolvable_metric_records_nothing(tmp_path: Path):
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-005", metrics={"notes": "no numbers here"})

    assert _score_event_for(ws, "E-005") is None


def test_the_writer_appends_and_derives_on_the_real_persist_path(tmp_path: Path):
    """End to end through the function the loop actually calls."""
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-001", metrics={"cv_rmse": 194.80})
    _competition_json(ws, "E-001", "rmse", "minimize")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("beat baseline")

        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-001"
        )

        _, state = load_budget_pair(store.get_session(session.id))
        assert [e.experiment_id for e in state.score_events] == ["E-001"]
        assert state.metric_history == [194.80]
        assert state.last_metric == 194.80
    finally:
        store.close()


def test_a_failed_experiment_records_no_score(tmp_path: Path):
    """The breaker still counts it; the series must not."""
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-001")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")

        _record_experiment_outcome(
            store,
            session.id,
            succeeded=False,
            error="boom",
            workspace=ws,
            execution_id="E-001",
        )

        _, state = load_budget_pair(store.get_session(session.id))
        assert state.score_events == []
        assert state.consecutive_failures == 1
    finally:
        store.close()


def test_a_session_with_no_events_keeps_its_stored_metric_history(tmp_path: Path):
    """The derivation runs only where the series changes. A campaign resumed
    from a session that predates `score_events` still has values in
    `metric_history`, and a step that records no score must not erase them."""
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session(
            "g", metadata={"budget_state": {"metric_history": [0.5, 0.6], "last_metric": 0.6}}
        )

        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-404"
        )

        _, state = load_budget_pair(store.get_session(session.id))
        assert state.metric_history == [0.5, 0.6]
        assert state.last_metric == 0.6
    finally:
        store.close()


def _event(metric_name: str, value: float) -> ScoreEvent:
    return ScoreEvent(
        experiment_id="E-001",
        metric_name=metric_name,
        value=value,
        maximize=False,
    )


def test_a_target_is_not_met_by_a_different_metric():
    """A cv_rmse of 190.97 must not satisfy an lb_auc target of 0.90 — that
    ends the campaign on a metric it was never measuring."""
    config = BudgetConfig(target_metric="lb_auc", target_value=0.90, maximize=True)
    state = BudgetState(
        score_events=[_event("cv_rmse", 190.97)],
        metric_history=[190.97],
        last_metric=190.97,
    )

    assert evaluate_stops(config, state) == "none"


def test_a_target_is_met_by_its_own_metric():
    """The guard must not cost the stop it protects."""
    config = BudgetConfig(target_metric="cv_rmse", target_value=200.0, maximize=False)
    state = BudgetState(
        score_events=[_event("cv_rmse", 190.97)],
        metric_history=[190.97],
        last_metric=190.97,
    )

    assert evaluate_stops(config, state) == "metric_target"


def test_a_session_without_events_keeps_the_older_looser_target_behaviour():
    """A `last_metric` that predates the series names no metric. Refusing to
    compare there would disarm a target that used to fire."""
    config = BudgetConfig(target_metric="lb", target_value=0.9, maximize=True)

    assert evaluate_stops(config, BudgetState(last_metric=0.91)) == "metric_target"
