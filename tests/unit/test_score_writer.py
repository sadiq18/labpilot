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

import pytest

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    evaluate_stops,
    metric_names_match,
)
from labpilot.research_engine.conductor.checkpoint import load_budget_pair
from labpilot.research_engine.conductor.loop import _record_experiment_outcome
from labpilot.research_engine.conductor.scoring import _direction_for, score_event_for
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.intelligence.competition.direction import _direction_to_maximize
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

    event = score_event_for(ws, "E-001")

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

    event = score_event_for(ws, "E-002")

    assert event is not None
    assert event.value == 190.97


def test_an_execution_with_no_outcome_records_nothing(tmp_path: Path):
    """`run_experiment` writes no execution outcome. Skipping is the answer,
    not inventing a score."""
    ws = _ws(tmp_path)

    assert score_event_for(ws, "E-404") is None


def test_a_placeholder_run_records_nothing(tmp_path: Path):
    """A run that never trained a model has no score, for the same reason it
    must not reach an evidence card."""
    ws = _ws(tmp_path)
    # The real marker a scaffolded run writes — a plausible-looking number is
    # exactly what fooled every check that only asked whether a score existed.
    _write_outcome(ws, "E-003", metrics={"status": "dry_run_stub", "cv_rmse": 0.5})

    assert score_event_for(ws, "E-003") is None


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

    assert score_event_for(ws, "E-004") is None


def test_an_unresolvable_metric_records_nothing(tmp_path: Path):
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-005", metrics={"notes": "no numbers here"})

    assert score_event_for(ws, "E-005") is None


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


@pytest.mark.parametrize("target_metric", ["rmse", "cv_rmse", "RMSE"])
def test_a_target_is_met_however_the_user_spelled_the_metric(target_metric):
    """The name the user types and the key the resolver records differ.

    `--target-metric` takes the competition's own metric (`rmse`, from
    `MetricSpec.key` — the only spelling the user has), while a `ScoreEvent`
    carries the resolver's cross-validated key (`cv_rmse`). An earlier version
    of this guard compared the two for equality, so the target a user could
    actually state never matched and `metric_target` could not fire at all —
    and the test missed it by passing the internal key, which no user types.
    """
    config = BudgetConfig(target_metric=target_metric, target_value=200.0, maximize=False)
    state = BudgetState(
        score_events=[_event("cv_rmse", 190.97)],
        metric_history=[190.97],
        last_metric=190.97,
    )

    assert evaluate_stops(config, state) == "metric_target"


@pytest.mark.parametrize(
    ("recorded", "requested", "expected"),
    [
        # An unqualified request matches any measurement of that metric — the
        # user naming `rmse` has no way to say which one.
        ("cv_rmse", "rmse", True),
        ("lb_auc", "auc", True),
        ("val_loss", "loss", True),
        ("rmse", "rmse", True),
        ("cv_rmse", "CV_RMSE", True),
        # A qualified request is taken literally: local and leaderboard are the
        # distinction this milestone keeps separate, not a spelling difference.
        ("cv_auc", "lb_auc", False),
        ("lb_auc", "cv_auc", False),
        # Different metrics never match.
        ("cv_rmse", "auc", False),
        ("cv_rmse", "lb_auc", False),
        ("", "rmse", False),
        ("cv_rmse", "", False),
    ],
)
def test_metric_names_match_semantics(recorded, requested, expected):
    """An earlier version stripped only `cv_`, so a recorded `lb_auc` never
    answered a target of `auc` — the same never-matching target the prefix
    handling was added to fix, one prefix over."""
    assert metric_names_match(recorded, requested) is expected


def test_a_session_without_events_keeps_the_older_looser_target_behaviour():
    """A `last_metric` that predates the series names no metric. Refusing to
    compare there would disarm a target that used to fire."""
    config = BudgetConfig(target_metric="lb", target_value=0.9, maximize=True)

    assert evaluate_stops(config, BudgetState(last_metric=0.91)) == "metric_target"


@pytest.mark.parametrize("body", ["[]", "null", '"nope"', "123"])
def test_a_malformed_outcome_file_records_nothing_instead_of_raising(tmp_path: Path, body):
    """These parse as JSON but are not objects.

    Raising here would not stay local: `_record_experiment_outcome` runs inside
    the dispatch try block, so the error would surface as a dispatch failure
    and count a *successful* experiment against the circuit breaker — three of
    them stop the campaign.
    """
    ws = _ws(tmp_path)
    out = _write_outcome(ws, "E-006")
    out.write_text(body, encoding="utf-8")

    assert score_event_for(ws, "E-006") is None


def _knowledge_competition_json(ws: Workspace, key: str, direction: str) -> None:
    """The knowledge copy of the spec — where `analyze` puts it."""
    root = ResearchPaths(ws.knowledge_dir, ws.competition).root
    root.mkdir(parents=True, exist_ok=True)
    (root / "competition.json").write_text(
        json.dumps(
            {
                "slug": ws.competition,
                "title": ws.competition,
                "evaluation_metric": {"name": key, "key": key, "direction": direction},
            }
        ),
        encoding="utf-8",
    )


def test_the_spec_is_found_in_the_knowledge_tree_too(tmp_path: Path):
    """`analyze` writes the knowledge copy, and a workspace can have only that.

    Searching just the run dir and the workspace root left the resolver with
    no spec, so it fell through to the alphabetically-first metric and
    defaulted the direction — recording `cv_mae`/maximize for a competition
    whose spec says `rmse`/minimize.
    """
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-001", metrics={"cv_mae": 12.0, "cv_rmse": 194.80})
    _knowledge_competition_json(ws, "rmse", "minimize")

    event = score_event_for(ws, "E-001")

    assert event is not None
    assert event.metric_name == "cv_rmse"
    assert event.value == 194.80
    assert event.maximize is False


@pytest.mark.parametrize(
    "raw", ["minimize", "Minimize", "MINIMIZE", "min", "maximize", "Maximize", "max", "MAX"]
)
@pytest.mark.parametrize("shape", ["metric", "evaluation_metric"])
def test_direction_resolution_agrees_with_the_canonical_reader(tmp_path: Path, raw, shape):
    """The conductor must not answer this question differently from the module
    that owns it.

    Both bugs this replaced were disagreements on an identical input: a
    hand-rolled `!= "minimize"` read `"Minimize"` as maximize, and pydantic's
    `direction` default turned an absent field into a confident answer. Asking
    the two readers the same question is the check that catches that class —
    testing this function alone never can.
    """
    ws = _ws(tmp_path)
    (ws.root / "competition.json").write_text(
        json.dumps({"slug": ws.competition, "title": ws.competition, shape: {"direction": raw}}),
        encoding="utf-8",
    )

    assert _direction_for(ws, "E-001", ResearchPaths(ws.knowledge_dir, ws.competition)) == (
        _direction_to_maximize(raw)
    )


def test_an_absent_direction_stays_unknowable(tmp_path: Path):
    """A spec naming a metric but no direction must not resolve.

    `MetricSpec.direction` defaults to "maximize", so parsing through the
    model turns silence into a confident wrong sign and stops the search
    before the Analyze profile artifact — which is where a real competition's
    `minimize` was actually found.
    """
    ws = _ws(tmp_path)
    (ws.root / "competition.json").write_text(
        json.dumps(
            {
                "slug": ws.competition,
                "title": ws.competition,
                "evaluation_metric": {"name": "rmse", "key": "rmse"},
            }
        ),
        encoding="utf-8",
    )

    assert _direction_for(ws, "E-001", ResearchPaths(ws.knowledge_dir, ws.competition)) is None


def test_the_direction_falls_back_to_the_campaigns_own_when_unknowable(tmp_path: Path):
    """With no spec anywhere, the event must not invent a second opinion — it
    takes the direction the campaign is already running under."""
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-001", metrics={"cv_rmse": 194.80})

    assert score_event_for(ws, "E-001", fallback_maximize=False).maximize is False
    assert score_event_for(ws, "E-001", fallback_maximize=True).maximize is True


def test_a_changed_primary_metric_narrows_the_window_without_losing_the_record(tmp_path: Path):
    """Two readings of different metrics are not a *comparison*, but they are
    both still experiments that happened.

    `analyze_competition` can correct which key is primary mid-campaign. The
    comparison window narrows to the readings that share a metric, so
    `plateau` never takes a max-minus-min across scales — but every event
    stays on record, because exit criterion 1 and the stagnation mint both
    cite experiments by id, and evicting them is what the design doc refused
    to do.
    """
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")
        _write_outcome(ws, "E-001", metrics={"cv_mae": 12.0})
        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-001"
        )
        # The spec arrives and names a different primary metric.
        _write_outcome(ws, "E-002", metrics={"cv_mae": 11.0, "cv_rmse": 190.9})
        _competition_json(ws, "E-002", "rmse", "minimize")

        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-002"
        )

        _, state = load_budget_pair(store.get_session(session.id))
        # Both experiments remain citable by id...
        assert [e.experiment_id for e in state.score_events] == ["E-001", "E-002"]
        # ...while only the comparable reading feeds the plateau window.
        assert state.metric_history == [190.9]
        assert state.last_metric == 190.9
    finally:
        store.close()


def test_the_window_reopens_as_the_new_metric_accumulates(tmp_path: Path):
    """After a metric change, later readings of the new metric join the window
    rather than each one resetting it."""
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")
        _write_outcome(ws, "E-001", metrics={"cv_mae": 12.0})
        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-001"
        )
        for eid, value in (("E-002", 190.9), ("E-003", 188.4)):
            _write_outcome(ws, eid, metrics={"cv_rmse": value})
            _competition_json(ws, eid, "rmse", "minimize")
            _record_experiment_outcome(
                store, session.id, succeeded=True, workspace=ws, execution_id=eid
            )

        _, state = load_budget_pair(store.get_session(session.id))
        assert len(state.score_events) == 3
        assert state.metric_history == [190.9, 188.4]
    finally:
        store.close()


def test_a_consistent_metric_keeps_accumulating(tmp_path: Path):
    """The reset must not cost the series it protects."""
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")
        for eid, value in (("E-001", 194.80), ("E-002", 190.97)):
            _write_outcome(ws, eid, metrics={"cv_rmse": value})
            _competition_json(ws, eid, "rmse", "minimize")
            _record_experiment_outcome(
                store, session.id, succeeded=True, workspace=ws, execution_id=eid
            )

        _, state = load_budget_pair(store.get_session(session.id))
        assert state.metric_history == [194.80, 190.97]
    finally:
        store.close()


@pytest.mark.parametrize("metrics", [["not", "a", "dict"], "a string", 7])
def test_a_non_dict_metrics_field_records_nothing_instead_of_raising(tmp_path: Path, metrics):
    """`outcome["metrics"]` is not guaranteed to be an object either.

    A non-empty non-dict slips past a plain `or {}` and reaches
    `is_placeholder_metrics`, whose `(metrics or {}).get(...)` then raises —
    with the same consequence as a malformed file, a successful experiment
    counted as a failure.
    """
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-007", metrics=metrics)

    assert score_event_for(ws, "E-007") is None


def test_the_first_recorded_score_replaces_a_legacy_history(tmp_path: Path):
    """Stored readings that predate the series name no metric, so they cannot
    be compared against a keyed one — the series becomes authoritative rather
    than being appended to a mixed-scale list."""
    ws = _ws(tmp_path)
    _write_outcome(ws, "E-001", metrics={"cv_rmse": 194.80})
    _competition_json(ws, "E-001", "rmse", "minimize")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session(
            "g", metadata={"budget_state": {"metric_history": [0.5, 0.6, 0.7], "last_metric": 0.7}}
        )

        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-001"
        )

        _, state = load_budget_pair(store.get_session(session.id))
        assert state.metric_history == [194.80]
        assert state.last_metric == 194.80
    finally:
        store.close()
