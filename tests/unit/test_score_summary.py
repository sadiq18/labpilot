"""M8-3/4/5: the campaign can tell whether it is getting anywhere.

`score_summary` is the one place the four progress numbers are derived, so
M17's `goal_progress(config, state)` renders from it rather than computing
them a second way — the primary-metric key already has four disagreeing
resolvers and that is the pattern being avoided.

The stagnant clause is deliberately a *gate* on the tool allowlist, not a
number in the policy prompt. `decide_next` is LLM-driven, and a metric the
model merely sees is the bet that already lost: `evaluate_stops` has read one
since M3 and changed no decision in nine campaigns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    evaluate_stops,
    score_summary,
)
from labpilot.research_engine.conductor.policy import (
    available_tools,
    build_observe_bundle,
    should_gather_evidence,
)
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _events(*values: float, metric: str = "cv_rmse", maximize: bool = False) -> list[ScoreEvent]:
    return [
        ScoreEvent(
            experiment_id=f"E-{i:03d}",
            metric_name=metric,
            value=value,
            maximize=maximize,
        )
        for i, value in enumerate(values, start=1)
    ]


def _state(*values: float, **kw) -> BudgetState:
    return BudgetState(score_events=_events(*values, **kw))


def _with_history(state: BudgetState) -> BudgetState:
    """The same state as the M8-2 writer leaves it — `metric_history` derived
    from the series, which is what `evaluate_stops` reads."""
    return state.model_copy(
        update={
            "metric_history": [e.value for e in state.score_events],
            "last_metric": state.score_events[-1].value if state.score_events else None,
        }
    )


def _stocked(ws: Workspace, count: int = 8, evidence_age_hours: float = 2.0) -> Workspace:
    """A workspace where only the stagnant clause can decide.

    `should_gather_evidence` is an independent-OR gate, so a test that leaves
    the other clauses free proves nothing about this one: a fresh scaffolded
    workspace has an empty backlog (the `viable` clause opens the gate and
    returns) and no artifacts at all (the "no evidence gathered yet" clause
    does the same). Either would make a stagnant assertion pass for the wrong
    reason.

    So the backlog is filled past `_VIABLE_TARGET`, and one artifact is dated
    into the window between `_MIN_RESWEEP_HOURS` and
    `_EVIDENCE_COOLDOWN_HOURS` — recent enough that staleness does not fire,
    old enough that the re-sweep floor does not veto. This is also the real
    shape of the situation the clause exists for: rogii sat on 46 proposed
    hypotheses, freshly gathered, while the score did not move.
    """
    from datetime import UTC, datetime, timedelta

    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.intelligence.models import (
        ResearchArtifact,
        ResearchArtifactType,
    )
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    store = HypothesisStore(ws.knowledge_dir, ws.competition)
    for i in range(count):
        store.create(
            observation=f"observation {i}",
            reason=f"reason {i}",
            prediction=f"prediction {i}",
            confidence=0.5,
            technique=f"technique_{i}",
        )

    stamp = (datetime.now(UTC) - timedelta(hours=evidence_age_hours)).isoformat()
    with KnowledgeStore(ws.knowledge_dir, ws.competition) as knowledge:
        knowledge.upsert_artifact(
            ResearchArtifact(
                id="art:test:1",
                type=ResearchArtifactType.NOTE,
                source="user",
                title="gathered evidence",
                competition_slug=ws.competition,
            )
        )
        knowledge._conn.execute(  # noqa: SLF001 — dating the row is the point
            "UPDATE research_artifacts SET created_at = ?", (stamp,)
        )
        knowledge._conn.commit()  # noqa: SLF001
    return ws


def test_the_stocked_fixture_leaves_only_the_stagnant_clause_free(tmp_path: Path):
    """Guards the fixture itself.

    Every gate assertion below is meaningless if another clause is deciding —
    and both of the others *open* the gate, so a broken fixture turns a
    stagnation test into a pass for the wrong reason. With no score series at
    all, this workspace must close the gate.
    """
    ws = _stocked(_ws(tmp_path))

    ok, reason = should_gather_evidence(ws)

    assert ok is False, f"another clause is deciding: {reason}"


# --- M8-3: the summary itself -------------------------------------------


def test_an_empty_series_says_nothing_rather_than_zero():
    """`0.0` and "no reading yet" are different answers. A consumer that saw
    a best of 0.0 for an untouched campaign would treat it as a real score."""
    summary = score_summary(BudgetState(), BudgetConfig())

    assert summary.best_so_far is None
    assert summary.delta_vs_best is None
    assert summary.last_3_scores == []
    assert summary.steps_since_improvement == 0
    assert summary.metric_name is None


def test_best_is_read_in_the_series_own_direction():
    """A minimised metric's best is its lowest reading. Taking max() would
    report the worst run as the best — the sign error that recorded this
    system's only real improvement as a rejection."""
    minimised = score_summary(_state(194.8, 190.9, 192.0), BudgetConfig())
    maximised = score_summary(
        _state(0.80, 0.86, 0.83, metric="cv_auc", maximize=True), BudgetConfig()
    )

    assert minimised.best_so_far == 190.9
    assert maximised.best_so_far == 0.86


@pytest.mark.parametrize(
    ("values", "maximize", "expected"),
    [
        # Latest is the best -> no gap, either direction.
        ((194.8, 190.9), False, 0.0),
        ((0.80, 0.86), True, 0.0),
        # Latest is worse than best -> negative, either direction.
        ((190.9, 194.8), False, -3.9000000000000057),
        ((0.86, 0.80), True, -0.06000000000000005),
    ],
)
def test_delta_vs_best_is_signed_so_improvement_is_positive(values, maximize, expected):
    """The caller must not re-derive the sign from the metric — that
    re-derivation is what `ScoreEvent.maximize` exists to prevent."""
    summary = score_summary(_state(*values, maximize=maximize), BudgetConfig())

    assert summary.delta_vs_best == pytest.approx(expected)


def test_last_three_scores_read_in_the_order_they_happened():
    summary = score_summary(_state(1.0, 2.0, 3.0, 4.0, 5.0), BudgetConfig())

    assert summary.last_3_scores == [3.0, 4.0, 5.0]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((194.8,), 0),  # one reading cannot have stalled
        ((194.8, 190.9), 0),  # improved on the last one
        ((194.8, 195.0), 1),  # worse
        ((194.8, 195.0, 196.0), 2),  # worse twice
        ((194.8, 195.0, 190.0), 0),  # recovered
        ((190.0, 195.0, 194.0), 2),  # never beat the first
    ],
)
def test_steps_since_improvement_counts_experiments_not_steps(values, expected):
    """Compared against the best of the *preceding* readings. Against a
    running best that includes itself, every event ties its own best and
    nothing ever counts as an improvement."""
    assert score_summary(_state(*values), BudgetConfig()).steps_since_improvement == expected


def test_a_change_of_metric_does_not_leak_into_the_summary():
    """Readings either side of a metric change are on different scales, so a
    "best" across them would compare an RMSE against an accuracy."""
    state = BudgetState(
        score_events=_events(0.91, 0.93, metric="cv_accuracy", maximize=True)
        + _events(194.8, metric="cv_rmse", maximize=False)
    )

    summary = score_summary(state, BudgetConfig())

    assert summary.metric_name == "cv_rmse"
    assert summary.best_so_far == 194.8
    assert summary.last_3_scores == [194.8]


def test_the_newest_reading_settles_the_direction_for_the_window():
    """Flags within one metric can disagree: M8-2 falls back to the campaign's
    configured direction when the competition profile cannot answer, so an
    early experiment may carry a guess and a later one the resolved answer.

    Direction belongs to the metric rather than the reading, so the
    better-informed newest event decides — rather than splitting a series that
    measures a single thing into two windows.
    """
    guessed_then_resolved = BudgetState(
        score_events=[
            _events(194.8, maximize=True)[0],  # recorded before the spec existed
            _events(190.9, maximize=False)[0],  # recorded after `analyze` ran
        ]
    )

    summary = score_summary(guessed_then_resolved, BudgetConfig())

    assert summary.best_so_far == 190.9  # minimised, per the newest flag
    assert summary.steps_since_improvement == 0


def test_the_noise_floor_is_the_one_plateau_uses():
    """A gain smaller than `plateau_epsilon` is not an improvement, so the
    gate and the stop cannot disagree about what "no change" means."""
    config = BudgetConfig(plateau_epsilon=0.5)

    assert score_summary(_state(194.8, 194.5), config).steps_since_improvement == 1
    assert score_summary(_state(194.8, 194.0), config).steps_since_improvement == 0


# --- M8-4: the policy sees it -------------------------------------------


def test_the_observe_bundle_carries_the_progress_numbers(tmp_path: Path):
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        state = _state(194.8, 195.0, 196.0)
        session = store.create_session(
            "g", metadata={"budget_state": state.model_dump(mode="json")}
        )

        observe = build_observe_bundle(store, ws, session.id, include_context=False)

        assert observe["best_so_far"] == 194.8
        assert observe["last_3_scores"] == [194.8, 195.0, 196.0]
        assert observe["steps_since_improvement"] == 2
        assert observe["score_metric"] == "cv_rmse"
    finally:
        store.close()


def test_the_bundle_reflects_the_state_the_caller_is_acting_on(tmp_path: Path):
    """`decide_next` hands the same pair to the gate and to this bundle.

    If the bundle re-derived from the session instead, the model would read
    one campaign's numbers while the allowlist was built from another's —
    two sources for one number, which is the shape this milestone has already
    been bitten by three times.
    """
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        persisted = _state(197.0, 196.0, 195.0, 194.0)  # improving
        session = store.create_session(
            "g", metadata={"budget_state": persisted.model_dump(mode="json")}
        )
        in_hand = _state(194.8, 195.0, 196.0, 197.0)  # stagnant

        observe = build_observe_bundle(
            store, ws, session.id, include_context=False, budget_state=in_hand
        )

        assert observe["steps_since_improvement"] == 3
        assert observe["best_so_far"] == 194.8
    finally:
        store.close()


def test_the_bundle_keeps_its_shape_when_the_session_is_missing(tmp_path: Path):
    """Every other field here degrades to a value rather than disappearing.
    A consumer that subscripts a key present on every real session would
    otherwise raise only on the rare path — the one least exercised."""
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        observe = build_observe_bundle(store, ws, "S-does-not-exist", include_context=False)

        assert observe["best_so_far"] is None
        assert observe["last_3_scores"] == []
        assert observe["delta_vs_best"] is None
        assert observe["steps_since_improvement"] == 0
        assert observe["score_metric"] is None
    finally:
        store.close()


def test_a_campaign_with_no_scores_still_produces_a_usable_bundle(tmp_path: Path):
    """Observe must stay usable for every campaign, not only those that have
    already run an experiment."""
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")

        observe = build_observe_bundle(store, ws, session.id, include_context=False)

        assert observe["best_so_far"] is None
        assert observe["steps_since_improvement"] == 0
        assert observe["score_metric"] is None
    finally:
        store.close()


# --- M8-5: it changes a decision ----------------------------------------


def test_a_stagnant_campaign_reopens_evidence_gathering(tmp_path: Path):
    """The exit criterion is not "the field exists", it is "a decision
    changes". A backlog that keeps producing hypotheses which do not move the
    score is, at the allowlist level, the same as a thin or stale one."""
    ws = _stocked(_ws(tmp_path))
    config = BudgetConfig(plateau_window=3)

    stagnant = _state(194.8, 195.0, 196.0, 197.0)
    improving = _state(197.0, 196.0, 195.0, 194.0)

    ok_stagnant, reason = should_gather_evidence(ws, stagnant, config)
    ok_improving, _ = should_gather_evidence(ws, improving, config)

    assert ok_stagnant is True
    assert "no improvement" in reason
    assert ok_improving is False


def test_the_stagnant_gate_changes_the_allowlist(tmp_path: Path):
    """A pure code assertion, no LLM. Asserting on `decide_next`'s chosen
    action instead would only prove the offline fallback reacted, or require
    asserting on model output."""
    ws = _stocked(_ws(tmp_path))
    config = BudgetConfig(plateau_window=3)
    allowlist = {"analyze_competition", "run_plan"}

    stagnant = available_tools(ws, allowlist, _state(194.8, 195.0, 196.0, 197.0), config)
    improving = available_tools(ws, allowlist, _state(197.0, 196.0, 195.0, 194.0), config)

    assert "analyze_competition" in stagnant
    assert "analyze_competition" not in improving


def test_a_campaign_below_the_window_is_not_yet_stagnant(tmp_path: Path):
    """The gate must not fire on the first flat result — `plateau_window` is
    the same count the stop uses, so the two agree about how long is long."""
    ws = _stocked(_ws(tmp_path))
    config = BudgetConfig(plateau_window=3)

    ok, _ = should_gather_evidence(ws, _state(194.8, 195.0, 196.0), config)

    assert ok is False


def test_a_zero_window_does_not_make_every_campaign_stagnant(tmp_path: Path):
    """`plateau_window=0` would otherwise fire the gate on a campaign that has
    never run an experiment — `0 >= 0`. `evaluate_stops` normalises the same
    value with `max(1, ...)`; the gate has to agree or the two disagree about
    a config the user can set."""
    ws = _stocked(_ws(tmp_path))

    ok, reason = should_gather_evidence(ws, BudgetState(), BudgetConfig(plateau_window=0))

    assert ok is False, f"a zero window opened the gate on an empty series: {reason}"


def test_where_the_gate_sits_relative_to_the_plateau_stop(tmp_path: Path):
    """Pins the relationship between the two, because it is not the obvious one.

    They share `plateau_window` but measure different things: this counts
    experiments since the last record, `plateau` measures the spread of the
    last n readings. On a perfectly flat series `plateau` stops the campaign
    at n readings while the gate needs n+1, so on that path the gate never
    acts — acceptable only because `plateau` needs near-exact ties that real
    CV scores do not produce. On the realistic drifting series `plateau` never
    fires and the gate is the only signal.
    """
    ws = _stocked(_ws(tmp_path))
    config = BudgetConfig(plateau_window=3)

    flat_at_window = _state(194.8, 194.8, 194.8)
    assert evaluate_stops(config, _with_history(flat_at_window)) == "plateau"
    assert should_gather_evidence(ws, flat_at_window, config)[0] is False

    drifting = _state(194.8, 195.0, 196.0, 197.0)
    assert evaluate_stops(config, _with_history(drifting)) == "none"
    assert should_gather_evidence(ws, drifting, config)[0] is True


def test_no_series_is_not_the_same_answer_as_improving(tmp_path: Path):
    """A caller that passes no state must not read as "the campaign is
    improving" — the clause has nothing to say, and the other two decide."""
    ws = _stocked(_ws(tmp_path))

    assert should_gather_evidence(ws, None, None) == should_gather_evidence(ws)
    assert should_gather_evidence(ws, BudgetState(), BudgetConfig()) == should_gather_evidence(ws)
