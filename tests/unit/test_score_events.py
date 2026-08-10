"""M8-1: the score series has somewhere durable to live, and a derived view.

`BudgetState.metric_history`/`.last_metric` were read in `evaluate_stops` and
the run loop but written nowhere (docs/research-os/autonomy-roadmap/design/02-objective-loop.md
§1.1) — a flat float list also has nowhere to carry `experiment_id`, which a
later hypothesis needs to cite by id. `score_events` is the structured series;
`metric_history`/`.last_metric` are recomputed from it on every persist, never
stepped, so a stale or manually-corrupted value never survives a write.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.conductor.budgets import (
    _SCORE_EVENTS_WARN_THRESHOLD,
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    budgets_from_metadata,
    budgets_to_metadata,
    recompute_metric_history,
)
from labpilot.research_engine.conductor.checkpoint import load_budget_pair, persist_budgets
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _event(experiment_id: str, value: float, **overrides: object) -> ScoreEvent:
    defaults: dict[str, object] = {
        "experiment_id": experiment_id,
        "hypothesis_id": "H-001",
        "technique": "target_encoding",
        "metric_name": "cv_rmse",
        "value": value,
        "maximize": False,
        "timestamp": "2026-08-11T00:00:00+00:00",
    }
    defaults.update(overrides)
    return ScoreEvent(**defaults)


def test_recompute_derives_metric_history_from_score_events():
    state = BudgetState(score_events=[_event("E-001", 194.80), _event("E-002", 190.97)])

    recompute_metric_history(state)

    assert state.metric_history == [194.80, 190.97]
    assert state.last_metric == 190.97


def test_recompute_overwrites_a_stale_value_rather_than_trusting_it():
    """Recompute, not step — a pre-existing wrong value must not survive.

    Proves the fix does what AGENTS.md rule 2 asks: derived state is a
    function of current inputs, so it stays correct even after those inputs
    are corrected. A test that only checked "metric_history is non-empty"
    could pass on a stepped counter that was merely never reset — this checks
    the actual value changes to match the current events.
    """
    state = BudgetState(
        metric_history=[999.0, 999.0],
        last_metric=999.0,
        score_events=[_event("E-001", 190.97)],
    )

    recompute_metric_history(state)

    assert state.metric_history == [190.97]
    assert state.last_metric == 190.97


def test_recompute_on_an_empty_series_clears_rather_than_leaves_stale():
    state = BudgetState(metric_history=[194.80], last_metric=194.80, score_events=[])

    recompute_metric_history(state)

    assert state.metric_history == []
    assert state.last_metric is None


def test_score_events_round_trip_through_session_metadata():
    """The nested model, not just floats, must survive the metadata blob."""
    config = BudgetConfig()
    state = BudgetState(
        score_events=[
            _event("E-001", 194.80, technique="target_encoding"),
            _event(
                "E-002",
                190.97,
                technique=None,
                combo_techniques=["mixup", "cutout"],
                hypothesis_id="H-002",
            ),
        ]
    )

    meta = budgets_to_metadata({}, config, state)
    restored_config, restored_state = budgets_from_metadata(meta)

    assert restored_config == config
    assert [e.experiment_id for e in restored_state.score_events] == ["E-001", "E-002"]
    assert restored_state.score_events[1].combo_techniques == ["mixup", "cutout"]
    assert restored_state.score_events[1].technique is None
    assert restored_state.score_events[1].hypothesis_id == "H-002"


def test_persist_budgets_derives_metric_history_before_writing(tmp_path: Path):
    """The recompute must actually run on the real write path, not just the
    helper function in isolation — this is what `persist_budgets` is for."""
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("beat baseline")
        config, state = load_budget_pair(session)
        state.score_events = [_event("E-001", 194.80), _event("E-002", 190.97)]
        # A stale value must not be trusted just because it was already there.
        state.metric_history = [1.0]
        state.last_metric = 1.0

        persist_budgets(store, session.id, config, state)

        reloaded_session = store.get_session(session.id)
        assert reloaded_session is not None
        _, reloaded_state = load_budget_pair(reloaded_session)
        assert reloaded_state.metric_history == [194.80, 190.97]
        assert reloaded_state.last_metric == 190.97
        assert [e.experiment_id for e in reloaded_state.score_events] == [
            "E-001",
            "E-002",
        ]
    finally:
        store.close()


def test_stagnation_mint_fired_defaults_false_and_round_trips():
    state = BudgetState(stagnation_mint_fired=True)

    meta = budgets_to_metadata({}, BudgetConfig(), state)
    _, restored = budgets_from_metadata(meta)

    assert BudgetState().stagnation_mint_fired is False
    assert restored.stagnation_mint_fired is True


def _warnings(caplog) -> list:
    return [r for r in caplog.records if "score_events" in r.message]


def test_score_events_past_the_warn_threshold_logs_once(caplog):
    """A runaway campaign must be visible, not just slower.

    `score_events` is deliberately not capped (truncating it would break
    citing an arbitrary prior experiment by id) — so past a generous
    threshold, recompute logs instead of silently degrading. Once, not every
    call after — a long campaign must not spam the log for every step past
    the threshold.
    """
    state = BudgetState(
        score_events=[_event(f"E-{i:03d}", float(i)) for i in range(_SCORE_EVENTS_WARN_THRESHOLD)]
    )

    with caplog.at_level("WARNING", logger="labpilot.research_engine.conductor.budgets"):
        recompute_metric_history(state)
        recompute_metric_history(state)
        recompute_metric_history(state)

    assert len(_warnings(caplog)) == 1
    assert state.score_events_warned is True


def test_the_warning_is_not_skipped_when_the_series_jumps_past_the_threshold(caplog):
    """The two prior attempts at this both broke here.

    A length-diff proxy for "already warned" (`grew and len(...) == threshold`)
    fires only if the series passes through the threshold one entry at a
    time. A resumed session that already starts above it, or several
    experiments logged before the next `persist_budgets` call, jumps straight
    past 500 without ever landing on it exactly — and both prior versions of
    this check silently never warned for the rest of that campaign. A plain
    latch (`score_events_warned`) has no "landed exactly on N" requirement to
    miss.
    """
    state = BudgetState(
        score_events=[
            _event(f"E-{i:03d}", float(i)) for i in range(_SCORE_EVENTS_WARN_THRESHOLD + 100)
        ]
    )

    with caplog.at_level("WARNING", logger="labpilot.research_engine.conductor.budgets"):
        recompute_metric_history(state)

    assert len(_warnings(caplog)) == 1


def test_the_warning_never_refires_once_the_latch_is_set(caplog):
    """Growing further past the threshold, or shrinking and regrowing through
    it, must not refire — the latch is a one-time notice, not a re-armable
    trigger keyed on the current length."""
    state = BudgetState(score_events=[], score_events_warned=True)

    with caplog.at_level("WARNING", logger="labpilot.research_engine.conductor.budgets"):
        state.score_events = [
            _event(f"E-{i:03d}", float(i)) for i in range(_SCORE_EVENTS_WARN_THRESHOLD)
        ]
        recompute_metric_history(state)

    assert _warnings(caplog) == []


def test_budgets_from_metadata_recomputes_on_load():
    """A read-only caller (`conduct_status`) must not trust a stale stored
    value just because nothing has written it back yet — the load path needs
    the same guarantee the write paths already have."""
    stale_meta = {
        "budgets": BudgetConfig().model_dump(),
        "budget_state": BudgetState(
            score_events=[_event("E-001", 194.80), _event("E-002", 190.97)],
            metric_history=[1.0],
            last_metric=1.0,
        ).model_dump(),
    }

    _, state = budgets_from_metadata(stale_meta)

    assert state.metric_history == [194.80, 190.97]
    assert state.last_metric == 190.97


def test_budget_metadata_recomputes_from_an_existing_session():
    """`_budget_metadata` (cli/conduct.py) is a second write path, not routed
    through `persist_budgets` — it must not skip the recompute it exists to
    guarantee, or a resumed session's `metric_history` goes stale the moment
    this is the function that writes it back out."""
    from labpilot.cli.conduct import _budget_metadata

    existing_state = BudgetState(
        score_events=[_event("E-001", 194.80), _event("E-002", 190.97)],
        metric_history=[1.0],
        last_metric=1.0,
    )
    existing = {"budget_state": existing_state.model_dump()}

    meta = _budget_metadata(
        max_submissions=None,
        max_wall_s=None,
        max_cost_usd=None,
        target_metric=None,
        target_value=None,
        plateau_window=3,
        maximize=None,
        existing=existing,
    )

    assert meta["budget_state"]["metric_history"] == [194.80, 190.97]
    assert meta["budget_state"]["last_metric"] == 190.97
