"""M17: a campaign ends on its objective, not on a step counter.

Three things are pinned here, and the first is the one with the widest blast
radius: **`plateau` must mean the same thing whatever units the metric is
measured in.** It did not. `plateau_epsilon` was an absolute `1e-6` compared
against a raw spread, so three readings within a whisker of each other was a
plateau on an accuracy near 0.9 and an impossibility on an RMSE near 1380 —
a domain assumption sitting in a control-plane stop. Harmless while
`max_steps` ended every campaign; load-bearing the moment it does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from labpilot.cli.main import app
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    budgets_to_metadata,
    evaluate_stops,
    goal_progress,
    score_summary,
)
from labpilot.research_engine.conductor.checkpoint import latest_active_session, load_budget_pair
from labpilot.research_engine.conductor.loop import (
    _STOP_SESSION_STATUS,
    _record_experiment_outcome,
    run_until_stop,
)
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace
from tests.helpers.cli import cli_runner


def _ws(tmp_path: Path, slug: str = "m17") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _events(*values: float, metric: str = "cv_rmse", maximize: bool = False) -> list[ScoreEvent]:
    return [
        ScoreEvent(experiment_id=f"E-{i:03d}", metric_name=metric, value=v, maximize=maximize)
        for i, v in enumerate(values, start=1)
    ]


def _state(*values: float, **kw) -> BudgetState:
    return BudgetState(score_events=_events(*values, **kw))


# -- the noise floor no longer depends on the metric's units ---------------


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1380.0])
def test_a_flat_series_plateaus_at_every_magnitude(scale: float) -> None:
    """The same three readings, moved up and down the number line.

    This is the test that would have caught the absolute epsilon. Under it,
    only `scale=1e-3` plateaued: the identical *shape* of series was a stop
    on one metric and invisible on another.
    """
    flat = [1.0 * scale, 1.0002 * scale, 1.0001 * scale]

    assert evaluate_stops(BudgetConfig(), BudgetState(metric_history=flat)) == "plateau"


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1380.0])
def test_a_moving_series_is_not_a_plateau_at_any_magnitude(scale: float) -> None:
    moving = [1.0 * scale, 0.9 * scale, 0.8 * scale]

    assert evaluate_stops(BudgetConfig(), BudgetState(metric_history=moving)) == "none"


def test_zeroing_the_relative_floor_restores_the_absolute_comparison() -> None:
    """The documented rollback for this change, exercised rather than asserted.

    `plateau_rel_epsilon=0` leaves `max(abs, 0 * scale) == abs`, which is the
    comparison the stop made before — so a spread that is large in absolute
    terms and tiny in relative ones stops being a plateau again.
    """
    flat_but_large = [1380.0, 1380.2, 1380.1]

    assert evaluate_stops(BudgetConfig(), BudgetState(metric_history=flat_but_large)) == "plateau"
    assert (
        evaluate_stops(
            BudgetConfig(plateau_rel_epsilon=0.0),
            BudgetState(metric_history=flat_but_large),
        )
        == "none"
    )


def test_a_series_straddling_zero_uses_the_absolute_floor() -> None:
    """Where the relative test degenerates, the absolute one still answers.

    Magnitude near zero makes `rel * scale` vanish, which is exactly when a
    relative-only floor would call every series a plateau.
    """
    near_zero_but_moving = [-1e-4, 0.0, 1e-4]

    state = BudgetState(metric_history=near_zero_but_moving)

    assert evaluate_stops(BudgetConfig(), state) == "none"


# -- needs_guidance, and where it sits ------------------------------------


def test_no_new_score_asks_for_guidance() -> None:
    config = BudgetConfig()
    state = BudgetState(steps_since_new_score=config.max_steps_without_score)

    assert evaluate_stops(config, state) == "needs_guidance"


def test_nothing_eligible_to_run_asks_for_guidance() -> None:
    state = BudgetState(consecutive_unmapped=3)

    assert evaluate_stops(BudgetConfig(), state) == "needs_guidance"


def test_a_stalled_campaign_is_not_reported_as_a_plateau() -> None:
    """Both conditions hold; only one of them is a claim about results.

    A campaign that has written no score for the whole window has a flat one
    for the trivial reason that nothing wrote to it. Calling that `plateau` is
    a stop asserting something it never measured.
    """
    config = BudgetConfig()
    stalled_and_flat = BudgetState(
        steps_since_new_score=config.max_steps_without_score,
        metric_history=[1380.0, 1380.0, 1380.0],
    )

    assert evaluate_stops(config, stalled_and_flat) == "needs_guidance"


def test_a_campaign_that_reached_its_goal_is_finished_not_stuck() -> None:
    reached_and_stalled = BudgetState(
        last_metric=0.91,
        steps_since_new_score=BudgetConfig().max_steps_without_score,
        score_events=_events(0.91, metric="cv_auc", maximize=True),
    )
    config = BudgetConfig(target_metric="auc", target_value=0.9, maximize=True)

    assert evaluate_stops(config, reached_and_stalled) == "metric_target"


@pytest.mark.parametrize(
    "config",
    [
        BudgetConfig(max_steps_without_score=None),
        BudgetConfig(max_consecutive_unmapped=None),
    ],
)
def test_either_guidance_threshold_can_be_switched_off(config: BudgetConfig) -> None:
    """Same opt-out the M20 breaker has, for the same reason: a campaign
    deliberately probing a broken workspace is a legitimate thing to run."""
    state = BudgetState(
        steps_since_new_score=99 if config.max_steps_without_score is None else 0,
        consecutive_unmapped=99 if config.max_consecutive_unmapped is None else 0,
    )

    assert evaluate_stops(config, state) == "none"


# -- goal_progress ---------------------------------------------------------


def test_the_progress_line_reads_as_the_plan_specified() -> None:
    config = BudgetConfig(target_metric="rmse", target_value=5.0, maximize=False)

    line = goal_progress(config, _state(199.9, 150.0, 120.0))

    assert line == (
        "goal cv_rmse: best 120 → target 5 · 41% closed · 3 result(s) · 0 since improvement"
    )


def test_a_campaign_with_no_result_yet_says_so() -> None:
    config = BudgetConfig(target_metric="rmse", target_value=2.236)

    assert goal_progress(config, BudgetState()) == "goal rmse: no result yet · target 2.236"


def test_a_campaign_with_no_target_reports_progress_without_one() -> None:
    line = goal_progress(BudgetConfig(), _state(194.8, 190.97, 192.0))

    assert line == "goal cv_rmse: best 190.97 · 3 result(s) · 1 since improvement"


def test_nothing_to_report_renders_nothing() -> None:
    assert goal_progress(BudgetConfig(), BudgetState()) is None


def test_a_first_result_past_the_target_is_not_a_percentage() -> None:
    config = BudgetConfig(target_metric="rmse", target_value=5.0, maximize=False)

    assert "target met at first result" in goal_progress(config, _state(4.0, 3.0))


def test_a_target_for_another_metric_is_not_shown() -> None:
    """`lb_auc` and `cv_rmse` are not the same question. Putting a threshold
    for one beside a reading of the other is the mistake
    `_last_metric_matches_target` keeps the stop from making, and worse here,
    because a person reads this line."""
    config = BudgetConfig(target_metric="lb_auc", target_value=0.9)

    line = goal_progress(config, _state(194.8, 190.9))

    assert "target" not in line


def test_direction_comes_from_the_series_not_the_config() -> None:
    """M12's trap: `maximize` is a competition property today and must become
    a property of the result. `ScoreEvent` is where it travels, so a config
    that disagrees with the series does not get to invert the arithmetic."""
    config = BudgetConfig(target_metric="auc", target_value=0.95, maximize=False)

    line = goal_progress(config, _state(0.50, 0.70, 0.95, metric="cv_auc", maximize=True))

    assert "best 0.95" in line and "100% closed" in line


def test_two_different_series_render_two_different_lines() -> None:
    """M15's contract test. A renderer that ignores its input is the hollow
    layer this roadmap is named for."""
    config = BudgetConfig(target_metric="rmse", target_value=5.0, maximize=False)

    assert goal_progress(config, _state(199.9, 120.0)) != goal_progress(config, _state(199.9, 60.0))


def test_the_line_names_nothing_kaggle_specific() -> None:
    """A campaign scored by a benchmark harness or a simulator reads the same
    as one scored by a competition (M12)."""
    line = goal_progress(BudgetConfig(), _state(194.8, 190.9))

    assert not {"submission", "leaderboard", "competition", "kaggle"} & set(line.lower().split())


# -- the counter is keyed on the series, not on a tool list ---------------


def _outcome(ws: Workspace, execution_id: str, metrics: dict) -> None:
    paths = ResearchPaths(ws.knowledge_dir, ws.competition)
    out = paths.executions_dir / execution_id / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "execution_outcome.json").write_text(
        json.dumps(
            {
                "competition": ws.competition,
                "execution_id": execution_id,
                "plan_id": "P-001",
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )


def test_steps_since_new_score_resets_only_when_the_series_grows(tmp_path: Path) -> None:
    """The defect this counter exists for, asserted directly.

    Both calls report a *successful* execution. One writes a comparable score
    and one writes an unusable metric — which is the case `steps_since_success`
    cannot see, because it resets on the success either way.
    """
    ws = _ws(tmp_path)
    _outcome(ws, "E-001", {"cv_rmse": 194.80})
    _outcome(ws, "E-002", {"notes": "no numbers here"})
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g", metadata={"budget_state": {"steps_since_new_score": 4}})

        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-002"
        )
        _, stalled = load_budget_pair(store.get_session(session.id))
        assert stalled.steps_since_new_score == 4, "a success with no score is not progress"
        assert stalled.steps_since_success == 0, "the older counter reset — this is the gap"

        _record_experiment_outcome(
            store, session.id, succeeded=True, workspace=ws, execution_id="E-001"
        )
        _, scored = load_budget_pair(store.get_session(session.id))
        assert scored.steps_since_new_score == 0
    finally:
        store.close()


# -- the loop itself -------------------------------------------------------


def _echo(workspace: Workspace, **kwargs: object) -> ToolResult:
    return ToolResult(ok=True, summary="ok")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name in ("analyze_competition", "query_memory", "reflect"):
        reg.register(ToolDescriptor(name=name, handler=_echo))
    return reg


def _always_chooses(tool: str, *, ceiling: int = 20):
    """A policy that keeps asking for one tool, with a ceiling that raises.

    An unbounded loop that fails to stop is a hang, not an assertion failure,
    and a hang in a suite with no per-test timeout is indistinguishable from
    an infrastructure problem. The ceiling turns it back into a test result.
    """
    calls = {"n": 0}

    def _decide(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > ceiling:
            raise AssertionError(f"unbounded loop ran {calls['n']} steps without stopping")
        return SimpleNamespace(tool=tool, stop=False, rationale="test"), {}

    return _decide


def test_an_unbounded_campaign_stops_when_nothing_is_eligible(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "unbounded")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("win")

        with patch(
            "labpilot.research_engine.conductor.loop.decide_next",
            _always_chooses("no_such_tool"),
        ):
            decisions = run_until_stop(
                store,
                ws,
                session.id,
                _registry(),
                llm_client=object(),
                max_steps=None,
                auto_approve=True,
                autonomy=1,
            )

        assert any(d.stop and "needs_guidance" in (d.rationale or "") for d in decisions)
        assert store.get_session(session.id).status == "paused"
    finally:
        store.close()


def test_a_guidance_pause_is_resumable_without_naming_the_session(tmp_path: Path) -> None:
    """`conduct continue` resolves the latest active session. A stop that
    parked the campaign anywhere else would need `--session` to pick up."""
    ws = _ws(tmp_path, "resumable")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("win")

        with patch(
            "labpilot.research_engine.conductor.loop.decide_next",
            _always_chooses("no_such_tool"),
        ):
            run_until_stop(
                store,
                ws,
                session.id,
                _registry(),
                llm_client=object(),
                max_steps=None,
                auto_approve=True,
                autonomy=1,
            )

        assert latest_active_session(store) is not None
        suggestions = store.list_suggestions(session.id)
        assert any(s.kind == "needs_guidance" for s in suggestions)
        # A guidance pause is not a missing capability, and must not be
        # counted as one.
        assert store.get_metrics(session.id).no_capability == 3

        # …and resuming it must actually run. Asserting the session is merely
        # *findable* passed while `conduct continue` was a no-op: the counters
        # that tripped the stop are persisted, so the resumed run re-fired it
        # before dispatching anything.
        store.update_session_status(session.id, "running")
        with patch(
            "labpilot.research_engine.conductor.loop.decide_next",
            _always_chooses("analyze_competition"),
        ):
            resumed = run_until_stop(
                store,
                ws,
                session.id,
                _registry(),
                llm_client=object(),
                max_steps=3,
                auto_approve=True,
                autonomy=1,
            )

        assert [d.tool_name for d in resumed if d.tool_name], "resumed run dispatched nothing"
        assert not any(d.stop for d in resumed), "resumed run re-fired the stop it was resuming"
    finally:
        store.close()


def test_a_bounded_run_still_stops_on_max_steps(tmp_path: Path) -> None:
    """The `for/else` this replaced is where `max_steps` used to be reported."""
    ws = _ws(tmp_path, "bounded")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("win")

        with patch(
            "labpilot.research_engine.conductor.loop.decide_next",
            _always_chooses("analyze_competition"),
        ):
            run_until_stop(
                store,
                ws,
                session.id,
                _registry(),
                llm_client=object(),
                max_steps=2,
                auto_approve=True,
                autonomy=1,
            )

        assert store.get_session(session.id).status == "paused"
        checkpoint = store.get_session(session.id).metadata["checkpoint"]
        assert checkpoint["stop_reason"] == "max_steps"
    finally:
        store.close()


def test_the_barren_breaker_still_fires_before_the_guidance_stop() -> None:
    """A score append also resets `steps_since_success`, so the no-score
    counter is never behind it. Set below `max_barren_steps` it therefore
    fires first on every campaign and M20's `failing` becomes unreachable —
    a broken campaign parked in `paused` reading like a normal end.
    """
    config = BudgetConfig()
    assert config.max_steps_without_score > config.max_barren_steps

    stops = [
        evaluate_stops(config, BudgetState(steps_since_success=n, steps_since_new_score=n))
        for n in range(1, config.max_steps_without_score + 1)
    ]

    assert "failing" in stops
    assert stops.index("failing") < (
        stops.index("needs_guidance") if "needs_guidance" in stops else len(stops)
    )


def test_the_no_score_counter_catches_what_the_barren_breaker_cannot() -> None:
    """The case the margin is for: executions keep succeeding — so
    `steps_since_success` keeps resetting and `failing` never fires — while
    every one of them writes a placeholder metric the score writer skips."""
    succeeding_but_scoring_nothing = BudgetState(steps_since_success=0, steps_since_new_score=10)

    assert evaluate_stops(BudgetConfig(), succeeding_but_scoring_nothing) == "needs_guidance"


# -- an improvement and a plateau are different questions ------------------


def test_a_campaign_improving_every_run_is_not_stagnant() -> None:
    """The plateau band is 0.1%; a real gain can be smaller than that.

    Sharing one band made this series — four accuracy readings, each beating
    the last — report three experiments with no improvement, which is what
    `available_tools`' stagnant clause and the stagnation mint read. The
    campaign was working and was told it was stuck.
    """
    improving = _state(0.9100, 0.9105, 0.9110, 0.9115, metric="cv_accuracy", maximize=True)

    assert score_summary(improving, BudgetConfig()).steps_since_improvement == 0


def test_a_genuinely_flat_series_is_still_stagnant() -> None:
    flat = _state(0.91, 0.91, 0.91, 0.91, metric="cv_accuracy", maximize=True)

    assert score_summary(flat, BudgetConfig()).steps_since_improvement == 3


def test_the_two_bands_are_read_from_their_own_fields() -> None:
    """Widening one must not move the other. A single shared field is how the
    plateau fix reached the policy's view of progress in the first place."""
    wide_plateau = BudgetConfig(plateau_rel_epsilon=0.5)
    improving = _state(0.9100, 0.9105, metric="cv_accuracy", maximize=True)

    assert score_summary(improving, wide_plateau).steps_since_improvement == 0
    assert evaluate_stops(wide_plateau, BudgetState(metric_history=[0.91, 0.92, 0.93])) == "plateau"


# -- what `conduct status` still has to say --------------------------------


def _status(tmp_path: Path, slug: str, metadata: dict) -> str:
    ws = _ws(tmp_path, slug)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        store.create_session("beat the baseline", metadata=metadata)
    finally:
        store.close()
    result = cli_runner().invoke(
        app, ["conduct", "status", "--workspace", str(ws.root), "-c", ws.competition]
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_status_still_reports_the_value_the_target_stop_compares(tmp_path: Path) -> None:
    """`metric_target` fires on `last_metric`, and the goal line shows `best`.

    They differ the moment a run regresses, so dropping the raw field left the
    number the stop is actually evaluated against displayed nowhere.
    """
    config = BudgetConfig(target_metric="rmse", target_value=5.0, maximize=False)
    state = _state(199.9, 120.0, 150.0)
    state.last_metric = 150.0

    out = _status(tmp_path, "raw", budgets_to_metadata({}, config, state))

    assert "last_metric=150.0" in out
    assert "best 120" in out


def test_status_reports_a_session_that_predates_the_score_series(tmp_path: Path) -> None:
    """The back-compat case `_record_experiment_outcome` deliberately keeps:
    readings in `metric_history`, an empty `score_events`, no target. The goal
    line has nothing to render, so the raw field is the only thing that can
    report the campaign has a metric at all."""
    legacy = BudgetState(metric_history=[0.5, 0.6], last_metric=0.6)

    out = _status(tmp_path, "legacy", budgets_to_metadata({}, BudgetConfig(), legacy))

    assert "last_metric=0.6" in out
    assert "goal " not in out


def test_the_stop_status_map_cannot_be_rewritten() -> None:
    """A lookup table that decides how every campaign's end is recorded, held
    the way `_EXPERIMENT_TOOLS` beside it is held."""
    with pytest.raises(TypeError):
        _STOP_SESSION_STATUS["failing"] = "completed"  # type: ignore[index]


# -- the rule the thresholds follow ---------------------------------------


def test_no_threshold_this_milestone_adds_is_in_domain_units() -> None:
    """A threshold may be a count of decisions. It may never be a quantity in
    domain units — seconds, currency, or the metric's own magnitude.

    Six steps without a score means the same thing whether a step takes four
    seconds or four hours; six *hours* does not. The pre-existing `max_wall_s`
    and `max_cost_usd` are the contrast, and they are operator-supplied with
    no default for exactly this reason.
    """
    added = {
        "max_steps_without_score",
        "max_consecutive_unmapped",
        "plateau_rel_epsilon",
        "improvement_rel_epsilon",
    }
    fields = BudgetConfig.model_fields

    for name in added:
        assert name in fields, f"{name} is no longer a field — update this guard"
        assert not name.endswith(("_s", "_usd", "_seconds", "_ms")), name

    counts = added - {"plateau_rel_epsilon", "improvement_rel_epsilon"}
    for name in counts:
        assert isinstance(getattr(BudgetConfig(), name), int)

    # A ratio is dimensionless by construction; the proof it carries no units
    # is the scale-invariance test at the top of this file.
    assert isinstance(BudgetConfig().plateau_rel_epsilon, float)
    assert isinstance(BudgetConfig().improvement_rel_epsilon, float)
    assert BudgetConfig().max_wall_s is None and BudgetConfig().max_cost_usd is None
