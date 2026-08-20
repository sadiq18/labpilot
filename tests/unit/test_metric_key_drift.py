"""A correct score, recorded under a name its target cannot match.

Measured on a live campaign 2026-08-20. `gemini-pro-latest` produced a real
result against a target of 2.0:

    {"cv_score": 1.6523199050168489, "metric": "rmse"}

1.65 is inside the target and `metric_target` could not fire, because the
series recorded `cv_score` and `metric_names_match('cv_score', 'rmse')` is
False. The metric's identity was in the payload the whole time — in a sibling
field that `metrics_as_experiment`'s numeric filter drops — while the resolver
fell through to "the first shared metric key (sorted)".

M17's plan named this hazard ("resolution order should be `cv_<target>` →
`<target>` → generic fallbacks"); the drift happens at *emission*, upstream of
the matching logic built to absorb it.
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
from labpilot.research_engine.conductor.scoring import score_event_for
from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    name_self_declared_metrics,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace

#: Verbatim from the campaign that surfaced this.
LIVE_PAYLOAD = {"cv_score": 1.6523199050168489, "metric": "rmse"}


def _ws(tmp_path: Path, slug: str = "drift") -> Workspace:
    return Workspace.from_client(scaffold_workspace(tmp_path / slug, slug)).ensure_roots()


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


def _competition(ws: Workspace, execution_id: str, key: str, direction: str) -> None:
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


# -- the rename itself ------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (LIVE_PAYLOAD, "cv_rmse"),
        ({"score": 0.91, "metric_name": "accuracy"}, "accuracy"),
        ({"val_score": 1.0, "metric": "rmse"}, "val_rmse"),
    ],
)
def test_a_generic_key_takes_the_name_the_payload_declares(payload, expected) -> None:
    assert expected in name_self_declared_metrics(payload)


def test_the_declaration_field_is_not_itself_renamed() -> None:
    """`{"metric": "rmse"}` must not become `{"rmse": "rmse"}` — a string under
    a metric's name is something every consumer would have to defend against."""
    renamed = name_self_declared_metrics(LIVE_PAYLOAD)

    assert renamed["metric"] == "rmse"
    assert renamed["cv_rmse"] == pytest.approx(1.6523199050168489)


def test_a_specific_key_is_left_alone() -> None:
    """This closes a gap; it does not second-guess a run that was specific."""
    assert name_self_declared_metrics({"cv_rmse": 1.65}) == {"cv_rmse": 1.65}


def test_an_existing_canonical_key_is_never_clobbered() -> None:
    """Two different numbers under one name is worse than one under a vague one."""
    both = {"cv_score": 1.0, "cv_rmse": 2.0, "metric": "rmse"}

    assert name_self_declared_metrics(both)["cv_rmse"] == 2.0


def test_an_unknown_declared_name_changes_nothing() -> None:
    payload = {"cv_score": 1.0, "metric": "not a metric anyone catalogued"}

    assert name_self_declared_metrics(payload) == payload


def test_non_metric_numbers_keep_their_names() -> None:
    renamed = name_self_declared_metrics({"cv_score": 1.0, "train_time_s": 3.2, "metric": "rmse"})

    assert renamed["train_time_s"] == 3.2


# -- what it unblocks -------------------------------------------------------


def test_the_live_payload_now_resolves_to_the_competition_metric(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    _outcome(ws, "E-001", LIVE_PAYLOAD)
    _competition(ws, "E-001", "rmse", "minimize")

    event = score_event_for(ws, "E-001", fallback_maximize=False)

    assert event is not None
    assert event.metric_name == "cv_rmse"
    assert event.value == pytest.approx(1.6523199050168489)


def test_the_campaign_that_reached_its_goal_can_now_stop_on_it(tmp_path: Path) -> None:
    """The whole point. 1.65 against a target of 2.0 is a met objective, and
    before this it could not end the campaign."""
    ws = _ws(tmp_path, "reached")
    _outcome(ws, "E-001", LIVE_PAYLOAD)
    _competition(ws, "E-001", "rmse", "minimize")
    event = score_event_for(ws, "E-001", fallback_maximize=False)
    assert event is not None

    config = BudgetConfig(target_metric="rmse", target_value=2.0, maximize=False)
    state = BudgetState(score_events=[event], last_metric=event.value)

    assert metric_names_match(event.metric_name, "rmse")
    assert evaluate_stops(config, state) == "metric_target"


def test_a_generic_key_with_no_declaration_still_cannot_answer_a_target() -> None:
    """The guard stays: a reading that names no metric must not satisfy a
    threshold for one. This closes the drift, not the check."""
    anonymous = ScoreEvent(
        experiment_id="E-001", metric_name="cv_score", value=1.65, maximize=False
    )
    config = BudgetConfig(target_metric="rmse", target_value=2.0, maximize=False)

    assert evaluate_stops(config, BudgetState(score_events=[anonymous], last_metric=1.65)) == "none"
