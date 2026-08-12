"""Turn one execution into the `ScoreEvent` the budget series is built from.

Extracted from `loop.py` (M11). The competition-aware part — which metric,
which direction — lives in `shared/experiments/scoring.py` because M11's
promotion needs the same answer and specialists may not import the conductor.
What stays here is the conductor's own share: reading the execution outcome
and deciding what counts as a comparable score.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from labpilot.research_engine.conductor.budgets import ScoreEvent
from labpilot.research_engine.shared.experiments.scoring import resolve_metric_and_direction
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


def score_event_for(
    workspace: Workspace, execution_id: str, *, fallback_maximize: bool = True
) -> ScoreEvent | None:
    """The comparable score this execution produced, or None with a reason logged.

    Reads `execution_outcome.json` for *this* execution id rather than the
    `metrics.json` at the workspace root. The root file survives a failed run,
    so "is there a file?" and "did this run write one?" are different
    questions — `run_experiment` needed an explicit freshness guard for
    exactly that. Keyed by execution id, the outcome artifact cannot belong to
    a different run.

    `fallback_maximize` is the direction the campaign is already running under
    (`BudgetConfig.maximize`, resolved once at session start).

    Returns None — never a partial event — when the execution produced nothing
    comparable. Each reason is logged, because a silent skip here is
    invisible from outside and leaves the series quietly short.
    """
    from labpilot.research_engine.evidence.builder import (
        is_placeholder_metrics,
        metrics_as_experiment,
    )
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(workspace.knowledge_dir, workspace.competition)
    outcome_path = paths.executions_dir / execution_id / "artifacts" / "execution_outcome.json"
    if not outcome_path.is_file():
        # The specialist `run_experiment` path writes no execution outcome, so
        # this is the ordinary way a non-`run_plan` experiment lands here.
        logger.info("no execution outcome for %s; no score recorded", execution_id)
        return None
    try:
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable execution outcome for %s; no score recorded", execution_id)
        return None
    if not isinstance(outcome, dict):
        # A truncated or half-written file can still parse — as `null`, a
        # list, a bare string. Letting that raise here would surface as a
        # dispatch error and record a *successful* experiment as a failure
        # against the circuit breaker.
        logger.warning("malformed execution outcome for %s; no score recorded", execution_id)
        return None

    metrics = outcome.get("metrics")
    if not isinstance(metrics, dict):
        logger.info("execution %s recorded no metrics; no score recorded", execution_id)
        return None
    if is_placeholder_metrics(metrics):
        # A run that never trained a model has no score to compare, for the
        # same reason it must not reach an evidence card.
        logger.info("execution %s produced placeholder metrics; no score recorded", execution_id)
        return None

    metric_name, maximize = resolve_metric_and_direction(
        workspace, execution_id, metrics, fallback_maximize=fallback_maximize
    )
    if metric_name is None:
        logger.info("no resolvable primary metric for %s; no score recorded", execution_id)
        return None

    experiment = metrics_as_experiment(execution_id, workspace.competition, metrics)
    hypothesis_id = outcome.get("hypothesis_id") or None
    technique, combo = _techniques_for(workspace, hypothesis_id)
    try:
        return ScoreEvent(
            experiment_id=execution_id,
            hypothesis_id=hypothesis_id,
            technique=technique,
            combo_techniques=combo,
            metric_name=metric_name,
            value=float(experiment.metrics[metric_name]),
            maximize=maximize,
        )
    except ValidationError:
        # `ScoreEvent` refuses a non-finite value: a diverged run's NaN is not
        # a comparable score, and admitting one would silently disable the
        # plateau and metric_target stops that read this series.
        logger.info(
            "execution %s scored %r on %s, which is not a comparable value; no score recorded",
            execution_id,
            metrics.get(metric_name),
            metric_name,
        )
        return None


def _techniques_for(
    workspace: Workspace, hypothesis_id: str | None
) -> tuple[str | None, list[str]]:
    """`(technique, combo_techniques)` for the hypothesis under test.

    `combo_techniques`, not `technique_stack`: the stack is cumulative
    lineage, so a five-generation chain would name every ancestor for a change
    that tested one thing.
    """
    if not hypothesis_id:
        return None, []
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    try:
        hypothesis = HypothesisStore(workspace.knowledge_dir, workspace.competition).get(
            hypothesis_id
        )
        if hypothesis is None:
            return None, []
        # Inside the guard with the lookup: reading the fields is as able to
        # fail as fetching them, and an escape here does not stay local —
        # `_record_experiment_outcome` runs inside the dispatch try block, so
        # it would land as a dispatch error and count a *successful*
        # experiment against the circuit breaker.
        return hypothesis.technique, list(hypothesis.combo_techniques)
    except Exception:  # noqa: BLE001 — a missing hypothesis must not lose the score
        logger.info("cannot read hypothesis %s; recording the score without it", hypothesis_id)
        return None, []
