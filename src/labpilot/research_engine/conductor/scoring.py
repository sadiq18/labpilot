"""Turn one execution into the comparable score a campaign ranks it by.

Extracted from `loop.py` (M11): the sequential loop asks this per step to
feed the budget series, and K-way fan-out's promotion asks it per branch to
pick a cohort winner. Both must resolve the same metric key and the same
direction — a second, ad-hoc resolver is how a cohort ends up ranked on a
different metric than the campaign is steering by.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from labpilot.research_engine.conductor.budgets import ScoreEvent
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
    (`BudgetConfig.maximize`, resolved once at session start). It is used only
    when the competition profile cannot answer, so the event agrees with the
    campaign rather than inventing a second opinion.

    Returns None — never a partial event — when the execution produced nothing
    comparable. Each reason is logged, because a silent skip here is
    invisible from outside and leaves the series quietly short.
    """
    from labpilot.research_engine.evidence.builder import (
        is_placeholder_metrics,
        metrics_as_experiment,
    )
    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.shared.experiments.comparator import (
        resolve_primary_metric_key_and_direction,
    )

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

    experiment = metrics_as_experiment(execution_id, workspace.competition, metrics)
    # The one competition-aware resolver, called with the single execution on
    # both sides: `shared` degenerates to this run's own metric keys, which is
    # the lookup wanted here. Using a second, ad-hoc resolver is how four of
    # them ended up disagreeing about the "primary" key.
    #
    # The search has to cover everywhere a spec is kept, not just the run
    # directory: `analyze` writes the knowledge copy under `paths.root`, and a
    # workspace with only that copy otherwise falls through to the
    # alphabetically-first metric — picking `cv_mae` over `cv_rmse` and
    # calling it primary.
    metric_name, _ = resolve_primary_metric_key_and_direction(
        experiment,
        experiment,
        competition_dirs=(
            workspace.effective_runs_dir / execution_id,
            workspace.root,
            paths.root,
        ),
    )
    if metric_name is None:
        logger.info("no resolvable primary metric for %s; no score recorded", execution_id)
        return None

    # The comparator's own direction flag is discarded: it defaults to `True`
    # when it finds no spec, so trusting it records "higher is better" for an
    # error metric — and the whole reason `maximize` travels with the value is
    # that the sign is not re-derived later.
    #
    # `None` is a real answer here. Rather than guess, fall back to the
    # direction the campaign is already running under, so the event and the
    # campaign cannot disagree.
    resolved = _direction_for(workspace, execution_id, paths)
    maximize = resolved if resolved is not None else fallback_maximize

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


def _direction_for(workspace: Workspace, execution_id: str, paths: Any) -> bool | None:
    """Whether this competition maximises its metric, or None if unknowable.

    Chooses *where* to look and leaves *how to read it* to `resolve_maximize`,
    which owns that question — the conductor must not answer it differently
    from the module that defines it.

    Two calls because `resolve_maximize` takes a nearest-first pair of
    directories, and there are three worth asking before the profile artifact.
    """
    from labpilot.research_engine.intelligence.competition.direction import resolve_maximize

    resolved = resolve_maximize(
        competition=workspace.competition,
        workspace_root=workspace.effective_runs_dir / execution_id,
        knowledge_root=workspace.root,
    )
    if resolved is not None:
        return resolved
    return resolve_maximize(
        competition=workspace.competition,
        workspace_root=paths.root,
        extracted_dir=paths.extracted_dir,
    )


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
