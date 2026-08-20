"""Which metric a competition is judged on, and which way is better.

Lives under `shared/` rather than in the conductor because two layers need
the same answer: the conductor turns it into the budget series' `ScoreEvent`,
and M11's promotion ranks a fan-out cohort by it. Specialists may not import
the conductor (`test_agents_do_not_import_conductor`), so a shared home is
what lets both ask the same resolver instead of writing a second one — which
is how four disagreeing "primary metric" lookups happened before.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


def comparable_metric_value(metrics: Any, metric_key: str) -> float | None:
    """This run's score on `metric_key`, or None if it is not comparable.

    One definition of "comparable", because every consumer that grew its own
    kept a different subset and the missing one was always the placeholder
    check. Refuses, in order:

    * metrics from a run that never trained a model — the marker is explicit
      and reading past it is how a stub's 0.5 was compared against a real
      run's 194.80 (`evidence/builder.py`, measured rogii 2026-08-07);
    * a `bool`, which is an `int` in Python and would rank as 1.0;
    * anything non-numeric, or NaN/inf — every NaN comparison is False, so an
      admitted NaN wins a `min()` by default rather than losing.
    """
    from labpilot.research_engine.evidence.builder import is_placeholder_metrics
    from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
        name_self_declared_metrics,
    )

    if not isinstance(metrics, dict) or is_placeholder_metrics(metrics):
        return None
    # The same renaming `metrics_as_experiment` applies, because `metric_key`
    # was resolved against those canonical names. Reading the raw dict alone
    # would miss every self-declared payload — promotion would find no
    # comparable value and decline a cohort it could rank.
    #
    # Only when the direct lookup misses, though: this runs once per candidate
    # inside promotion's filter, and renaming builds a fresh dict every call.
    # A payload that already names its metric — the common case — pays nothing.
    if metric_key in metrics:
        raw = metrics[metric_key]
    else:
        raw = name_self_declared_metrics(metrics).get(metric_key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def resolve_metric_and_direction(
    workspace: Workspace,
    execution_id: str,
    metrics: dict[str, Any],
    *,
    fallback_maximize: bool = True,
) -> tuple[str | None, bool]:
    """`(metric_key, maximize)` for this execution's metrics.

    `metric_key` is None when nothing in `metrics` resolves to the
    competition's primary metric — callers must treat that as "not
    comparable" rather than picking a key themselves.

    `fallback_maximize` is the direction the campaign is already running
    under. It is used only when the competition profile cannot answer, so
    every caller agrees with the campaign rather than inventing a second
    opinion.

    A caller with no such direction to fall back on must use
    :func:`resolve_metric_key_and_optional_direction` and decline, rather than
    letting this function's default stand in for an answer nobody has.
    """
    metric_name, resolved = resolve_metric_key_and_optional_direction(
        workspace, execution_id, metrics
    )
    return metric_name, resolved if resolved is not None else fallback_maximize


def resolve_metric_key_and_optional_direction(
    workspace: Workspace,
    execution_id: str,
    metrics: dict[str, Any],
) -> tuple[str | None, bool | None]:
    """`(metric_key, maximize)` where `maximize` is None if truly unknowable.

    The honest shape of the answer. `resolve_metric_and_direction` collapses
    the `None` into a caller-supplied default, which is right for the
    conductor — it is already steering by `BudgetConfig.maximize` and wants
    every reading to agree with that — and wrong for anyone without such a
    direction of their own.

    Promotion is the second kind of caller, and taking the collapsed form's
    default meant assuming "higher is better" for a metric it could not
    classify: measured on a two-branch cohort, `cv_rmse` 0.50 was promoted over
    0.20. Distinguishing "maximise" from "nobody knows" is what lets it decline
    instead, and a cohort with no verdict is worth far more than one with the
    wrong winner.
    """
    from labpilot.research_engine.evidence.builder import (
        is_placeholder_metrics,
        metrics_as_experiment,
    )
    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.shared.experiments.comparator import (
        resolve_primary_metric_key_and_direction,
    )

    if is_placeholder_metrics(metrics):
        # A run that never trained a model has no key worth agreeing on, and
        # letting it name the cohort's metric would be that run steering a
        # comparison it cannot take part in.
        return None, None

    paths = ResearchPaths(workspace.knowledge_dir, workspace.competition)
    experiment = metrics_as_experiment(execution_id, workspace.competition, metrics)
    # The one competition-aware resolver, called with the single execution on
    # both sides: `shared` degenerates to this run's own metric keys, which is
    # the lookup wanted here.
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
        return None, None

    # The comparator's own direction flag is discarded: it defaults to `True`
    # when it finds no spec, so trusting it records "higher is better" for an
    # error metric — and the whole reason `maximize` travels with the value is
    # that the sign is not re-derived later.
    #
    # `None` is a real answer here, and it is returned as one — the caller
    # decides whether it has a direction to fall back on or should decline.
    return metric_name, _direction_for(workspace, execution_id, paths)


def _direction_for(workspace: Workspace, execution_id: str, paths: Any) -> bool | None:
    """Whether this competition maximises its metric, or None if unknowable.

    Chooses *where* to look and leaves *how to read it* to `resolve_maximize`,
    which owns that question — no caller may answer it differently from the
    module that defines it.

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
