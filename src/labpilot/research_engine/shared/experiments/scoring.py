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
from typing import Any

from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


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
    """
    from labpilot.research_engine.evidence.builder import metrics_as_experiment
    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.shared.experiments.comparator import (
        resolve_primary_metric_key_and_direction,
    )

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
        return None, fallback_maximize

    # The comparator's own direction flag is discarded: it defaults to `True`
    # when it finds no spec, so trusting it records "higher is better" for an
    # error metric — and the whole reason `maximize` travels with the value is
    # that the sign is not re-derived later.
    #
    # `None` is a real answer here. Rather than guess, fall back to the
    # direction the caller is already running under.
    resolved = _direction_for(workspace, execution_id, paths)
    return metric_name, resolved if resolved is not None else fallback_maximize


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
