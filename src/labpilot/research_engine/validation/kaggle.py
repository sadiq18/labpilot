"""The validator this repo has always had, given its name.

Nothing here is new behaviour. Every source it reads — `metrics.json`, the
registry-ordered primary-metric search, `resolve_maximize` over
`competition.json` and the Analyze profile artifact — is the path
`build_evidence_card` already took. The only change is that the three facts
come back as one object that states its own direction, instead of three
arguments the card assembles from three different places.

That matters for the *next* validator rather than this one: a benchmark harness
has no `competition.json`, so under the old arrangement it had nowhere to put a
direction it already knew.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.builder import _primary_cv_keyed
from labpilot.research_engine.intelligence.competition.direction import resolve_maximize
from labpilot.research_engine.validation.models import ValidationResult

SOURCE = "kaggle_cv"


def result_from_metrics(
    metrics: dict[str, Any],
    *,
    maximize: bool | None,
    secondary: float | None = None,
    artifacts: dict[str, str] | None = None,
) -> ValidationResult:
    """Wrap an already-loaded metrics blob, without re-deciding anything.

    Split out from `KaggleCvValidator.validate` because the control side of a
    comparison is a blob recovered from a stored artifact, not a run anybody is
    about to perform. Both sides must be described the same way or the
    comparison is between two different kinds of thing.
    """
    found = _primary_cv_keyed(metrics)
    provenance: list[str] = []
    if found is None:
        provenance.append("no primary metric found in the metrics blob")
    else:
        provenance.append(f"primary metric {found[1]!r} read from the metrics blob")
    if maximize is None:
        provenance.append("direction unresolved")
    else:
        provenance.append(f"direction {'maximize' if maximize else 'minimize'} from the workspace")

    return ValidationResult(
        score=found[0] if found else None,
        metric=found[1] if found else "",
        direction=None if maximize is None else ("maximize" if maximize else "minimize"),
        source=SOURCE,
        provenance=provenance,
        artifacts=dict(artifacts or {}),
        raw=dict(metrics),
        secondary=secondary,
    )


def _workspace_root(workspace: Any) -> Path:
    """The directory a run wrote into.

    `getattr(workspace, "root", workspace)` reads as "use `.root` if this is a
    Workspace, else treat it as a path" and is not: `pathlib.Path` *has* a `root`
    property, and it returns ``"/"``. Passing the obvious thing — a `Path`, which
    is what `TaskContext.workspace_root` holds — made the validator read
    ``/metrics.json`` and return a scoreless result with no error.

    Asking the type rather than probing for an attribute both types happen to
    have is the whole fix. `TaskContext` also carries `workspace_root`, so
    `workspace` is a redundant parameter; it stays because the protocol signature
    is fixed by the plan, and preferring the context here would only add a branch
    that answers identically.
    """
    if isinstance(workspace, str | Path):
        return Path(workspace)
    return Path(workspace.root)


class KaggleCvValidator:
    """Cross-validated training in a workspace, scored by the competition metric.

    Implements `HypothesisValidator`. Direction is resolved nearest-first —
    the run's own `competition.json`, then the knowledge copy, then the Analyze
    profile artifact — and stays `None` when none of them answers, because
    `resolve_maximize` returning `None` is a real answer and defaulting it to
    `True` is what cost rogii fifteen evidence cards.
    """

    source = SOURCE

    def validate(self, hypothesis: Any, workspace: Any, context: Any) -> ValidationResult:
        root = _workspace_root(workspace)
        competition = str(getattr(context, "competition", "") or "")
        knowledge_dir = getattr(getattr(context, "paths", None), "base_dir", None)

        from labpilot.research_engine.evidence.compare_service import _load_metrics

        metrics_path = root / "metrics.json"
        maximize = resolve_maximize(
            competition=competition,
            workspace_root=root,
            knowledge_root=Path(knowledge_dir) if knowledge_dir else None,
        )
        return result_from_metrics(
            _load_metrics(metrics_path),
            maximize=maximize,
            artifacts={"metrics": str(metrics_path)},
        )
