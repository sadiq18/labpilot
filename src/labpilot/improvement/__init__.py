"""Legacy improvement helpers (quarantined).

``research improve`` and the linear Pipeline are removed. Prefer:

    research plan create <slug> --hypothesis H-xxx
    research run --plan P-xxx --competition <slug>

``models`` (TrainingOverrides, DEFAULT_TABULAR_MODEL_PARAMS) remain shared by
codegen / experiment graph. Fork/planner/tuner are retained only as helpers for
historical tests and eventual Planner constraint porting — not CLI entry points.
"""

from labpilot.improvement.models import ImprovementAction, ImprovementPlan, TrainingOverrides

__all__ = [
    "ImprovementAction",
    "ImprovementPlan",
    "TrainingOverrides",
]
