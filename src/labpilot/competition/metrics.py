"""Best-effort mapping from Kaggle's free-text evaluation metric to a `MetricSpec`.

Kaggle's API returns the evaluation metric as a human-readable string (e.g.
"Root Mean Squared Error", "Categorization Accuracy", "AUC"), not a
structured enum. This is used only as informational context for the brief
and reflection — the actual `cv_<metric>` key the pipeline checks always
comes from `baseline.selector.DEFAULT_METRIC_BY_PROBLEM_TYPE`, since P0's
templates only ever emit `cv_accuracy` or `cv_rmse`.
"""

from labpilot.competition.models import MetricSpec

# Longer/more specific phrases first so substring matching doesn't get
# short-circuited by a more generic hint (e.g. "log loss" before "loss").
_MINIMIZE_HINTS = (
    "root mean squared logarithmic error",
    "root mean squared error",
    "mean squared error",
    "mean absolute error",
    "log loss",
    "logloss",
    "rmsle",
    "rmse",
    "mae",
    "mse",
    "error",
    "loss",
)
_MAXIMIZE_HINTS = (
    "area under the roc curve",
    "categorization accuracy",
    "accuracy",
    "auc",
    "f1",
    "precision",
    "recall",
    "kappa",
    "dice",
    "iou",
)


def normalize_metric(raw: str) -> MetricSpec | None:
    """Turn a Kaggle evaluation-metric string into a `MetricSpec`, or None."""
    cleaned = raw.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    direction = "maximize"
    for hint in _MINIMIZE_HINTS:
        if hint in lowered:
            direction = "minimize"
            break
    else:
        for hint in _MAXIMIZE_HINTS:
            if hint in lowered:
                direction = "maximize"
                break

    name = lowered.replace(" ", "_").replace("-", "_")
    return MetricSpec(name=name, direction=direction, description=cleaned)
