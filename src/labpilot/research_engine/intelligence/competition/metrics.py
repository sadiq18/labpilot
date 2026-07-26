"""Best-effort mapping from Kaggle's free-text evaluation metric to a `MetricSpec`.

Kaggle's API returns the evaluation metric as a human-readable string (e.g.
"Root Mean Squared Error", "Categorization Accuracy", "AUC"), not a
structured enum. P1 maps these to canonical `key` values used by the
baseline selector and training templates for `cv_<key>` in metrics.json.
"""

import logging
import re
from typing import TYPE_CHECKING

from labpilot.competition.models import MetricSpec

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

logger = logging.getLogger(__name__)

CANONICAL_METRIC_KEYS = frozenset(
    {"accuracy", "auc", "logloss", "f1", "rmse", "mse", "mae", "rmsle"}
)

# Longer/more specific phrases first so substring matching doesn't get
# short-circuited by a more generic hint (e.g. "log loss" before "loss").
_MINIMIZE_HINTS: tuple[tuple[str, str], ...] = (
    ("root mean squared logarithmic error", "rmsle"),
    ("root mean squared error", "rmse"),
    ("mean squared error", "mse"),
    ("mean absolute error", "mae"),
    ("log loss", "logloss"),
    ("logloss", "logloss"),
    ("rmsle", "rmsle"),
    ("rmse", "rmse"),
    ("mae", "mae"),
    ("mse", "mse"),
)
_MAXIMIZE_HINTS: tuple[tuple[str, str], ...] = (
    ("area under the roc curve", "auc"),
    ("categorization accuracy", "accuracy"),
    ("accuracy", "accuracy"),
    ("auc", "auc"),
    ("f1", "f1"),
    ("f1 score", "f1"),
    ("f1-score", "f1"),
)


def _derive_key(lowered: str) -> str | None:
    for hint, key in _MINIMIZE_HINTS:
        if hint in lowered:
            return key
    for hint, key in _MAXIMIZE_HINTS:
        if hint in lowered:
            return key
    return None


def normalize_metric(raw: str) -> MetricSpec | None:
    """Turn a Kaggle evaluation-metric string into a `MetricSpec`, or None."""
    cleaned = raw.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    direction = "maximize"
    key: str | None = None

    for hint, hint_key in _MINIMIZE_HINTS:
        if hint in lowered:
            direction = "minimize"
            key = hint_key
            break
    else:
        for hint, hint_key in _MAXIMIZE_HINTS:
            if hint in lowered:
                direction = "maximize"
                key = hint_key
                break

    name = lowered.replace(" ", "_").replace("-", "_")
    return MetricSpec(name=name, direction=direction, description=cleaned, key=key)


def resolve_metric_key_with_llm(
    raw: str,
    supported_keys: list[str],
    llm_client: "LLMClient | None",
) -> str | None:
    """Ask an optional LLM to pick one supported canonical metric key."""
    if llm_client is None or not raw.strip() or not supported_keys:
        return None

    options = ", ".join(sorted(supported_keys))
    system = (
        "You map Kaggle competition evaluation metrics to exactly one canonical key. "
        "Reply with only the key token, nothing else."
    )
    user = (
        f"Kaggle evaluation metric: {raw.strip()}\n"
        f"Choose exactly one key from: {options}\n"
        "Reply with only the key."
    )

    try:
        response = llm_client.complete(system, user).strip().lower()
    except Exception:
        logger.warning(
            "LLM metric tie-breaker failed for %r; falling back to default metric.",
            raw,
            exc_info=True,
        )
        return None

    token = re.split(r"[\s,.]+", response)[0] if response else ""
    if token in supported_keys:
        return token

    logger.warning(
        "LLM metric tie-breaker returned unrecognized key %r for %r.",
        token,
        raw,
    )
    return None


def enrich_metric_spec(
    metric: MetricSpec,
    raw: str,
    supported_keys: list[str] | None = None,
    llm_client: "LLMClient | None" = None,
) -> MetricSpec:
    """Ensure `metric.key` is set, optionally using an LLM tie-breaker."""
    if metric.key is not None:
        return metric

    keys = supported_keys or list(CANONICAL_METRIC_KEYS)
    resolved = resolve_metric_key_with_llm(raw, keys, llm_client)
    if resolved is None:
        return metric

    return metric.model_copy(update={"key": resolved})
