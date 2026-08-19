"""Map a competition's free-text evaluation metric onto a `MetricSpec`.

Kaggle's API returns the evaluation metric as a human-readable string (e.g.
"Root Mean Squared Error", "Categorization Accuracy", "AUC"), not a structured
enum. The canonical `key` is what the baseline selector and training templates
use for `cv_<key>` in metrics.json.

Identity and direction both come from `metric_vocabulary`. What this module used
to do instead is worth recording, because it is the bug class the registry
exists to close:

* Two ordered tuples of `(substring, key)` pairs, scanned in order. Substring
  matching made `"mse" in "rmse"` true, so the list survived only by spelling
  `rmse` before `mse` — position standing in for evidence, and one reordering
  away from mapping every RMSE competition to MSE.
* Direction was read off *which tuple matched*, so it was a property of where a
  name was written down rather than of the metric.
* The initial value was `direction = "maximize"`, so a metric matching **nothing**
  came back confidently maximizing. An unrecognised name is exactly the case with
  no evidence for either direction, and it got the answer that silently inverts
  every verdict for a loss.
"""

import logging
import re
from typing import TYPE_CHECKING

from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    direction_of,
    normalize_metric_key,
    scorable_keys,
)
from labpilot.research_engine.intelligence.competition.models import MetricSpec

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

logger = logging.getLogger(__name__)

#: Kept as a name because callers import it; the set itself is the registry's.
CANONICAL_METRIC_KEYS: frozenset[str] = scorable_keys()


def normalize_metric(raw: str) -> MetricSpec | None:
    """Turn a competition's evaluation-metric string into a `MetricSpec`, or None.

    `key` is None when no catalogued metric matches, and `direction` is
    `"unknown"` rather than a guess. Both are honest answers a caller can act on:
    the objective resolver probes an evaluator to settle direction, and the
    baseline selector falls back to its own default and logs that it did.
    """
    cleaned = raw.strip()
    if not cleaned:
        return None

    key = normalize_metric_key(cleaned)
    direction = direction_of(key) or "unknown"
    name = cleaned.lower().replace(" ", "_").replace("-", "_")
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
