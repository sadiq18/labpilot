"""One canonical vocabulary for evaluation metrics.

Three overlapping vocabularies existed before this module and none was
authoritative: ``CANONICAL_METRIC_KEYS`` in `metrics.py`,
``SUPPORTED_METRICS_BY_PROBLEM_TYPE`` in the baseline selector, and the priority
list inside ``evidence/builder._primary_cv_keyed`` — which privileges
``cv_balanced_accuracy`` and ``cv_roc_auc``, neither of them canonical anywhere.

Two defects that split produced, both measured on disk 2026-08-13:

* ``playground-series-s6e7/competition.json`` carries
  ``{"name": "balanced_accuracy_score", "key": "accuracy"}``. Balanced accuracy
  is not accuracy, and nothing noticed.
* ``metric_names_match("cv_mse", "mean_squared_error")`` is False, so a campaign
  targeting the competition's own stated metric could never fire its stop.

**Matching is exact, against declared aliases — never a substring.** Substring
matching is why ``mean_squared_error`` matched nothing at all while
``"mse" in "rmse"`` is True; the hint tuple this replaces survived that only by
listing ``rmse`` before ``mse``, which is a coincidence rather than a design.

**Direction belongs to the metric, not to the spec that names it.** See
`direction.py`, which exists because ``MetricSpec.direction`` defaults to
``"maximize"`` — and that default caused every one of rogii's fifteen evidence
cards to be built as though MSE were maximised, recording its single genuine
improvement as ``rejected``.

An unrecognised metric resolves to ``None``, and callers must carry that through
as *unknown*. It must never fall back to a direction: a guessed direction inverts
every conclusion drawn from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from labpilot.research_engine.intelligence.competition.models import ProblemType

MetricDirection = Literal["minimize", "maximize"]

#: Qualifiers a recorded key may carry to say *where* it was measured. ``cv_rmse``
#: and ``lb_rmse`` are one metric read in two places. Owned here rather than in
#: `conductor/budgets.py` so the vocabulary and the matcher cannot fork.
MEASUREMENT_PREFIXES: tuple[str, ...] = ("cv_", "lb_", "val_", "test_", "train_")

_CLASSIFICATION = frozenset(
    {
        ProblemType.TABULAR_CLASSIFICATION,
        ProblemType.TEXT_CLASSIFICATION,
        ProblemType.IMAGE_CLASSIFICATION,
    }
)
_REGRESSION = frozenset({ProblemType.TABULAR_REGRESSION})


@dataclass(frozen=True)
class CanonicalMetric:
    """One metric the system can name, and possibly score against.

    ``scorable`` is the distinction that keeps this table honest. Naming a metric
    correctly and being able to *compute* it are different capabilities:
    `execution/metrics.py:compute_metric` raises on anything outside its dispatch,
    and the generated ``train.py`` is what calls it. So a metric may be named here
    — which is strictly better than mis-mapping it to a neighbour — without the
    pipeline claiming it can produce that number.

    ``cv_priority`` orders the search `_primary_cv_keyed` performs when a
    ``metrics.json`` carries several readings and none is declared primary. Lower
    wins; the order is inherited from that function's hand-written list so moving
    it here preserves behaviour.
    """

    key: str
    direction: MetricDirection
    aliases: frozenset[str]
    problem_types: frozenset[ProblemType]
    cv_priority: int
    scorable: bool = True
    #: The scorer needs probabilities, not hard predictions. Declared because
    #: `compute_metric` *raises* without them for some metrics and silently
    #: computes accuracy instead for others (AUC and log loss on multiclass) —
    #: an undeclared input requirement that turns `cv_auc` into a number that is
    #: not AUC, recorded in a log line and nowhere else.
    requires_probabilities: bool = False


#: Every entry is grounded in a metric this repository has actually seen — in a
#: real fixture, a live workspace's ``competition.json``, or a hardcoded list
#: being replaced. Speculative aliases are deliberately absent: an alias nobody
#: has observed is an untested branch, and ``None`` (unknown) is a safe answer
#: where a wrong mapping is not.
_METRICS: tuple[CanonicalMetric, ...] = (
    # Named but not scorable: `compute_metric` has no branch for it. Live on disk
    # in playground-series-s6e7, where it is currently mis-mapped to `accuracy`.
    CanonicalMetric(
        key="balanced_accuracy",
        direction="maximize",
        aliases=frozenset({"balanced_accuracy", "balanced_accuracy_score"}),
        problem_types=_CLASSIFICATION,
        cv_priority=10,
        scorable=False,
    ),
    CanonicalMetric(
        key="accuracy",
        direction="maximize",
        aliases=frozenset({"accuracy", "acc", "categorization_accuracy"}),
        problem_types=_CLASSIFICATION,
        cv_priority=20,
    ),
    CanonicalMetric(
        key="auc",
        direction="maximize",
        aliases=frozenset(
            {
                "auc",
                "roc_auc",
                "area_under_curve",
                "area_under_the_roc_curve",
                "area_under_the_receiver_operating_characteristic_curve",
            }
        ),
        problem_types=_CLASSIFICATION,
        cv_priority=30,
        requires_probabilities=True,
    ),
    CanonicalMetric(
        key="rmse",
        direction="minimize",
        aliases=frozenset({"rmse", "root_mean_squared_error"}),
        problem_types=_REGRESSION,
        cv_priority=40,
    ),
    CanonicalMetric(
        key="f1",
        direction="maximize",
        aliases=frozenset({"f1", "f1_score"}),
        problem_types=_CLASSIFICATION,
        cv_priority=50,
    ),
    CanonicalMetric(
        key="logloss",
        direction="minimize",
        aliases=frozenset({"logloss", "log_loss", "logarithmic_loss"}),
        problem_types=_CLASSIFICATION,
        cv_priority=60,
        requires_probabilities=True,
    ),
    CanonicalMetric(
        key="mse",
        direction="minimize",
        aliases=frozenset({"mse", "mean_squared_error"}),
        problem_types=_REGRESSION,
        cv_priority=70,
    ),
    CanonicalMetric(
        key="mae",
        direction="minimize",
        aliases=frozenset({"mae", "mean_absolute_error"}),
        problem_types=_REGRESSION,
        cv_priority=80,
    ),
    CanonicalMetric(
        key="rmsle",
        direction="minimize",
        aliases=frozenset(
            {"rmsle", "root_mean_squared_logarithmic_error", "root_mean_squared_log_error"}
        ),
        problem_types=_REGRESSION,
        cv_priority=90,
    ),
)

_BY_KEY: dict[str, CanonicalMetric] = {m.key: m for m in _METRICS}
_BY_ALIAS: dict[str, CanonicalMetric] = {
    alias: metric for metric in _METRICS for alias in metric.aliases
}


def _slug(raw: str) -> str:
    """Lowercase, collapsing every run of non-alphanumerics to one underscore.

    So ``"Root Mean Squared Error"``, ``"root-mean-squared-error"`` and
    ``"ROOT_MEAN_SQUARED_ERROR"`` are one token. Nothing else is normalised —
    stemming or singularising would reintroduce the fuzzy matching this module
    exists to remove.
    """
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")


def strip_measurement_prefix(name: str) -> str:
    """Drop one leading ``cv_`` / ``lb_`` / … qualifier, if present.

    One level only. ``cv_lb_rmse`` is not a thing, and stripping repeatedly would
    turn a column literally named ``test_score`` into ``score``.
    """
    for prefix in MEASUREMENT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def normalize_metric_key(raw: str | None) -> str | None:
    """The canonical key for a metric name, or None when it is not one we know.

    Accepts a Kaggle metric string (``"Root Mean Squared Error"``), a canonical
    key (``"rmse"``), or a measured key (``"cv_rmse"``). Returns None for anything
    else — including genuinely real metrics the catalogue has not been taught,
    such as m5's WRMSSE. None means *unknown*, and callers must carry that rather
    than substituting a default.
    """
    if not raw:
        return None
    slug = _slug(raw)
    if not slug:
        return None
    metric = _BY_ALIAS.get(slug) or _BY_ALIAS.get(strip_measurement_prefix(slug))
    return metric.key if metric else None


def direction_of(key: str | None) -> MetricDirection | None:
    """Whether a metric is maximised or minimised, or None when unknown.

    None is a real answer and the only honest one for an unmapped metric. The
    caller decides what to do with it — `evidence/builder` refuses to build a
    card, which is correct, since the sign of every conclusion on it would be a
    guess.
    """
    normalized = normalize_metric_key(key)
    return _BY_KEY[normalized].direction if normalized else None


def is_scorable(key: str | None) -> bool:
    """Whether `compute_metric` can actually produce this number."""
    normalized = normalize_metric_key(key)
    return bool(normalized) and _BY_KEY[normalized].scorable


def metrics_for_problem_type(problem_type: ProblemType | str) -> frozenset[str]:
    """Scorable canonical keys for this problem type.

    Scorable only: this feeds the baseline selector, which tells codegen which
    ``cv_<key>`` to emit. Naming a metric the pipeline cannot compute would turn
    an honest mapping into a runtime failure inside generated code.
    """
    wanted = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    return frozenset(m.key for m in _METRICS if m.scorable and wanted in m.problem_types)


def cv_search_order() -> tuple[str, ...]:
    """Canonical keys by ``cv_priority``, for picking a primary metric from a blob."""
    return tuple(m.key for m in sorted(_METRICS, key=lambda m: m.cv_priority))


def known_keys() -> frozenset[str]:
    """Every canonical key, scorable or not. Replaces ``CANONICAL_METRIC_KEYS``."""
    return frozenset(_BY_KEY)


def scorable_keys() -> frozenset[str]:
    """Canonical keys `compute_metric` can produce."""
    return frozenset(m.key for m in _METRICS if m.scorable)


def requires_probabilities(key: str | None) -> bool:
    """Whether the scorer needs probabilities rather than hard predictions."""
    normalized = normalize_metric_key(key)
    return bool(normalized) and _BY_KEY[normalized].requires_probabilities
