import numpy as np
import pytest

from labpilot.research_engine.execution.metrics import compute_metric
from labpilot.research_engine.intelligence.competition.metrics import (
    enrich_metric_spec,
    normalize_metric,
    resolve_metric_key_with_llm,
)
from labpilot.research_engine.intelligence.competition.models import MetricSpec


def test_normalize_metric_sets_canonical_key():
    metric = normalize_metric("Root Mean Squared Error")
    assert metric is not None
    assert metric.key == "rmse"
    assert metric.direction == "minimize"

    auc = normalize_metric("Area Under the ROC Curve")
    assert auc is not None
    assert auc.key == "auc"


def test_normalize_metric_key_none_for_unrecognized():
    metric = normalize_metric("Quadratic Weighted Kappa")
    assert metric is not None
    assert metric.key is None


class FakeLLMClient:
    def __init__(self, response: str = "auc", error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def complete(self, system: str, user: str) -> str:
        if self.error:
            raise self.error
        return self.response


def test_resolve_metric_key_with_llm_valid():
    client = FakeLLMClient("auc")
    assert resolve_metric_key_with_llm("Some weird metric", ["auc", "accuracy"], client) == "auc"


def test_resolve_metric_key_with_llm_malformed():
    client = FakeLLMClient("not_a_real_metric")
    assert resolve_metric_key_with_llm("weird", ["auc", "accuracy"], client) is None


def test_resolve_metric_key_with_llm_no_client():
    assert resolve_metric_key_with_llm("weird", ["auc"], None) is None


def test_resolve_metric_key_with_llm_error():
    client = FakeLLMClient(error=RuntimeError("network"))
    assert resolve_metric_key_with_llm("weird", ["auc"], client) is None


def test_enrich_metric_spec_skips_when_key_present():
    metric = MetricSpec(name="auc", direction="maximize", key="auc")
    enriched = enrich_metric_spec(metric, "AUC", llm_client=FakeLLMClient("rmse"))
    assert enriched.key == "auc"


def test_compute_metric_rmse_no_squared_kwarg():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.2, 2.8])
    score = compute_metric(y_true, y_pred, "rmse")
    assert score == pytest.approx(0.173205, rel=1e-3)


def test_compute_metric_mae_and_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert compute_metric(y_true, y_pred, "mae") == pytest.approx(1 / 3)
    assert compute_metric(y_true, y_pred, "mse") == pytest.approx(1 / 3)


def test_compute_metric_rmsle_negative_fallback():
    y_true = np.array([-1.0, 2.0])
    y_pred = np.array([1.0, 3.0])
    score = compute_metric(y_true, y_pred, "rmsle")
    assert isinstance(score, float)


def test_compute_metric_auc_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.4, 0.9, 0.2])
    score = compute_metric(y_true, y_pred, "auc", y_proba=y_proba, num_classes=2)
    assert 0.0 <= score <= 1.0


def test_compute_metric_auc_multiclass_falls_back_to_accuracy():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    score = compute_metric(y_true, y_pred, "auc", y_proba=None, num_classes=3)
    assert score == pytest.approx(1.0)


# --- an unrecognised metric has no direction --------------------------------


def test_an_unrecognised_metric_is_not_read_as_maximizing() -> None:
    """`normalize_metric` opened with `direction = "maximize"` and only moved off
    it on a substring hit, so a metric matching *nothing* came back confidently
    maximizing. An unrecognised name is precisely the case with no evidence for
    either direction, and maximize is the answer that inverts every verdict for a
    loss — which is what rogii's fifteen evidence cards were built on.
    """
    spec = normalize_metric("Wellbore Misfit Score")

    assert spec is not None
    assert spec.key is None
    assert spec.direction == "unknown"


def test_a_recognised_metric_takes_the_registry_direction() -> None:
    assert normalize_metric("Root Mean Squared Error").direction == "minimize"
    assert normalize_metric("Area Under the ROC Curve").direction == "maximize"


def test_the_metric_spec_direction_comes_from_its_key() -> None:
    """Filled by the model itself, so a spec built anywhere — including one
    deserialized from a `competition.json` that omits the field — is oriented."""
    assert MetricSpec(name="rmse", key="rmse").direction == "minimize"
    assert MetricSpec(name="auc", key="auc").direction == "maximize"
    assert MetricSpec(name="mystery").direction == "unknown", "guessed with no key at all"


def test_a_direction_contradicting_the_registry_is_left_standing() -> None:
    """Deliberately *not* corrected here. It is the contradiction
    `resolve_objective` blocks the campaign on, and silently rewriting it would
    delete the detection while leaving the wrong contract on disk."""
    assert MetricSpec(name="rmse", key="rmse", direction="maximize").direction == "maximize"


def test_a_direction_that_is_not_one_of_the_three_is_rejected() -> None:
    """`direction: str` accepted "sideways", "min", "Minimize" and "" alike."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        MetricSpec(name="x", direction="sideways")
