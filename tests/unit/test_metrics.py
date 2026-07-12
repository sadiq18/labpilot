import numpy as np
import pytest

from labpilot.competition.metrics import (
    enrich_metric_spec,
    normalize_metric,
    resolve_metric_key_with_llm,
)
from labpilot.competition.models import MetricSpec
from labpilot.evaluation.metrics import compute_metric


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
