from labpilot.competition.metrics import normalize_metric


def test_normalize_metric_detects_minimize_metrics():
    for raw in [
        "Root Mean Squared Error",
        "RMSE",
        "Root-Mean-Squared-Logarithmic-Error (RMSLE)",
        "Mean Absolute Error",
        "Log Loss",
    ]:
        metric = normalize_metric(raw)
        assert metric is not None, raw
        assert metric.direction == "minimize", raw


def test_normalize_metric_detects_maximize_metrics():
    for raw in ["Categorization Accuracy", "Accuracy", "AUC", "F1 Score"]:
        metric = normalize_metric(raw)
        assert metric is not None, raw
        assert metric.direction == "maximize", raw


def test_normalize_metric_returns_none_for_blank_input():
    assert normalize_metric("") is None
    assert normalize_metric("   ") is None


def test_normalize_metric_preserves_original_text_as_description():
    metric = normalize_metric("Root Mean Squared Error")
    assert metric is not None
    assert metric.description == "Root Mean Squared Error"
