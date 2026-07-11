import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    y_proba: np.ndarray | None = None,
) -> float:
    name = metric_name.lower()

    if name in ("auc", "roc_auc", "roc-auc"):
        if y_proba is None:
            raise ValueError("AUC requires probability predictions")
        return float(roc_auc_score(y_true, y_proba))

    if name in ("logloss", "log_loss"):
        if y_proba is None:
            raise ValueError("Log loss requires probability predictions")
        return float(log_loss(y_true, y_proba))

    if name in ("accuracy", "acc"):
        return float(accuracy_score(y_true, y_pred))

    if name in ("rmse", "mse", "root_mean_squared_error"):
        return float(mean_squared_error(y_true, y_pred, squared=False))

    raise ValueError(f"Unsupported metric: {metric_name}")
