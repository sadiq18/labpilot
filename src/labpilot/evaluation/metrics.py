import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    y_proba: np.ndarray | None = None,
    *,
    num_classes: int | None = None,
) -> float:
    name = metric_name.lower()

    if name in ("auc", "roc_auc", "roc-auc"):
        if num_classes is not None and num_classes > 2:
            logger.warning(
                "AUC requested for %d classes; computing accuracy instead.",
                num_classes,
            )
            return float(accuracy_score(y_true, y_pred))
        if y_proba is None:
            raise ValueError("AUC requires probability predictions")
        return float(roc_auc_score(y_true, y_proba))

    if name in ("logloss", "log_loss"):
        if num_classes is not None and num_classes > 2:
            logger.warning(
                "Log loss requested for %d classes; computing accuracy instead.",
                num_classes,
            )
            return float(accuracy_score(y_true, y_pred))
        if y_proba is None:
            raise ValueError("Log loss requires probability predictions")
        return float(log_loss(y_true, y_proba))

    if name in ("accuracy", "acc"):
        return float(accuracy_score(y_true, y_pred))

    if name == "f1":
        average = "binary" if (num_classes is None or num_classes <= 2) else "macro"
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))

    if name in ("rmse", "root_mean_squared_error"):
        return float(root_mean_squared_error(y_true, y_pred))

    if name == "mse":
        return float(mean_squared_error(y_true, y_pred))

    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))

    if name == "rmsle":
        if np.any(y_true < 0) or np.any(y_pred < 0):
            logger.warning("RMSLE undefined for negative values; computing RMSE instead.")
            return float(root_mean_squared_error(y_true, y_pred))
        return float(
            root_mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))
        )

    raise ValueError(f"Unsupported metric: {metric_name}")
