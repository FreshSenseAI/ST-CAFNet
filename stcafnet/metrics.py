from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = ("tvc", "tvbn", "tbars")
    result = {}
    for index, name in enumerate(names):
        actual, predicted = y_true[:, index], y_pred[:, index]
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        result[name] = {
            "r2": float(r2_score(actual, predicted)),
            "rmse": rmse,
            "mae": float(mean_absolute_error(actual, predicted)),
            "rpd": float(np.std(actual, ddof=1) / rmse) if rmse > 0 else float("inf"),
        }
    return result

