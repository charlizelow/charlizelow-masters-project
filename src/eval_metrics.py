# shared evaluation utilities for the prediction models (linear regression, lstm, tcn, tcn-lstm hybrid). 
# keeps the metric set consistent with the project plan: mae, rmse, r2, mape.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """compute mae, rmse, r2, and mape for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def print_metrics(model_name: str, split_name: str, metrics: dict[str, float]) -> None:
    """print a metrics dict in a consistent, readable format."""
    print(
        f"{model_name} [{split_name}] -> "
        f"mae: {metrics['mae']:.4f}  "
        f"rmse: {metrics['rmse']:.4f}  "
        f"r2: {metrics['r2']:.4f}  "
        f"mape: {metrics['mape']:.2f}%"
    )


def log_results(
    results_path: Path,
    model_name: str,
    split_name: str,
    metrics: dict[str, float],
    training_time_sec: float | None = None,
) -> None:
    """append a row of results to a shared csv, creating it if needed.

    this is the file notebook 08 (model comparison) will read from, so every
    model notebook should call this once per split (val and test) after evaluation."""
    row = {
        "model_name": model_name,
        "split": split_name,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mape": metrics["mape"],
        "training_time_sec": training_time_sec,
    }

    if results_path.exists():
        results_df = pd.read_csv(results_path)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([row])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)