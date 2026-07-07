# shared helpers for turning traffic_density.csv rows into fixed-length sequences for the sequence models (lstm, tcn, tcn-lstm hybrid).
# sequences are built separately per (scene_id, camera_id) group so a window never crosses from one camera's timeline into another's.

import numpy as np
import pandas as pd

from src.data_utils import feature_columns, group_columns, target_column


def create_sequences(
    df: pd.DataFrame,
    window_size: int,
    columns: list[str] = feature_columns,
    target: str = target_column,
) -> tuple[np.ndarray, np.ndarray]:
    """build (x, y) sequence arrays from a dataframe."""
    x_windows = []
    y_targets = []

    for _, group_df in df.groupby(group_columns, sort=False):
        group_df = group_df.sort_values("interval_start_sec").reset_index(drop=True)
        features = group_df[columns].to_numpy()
        targets = group_df[target].to_numpy()

        n_rows = len(group_df)
        if n_rows <= window_size:
            continue  # not enough intervals in this camera's clip for one full window

        for start in range(n_rows - window_size):
            end = start + window_size
            x_windows.append(features[start:end])
            y_targets.append(targets[end])

    x = np.stack(x_windows) if x_windows else np.empty((0, window_size, len(columns)))
    y = np.array(y_targets)
    return x, y


def flatten_for_baseline(x: np.ndarray) -> np.ndarray:
    """flatten a (n_samples, window_size, n_features) array to 2d for models
    that expect flat tabular input, e.g. linear regression."""
    n_samples = x.shape[0]
    return x.reshape(n_samples, -1)