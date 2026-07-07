# shared helpers for loading traffic_density.csv and preparing train/val/test splits 
# for the prediction notebooks (04 linear regression, 05 lstm, 06 tcn, 07 tcn-lstm hybrid).

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

feature_columns = [
    "elapsed_time_sec",
    "relative_position",
    "vehicle_count_smoothed",
    "lag_1",
    "lag_2",
    "rate_of_change",
    "peak_density",
    "rolling_mean",
    "rolling_std",
]
 
target_column = "vehicle_count"
group_columns = ["scene_id", "camera_id"]
 
 
def load_density_data(path: Path) -> pd.DataFrame:
    """load traffic_density.csv and sort rows so each camera's timeline is contiguous."""
    df = pd.read_csv(path)
    df = df.sort_values(group_columns + ["interval_start_sec"]).reset_index(drop=True)
    return df
 
 
def drop_incomplete_lag_rows(df: pd.DataFrame) -> pd.DataFrame:
    """drop rows with nan lag features (the first few intervals per camera)."""
    return df.dropna(subset=["lag_1", "lag_2"]).reset_index(drop=True)
 
 
def split_by_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split into train/val/test dataframes using the existing 'split' column."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df
 
 
def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str] = feature_columns,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """fit a standard scaler on the train split only, then transform all three splits."""
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
 
    train_df[columns] = scaler.fit_transform(train_df[columns])
    val_df[columns] = scaler.transform(val_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
 
    return train_df, val_df, test_df, scaler