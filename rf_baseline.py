"""
Random Forest baseline on dataset.csv (no CLI args required)

- Input: dataset.csv
- Target column: Price
- Preprocessing: remove input columns containing 'price' (case-insensitive),
  keep only numeric/boolean features (booleans cast to float); drop rows with
  non-finite values in X or y.
- Split: train/test = 80%/20%, random_state=42
- Model:RandomForestRegressor  
- Metrics: RMSE, MAE, R^2 (sklearn.metrics)

Run:
  python rf_baseline.py
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


 

DATA_PATH: str = "dataset.csv"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# RandomForest parameters (single run, no search)
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_leaf": 1,
    "max_features": 1.0,  # default for regressor; can try 0.6 later
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_and_prepare(path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(path)
    if "Price" not in df.columns:
        raise ValueError("Target column 'Price' not found in dataset")

    # identify leakage features containing 'price' except the target itself
    lower_map = {c: c.lower() for c in df.columns}
    drop_cols: List[str] = [c for c in df.columns if ("price" in lower_map[c] and c != "Price")]

    # X: drop target and leakage columns; keep numeric/boolean only
    X_df = df.drop(columns=drop_cols)
    feature_cols = [c for c in X_df.columns if c != "Price"]
    X_df = X_df[feature_cols]
    # Keep numeric/boolean columns
    X_df = X_df.select_dtypes(include=[np.number, bool]).copy()
    # Convert bool to float
    for c in X_df.columns:
        if X_df[c].dtype == bool:
            X_df[c] = X_df[c].astype(float)

    y = df["Price"].astype(float)

    # remove rows with non-finite values
    mask = np.isfinite(X_df.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
    X_df = X_df.loc[mask]
    y = y.loc[mask]

    return X_df, y, drop_cols


def main():
    try:
        X, y, removed = load_and_prepare(DATA_PATH)
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        sys.exit(1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Logging
    print("Removed feature columns containing 'price' (case-insensitive), excluding target 'Price':")
    if removed:
        print("- " + ", ".join(removed))
    else:
        print("- None")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print("RandomForest params:")
    print(
        f"- n_estimators={RF_PARAMS['n_estimators']}, max_depth={RF_PARAMS['max_depth']}, "
        f"min_samples_leaf={RF_PARAMS['min_samples_leaf']}, max_features={RF_PARAMS['max_features']}, "
        f"random_state={RF_PARAMS['random_state']}, n_jobs={RF_PARAMS['n_jobs']}"
    )

    # Train and evaluate
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("Metrics:")
    print(f"- RMSE={rmse:.4f}")
    print(f"- MAE={mae:.4f}")
    print(f"- R2={r2:.4f}")


if __name__ == "__main__":
    main()

