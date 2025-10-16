"""
Ridge baseline on dataset.csv (no CLI args required)

- Input: dataset.csv
- Target column: Price
 - Behavior: remove input features whose names contain 'price' (case-insensitive),
   while keeping target 'Price' itself.
- Split: train/test = 80%/20%, random_state=42
- Model: sklearn Ridge; alphas = [0.1, 1.0, 10.0]
- Metrics: RMSE, MAE, R^2 (sklearn.metrics)

Run:
  python ridge_baseline.py
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



DATA_PATH: str = "dataset.csv"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
ALPHAS: List[float] = [0.1, 1.0, 10.0]



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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Logging
    print("Removed feature columns containing 'price' (case-insensitive), excluding target 'Price':")
    if removed:
        print("- " + ", ".join(removed))
    else:
        print("- None")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Try each alpha and evaluate on the test set
    results = []
    for a in ALPHAS:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=float(a)))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        print(f"alpha={a:.3f} -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        results.append((a, rmse, mae, r2))

    # pick best by RMSE
    best = min(results, key=lambda t: t[1])
    best_alpha, best_rmse, best_mae, best_r2 = best
    print("Best alpha by RMSE:")
    print(f"alpha={best_alpha:.3f} -> RMSE={best_rmse:.4f}, MAE={best_mae:.4f}, R2={best_r2:.4f}")


if __name__ == "__main__":
    main()
