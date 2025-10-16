"""
Data loading utilities for cross-region Ridge evaluation.

- Reads dataset.csv without dropping any price-related features (already cleaned).
- Filters to four metropolitan regions: Eastern, Western, Southern, Northern.
- Restores region label per row from Regionname_* one-hot columns.
- Produces numeric feature matrix X, target vector y, and region labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

MAJOR_REGIONS: Tuple[str, ...] = (
    "Eastern Metropolitan",
    "Western Metropolitan",
    "Southern Metropolitan",
    "Northern Metropolitan",
)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    regions: np.ndarray
    feature_names: List[str]


def _find_region_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c.startswith("Regionname_")]


def _infer_regions(df: pd.DataFrame, region_cols: Sequence[str]) -> np.ndarray:
    region_values = df.loc[:, region_cols].to_numpy(dtype=float)
    positive_mask = region_values > 0.5
    counts = positive_mask.sum(axis=1)

    if not np.all((counts == 0) | (counts == 1)):
        dup_rows = np.where(counts > 1)[0][:5]
        raise ValueError(
            "Some rows map to multiple regions; please inspect columns",  # noqa: TRY003
            dup_rows.tolist(),
        )

    indices = np.argmax(region_values, axis=1)
    regions = np.array(
        [region_cols[i].replace("Regionname_", "") for i in indices],
        dtype=object,
    )
    regions[counts == 0] = None
    return regions


def load_dataset(path: str = "dataset.csv") -> Dataset:
    df = pd.read_csv(path)
    if "Price" not in df.columns:
        raise ValueError("Expected column 'Price' in dataset")

    region_cols = _find_region_columns(df.columns)
    if not region_cols:
        raise ValueError("No Regionname_* columns found; cannot derive regions")

    regions = _infer_regions(df, region_cols)
    mask_major = np.isin(regions, MAJOR_REGIONS)

    df_major = df.loc[mask_major].reset_index(drop=True)
    regions_major = regions[mask_major]

    # Drop region indicator columns from features to avoid leakage.
    feature_cols = [c for c in df_major.columns if c not in {"Price", *region_cols}]
    X_df = df_major.loc[:, feature_cols].select_dtypes(include=[np.number, bool]).copy()
    bool_cols = list(X_df.select_dtypes(include=["bool"]).columns)
    if bool_cols:
        X_df[bool_cols] = X_df[bool_cols].astype(float)

    y_series = df_major["Price"].astype(float)

    finite_mask = np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
        y_series.to_numpy(dtype=float)
    )
    if not np.all(finite_mask):
        X_df = X_df.loc[finite_mask].reset_index(drop=True)
        y_series = y_series.loc[finite_mask].reset_index(drop=True)
        regions_major = regions_major[finite_mask]

    return Dataset(
        X=X_df.to_numpy(dtype=float),
        y=y_series.to_numpy(dtype=float),
        regions=regions_major.astype(object),
        feature_names=list(X_df.columns),
    )


def leave_one_region_out_splits(regions: Sequence[str]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    regions = np.asarray(regions)
    splits: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for test_region in MAJOR_REGIONS:
        test_idx = np.where(regions == test_region)[0]
        train_idx = np.where(regions != test_region)[0]
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx, test_region))
    return splits


def summarize_regions(regions: Sequence[str]) -> List[Tuple[str, int]]:
    regions = np.asarray(regions)
    summary: List[Tuple[str, int]] = []
    for region in MAJOR_REGIONS:
        summary.append((region, int(np.sum(regions == region))))
    return summary


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0.0] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform()")
        return (X - self.mean_) / self.std_


class RidgeRegressor:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._coef: np.ndarray | None = None  # includes intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        n_samples, n_features = X.shape
        X_aug = np.hstack([np.ones((n_samples, 1)), X])
        gram = X_aug.T @ X_aug
        reg = np.eye(n_features + 1, dtype=float)
        reg[0, 0] = 0.0  # do not regularise intercept
        gram = gram + self.alpha * reg
        rhs = X_aug.T @ y
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs
        self._coef = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("Model not fitted")
        n_samples = X.shape[0]
        X_aug = np.hstack([np.ones((n_samples, 1)), X])
        return X_aug @ self._coef


def kfold_indices(n_samples: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if k <= 1 or k > n_samples:
        raise ValueError(f"k-fold requires 1 < k <= n_samples, got k={k}, n={n_samples}")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_idx = folds[i]
        if val_idx.size == 0:
            continue
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def inner_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
    k: int,
    seed: int,
) -> Tuple[float, List[Dict[str, object]]]:
    splits = kfold_indices(len(X), k, seed)
    alpha_results: List[Dict[str, object]] = []
    alpha_to_rmse: Dict[float, float] = {}

    for alpha in alphas:
        fold_rmses: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx])
            y_train = y[train_idx]
            X_val = scaler.transform(X[val_idx])
            y_val = y[val_idx]

            model = RidgeRegressor(alpha=alpha).fit(X_train, y_train)
            preds = model.predict(X_val)
            fold_rmses.append(rmse(y_val, preds))
        avg_rmse = float(np.mean(fold_rmses))
        alpha_results.append(
            {
                "alpha": float(alpha),
                "fold_rmses": [float(v) for v in fold_rmses],
                "avg_rmse": avg_rmse,
            }
        )
        alpha_to_rmse[float(alpha)] = avg_rmse

    best_alpha = min(alpha_to_rmse, key=alpha_to_rmse.get)
    return best_alpha, alpha_results


def outer_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
    outer_k: int,
    inner_k: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[float, List[float]], Dict[str, Dict[str, float]]]:
    splits = kfold_indices(len(X), outer_k, seed)
    outer_results: List[Dict[str, object]] = []
    alpha_rmse_tracker: Dict[float, List[float]] = {float(a): [] for a in alphas}
    metric_tracker: Dict[str, List[float]] = {"rmse": [], "mae": [], "r2": []}
    inner_avg_records: Dict[float, List[float]] = {float(a): [] for a in alphas}

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        inner_seed = seed + 1000 + fold_idx
        best_alpha, inner_details = inner_ridge_cv(
            X[train_idx],
            y[train_idx],
            alphas=alphas,
            k=inner_k,
            seed=inner_seed,
        )

        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        y_train = y[train_idx]
        X_val = scaler.transform(X[val_idx])
        y_val = y[val_idx]

        model = RidgeRegressor(alpha=best_alpha).fit(X_train, y_train)
        preds = model.predict(X_val)
        fold_rmse = rmse(y_val, preds)
        fold_mae = mae(y_val, preds)
        fold_r2 = r2_score(y_val, preds)

        alpha_rmse_tracker[float(best_alpha)].append(fold_rmse)
        metric_tracker["rmse"].append(fold_rmse)
        metric_tracker["mae"].append(fold_mae)
        metric_tracker["r2"].append(fold_r2)
        for inner in inner_details:
            inner_avg_records[float(inner["alpha"])].append(float(inner["avg_rmse"]))
        outer_results.append(
            {
                "fold": fold_idx,
                "alpha": float(best_alpha),
                "rmse": fold_rmse,
                "mae": fold_mae,
                "r2": fold_r2,
                "inner_summary": inner_details,
            }
        )

    metric_summary: Dict[str, Dict[str, float]] = {}
    for key, values in metric_tracker.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            mean_val = float("nan")
            std_val = float("nan")
        metric_summary[key] = {"mean": mean_val, "std": std_val}

    return outer_results, alpha_rmse_tracker, metric_summary, inner_avg_records


def select_final_alpha(alpha_rmse_tracker: Dict[float, List[float]]) -> Tuple[float, Dict[float, float]]:
    avg_rmse: Dict[float, float] = {}
    for alpha, rmses in alpha_rmse_tracker.items():
        if rmses:
            avg_rmse[alpha] = float(np.mean(rmses))
        else:
            avg_rmse[alpha] = float("inf")
    best_alpha = min(avg_rmse, key=avg_rmse.get)
    return best_alpha, avg_rmse


def evaluate_region_split(
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_region: str,
    alphas: Sequence[float],
    outer_k: int,
    inner_k: int,
    seed: int,
) -> Dict[str, object]:
    X_train = dataset.X[train_idx]
    y_train = dataset.y[train_idx]
    X_test = dataset.X[test_idx]
    y_test = dataset.y[test_idx]

    outer_results, alpha_tracker, outer_summary, inner_avg_records = outer_ridge_cv(
        X_train, y_train, alphas=alphas, outer_k=outer_k, inner_k=inner_k, seed=seed
    )
    alpha_final, alpha_avg_rmse = select_final_alpha(inner_avg_records)

    alpha_frequency: Dict[float, int] = {float(a): 0 for a in alphas}
    for res in outer_results:
        alpha_frequency[float(res["alpha"])] += 1

    scaler = StandardScaler().fit(X_train)
    model = RidgeRegressor(alpha=alpha_final).fit(scaler.transform(X_train), y_train)
    preds = model.predict(scaler.transform(X_test))
    test_metrics = {
        "rmse": rmse(y_test, preds),
        "mae": mae(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
    # Evaluate variability within the held-out region by chunking test samples.
    test_subsplit_details: List[Dict[str, float]] = []
    subsplit_metrics_collectors: Dict[str, List[float]] = {"rmse": [], "mae": [], "r2": []}
    if len(test_idx) > 0:
        rng = np.random.default_rng(seed + 12345)
        positions = np.arange(len(test_idx))
        permuted = rng.permutation(positions)
        split_count = min(len(permuted), 10)
        subsplit_positions = np.array_split(permuted, split_count)
        for subsplit_id, subset_pos in enumerate(subsplit_positions):
            if subset_pos.size == 0:
                continue
            y_subset = y_test[subset_pos]
            preds_subset = preds[subset_pos]
            subset_rmse = rmse(y_subset, preds_subset)
            subset_mae = mae(y_subset, preds_subset)
            subset_r2 = r2_score(y_subset, preds_subset)
            test_subsplit_details.append(
                {
                    "split": int(subsplit_id),
                    "size": int(subset_pos.size),
                    "rmse": subset_rmse,
                    "mae": subset_mae,
                    "r2": subset_r2,
                }
            )
            subsplit_metrics_collectors["rmse"].append(subset_rmse)
            subsplit_metrics_collectors["mae"].append(subset_mae)
            subsplit_metrics_collectors["r2"].append(subset_r2)
    test_subsplit_summary: Dict[str, Dict[str, float]] = {}
    for metric_name, values in subsplit_metrics_collectors.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            mean_val = float("nan")
            std_val = float("nan")
        test_subsplit_summary[metric_name] = {"mean": mean_val, "std": std_val}

    return {
        "test_region": test_region,
        "alpha_final": float(alpha_final),
        "alpha_avg_rmse": alpha_avg_rmse,
        "outer_results": outer_results,
        "outer_summary": outer_summary,
        "alpha_frequency": alpha_frequency,
        "inner_avg_records": inner_avg_records,
        "test_metrics": test_metrics,
        "test_subsplit_details": test_subsplit_details,
        "test_subsplit_summary": test_subsplit_summary,
        "test_samples": int(len(test_idx)),
        "train_samples": int(len(train_idx)),
    }


def evaluate_all_regions(
    dataset: Dataset,
    alphas: Sequence[float],
    outer_k: int = 10,
    inner_k: int = 3,
    seed: int = 42,
) -> List[Dict[str, object]]:
    splits = leave_one_region_out_splits(dataset.regions)
    results: List[Dict[str, object]] = []
    for split_idx, (train_idx, test_idx, test_region) in enumerate(splits):
        region_seed = seed + 10_000 * split_idx
        results.append(
            evaluate_region_split(
                dataset,
                train_idx,
                test_idx,
                test_region,
                alphas=alphas,
                outer_k=outer_k,
                inner_k=inner_k,
                seed=region_seed,
            )
        )
    return results


if __name__ == "__main__":
    ds = load_dataset("dataset.csv")
    print("Dataset summary:")
    print(f"- samples: {ds.X.shape[0]}, features: {ds.X.shape[1]}")
    for region, count in summarize_regions(ds.regions):
        print(f"  {region}: {count}")

    alpha_grid = [1.0, 10.0,100]
    results = evaluate_all_regions(ds, alphas=alpha_grid, outer_k=10, inner_k=3, seed=42)
    print("\nCross-region Ridge evaluation:")
    for res in results:
        test_region = res["test_region"]
        alpha_final = res["alpha_final"]
        metrics = res["test_metrics"]
        outer_summary = res["outer_summary"]
        alpha_frequency = res["alpha_frequency"]
        inner_avg_records = res["inner_avg_records"]
        print(f"\n=== Test region: {test_region} ===")
        print(
            f"Final alpha: {alpha_final:.3f} | Test RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}"
        )
        subsplit_summary = res["test_subsplit_summary"]
        subsplit_details = res["test_subsplit_details"]
        print(
            "Test sub-splits (up to 10 chunks): "
            f"RMSE={subsplit_summary['rmse']['mean']:.2f} ± {subsplit_summary['rmse']['std']:.2f}, "
            f"MAE={subsplit_summary['mae']['mean']:.2f} ± {subsplit_summary['mae']['std']:.2f}, "
            f"R2={subsplit_summary['r2']['mean']:.4f} ± {subsplit_summary['r2']['std']:.4f}"
        )
        if subsplit_details:
            print("Test sub-split details:")
            for detail in subsplit_details:
                print(
                    f"  Chunk {detail['split']:02d} (n={detail['size']}): "
                    f"RMSE={detail['rmse']:.2f}, MAE={detail['mae']:.2f}, R2={detail['r2']:.4f}"
                )
        print("Inner-CV average RMSE across folds (per alpha):")
        for alpha in sorted(inner_avg_records.keys()):
            values = inner_avg_records[alpha]
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                print(
                    f"  alpha={alpha:.3f}: avg RMSE={mean_val:.2f} ± {std_val:.2f} "
                    f"(n={len(values)})"
                )
            else:
                print(f"  alpha={alpha:.3f}: no data")
        print(
            "Outer 10-fold validation (mean ± std): "
            f"RMSE={outer_summary['rmse']['mean']:.2f} ± {outer_summary['rmse']['std']:.2f}, "
            f"MAE={outer_summary['mae']['mean']:.2f} ± {outer_summary['mae']['std']:.2f}, "
            f"R2={outer_summary['r2']['mean']:.4f} ± {outer_summary['r2']['std']:.4f}"
        )
        print("Alpha frequency across outer folds:")
        for alpha in sorted(alpha_frequency.keys()):
            print(f"  alpha={alpha:.3f}: {alpha_frequency[alpha]} folds")
        print("Outer fold details:")
        for fold_info in res["outer_results"]:
            fold = fold_info["fold"]
            alpha = fold_info["alpha"]
            fold_rmse = fold_info["rmse"]
            fold_mae = fold_info["mae"]
            fold_r2 = fold_info["r2"]
            print(
                f"  Fold {fold:02d}: alpha={alpha:.3f}, "
                f"RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}, R2={fold_r2:.4f}"
            )
            print("    Inner 3-fold RMSE per alpha:")
            for inner in fold_info["inner_summary"]:
                a = inner["alpha"]
                fold_rmses = ", ".join(f"{v:.2f}" for v in inner["fold_rmses"])
                print(f"      alpha={a:.3f} -> folds [{fold_rmses}] | avg={inner['avg_rmse']:.2f}")
