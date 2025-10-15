"""
Cross-region Random Forest nested cross-validation.

- Loads dataset.csv via cross_region_ridge.load_dataset (already filters metro regions).
- Performs leave-one-region-out evaluation with outer 10-fold / inner 3-fold nested CV.
- Hyperparameter grid:
    max_depth ∈ {None, 12, 24}
    min_samples_leaf ∈ {1, 5, 10}
    max_features ∈ {1.0, "sqrt", 0.5}
- Uses manual loops for CV and manual RMSE/MAE/R2 metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from cross_region_ridge import (
    Dataset,
    load_dataset,
    leave_one_region_out_splits,
    kfold_indices,
    rmse,
    mae,
    r2_score,
    summarize_regions,
)


@dataclass(frozen=True)
class RFParams:
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | float

    def to_kwargs(self, n_features: int, random_state: int) -> Dict[str, object]:
        if isinstance(self.max_features, str):
            mf = self.max_features
        else:
            frac = float(self.max_features)
            if frac <= 0 or frac > 1:
                raise ValueError(f"max_features fraction must be (0,1], got {frac}")
            mf = max(1, int(np.floor(frac * n_features)))
        return {
            "n_estimators": 200,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": mf,
            "random_state": random_state,
            "n_jobs": -1,
        }


def rf_param_grid() -> List[RFParams]:
    grid = []
    depth_options = [8, 16, None]
    leaf_options = [1, 10, 20]
    feature_options = [0.3, "sqrt", 1.0]
    for depth, leaf, feat in product(depth_options, leaf_options, feature_options):
        grid.append(RFParams(depth, leaf, feat))
    return grid


def model_priority(params: RFParams) -> Tuple[int, int, int]:
    depth_order = {12: 0, 24: 1, None: 2}
    leaf_order = {10: 0, 5: 1, 1: 2}
    feat_order = {0.5: 0, "sqrt": 1, 1.0: 2}
    return (
        depth_order.get(params.max_depth, 1),
        leaf_order.get(params.min_samples_leaf, 1),
        feat_order.get(params.max_features, 1),
    )


def inner_rf_cv(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Sequence[RFParams],
    k: int,
    seed: int,
) -> Tuple[RFParams, List[Dict[str, object]]]:
    splits = kfold_indices(len(X), k, seed)
    results: List[Dict[str, object]] = []
    avg_rmse_map: Dict[RFParams, float] = {}

    for params in param_grid:
        fold_rmses: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            kwargs = params.to_kwargs(n_features=X.shape[1], random_state=seed + fold_idx)
            model = RandomForestRegressor(**kwargs)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            fold_rmses.append(rmse(y[val_idx], preds))
        avg_rmse = float(np.mean(fold_rmses))
        results.append(
            {
                "params": params,
                "fold_rmses": [float(v) for v in fold_rmses],
                "avg_rmse": avg_rmse,
            }
        )
        avg_rmse_map[params] = avg_rmse

    best_params = min(
        param_grid,
        key=lambda p: (avg_rmse_map[p], model_priority(p)),
    )
    return best_params, results


def outer_rf_cv(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Sequence[RFParams],
    outer_k: int,
    inner_k: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[RFParams, int], Dict[str, Dict[str, float]], Dict[RFParams, List[float]], Dict[RFParams, List[float]]]:
    splits = kfold_indices(len(X), outer_k, seed)
    outer_results: List[Dict[str, object]] = []
    metric_tracker: Dict[str, List[float]] = {"rmse": [], "mae": [], "r2": []}
    outer_frequency: Dict[RFParams, int] = {p: 0 for p in param_grid}
    inner_avg_records: Dict[RFParams, List[float]] = {p: [] for p in param_grid}
    outer_selected_rmse: Dict[RFParams, List[float]] = {p: [] for p in param_grid}

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        inner_seed = seed + 1000 + fold_idx
        best_params, inner_details = inner_rf_cv(
            X[train_idx],
            y[train_idx],
            param_grid=param_grid,
            k=inner_k,
            seed=inner_seed,
        )

        for entry in inner_details:
            params = entry["params"]
            inner_avg_records[params].append(float(entry["avg_rmse"]))

        kwargs = best_params.to_kwargs(n_features=X.shape[1], random_state=inner_seed)
        model = RandomForestRegressor(**kwargs)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        fold_rmse = rmse(y[val_idx], preds)
        fold_mae = mae(y[val_idx], preds)
        fold_r2 = r2_score(y[val_idx], preds)

        metric_tracker["rmse"].append(fold_rmse)
        metric_tracker["mae"].append(fold_mae)
        metric_tracker["r2"].append(fold_r2)
        outer_frequency[best_params] += 1
        outer_selected_rmse[best_params].append(fold_rmse)

        outer_results.append(
            {
                "fold": fold_idx,
                "params": best_params,
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

    return outer_results, outer_frequency, metric_summary, inner_avg_records, outer_selected_rmse


def select_final_params(inner_avg_records: Dict[RFParams, List[float]]) -> Tuple[RFParams, Dict[RFParams, float]]:
    avg_map: Dict[RFParams, float] = {}
    for params, values in inner_avg_records.items():
        avg_map[params] = float(np.mean(values)) if values else float("inf")
    best_params = min(avg_map.keys(), key=lambda p: (avg_map[p], model_priority(p)))
    return best_params, avg_map


def evaluate_region_split(
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_region: str,
    param_grid: Sequence[RFParams],
    outer_k: int,
    inner_k: int,
    seed: int,
) -> Dict[str, object]:
    X_train = dataset.X[train_idx]
    y_train = dataset.y[train_idx]
    X_test = dataset.X[test_idx]
    y_test = dataset.y[test_idx]

    (
        outer_results,
        outer_frequency,
        outer_summary,
        inner_avg_records,
        outer_selected_rmse,
    ) = outer_rf_cv(
        X_train,
        y_train,
        param_grid=param_grid,
        outer_k=outer_k,
        inner_k=inner_k,
        seed=seed,
    )
    best_params, avg_map = select_final_params(inner_avg_records)

    kwargs = best_params.to_kwargs(n_features=X_train.shape[1], random_state=seed + 99)
    model = RandomForestRegressor(**kwargs)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_metrics = {
        "rmse": rmse(y_test, preds),
        "mae": mae(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
    # Chunk the held-out region into up to 10 subsets to inspect variability.
    test_subsplit_details: List[Dict[str, float]] = []
    subsplit_metrics_collectors: Dict[str, List[float]] = {"rmse": [], "mae": [], "r2": []}
    if len(test_idx) > 0:
        rng = np.random.default_rng(seed + 23456)
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
        "best_params": best_params,
        "outer_results": outer_results,
        "outer_frequency": outer_frequency,
        "outer_summary": outer_summary,
        "inner_avg_records": inner_avg_records,
        "outer_selected_rmse": outer_selected_rmse,
        "avg_map": avg_map,
        "test_metrics": test_metrics,
        "test_subsplit_details": test_subsplit_details,
        "test_subsplit_summary": test_subsplit_summary,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
    }


def evaluate_all_regions(
    dataset: Dataset,
    param_grid: Sequence[RFParams],
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
                param_grid=param_grid,
                outer_k=outer_k,
                inner_k=inner_k,
                seed=region_seed,
            )
        )
    return results


def run_single_region(
    dataset: Dataset,
    target_region: str,
    param_grid: Sequence[RFParams],
    outer_k: int = 10,
    inner_k: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    for split_idx, (train_idx, test_idx, test_region) in enumerate(
        leave_one_region_out_splits(dataset.regions)
    ):
        if test_region == target_region:
            region_seed = seed + 10_000 * split_idx
            return evaluate_region_split(
                dataset,
                train_idx,
                test_idx,
                test_region,
                param_grid=param_grid,
                outer_k=outer_k,
                inner_k=inner_k,
                seed=region_seed,
            )
    raise ValueError(f"Target region '{target_region}' not present in dataset")


if __name__ == "__main__":
    ds = load_dataset("dataset.csv")
    print("Dataset summary:")
    print(f"- samples: {ds.X.shape[0]}, features: {ds.X.shape[1]}")
    for region, count in summarize_regions(ds.regions):
        print(f"  {region}: {count}")

    grid = rf_param_grid()
    results = evaluate_all_regions(ds, param_grid=grid, outer_k=10, inner_k=3, seed=42)

    print("\nCross-region Random Forest evaluation:")
    for res in results:
        region = res["test_region"]
        best_params: RFParams = res["best_params"]
        metrics = res["test_metrics"]
        outer_summary = res["outer_summary"]
        freq = res["outer_frequency"]
        inner_avg = res["inner_avg_records"]

        print(f"\n=== Test region: {region} ===")
        print(
            f"Final params: max_depth={best_params.max_depth}, min_samples_leaf={best_params.min_samples_leaf}, "
            f"max_features={best_params.max_features}"
        )
        print(
            f"Test metrics -> RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}"
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
        print("Inner-CV average RMSE across folds (top 5 combos):")
        sorted_avg = sorted(
            (
                (
                    params,
                    np.mean(values),
                    np.std(values, ddof=1) if len(values) > 1 else 0.0,
                    len(values),
                )
                for params, values in inner_avg.items()
                if values
            ),
            key=lambda item: (item[1], model_priority(item[0])),
        )
        for params, mean_val, std_val, count_val in sorted_avg[:5]:
            print(
                f"  max_depth={params.max_depth}, min_samples_leaf={params.min_samples_leaf}, max_features={params.max_features}"
                f" -> avg RMSE={mean_val:.2f} ± {std_val:.2f} (n={count_val})"
            )
        print(
            "Outer 10-fold validation (mean ± std): "
            f"RMSE={outer_summary['rmse']['mean']:.2f} ± {outer_summary['rmse']['std']:.2f}, "
            f"MAE={outer_summary['mae']['mean']:.2f} ± {outer_summary['mae']['std']:.2f}, "
            f"R2={outer_summary['r2']['mean']:.4f} ± {outer_summary['r2']['std']:.4f}"
        )
        print("Parameter selection frequency across outer folds:")
        for params, count_sel in freq.items():
            if count_sel:
                print(
                    f"  max_depth={params.max_depth}, min_samples_leaf={params.min_samples_leaf}, "
                    f"max_features={params.max_features}: {count_sel} folds"
                )
        print("Outer fold details:")
        for entry in res["outer_results"]:
            params = entry["params"]
            print(
                f"  Fold {entry['fold']:02d}: params=(max_depth={params.max_depth}, min_samples_leaf={params.min_samples_leaf}, max_features={params.max_features}), "
                f"RMSE={entry['rmse']:.2f}, MAE={entry['mae']:.2f}, R2={entry['r2']:.4f}"
            )
            print("    Inner 3-fold RMSE per param:")
            for inner in entry["inner_summary"]:
                params_i: RFParams = inner["params"]
                fold_rmses = ", ".join(f"{v:.2f}" for v in inner["fold_rmses"])
                print(
                    f"      max_depth={params_i.max_depth}, min_samples_leaf={params_i.min_samples_leaf}, max_features={params_i.max_features}"
                    f" -> folds [{fold_rmses}] | avg={inner['avg_rmse']:.2f}"
                )
