import pandas as pd
from typing import Dict, Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.types import ProjectConfig, TrainConfig

def clean_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")

    if y.name is None:
        y = y.rename("target")

    dataset = X.join(y, how="inner").dropna()

    if dataset.empty:
        raise ValueError("No data left after joining and dropping missing values")

    X_clean = dataset[X.columns].copy()
    y_clean = dataset[y.name].copy()

    if not X_clean.index.equals(y_clean.index):
        raise ValueError("X and y indices are not aligned after cleaning")

    return X_clean, y_clean


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Split index is invalid; check dataset size and test_size")

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def build_models(config: TrainConfig) -> Dict[str, Any]:
    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=config.ridge_alpha)),
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        random_state=config.random_state,
        n_jobs=-1,
    )

    return {
        "ridge": ridge_model,
        "rf": rf_model,
    }


def compute_metrics(y_true: pd.Series, y_pred: pd.Series | pd.Index | list) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> Dict[str, Any]:
    X, y = clean_data(X, y)
    X_train, X_test, y_train, y_test = chronological_split(X, y, config.train.test_size)

    if config.train.baseline_column not in X_test.columns:
        raise KeyError(f"Baseline column '{config.train.baseline_column}' not found in X_test")

    baseline_pred = X_test[config.train.baseline_column].copy()

    models = build_models(config.train)
    fitted_models = {}
    predictions = {
        "baseline": baseline_pred
    }

    for name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        predictions[name] = pd.Series(
            fitted_model.predict(X_test),
            index=y_test.index,
            name=f"{name}_pred"
        )
        fitted_models[name] = fitted_model

    results = pd.DataFrame(
        {
            "actual": y_test,
            "baseline_pred": predictions["baseline"],
            "ridge_pred": predictions["ridge"],
            "rf_pred": predictions["rf"],
        }
    )

    metrics = {
        "baseline": compute_metrics(y_test, predictions["baseline"]),
        "ridge": compute_metrics(y_test, predictions["ridge"]),
        "rf": compute_metrics(y_test, predictions["rf"]),
    }

    return {
        "results": results,
        "metrics": metrics,
        "models": fitted_models,
        "split_index": X_train.index[-1],
    }

def walk_forward_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig,
    train_fraction: float = 0.6,
    step_size: int = 20,
    test_window: int = 20,
) -> Dict[str, Any]:
    X, y = clean_data(X, y)

    n_samples = len(X)
    initial_train_size = int(n_samples * train_fraction)

    if initial_train_size <= 0 or initial_train_size >= n_samples:
        raise ValueError("Invalid initial train size")

    if step_size <= 0 or test_window <= 0:
        raise ValueError("step_size and test_window must be positive")

    baseline_column = config.train.baseline_column
    if baseline_column not in X.columns:
        raise KeyError(f"Baseline column '{baseline_column}' not found in X")

    models = build_models(config.train)

    all_predictions = []
    all_actuals = []

    for start_test in range(initial_train_size, n_samples - test_window + 1, step_size):
        end_test = start_test + test_window

        X_train = X.iloc[:start_test].copy()
        y_train = y.iloc[:start_test].copy()

        X_test = X.iloc[start_test:end_test].copy()
        y_test = y.iloc[start_test:end_test].copy()

        fold_df = pd.DataFrame(index=y_test.index)
        fold_df["actual"] = y_test
        fold_df["baseline_pred"] = X_test[baseline_column]

        for name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)

            preds = fitted_model.predict(X_test)
            fold_df[f"{name}_pred"] = preds

        all_predictions.append(fold_df)
        all_actuals.append(y_test)

    if not all_predictions:
        raise ValueError("No walk-forward folds were created")

    results = pd.concat(all_predictions).sort_index()

    metrics = {
        "baseline": compute_metrics(results["actual"], results["baseline_pred"]),
        "ridge": compute_metrics(results["actual"], results["ridge_pred"]),
        "rf": compute_metrics(results["actual"], results["rf_pred"]),
    }

    return {
        "results": results,
        "metrics": metrics,
        "n_folds": len(all_predictions),
        "initial_train_size": initial_train_size,
        "step_size": step_size,
        "test_window": test_window,
    }
