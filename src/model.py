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
