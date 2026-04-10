import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def clean_data(X, y):
    dataset = X.join(y, how="inner").dropna()
    X_clean = dataset[X.columns]
    y_clean = dataset["target"]
    print("INFO: dataset cleaning done")
    return X_clean, y_clean

def train_and_evaluate(X, y, config):
    X, y = clean_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False  # you don't want to shuffle the time series ;)
    )

    print("INFO: testing models ..")
    
    # Baseline
    baseline_pred = X_test["std_20"]
    print("  .. Baseline")
    
    # Linear model (Ridge)
    lin_regr = linear_model.Ridge(alpha=0.5)
    lin_regr.fit(X_train, y_train)
    lin_regr_pred = lin_regr.predict(X_test)
    print("  .. Linear model")

    # Random forest
    rf_regr = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=0
    )
    rf_regr.fit(X_train, y_train)
    rf_regr_pred = rf_regr.predict(X_test)
    print("  .. Random forest")
    
    results = pd.DataFrame({
        "actual": y_test,
        "baseline_pred": baseline_pred,
        "ridge_pred": lin_regr_pred,
        "rf_pred": rf_regr_pred,
    }, index=y_test.index)

    metrics = {
        "baseline_mae": mean_absolute_error(y_test, baseline_pred),
        "baseline_mse": mean_squared_error(y_test, baseline_pred),
        "ridge_mae": mean_absolute_error(y_test, lin_regr_pred),
        "ridge_mse": mean_squared_error(y_test, lin_regr_pred),
        "rf_mae": mean_absolute_error(y_test, rf_regr_pred),
        "rf_mse": mean_squared_error(y_test, rf_regr_pred)
    }

    return {"results": results,
            "metrics": metrics,
            "models": {
                "ridge": lin_regr,
                "rf": rf_regr
                }
            }
