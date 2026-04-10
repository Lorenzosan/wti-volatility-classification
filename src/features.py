import pandas as pd
from src.types import DataConfig, FeaturesConfig, ProjectConfig

def build_features(returns: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    asset    = config.data.asset 
    series   = returns[asset]
    features = pd.DataFrame(index=series.index)
    
    for lag in config.features.lags:
        features[f"return_lag_{lag}"] = series.shift(lag)
    features["return_lag_1_abs"] = series.abs().shift(1)
    
    short_window = config.features.short_volatility_window
    long_window  = config.features.long_volatility_window        
    features["mean_"+str(short_window)] = series.shift(1).rolling(window=short_window).mean()
    features["std_"+str(short_window)] = series.shift(1).rolling(window=short_window).std()
    features["mean_"+str(long_window)] = series.shift(1).rolling(window=long_window).mean()
    features["std_"+str(long_window)] = series.shift(1).rolling(window=long_window).std()

    print("INFO: features built successfully")
    return features


def build_target(returns: pd.DataFrame, config: ProjectConfig) -> pd.Series:
    asset = config.data.asset
    series = returns[asset]

    horizon = config.features.forecast_horizon

    target = series.rolling(window=horizon).std().shift(-horizon)
    target.name = "target"

    print("INFO: target built successfully")
    return target
