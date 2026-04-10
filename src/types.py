from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    asset: str
    start_date: str
    end_date: Optional[str] = None

    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.today().strftime("%Y-%m-%d")

@dataclass            
class FeaturesConfig:
    lags: list[int]
    short_volatility_window: Optional[int] = None
    long_volatility_window: Optional[int] = None
    forecast_horizon: Optional[int]  = None

    def __post_init__(self):
        if self.short_volatility_window is None:
            self.short_volatility_window = 5
        if self.long_volatility_window is None:
            self.long_volatility_window = 20
        if self.forecast_horizon is None:
            self.forecast_horizon = 5

@dataclass
class TrainConfig:
    test_size: float = 0.2
    ridge_alpha: float = 0.5
    rf_n_estimators: int = 200
    rf_max_depth: int = 5
    random_state: int = 0
    baseline_column: str = "std_20"
            
@dataclass
class ProjectConfig:
    data: DataConfig
    features: FeaturesConfig
    train: TrainConfig
