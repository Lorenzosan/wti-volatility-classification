import yaml
from src.types import DataConfig, FeaturesConfig, ProjectConfig

def load_config(path: str) -> ProjectConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = DataConfig(**cfg["data"])
    feat_cfg = FeaturesConfig(**cfg["features"])
    return ProjectConfig(data=data_cfg, features=feat_cfg)

