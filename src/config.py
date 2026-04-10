import yaml
from src.types import DataConfig, FeaturesConfig, TrainConfig, ProjectConfig

def load_config(path: str) -> ProjectConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = DataConfig(**cfg["data"])
    feat_cfg = FeaturesConfig(**cfg["features"])
    train_cfg = TrainConfig(**cfg["train"])
    return ProjectConfig(data=data_cfg, features=feat_cfg, train=train_cfg)

