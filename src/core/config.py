"""Configuration management using Pydantic for type-safe settings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class PathConfig(BaseModel):
    """Configuration for data and model paths."""

    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    processed_dir: str = "data/processed"
    model_dir: str = "models"
    metrics_dir: str = "models/metrics"

    @field_validator("*", mode="before")
    @classmethod
    def ensure_path_exists(cls, v):
        """Ensure paths exist when accessed."""
        if isinstance(v, str):
            Path(v).mkdir(parents=True, exist_ok=True)
        return v


class TuneConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    grid: list[dict[str, Any]] = Field(default_factory=list)
    cv: int = 5
    scoring: str = "accuracy"
    n_jobs: int = -1


class ModelConfig(BaseModel):
    """Configuration for model training."""

    name: str = "RandomForestClassifier"
    params: dict[str, Any] = Field(default_factory=dict)
    tune: TuneConfig | None = None


class FeatureConfig(BaseModel):
    """Configuration for feature processing."""

    numeric_features: list[str] = Field(
        default_factory=lambda: [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "year",
        ]
    )
    categorical_features: list[str] = Field(default_factory=lambda: ["island", "sex"])
    target: str = "species"
    test_size: float = 0.2
    stratify: bool = True


class ExperimentConfig(BaseSettings):
    """Main experiment configuration."""

    seed: int = 42
    paths: PathConfig = Field(default_factory=PathConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


class APIConfig(BaseSettings):
    """Configuration for API serving."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Enable reload")
    workers: int = Field(default=1, description="Number of workers")

    class Config:
        env_file = ".env"


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Log level")
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    rotation: str = "1 day"
    retention: str = "30 days"
    log_file: str | None = "logs/app.log"

    class Config:
        env_file = ".env"


class ServingConfig(BaseSettings):
    """Configuration for model serving."""

    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model_path: str = "models/latest"

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path: str, config_class=ExperimentConfig):
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        # Return default configuration if file doesn't exist
        return config_class()

    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    return config_class(**config_data)


def save_config(config: BaseModel, config_path: str):
    """Save configuration to YAML file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
