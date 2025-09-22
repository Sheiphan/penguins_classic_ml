"""Data loading and schema validation utilities."""

from .dataset import PenguinDataLoader, load_penguins_data, get_sample_data
from .schema import (
    PenguinRecord,
    PenguinDataset,
    validate_dataframe_schema,
    get_feature_info,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ALL_FEATURES
)

__all__ = [
    "PenguinDataLoader",
    "load_penguins_data", 
    "get_sample_data",
    "PenguinRecord",
    "PenguinDataset",
    "validate_dataframe_schema",
    "get_feature_info",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES", 
    "TARGET",
    "ALL_FEATURES"
]