"""Data loading and schema validation utilities."""

from .dataset import PenguinDataLoader, get_sample_data, load_penguins_data
from .schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    PenguinDataset,
    PenguinRecord,
    get_feature_info,
    validate_dataframe_schema,
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
    "ALL_FEATURES",
]
