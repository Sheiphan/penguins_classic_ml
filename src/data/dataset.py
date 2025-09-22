"""Data loading utilities for the penguins dataset."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    validate_dataframe_schema
)


class PenguinDataLoader:
    """Data loader for the penguins dataset."""
    
    def __init__(self, data_path: str = "data/raw/penguins_lter.csv"):
        """Initialize the data loader.
        
        Args:
            data_path: Path to the raw penguins CSV file
        """
        self.data_path = Path(data_path)
        self._raw_data: Optional[pd.DataFrame] = None
        self._clean_data: Optional[pd.DataFrame] = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if self._raw_data is None:
            self._raw_data = pd.read_csv(self.data_path)
        
        return self._raw_data.copy()
    
    def load_clean_data(self) -> pd.DataFrame:
        """Load and clean the data according to schema."""
        if self._clean_data is None:
            raw_data = self.load_raw_data()
            self._clean_data = validate_dataframe_schema(raw_data)
        
        return self._clean_data.copy()
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get features (X) and target (y) from clean data.
        
        Returns:
            Tuple of (features_df, target_series)
        """
        clean_data = self.load_clean_data()
        
        # Remove rows where target is missing
        clean_data = clean_data.dropna(subset=[TARGET])
        
        X = clean_data[ALL_FEATURES]
        y = clean_data[TARGET]
        
        return X, y
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by target variable
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.get_features_and_target()
        
        # Only stratify if we have enough samples per class
        stratify_param = y if stratify else None
        if stratify:
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            if min_class_count < 2:
                print(f"Warning: Minimum class count is {min_class_count}, disabling stratification")
                stratify_param = None
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
    
    def get_data_info(self) -> dict:
        """Get information about the loaded data."""
        clean_data = self.load_clean_data()
        X, y = self.get_features_and_target()
        
        info = {
            "total_rows": len(clean_data),
            "rows_with_target": len(X),
            "features": {
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
                "total": len(ALL_FEATURES)
            },
            "target_distribution": y.value_counts().to_dict(),
            "missing_values": {}
        }
        
        # Check for missing values in features
        for feature in ALL_FEATURES:
            if feature in X.columns:
                missing_count = X[feature].isna().sum()
                if missing_count > 0:
                    info["missing_values"][feature] = missing_count
        
        return info
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to CSV file.
        
        Args:
            output_path: Path where to save the processed data
        """
        clean_data = self.load_clean_data()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        clean_data.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")


def load_penguins_data(
    data_path: str = "data/raw/penguins_lter.csv",
    return_split: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience function to load penguins data.
    
    Args:
        data_path: Path to the raw data file
        return_split: If True, return train/test split
        test_size: Test set size (only used if return_split=True)
        random_state: Random seed
        stratify: Whether to stratify the split
        
    Returns:
        If return_split=False: (X, y)
        If return_split=True: (X_train, X_test, y_train, y_test)
    """
    loader = PenguinDataLoader(data_path)
    
    if return_split:
        return loader.train_test_split(
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
    else:
        return loader.get_features_and_target()


def get_sample_data(n_samples: int = 10) -> pd.DataFrame:
    """Get a sample of the penguins data for testing.
    
    Args:
        n_samples: Number of samples to return
        
    Returns:
        DataFrame with sample data
    """
    loader = PenguinDataLoader()
    clean_data = loader.load_clean_data()
    
    # Remove rows with missing target
    clean_data = clean_data.dropna(subset=[TARGET])
    
    # Sample data, ensuring we get different species if possible
    if len(clean_data) >= n_samples:
        # Try to get balanced samples across species
        sample_data = clean_data.groupby(TARGET).apply(
            lambda x: x.sample(min(len(x), max(1, n_samples // 3)), random_state=42)
        ).reset_index(drop=True)
        
        # If we don't have enough, just sample randomly
        if len(sample_data) < n_samples:
            additional_needed = n_samples - len(sample_data)
            remaining_data = clean_data[~clean_data.index.isin(sample_data.index)]
            if len(remaining_data) > 0:
                additional_samples = remaining_data.sample(
                    min(len(remaining_data), additional_needed), 
                    random_state=42
                )
                sample_data = pd.concat([sample_data, additional_samples]).reset_index(drop=True)
    else:
        sample_data = clean_data.copy()
    
    return sample_data.head(n_samples)