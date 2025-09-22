"""Tests for preprocessing pipeline."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.preprocess import (PenguinPreprocessor, create_preprocessor,
                                     get_preprocessing_info, preprocess_data)


class TestPenguinPreprocessor:
    """Test PenguinPreprocessor functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "bill_length_mm": [39.1, 39.5, None, 36.7, 40.3],
                "bill_depth_mm": [18.7, 17.4, 18.0, 19.3, 20.6],
                "flipper_length_mm": [181.0, 186.0, 195.0, 193.0, 190.0],
                "body_mass_g": [3750.0, 3800.0, 3250.0, 3450.0, 3650.0],
                "year": [2007, 2007, 2007, 2007, 2007],
                "island": ["Torgersen", "Torgersen", None, "Torgersen", "Biscoe"],
                "sex": ["MALE", "FEMALE", "FEMALE", "FEMALE", "MALE"],
            }
        )

    def test_build_preprocessor(self):
        """Test building the preprocessing pipeline."""
        preprocessor = PenguinPreprocessor()
        sklearn_preprocessor = preprocessor.build_preprocessor()

        assert isinstance(sklearn_preprocessor, ColumnTransformer)
        assert len(sklearn_preprocessor.transformers) == 2

        # Check transformer names
        transformer_names = [name for name, _, _ in sklearn_preprocessor.transformers]
        assert "numeric" in transformer_names
        assert "categorical" in transformer_names

    def test_fit_transform(self, sample_data):
        """Test fitting and transforming data."""
        preprocessor = PenguinPreprocessor()

        # Should work with fit_transform
        X_transformed = preprocessor.fit_transform(sample_data)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(sample_data)
        assert X_transformed.shape[1] > len(
            NUMERIC_FEATURES
        )  # Should have more columns due to one-hot encoding

    def test_fit_then_transform(self, sample_data):
        """Test separate fit and transform calls."""
        preprocessor = PenguinPreprocessor()

        # Fit first
        preprocessor.fit(sample_data)

        # Then transform
        X_transformed = preprocessor.transform(sample_data)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(sample_data)

    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        preprocessor = PenguinPreprocessor()

        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_data)

    def test_handle_missing_values(self, sample_data):
        """Test handling of missing values."""
        preprocessor = PenguinPreprocessor()
        X_transformed = preprocessor.fit_transform(sample_data)

        # Should not have any NaN values after preprocessing
        assert not X_transformed.isna().any().any()

    def test_numeric_preprocessing(self, sample_data):
        """Test numeric feature preprocessing."""
        preprocessor = PenguinPreprocessor()
        X_transformed = preprocessor.fit_transform(sample_data)

        # Get numeric columns (first few columns should be numeric features)
        numeric_cols = [col for col in X_transformed.columns if col in NUMERIC_FEATURES]

        # Check that numeric features are standardized (approximately mean=0)
        for col in numeric_cols:
            values = X_transformed[col]
            assert abs(values.mean()) < 1e-10  # Should be close to 0

            # Check that values are finite (not NaN or inf)
            assert not values.isna().any()  # No NaN values
            assert not (values == float("inf")).any()  # No inf values
            assert not (values == float("-inf")).any()  # No -inf values

            # For features with variance, check standardization worked
            original_col_data = sample_data[col]
            if original_col_data.std() > 0:  # Only check if original had variance
                assert values.std() > 0  # Should maintain some variance after scaling

    def test_categorical_preprocessing(self, sample_data):
        """Test categorical feature preprocessing."""
        preprocessor = PenguinPreprocessor()
        X_transformed = preprocessor.fit_transform(sample_data)

        # Should have one-hot encoded categorical features
        categorical_cols = [
            col
            for col in X_transformed.columns
            if any(feat in col for feat in CATEGORICAL_FEATURES)
        ]

        assert len(categorical_cols) > len(
            CATEGORICAL_FEATURES
        )  # Should have more columns due to encoding

        # All categorical columns should be binary (0 or 1)
        for col in categorical_cols:
            unique_values = set(X_transformed[col].unique())
            assert unique_values.issubset({0.0, 1.0})

    def test_get_feature_names(self, sample_data):
        """Test getting feature names after fitting."""
        preprocessor = PenguinPreprocessor()
        preprocessor.fit(sample_data)

        feature_names = preprocessor.get_feature_names()

        assert feature_names is not None
        assert len(feature_names) > 0

        # Should include numeric feature names
        for numeric_feat in NUMERIC_FEATURES:
            if numeric_feat in sample_data.columns:
                assert numeric_feat in feature_names

    def test_different_imputation_strategies(self, sample_data):
        """Test different imputation strategies."""
        # Test median imputation for numeric features
        preprocessor = PenguinPreprocessor(numeric_strategy="median")
        X_transformed = preprocessor.fit_transform(sample_data)

        assert not X_transformed.isna().any().any()

        # Test most_frequent for categorical features
        preprocessor = PenguinPreprocessor(categorical_strategy="most_frequent")
        X_transformed = preprocessor.fit_transform(sample_data)

        assert not X_transformed.isna().any().any()

    def test_drop_first_encoding(self, sample_data):
        """Test drop_first option for one-hot encoding."""
        preprocessor_no_drop = PenguinPreprocessor(drop_first=False)
        X_no_drop = preprocessor_no_drop.fit_transform(sample_data)

        preprocessor_drop = PenguinPreprocessor(drop_first=True)
        X_drop = preprocessor_drop.fit_transform(sample_data)

        # Should have fewer columns when dropping first category
        assert X_drop.shape[1] < X_no_drop.shape[1]

    def test_transform_new_data(self, sample_data):
        """Test transforming new data with fitted preprocessor."""
        preprocessor = PenguinPreprocessor()
        preprocessor.fit(sample_data)

        # Create new data with same structure
        new_data = pd.DataFrame(
            {
                "bill_length_mm": [41.0, 42.0],
                "bill_depth_mm": [19.0, 20.0],
                "flipper_length_mm": [185.0, 190.0],
                "body_mass_g": [3600.0, 3700.0],
                "year": [2008, 2008],
                "island": ["Torgersen", "Dream"],
                "sex": ["MALE", "FEMALE"],
            }
        )

        X_new_transformed = preprocessor.transform(new_data)

        assert len(X_new_transformed) == 2
        assert (
            X_new_transformed.shape[1]
            == preprocessor.fit_transform(sample_data).shape[1]
        )


class TestCreatePreprocessor:
    """Test preprocessor creation function."""

    def test_create_preprocessor_default(self):
        """Test creating preprocessor with default parameters."""
        preprocessor = create_preprocessor()

        assert isinstance(preprocessor, PenguinPreprocessor)
        assert preprocessor.numeric_strategy == "mean"
        assert preprocessor.categorical_strategy == "constant"

    def test_create_preprocessor_custom(self):
        """Test creating preprocessor with custom parameters."""
        preprocessor = create_preprocessor(
            numeric_strategy="median",
            categorical_strategy="most_frequent",
            handle_unknown="error",
            drop_first=True,
        )

        assert preprocessor.numeric_strategy == "median"
        assert preprocessor.categorical_strategy == "most_frequent"
        assert preprocessor.handle_unknown == "error"
        assert preprocessor.drop_first is True


class TestPreprocessData:
    """Test convenience function for preprocessing data."""

    @pytest.fixture
    def train_test_data(self):
        """Create train and test data for testing."""
        train_data = pd.DataFrame(
            {
                "bill_length_mm": [39.1, 39.5, 40.3, 36.7],
                "bill_depth_mm": [18.7, 17.4, 18.0, 19.3],
                "flipper_length_mm": [181.0, 186.0, 195.0, 193.0],
                "body_mass_g": [3750.0, 3800.0, 3250.0, 3450.0],
                "year": [2007, 2007, 2007, 2007],
                "island": ["Torgersen", "Torgersen", "Torgersen", "Torgersen"],
                "sex": ["MALE", "FEMALE", "FEMALE", "FEMALE"],
            }
        )

        test_data = pd.DataFrame(
            {
                "bill_length_mm": [41.0, 42.0],
                "bill_depth_mm": [19.0, 20.0],
                "flipper_length_mm": [185.0, 190.0],
                "body_mass_g": [3600.0, 3700.0],
                "year": [2008, 2008],
                "island": ["Biscoe", "Dream"],
                "sex": ["MALE", "FEMALE"],
            }
        )

        return train_data, test_data

    def test_preprocess_data_train_only(self, train_test_data):
        """Test preprocessing with only training data."""
        train_data, _ = train_test_data

        X_train_processed, X_test_processed, preprocessor = preprocess_data(train_data)

        assert isinstance(X_train_processed, pd.DataFrame)
        assert X_test_processed is None
        assert isinstance(preprocessor, PenguinPreprocessor)
        assert len(X_train_processed) == len(train_data)

    def test_preprocess_data_train_and_test(self, train_test_data):
        """Test preprocessing with both training and test data."""
        train_data, test_data = train_test_data

        X_train_processed, X_test_processed, preprocessor = preprocess_data(
            train_data, test_data
        )

        assert isinstance(X_train_processed, pd.DataFrame)
        assert isinstance(X_test_processed, pd.DataFrame)
        assert isinstance(preprocessor, PenguinPreprocessor)

        assert len(X_train_processed) == len(train_data)
        assert len(X_test_processed) == len(test_data)

        # Should have same number of columns
        assert X_train_processed.shape[1] == X_test_processed.shape[1]

    def test_preprocess_data_with_custom_params(self, train_test_data):
        """Test preprocessing with custom parameters."""
        train_data, test_data = train_test_data

        X_train_processed, X_test_processed, preprocessor = preprocess_data(
            train_data, test_data, numeric_strategy="median", drop_first=True
        )

        assert preprocessor.numeric_strategy == "median"
        assert preprocessor.drop_first is True


class TestGetPreprocessingInfo:
    """Test preprocessing information function."""

    @pytest.fixture
    def fitted_preprocessor(self):
        """Create a fitted preprocessor for testing."""
        sample_data = pd.DataFrame(
            {
                "bill_length_mm": [39.1, 39.5, 40.3],
                "bill_depth_mm": [18.7, 17.4, 18.0],
                "flipper_length_mm": [181.0, 186.0, 195.0],
                "body_mass_g": [3750.0, 3800.0, 3250.0],
                "year": [2007, 2007, 2007],
                "island": ["Torgersen", "Torgersen", "Biscoe"],
                "sex": ["MALE", "FEMALE", "FEMALE"],
            }
        )

        preprocessor = PenguinPreprocessor()
        preprocessor.fit(sample_data)
        return preprocessor

    def test_get_preprocessing_info_fitted(self, fitted_preprocessor):
        """Test getting info from fitted preprocessor."""
        info = get_preprocessing_info(fitted_preprocessor)

        assert info["status"] == "fitted"
        assert "numeric_features" in info
        assert "categorical_features" in info
        assert "output_features" in info
        assert "transformers" in info

        assert info["numeric_features"] == NUMERIC_FEATURES
        assert info["categorical_features"] == CATEGORICAL_FEATURES

        # Check transformer info
        assert "numeric" in info["transformers"]
        assert "categorical" in info["transformers"]

        numeric_info = info["transformers"]["numeric"]
        assert "imputation_strategy" in numeric_info
        assert "scaling_mean" in numeric_info
        assert "scaling_scale" in numeric_info

        categorical_info = info["transformers"]["categorical"]
        assert "imputation_strategy" in categorical_info
        assert "categories" in categorical_info

    def test_get_preprocessing_info_not_fitted(self):
        """Test getting info from unfitted preprocessor."""
        preprocessor = PenguinPreprocessor()
        info = get_preprocessing_info(preprocessor)

        assert info["status"] == "not_fitted"
