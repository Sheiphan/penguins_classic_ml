"""Preprocessing pipeline for the penguins dataset using sklearn."""


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class PenguinPreprocessor:
    """Preprocessing pipeline for penguins dataset."""

    def __init__(
        self,
        numeric_strategy: str = "mean",
        categorical_strategy: str = "constant",
        categorical_fill_value: str = "Missing",
        handle_unknown: str = "ignore",
        drop_first: bool = False,
    ):
        """Initialize the preprocessor.

        Args:
            numeric_strategy: Strategy for imputing numeric features
            categorical_strategy: Strategy for imputing categorical features
            categorical_fill_value: Fill value for categorical imputation
            handle_unknown: How to handle unknown categories in encoding
            drop_first: Whether to drop first category in one-hot encoding
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first

        self._preprocessor: ColumnTransformer | None = None
        self._feature_names: list | None = None

    def build_preprocessor(self) -> ColumnTransformer:
        """Build the preprocessing pipeline.

        Returns:
            Configured ColumnTransformer
        """
        # Numeric preprocessing pipeline
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=self.numeric_strategy)),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical preprocessing pipeline
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.categorical_strategy,
                        fill_value=self.categorical_fill_value,
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        drop="first" if self.drop_first else None,
                        sparse_output=False,
                    ),
                ),
            ]
        )

        # Combine pipelines
        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ]
        )

        self._preprocessor = preprocessor
        return preprocessor

    def fit(self, X: pd.DataFrame) -> "PenguinPreprocessor":
        """Fit the preprocessor to the data.

        Args:
            X: Input features DataFrame

        Returns:
            Self for method chaining
        """
        if self._preprocessor is None:
            self.build_preprocessor()

        self._preprocessor.fit(X)
        self._extract_feature_names(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor.

        Args:
            X: Input features DataFrame

        Returns:
            Transformed features DataFrame
        """
        if self._preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transform")

        # Transform the data
        X_transformed = self._preprocessor.transform(X)

        # Convert back to DataFrame with proper column names
        if self._feature_names is not None:
            return pd.DataFrame(
                X_transformed, columns=self._feature_names, index=X.index
            )
        else:
            return pd.DataFrame(X_transformed, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the data.

        Args:
            X: Input features DataFrame

        Returns:
            Transformed features DataFrame
        """
        return self.fit(X).transform(X)

    def _extract_feature_names(self, X: pd.DataFrame) -> None:
        """Extract feature names after fitting."""
        if self._preprocessor is None:
            return

        feature_names = []

        # Get numeric feature names (same as input)
        numeric_features = [f for f in NUMERIC_FEATURES if f in X.columns]
        feature_names.extend(numeric_features)

        # Get categorical feature names (from one-hot encoder)
        categorical_features = [f for f in CATEGORICAL_FEATURES if f in X.columns]
        if categorical_features:
            # Get the encoder from the pipeline
            categorical_transformer = self._preprocessor.named_transformers_[
                "categorical"
            ]
            encoder = categorical_transformer.named_steps["encoder"]

            if hasattr(encoder, "get_feature_names_out"):
                # For newer sklearn versions
                cat_feature_names = encoder.get_feature_names_out(categorical_features)
            else:
                # Fallback for older versions
                cat_feature_names = []
                for i, feature in enumerate(categorical_features):
                    categories = encoder.categories_[i]
                    if self.drop_first:
                        categories = categories[1:]  # Skip first category
                    for category in categories:
                        cat_feature_names.append(f"{feature}_{category}")

            feature_names.extend(cat_feature_names)

        self._feature_names = feature_names

    def get_feature_names(self) -> list | None:
        """Get the names of the output features.

        Returns:
            List of feature names or None if not fitted
        """
        return self._feature_names

    def get_preprocessor(self) -> ColumnTransformer | None:
        """Get the underlying sklearn preprocessor.

        Returns:
            The fitted ColumnTransformer or None if not fitted
        """
        return self._preprocessor


def create_preprocessor(
    numeric_strategy: str = "mean",
    categorical_strategy: str = "constant",
    categorical_fill_value: str = "Missing",
    handle_unknown: str = "ignore",
    drop_first: bool = False,
) -> PenguinPreprocessor:
    """Create a new preprocessor instance.

    Args:
        numeric_strategy: Strategy for imputing numeric features
        categorical_strategy: Strategy for imputing categorical features
        categorical_fill_value: Fill value for categorical imputation
        handle_unknown: How to handle unknown categories in encoding
        drop_first: Whether to drop first category in one-hot encoding

    Returns:
        Configured PenguinPreprocessor instance
    """
    return PenguinPreprocessor(
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        categorical_fill_value=categorical_fill_value,
        handle_unknown=handle_unknown,
        drop_first=drop_first,
    )


def preprocess_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, **preprocessor_kwargs
) -> tuple[pd.DataFrame, pd.DataFrame | None, PenguinPreprocessor]:
    """Convenience function to preprocess training and test data.

    Args:
        X_train: Training features
        X_test: Test features (optional)
        **preprocessor_kwargs: Arguments for preprocessor configuration

    Returns:
        Tuple of (X_train_processed, X_test_processed, fitted_preprocessor)
    """
    preprocessor = create_preprocessor(**preprocessor_kwargs)

    # Fit on training data and transform
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data if provided
    X_test_processed = None
    if X_test is not None:
        X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor


def get_preprocessing_info(preprocessor: PenguinPreprocessor) -> dict:
    """Get information about the preprocessing pipeline.

    Args:
        preprocessor: Fitted preprocessor

    Returns:
        Dictionary with preprocessing information
    """
    if preprocessor.get_preprocessor() is None:
        return {"status": "not_fitted"}

    info = {
        "status": "fitted",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "output_features": preprocessor.get_feature_names(),
        "transformers": {},
    }

    # Get information about each transformer
    sklearn_preprocessor = preprocessor.get_preprocessor()

    # Numeric transformer info
    numeric_transformer = sklearn_preprocessor.named_transformers_["numeric"]
    imputer = numeric_transformer.named_steps["imputer"]
    scaler = numeric_transformer.named_steps["scaler"]

    info["transformers"]["numeric"] = {
        "imputation_strategy": imputer.strategy,
        "imputation_values": getattr(imputer, "statistics_", None),
        "scaling_mean": getattr(scaler, "mean_", None),
        "scaling_scale": getattr(scaler, "scale_", None),
    }

    # Categorical transformer info
    categorical_transformer = sklearn_preprocessor.named_transformers_["categorical"]
    cat_imputer = categorical_transformer.named_steps["imputer"]
    encoder = categorical_transformer.named_steps["encoder"]

    info["transformers"]["categorical"] = {
        "imputation_strategy": cat_imputer.strategy,
        "imputation_fill_value": getattr(cat_imputer, "fill_value", None),
        "categories": getattr(encoder, "categories_", None),
        "drop_first": encoder.drop == "first" if hasattr(encoder, "drop") else False,
    }

    return info
