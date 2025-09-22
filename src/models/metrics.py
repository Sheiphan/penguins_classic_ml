"""Metrics calculation and storage system for model evaluation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelMetrics:
    """Calculate and manage model evaluation metrics."""

    def __init__(self):
        """Initialize the metrics calculator."""
        pass

    def calculate_metrics(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_pred: pd.Series | np.ndarray | list,
        y_proba: pd.DataFrame | np.ndarray | None = None,
        average: str = "weighted",
    ) -> dict[str, Any]:
        """Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            average: Averaging strategy for multi-class metrics

        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert to numpy arrays for consistency
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get unique classes
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

        # Basic metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            ),
        }

        # Per-class metrics
        per_class_metrics = {
            "precision": precision_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist(),
            "recall": recall_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist(),
            "f1_score": f1_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist(),
        }

        # Map per-class metrics to class names
        metrics["per_class"] = {}
        for i, class_name in enumerate(classes):
            metrics["per_class"][str(class_name)] = {
                "precision": float(per_class_metrics["precision"][i]),
                "recall": float(per_class_metrics["recall"][i]),
                "f1_score": float(per_class_metrics["f1_score"][i]),
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        metrics["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": [str(c) for c in classes],
        }

        # Classification report
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report

        # ROC AUC for multi-class (if probabilities provided)
        if y_proba is not None:
            try:
                if len(classes) == 2:
                    # Binary classification
                    if isinstance(y_proba, pd.DataFrame):
                        # Use probability of positive class
                        y_proba_positive = y_proba.iloc[:, 1].values
                    else:
                        y_proba_positive = (
                            y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                        )

                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_positive))
                else:
                    # Multi-class classification
                    if isinstance(y_proba, pd.DataFrame):
                        y_proba_array = y_proba.values
                    else:
                        y_proba_array = y_proba

                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true, y_proba_array, multi_class="ovr", average=average
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics["roc_auc"] = None

        # Additional statistics
        metrics["support"] = {
            str(class_name): int(np.sum(y_true == class_name)) for class_name in classes
        }

        metrics["total_samples"] = int(len(y_true))
        metrics["num_classes"] = len(classes)
        metrics["classes"] = [str(c) for c in classes]

        return metrics

    def calculate_regression_metrics(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_pred: pd.Series | np.ndarray | list,
    ) -> dict[str, Any]:
        """Calculate regression metrics (for future use).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary containing regression metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "total_samples": int(len(y_true)),
        }

        return metrics

    def compare_metrics(
        self,
        metrics_list: list[dict[str, Any]],
        metric_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compare metrics across multiple models.

        Args:
            metrics_list: List of metrics dictionaries
            metric_names: Names for each metrics set

        Returns:
            DataFrame comparing metrics
        """
        if metric_names is None:
            metric_names = [f"Model_{i + 1}" for i in range(len(metrics_list))]

        # Extract key metrics for comparison
        comparison_data = []

        for i, metrics in enumerate(metrics_list):
            row = {
                "model": metric_names[i],
                "accuracy": metrics.get("accuracy", None),
                "precision": metrics.get("precision", None),
                "recall": metrics.get("recall", None),
                "f1_score": metrics.get("f1_score", None),
                "roc_auc": metrics.get("roc_auc", None),
                "num_classes": metrics.get("num_classes", None),
                "total_samples": metrics.get("total_samples", None),
            }
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_model(
        self,
        metrics_list: list[dict[str, Any]],
        metric_names: list[str] | None = None,
        primary_metric: str = "f1_score",
    ) -> dict[str, Any]:
        """Find the best model based on a primary metric.

        Args:
            metrics_list: List of metrics dictionaries
            metric_names: Names for each metrics set
            primary_metric: Metric to use for comparison

        Returns:
            Dictionary with best model information
        """
        if metric_names is None:
            metric_names = [f"Model_{i + 1}" for i in range(len(metrics_list))]

        best_score = -1
        best_idx = 0

        for i, metrics in enumerate(metrics_list):
            score = metrics.get(primary_metric, -1)
            if score is not None and score > best_score:
                best_score = score
                best_idx = i

        return {
            "best_model_name": metric_names[best_idx],
            "best_model_index": best_idx,
            "best_score": best_score,
            "primary_metric": primary_metric,
            "best_metrics": metrics_list[best_idx],
        }


class MetricsStorage:
    """Handle storage and retrieval of model metrics."""

    def __init__(self, storage_dir: str = "models/metrics"):
        """Initialize metrics storage.

        Args:
            storage_dir: Directory to store metrics files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self, metrics: dict[str, Any], model_id: str, suffix: str = ""
    ) -> Path:
        """Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary to save
            model_id: Model identifier
            suffix: Optional suffix for filename

        Returns:
            Path to saved metrics file
        """
        filename = f"{model_id}_metrics"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"

        filepath = self.storage_dir / filename

        # Add metadata
        metrics_with_metadata = {
            "model_id": model_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": metrics,
        }

        with open(filepath, "w") as f:
            json.dump(metrics_with_metadata, f, indent=2, default=str)

        logger.info(f"Metrics saved to: {filepath}")
        return filepath

    def load_metrics(self, model_id: str, suffix: str = "") -> dict[str, Any]:
        """Load metrics from JSON file.

        Args:
            model_id: Model identifier
            suffix: Optional suffix for filename

        Returns:
            Loaded metrics dictionary

        Raises:
            FileNotFoundError: If metrics file doesn't exist
        """
        filename = f"{model_id}_metrics"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"

        filepath = self.storage_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        return data

    def list_metrics_files(self) -> list[str]:
        """List all available metrics files.

        Returns:
            List of metrics file names
        """
        metrics_files = list(self.storage_dir.glob("*_metrics*.json"))
        return [f.stem for f in metrics_files]

    def load_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Load all available metrics.

        Returns:
            Dictionary mapping model IDs to their metrics
        """
        all_metrics = {}

        for metrics_file in self.storage_dir.glob("*_metrics*.json"):
            try:
                with open(metrics_file) as f:
                    data = json.load(f)

                model_id = data.get("model_id", metrics_file.stem)
                all_metrics[model_id] = data

            except Exception as e:
                logger.warning(f"Could not load metrics from {metrics_file}: {e}")

        return all_metrics

    def compare_all_models(self, primary_metric: str = "f1_score") -> pd.DataFrame:
        """Compare all stored models.

        Args:
            primary_metric: Primary metric for ranking

        Returns:
            DataFrame with model comparison
        """
        all_metrics = self.load_all_metrics()

        if not all_metrics:
            return pd.DataFrame()

        comparison_data = []

        for model_id, data in all_metrics.items():
            metrics = data.get("metrics", {})

            # Handle nested metrics structure
            if "test_metrics" in metrics:
                test_metrics = metrics["test_metrics"]
            else:
                test_metrics = metrics

            row = {
                "model_id": model_id,
                "timestamp": data.get("timestamp", "unknown"),
                "accuracy": test_metrics.get("accuracy", None),
                "precision": test_metrics.get("precision", None),
                "recall": test_metrics.get("recall", None),
                "f1_score": test_metrics.get("f1_score", None),
                "roc_auc": test_metrics.get("roc_auc", None),
            }

            # Add model-specific info if available
            if "model_name" in metrics:
                row["model_name"] = metrics["model_name"]

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by primary metric (descending)
        if primary_metric in df.columns and not df[primary_metric].isna().all():
            df = df.sort_values(primary_metric, ascending=False)

        return df

    def get_best_model_id(self, primary_metric: str = "f1_score") -> str | None:
        """Get the ID of the best performing model.

        Args:
            primary_metric: Metric to use for comparison

        Returns:
            Model ID of best performing model or None if no models found
        """
        comparison_df = self.compare_all_models(primary_metric)

        if comparison_df.empty:
            return None

        # Filter out rows with missing primary metric
        valid_rows = comparison_df.dropna(subset=[primary_metric])

        if valid_rows.empty:
            return None

        best_model_id = valid_rows.iloc[0]["model_id"]
        return best_model_id

    def cleanup_old_metrics(self, keep_latest: int = 10) -> None:
        """Clean up old metrics files, keeping only the latest ones.

        Args:
            keep_latest: Number of latest metrics files to keep
        """
        metrics_files = list(self.storage_dir.glob("*_metrics*.json"))

        if len(metrics_files) <= keep_latest:
            return

        # Sort by modification time (newest first)
        metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old files
        for old_file in metrics_files[keep_latest:]:
            old_file.unlink()
            logger.info(f"Removed old metrics file: {old_file}")


def calculate_model_metrics(
    y_true: pd.Series | np.ndarray | list,
    y_pred: pd.Series | np.ndarray | list,
    y_proba: pd.DataFrame | np.ndarray | None = None,
) -> dict[str, Any]:
    """Convenience function to calculate model metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)

    Returns:
        Dictionary containing calculated metrics
    """
    calculator = ModelMetrics()
    return calculator.calculate_metrics(y_true, y_pred, y_proba)


def save_model_metrics(
    metrics: dict[str, Any], model_id: str, storage_dir: str = "models/metrics"
) -> Path:
    """Convenience function to save model metrics.

    Args:
        metrics: Metrics dictionary to save
        model_id: Model identifier
        storage_dir: Directory to store metrics

    Returns:
        Path to saved metrics file
    """
    storage = MetricsStorage(storage_dir)
    return storage.save_metrics(metrics, model_id)


def load_model_metrics(
    model_id: str, storage_dir: str = "models/metrics"
) -> dict[str, Any]:
    """Convenience function to load model metrics.

    Args:
        model_id: Model identifier
        storage_dir: Directory containing metrics

    Returns:
        Loaded metrics dictionary
    """
    storage = MetricsStorage(storage_dir)
    return storage.load_metrics(model_id)
