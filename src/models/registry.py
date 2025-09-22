"""Model registry utilities for loading and saving trained models."""

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .metrics import MetricsStorage


class ModelRegistry:
    """Registry for managing trained models and their metadata."""

    def __init__(self, registry_dir: str = "models"):
        """Initialize the model registry.

        Args:
            registry_dir: Directory to store models and registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.registry_dir / "artifacts"
        self.models_dir.mkdir(exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.metrics_storage = MetricsStorage(self.registry_dir / "metrics")

        # Load or create registry
        self._registry = self._load_registry()

        # Save registry to create file if it doesn't exist
        if not self.registry_file.exists():
            self._save_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load the model registry from file.

        Returns:
            Registry dictionary
        """
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry file: {e}")
                return {
                    "models": {},
                    "metadata": {"created": datetime.now().isoformat()},
                }
        else:
            return {"models": {}, "metadata": {"created": datetime.now().isoformat()}}

    def _save_registry(self) -> None:
        """Save the registry to file."""
        self._registry["metadata"]["last_updated"] = datetime.now().isoformat()

        with open(self.registry_file, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def save_model(
        self,
        model: BaseEstimator | Pipeline,
        model_id: str,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Save a trained model to the registry.

        Args:
            model: Trained sklearn model or pipeline
            model_id: Unique identifier for the model
            metadata: Additional metadata to store
            overwrite: Whether to overwrite existing model

        Returns:
            Path to saved model file

        Raises:
            ValueError: If model_id already exists and overwrite=False
        """
        if model_id in self._registry["models"] and not overwrite:
            raise ValueError(
                f"Model {model_id} already exists. Use overwrite=True to replace."
            )

        # Create model file path
        model_filename = f"{model_id}.pkl"
        model_path = self.models_dir / model_filename

        # Save model using pickle
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise

        # Prepare model metadata
        model_metadata = {
            "model_id": model_id,
            "filename": model_filename,
            "created": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "file_size": model_path.stat().st_size,
        }

        # Add sklearn pipeline info if applicable
        if isinstance(model, Pipeline):
            model_metadata["pipeline_steps"] = [step[0] for step in model.steps]
            if hasattr(model, "classes_"):
                model_metadata["classes"] = model.classes_.tolist()
        elif hasattr(model, "classes_"):
            model_metadata["classes"] = model.classes_.tolist()

        # Add custom metadata
        if metadata:
            model_metadata.update(metadata)

        # Update registry
        self._registry["models"][model_id] = model_metadata
        self._save_registry()

        logger.info(f"Model {model_id} saved to: {model_path}")
        return model_path

    def load_model(self, model_id: str) -> BaseEstimator | Pipeline:
        """Load a model from the registry.

        Args:
            model_id: Model identifier

        Returns:
            Loaded sklearn model or pipeline

        Raises:
            ValueError: If model_id not found
            FileNotFoundError: If model file doesn't exist
        """
        if model_id not in self._registry["models"]:
            available_models = list(self._registry["models"].keys())
            raise ValueError(
                f"Model {model_id} not found. Available: {available_models}"
            )

        model_info = self._registry["models"][model_id]
        model_path = self.models_dir / model_info["filename"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            logger.info(f"Model {model_id} loaded from: {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def load_latest_model(self) -> BaseEstimator | Pipeline | None:
        """Load the most recently created model.

        Returns:
            Latest model or None if no models exist
        """
        if not self._registry["models"]:
            logger.warning("No models found in registry")
            return None

        # Find latest model by creation time
        latest_model_id = max(
            self._registry["models"].keys(),
            key=lambda x: self._registry["models"][x]["created"],
        )

        return self.load_model(latest_model_id)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get information about a registered model.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata dictionary

        Raises:
            ValueError: If model_id not found
        """
        if model_id not in self._registry["models"]:
            available_models = list(self._registry["models"].keys())
            raise ValueError(
                f"Model {model_id} not found. Available: {available_models}"
            )

        return self._registry["models"][model_id].copy()

    def list_models(self) -> list[str]:
        """List all registered model IDs.

        Returns:
            List of model identifiers
        """
        return list(self._registry["models"].keys())

    def get_models_summary(self) -> pd.DataFrame:
        """Get a summary of all registered models.

        Returns:
            DataFrame with model information
        """
        if not self._registry["models"]:
            return pd.DataFrame()

        summary_data = []

        for model_id, info in self._registry["models"].items():
            row = {
                "model_id": model_id,
                "model_type": info.get("model_type", "unknown"),
                "created": info.get("created", "unknown"),
                "file_size_mb": round(info.get("file_size", 0) / (1024 * 1024), 2),
                "classes": info.get("classes", []),
            }

            # Add pipeline info if available
            if "pipeline_steps" in info:
                row["pipeline_steps"] = ", ".join(info["pipeline_steps"])

            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Sort by creation time (newest first)
        if "created" in df.columns:
            df = df.sort_values("created", ascending=False)

        return df

    def delete_model(self, model_id: str, delete_metrics: bool = True) -> None:
        """Delete a model from the registry.

        Args:
            model_id: Model identifier
            delete_metrics: Whether to also delete associated metrics

        Raises:
            ValueError: If model_id not found
        """
        if model_id not in self._registry["models"]:
            available_models = list(self._registry["models"].keys())
            raise ValueError(
                f"Model {model_id} not found. Available: {available_models}"
            )

        model_info = self._registry["models"][model_id]
        model_path = self.models_dir / model_info["filename"]

        # Delete model file
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model file: {model_path}")

        # Delete metrics if requested
        if delete_metrics:
            try:
                metrics_files = list(
                    self.metrics_storage.storage_dir.glob(f"{model_id}_metrics*.json")
                )
                for metrics_file in metrics_files:
                    metrics_file.unlink()
                    logger.info(f"Deleted metrics file: {metrics_file}")
            except Exception as e:
                logger.warning(f"Could not delete metrics for {model_id}: {e}")

        # Remove from registry
        del self._registry["models"][model_id]
        self._save_registry()

        logger.info(f"Model {model_id} deleted from registry")

    def save_metrics(
        self, metrics: dict[str, Any], model_id: str, suffix: str = ""
    ) -> Path:
        """Save metrics for a model.

        Args:
            metrics: Metrics dictionary
            model_id: Model identifier
            suffix: Optional suffix for metrics file

        Returns:
            Path to saved metrics file
        """
        return self.metrics_storage.save_metrics(metrics, model_id, suffix)

    def load_metrics(self, model_id: str, suffix: str = "") -> dict[str, Any]:
        """Load metrics for a model.

        Args:
            model_id: Model identifier
            suffix: Optional suffix for metrics file

        Returns:
            Loaded metrics dictionary
        """
        return self.metrics_storage.load_metrics(model_id, suffix)

    def get_best_model_id(self, primary_metric: str = "f1_score") -> str | None:
        """Get the ID of the best performing model based on metrics.

        Args:
            primary_metric: Metric to use for comparison

        Returns:
            Model ID of best performing model or None if no models/metrics found
        """
        return self.metrics_storage.get_best_model_id(primary_metric)

    def load_best_model(
        self, primary_metric: str = "f1_score"
    ) -> BaseEstimator | Pipeline | None:
        """Load the best performing model based on metrics.

        Args:
            primary_metric: Metric to use for comparison

        Returns:
            Best performing model or None if no models found
        """
        best_model_id = self.get_best_model_id(primary_metric)

        if best_model_id is None:
            logger.warning("No best model found")
            return None

        return self.load_model(best_model_id)

    def export_model(
        self, model_id: str, export_path: str, include_metrics: bool = True
    ) -> None:
        """Export a model and its metadata to a directory.

        Args:
            model_id: Model identifier
            export_path: Path to export directory
            include_metrics: Whether to include metrics files
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        model_info = self.get_model_info(model_id)
        model_path = self.models_dir / model_info["filename"]

        if model_path.exists():
            shutil.copy2(model_path, export_dir / model_info["filename"])

        # Save model metadata
        metadata_path = export_dir / f"{model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(model_info, f, indent=2, default=str)

        # Copy metrics if requested
        if include_metrics:
            try:
                metrics_files = list(
                    self.metrics_storage.storage_dir.glob(f"{model_id}_metrics*.json")
                )
                for metrics_file in metrics_files:
                    shutil.copy2(metrics_file, export_dir / metrics_file.name)
            except Exception as e:
                logger.warning(f"Could not export metrics for {model_id}: {e}")

        logger.info(f"Model {model_id} exported to: {export_dir}")

    def import_model(
        self,
        model_path: str,
        model_id: str,
        metadata_path: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Import a model from a file.

        Args:
            model_path: Path to model pickle file
            model_id: Identifier for the imported model
            metadata_path: Optional path to metadata JSON file
            overwrite: Whether to overwrite existing model
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the model to validate it
        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not load model from {model_path}: {e}")  # noqa: B904

        # Load metadata if provided
        metadata = {}
        if metadata_path:
            metadata_file = Path(metadata_path)
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load metadata from {metadata_path}: {e}")

        # Save the model using the registry
        self.save_model(model, model_id, metadata, overwrite)

        logger.info(f"Model imported as {model_id} from: {model_path}")

    def cleanup_registry(self, keep_latest: int = 10) -> None:
        """Clean up old models and metrics, keeping only the latest ones.

        Args:
            keep_latest: Number of latest models to keep
        """
        if len(self._registry["models"]) <= keep_latest:
            return

        # Sort models by creation time (newest first)
        sorted_models = sorted(
            self._registry["models"].items(),
            key=lambda x: x[1]["created"],
            reverse=True,
        )

        # Delete old models
        for model_id, _ in sorted_models[keep_latest:]:
            try:
                self.delete_model(model_id, delete_metrics=True)
                logger.info(f"Cleaned up old model: {model_id}")
            except Exception as e:
                logger.warning(f"Could not delete old model {model_id}: {e}")

        # Clean up orphaned metrics
        self.metrics_storage.cleanup_old_metrics(keep_latest)


def create_model_registry(registry_dir: str = "models") -> ModelRegistry:
    """Create a new model registry instance.

    Args:
        registry_dir: Directory for the registry

    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(registry_dir)


def save_model_to_registry(
    model: BaseEstimator | Pipeline,
    model_id: str,
    registry_dir: str = "models",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Convenience function to save a model to registry.

    Args:
        model: Trained sklearn model or pipeline
        model_id: Model identifier
        registry_dir: Registry directory
        metadata: Additional metadata

    Returns:
        Path to saved model file
    """
    registry = ModelRegistry(registry_dir)
    return registry.save_model(model, model_id, metadata)


def load_model_from_registry(
    model_id: str, registry_dir: str = "models"
) -> BaseEstimator | Pipeline:
    """Convenience function to load a model from registry.

    Args:
        model_id: Model identifier
        registry_dir: Registry directory

    Returns:
        Loaded sklearn model or pipeline
    """
    registry = ModelRegistry(registry_dir)
    return registry.load_model(model_id)
