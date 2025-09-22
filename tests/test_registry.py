"""Unit tests for model registry."""

import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.registry import (
    ModelRegistry,
    create_model_registry,
    load_model_from_registry,
    save_model_to_registry,
)


@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    model = RandomForestClassifier(n_estimators=5, random_state=42)

    # Create some dummy training data
    import numpy as np

    X = np.random.randn(50, 4)
    y = np.random.choice(["A", "B", "C"], 50)

    model.fit(X, y)
    return model


@pytest.fixture
def sample_pipeline():
    """Create a sample sklearn pipeline."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
    )

    # Create some dummy training data
    import numpy as np

    X = np.random.randn(50, 4)
    y = np.random.choice(["A", "B", "C"], 50)

    pipeline.fit(X, y)
    return pipeline


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    def test_init(self):
        """Test registry initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            assert registry.registry_dir == Path(temp_dir)
            assert registry.models_dir.exists()
            assert registry.registry_file.exists()
            assert registry.metrics_storage is not None

    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "models"
            registry = ModelRegistry(str(registry_path))

            assert registry_path.exists()
            assert (registry_path / "artifacts").exists()

    def test_save_model(self, sample_model):
        """Test saving a model to registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            model_path = registry.save_model(sample_model, "test_model")

            # Check that model file was created
            assert model_path.exists()
            assert model_path.name == "test_model.pkl"

            # Check that registry was updated
            assert "test_model" in registry._registry["models"]
            model_info = registry._registry["models"]["test_model"]

            assert model_info["model_id"] == "test_model"
            assert model_info["filename"] == "test_model.pkl"
            assert model_info["model_type"] == "RandomForestClassifier"
            assert "created" in model_info
            assert "file_size" in model_info
            assert "classes" in model_info

    def test_save_pipeline(self, sample_pipeline):
        """Test saving a pipeline to registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            model_path = registry.save_model(sample_pipeline, "test_pipeline")

            # Check registry entry
            model_info = registry._registry["models"]["test_pipeline"]
            assert model_info["model_type"] == "Pipeline"
            assert "pipeline_steps" in model_info
            assert model_info["pipeline_steps"] == ["scaler", "classifier"]

    def test_save_model_with_metadata(self, sample_model):
        """Test saving model with custom metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            metadata = {
                "experiment_name": "test_experiment",
                "accuracy": 0.95,
                "notes": "Best model so far",
            }

            registry.save_model(sample_model, "test_model", metadata)

            model_info = registry._registry["models"]["test_model"]
            assert model_info["experiment_name"] == "test_experiment"
            assert model_info["accuracy"] == 0.95
            assert model_info["notes"] == "Best model so far"

    def test_save_model_overwrite_false(self, sample_model):
        """Test saving model with overwrite=False when model exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save model first time
            registry.save_model(sample_model, "test_model")

            # Try to save again with overwrite=False
            with pytest.raises(ValueError, match="Model test_model already exists"):
                registry.save_model(sample_model, "test_model", overwrite=False)

    def test_load_model(self, sample_model):
        """Test loading a model from registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save model
            registry.save_model(sample_model, "test_model")

            # Load model
            loaded_model = registry.load_model("test_model")

            assert isinstance(loaded_model, RandomForestClassifier)
            assert loaded_model.n_estimators == sample_model.n_estimators
            assert loaded_model.random_state == sample_model.random_state

    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            with pytest.raises(ValueError, match="Model nonexistent not found"):
                registry.load_model("nonexistent")

    def test_load_model_file_missing(self, sample_model):
        """Test loading model when file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save model
            registry.save_model(sample_model, "test_model")

            # Delete the model file
            model_path = registry.models_dir / "test_model.pkl"
            model_path.unlink()

            with pytest.raises(FileNotFoundError):
                registry.load_model("test_model")

    def test_load_latest_model(self, sample_model):
        """Test loading the latest model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save multiple models with slight delays to ensure different timestamps
            import time

            registry.save_model(sample_model, "model1")
            time.sleep(0.01)
            registry.save_model(sample_model, "model2")
            time.sleep(0.01)
            registry.save_model(sample_model, "model3")

            # Load latest model
            latest_model = registry.load_latest_model()

            assert isinstance(latest_model, RandomForestClassifier)

            # Verify it's the latest by checking the registry
            latest_id = max(
                registry._registry["models"].keys(),
                key=lambda x: registry._registry["models"][x]["created"],
            )
            assert latest_id == "model3"

    def test_load_latest_model_empty_registry(self):
        """Test loading latest model when registry is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            latest_model = registry.load_latest_model()
            assert latest_model is None

    def test_get_model_info(self, sample_model):
        """Test getting model information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            registry.save_model(sample_model, "test_model")

            model_info = registry.get_model_info("test_model")

            assert model_info["model_id"] == "test_model"
            assert model_info["model_type"] == "RandomForestClassifier"
            assert "created" in model_info
            assert "file_size" in model_info

    def test_get_model_info_nonexistent(self):
        """Test getting info for nonexistent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            with pytest.raises(ValueError, match="Model nonexistent not found"):
                registry.get_model_info("nonexistent")

    def test_list_models(self, sample_model):
        """Test listing all models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Initially empty
            models = registry.list_models()
            assert len(models) == 0

            # Save some models
            registry.save_model(sample_model, "model1")
            registry.save_model(sample_model, "model2")

            models = registry.list_models()
            assert len(models) == 2
            assert "model1" in models
            assert "model2" in models

    def test_get_models_summary(self, sample_model, sample_pipeline):
        """Test getting models summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Empty registry
            summary = registry.get_models_summary()
            assert isinstance(summary, pd.DataFrame)
            assert len(summary) == 0

            # Save some models
            registry.save_model(sample_model, "rf_model")
            registry.save_model(sample_pipeline, "pipeline_model")

            summary = registry.get_models_summary()

            assert len(summary) == 2
            assert "model_id" in summary.columns
            assert "model_type" in summary.columns
            assert "created" in summary.columns
            assert "file_size_mb" in summary.columns

            # Check specific values
            rf_row = summary[summary["model_id"] == "rf_model"].iloc[0]
            assert rf_row["model_type"] == "RandomForestClassifier"

            pipeline_row = summary[summary["model_id"] == "pipeline_model"].iloc[0]
            assert pipeline_row["model_type"] == "Pipeline"
            assert "scaler, classifier" in pipeline_row["pipeline_steps"]

    def test_delete_model(self, sample_model):
        """Test deleting a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save model
            registry.save_model(sample_model, "test_model")

            # Verify it exists
            assert "test_model" in registry.list_models()
            model_path = registry.models_dir / "test_model.pkl"
            assert model_path.exists()

            # Delete model
            registry.delete_model("test_model")

            # Verify it's gone
            assert "test_model" not in registry.list_models()
            assert not model_path.exists()

    def test_delete_nonexistent_model(self):
        """Test deleting a model that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            with pytest.raises(ValueError, match="Model nonexistent not found"):
                registry.delete_model("nonexistent")

    def test_save_and_load_metrics(self, sample_model):
        """Test saving and loading metrics through registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save model
            registry.save_model(sample_model, "test_model")

            # Save metrics
            metrics = {"accuracy": 0.95, "f1_score": 0.93}
            metrics_path = registry.save_metrics(metrics, "test_model")

            assert metrics_path.exists()

            # Load metrics
            loaded_metrics = registry.load_metrics("test_model")

            assert loaded_metrics["metrics"]["accuracy"] == 0.95
            assert loaded_metrics["metrics"]["f1_score"] == 0.93

    def test_get_best_model_id(self, sample_model):
        """Test getting best model ID based on metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save models with different performance
            registry.save_model(sample_model, "model1")
            registry.save_model(sample_model, "model2")

            # Save metrics
            metrics1 = {"test_metrics": {"accuracy": 0.85, "f1_score": 0.83}}
            metrics2 = {"test_metrics": {"accuracy": 0.92, "f1_score": 0.90}}

            registry.save_metrics(metrics1, "model1")
            registry.save_metrics(metrics2, "model2")

            best_id = registry.get_best_model_id("f1_score")
            assert best_id == "model2"

    def test_load_best_model(self, sample_model):
        """Test loading best model based on metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save models
            registry.save_model(sample_model, "model1")
            registry.save_model(sample_model, "model2")

            # Save metrics
            metrics1 = {"test_metrics": {"accuracy": 0.85}}
            metrics2 = {"test_metrics": {"accuracy": 0.92}}

            registry.save_metrics(metrics1, "model1")
            registry.save_metrics(metrics2, "model2")

            best_model = registry.load_best_model("accuracy")

            assert isinstance(best_model, RandomForestClassifier)

    def test_export_model(self, sample_model):
        """Test exporting a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save model and metrics
            registry.save_model(sample_model, "test_model")
            metrics = {"accuracy": 0.95}
            registry.save_metrics(metrics, "test_model")

            # Export model
            export_dir = Path(temp_dir) / "export"
            registry.export_model("test_model", str(export_dir))

            # Check exported files
            assert (export_dir / "test_model.pkl").exists()
            assert (export_dir / "test_model_metadata.json").exists()
            assert (export_dir / "test_model_metrics.json").exists()

    def test_import_model(self):
        """Test importing a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a model to import
            model = RandomForestClassifier(n_estimators=3, random_state=42)
            import numpy as np

            X = np.random.randn(20, 4)
            y = np.random.choice(["A", "B"], 20)
            model.fit(X, y)

            # Save model to a file
            model_path = Path(temp_dir) / "external_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Create registry and import
            registry_dir = Path(temp_dir) / "registry"
            registry = ModelRegistry(str(registry_dir))

            registry.import_model(str(model_path), "imported_model")

            # Verify import
            assert "imported_model" in registry.list_models()
            loaded_model = registry.load_model("imported_model")
            assert isinstance(loaded_model, RandomForestClassifier)
            assert loaded_model.n_estimators == 3

    def test_import_nonexistent_model(self):
        """Test importing a model that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            with pytest.raises(FileNotFoundError):
                registry.import_model("nonexistent.pkl", "test_model")

    def test_cleanup_registry(self, sample_model):
        """Test cleaning up old models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save multiple models
            import time

            for i in range(5):
                registry.save_model(sample_model, f"model_{i}")
                time.sleep(0.01)  # Ensure different timestamps

            # Verify all models exist
            assert len(registry.list_models()) == 5

            # Cleanup, keeping only 3
            registry.cleanup_registry(keep_latest=3)

            # Verify only 3 remain
            remaining_models = registry.list_models()
            assert len(remaining_models) == 3

            # Verify the latest models were kept
            assert "model_4" in remaining_models
            assert "model_3" in remaining_models
            assert "model_2" in remaining_models


class TestConvenienceFunctions:
    """Test convenience functions for registry."""

    def test_create_model_registry(self):
        """Test create_model_registry convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = create_model_registry(temp_dir)

            assert isinstance(registry, ModelRegistry)
            assert registry.registry_dir == Path(temp_dir)

    def test_save_model_to_registry(self, sample_model):
        """Test save_model_to_registry convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = save_model_to_registry(sample_model, "test_model", temp_dir)

            assert model_path.exists()
            assert model_path.name == "test_model.pkl"

    def test_load_model_from_registry(self, sample_model):
        """Test load_model_from_registry convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model first
            save_model_to_registry(sample_model, "test_model", temp_dir)

            # Load model
            loaded_model = load_model_from_registry("test_model", temp_dir)

            assert isinstance(loaded_model, RandomForestClassifier)
            assert loaded_model.n_estimators == sample_model.n_estimators


class TestRegistryIntegration:
    """Integration tests for model registry."""

    def test_full_registry_workflow(self, sample_model, sample_pipeline):
        """Test complete registry workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Save multiple models
            registry.save_model(sample_model, "rf_model")
            registry.save_model(sample_pipeline, "pipeline_model")

            # Save metrics for both
            rf_metrics = {"test_metrics": {"accuracy": 0.85, "f1_score": 0.83}}
            pipeline_metrics = {"test_metrics": {"accuracy": 0.92, "f1_score": 0.90}}

            registry.save_metrics(rf_metrics, "rf_model")
            registry.save_metrics(pipeline_metrics, "pipeline_model")

            # Get summary
            summary = registry.get_models_summary()
            assert len(summary) == 2

            # Find best model
            best_id = registry.get_best_model_id("accuracy")
            assert best_id == "pipeline_model"

            # Load best model
            best_model = registry.load_best_model("accuracy")
            assert isinstance(best_model, Pipeline)

            # Export best model
            export_dir = Path(temp_dir) / "export"
            registry.export_model(best_id, str(export_dir))

            # Verify export
            assert (export_dir / "pipeline_model.pkl").exists()
            assert (export_dir / "pipeline_model_metadata.json").exists()

            # Test cleanup
            registry.cleanup_registry(keep_latest=1)
            remaining_models = registry.list_models()
            assert len(remaining_models) == 1
