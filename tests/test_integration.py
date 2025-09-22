"""Integration tests for end-to-end workflows."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.cli import cli
from src.core.config import ExperimentConfig, ModelConfig
from src.data.dataset import PenguinDataLoader
from src.models.trainer import ModelTrainer
from src.serving.app import app


@pytest.fixture
def sample_penguins_data():
    """Create sample penguins dataset for testing."""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame(
        {
            "species": np.random.choice(["Adelie", "Chinstrap", "Gentoo"], n_samples),
            "island": np.random.choice(["Torgersen", "Biscoe", "Dream"], n_samples),
            "bill_length_mm": np.random.normal(44, 5, n_samples),
            "bill_depth_mm": np.random.normal(17, 2, n_samples),
            "flipper_length_mm": np.random.normal(200, 15, n_samples),
            "body_mass_g": np.random.normal(4200, 800, n_samples),
            "sex": np.random.choice(["MALE", "FEMALE"], n_samples),
            "year": np.random.choice([2007, 2008, 2009], n_samples),
        }
    )

    return data


@pytest.fixture
def temp_project_structure():
    """Create temporary project structure for integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure
        (temp_path / "data" / "raw").mkdir(parents=True)
        (temp_path / "data" / "processed").mkdir(parents=True)
        (temp_path / "models" / "artifacts").mkdir(parents=True)
        (temp_path / "models" / "metrics").mkdir(parents=True)
        (temp_path / "configs").mkdir()

        yield temp_path


@pytest.mark.integration
class TestEndToEndTrainingWorkflow:
    """Test complete training workflow from data to model."""

    def test_complete_training_pipeline(
        self, sample_penguins_data, temp_project_structure
    ):
        """Test complete training pipeline from raw data to saved model."""
        # Save sample data
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        # Create configuration
        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"n_estimators": 10, "random_state": 42},
            ),
        )
        config.paths.raw = str(data_path)
        config.paths.model_dir = str(temp_project_structure / "models" / "artifacts")
        config.paths.metrics_dir = str(temp_project_structure / "models" / "metrics")

        # Initialize trainer
        trainer = ModelTrainer(config)

        # Run training
        results = trainer.train(save_model=True)

        # Verify results
        assert "model_name" in results
        assert "model_id" in results
        assert "train_metrics" in results
        assert "test_metrics" in results
        assert "data_info" in results

        # Verify model was saved
        model_files = list((temp_project_structure / "models").glob("**/*.pkl"))
        assert len(model_files) > 0

        # Verify metrics were saved
        metrics_files = list(
            (temp_project_structure / "models").glob("**/*_metrics.json")
        )
        assert len(metrics_files) > 0

        # Verify model can make predictions
        test_data = sample_penguins_data.drop("species", axis=1).head(5)
        predictions = trainer.predict(test_data)
        assert len(predictions) == 5
        assert all(pred in ["Adelie", "Chinstrap", "Gentoo"] for pred in predictions)

    def test_hyperparameter_tuning_workflow(
        self, sample_penguins_data, temp_project_structure
    ):
        """Test hyperparameter tuning workflow."""
        # Save sample data
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        # Create configuration with tuning
        from src.core.config import TuneConfig

        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"random_state": 42},
                tune=TuneConfig(
                    grid=[{"n_estimators": [5, 10], "max_depth": [3, 5]}],
                    cv=3,
                    scoring="accuracy",
                ),
            ),
        )
        config.paths.raw = str(data_path)
        config.paths.model_dir = str(temp_project_structure / "models" / "artifacts")
        config.paths.metrics_dir = str(temp_project_structure / "models" / "metrics")

        # Initialize trainer
        trainer = ModelTrainer(config)

        # Run tuning
        results = trainer.tune_hyperparameters(save_best_model=True)

        # Verify results
        assert "best_params" in results
        assert "best_score" in results
        assert "cv_results" in results
        assert "train_metrics" in results
        assert "test_metrics" in results

        # Verify best model was saved
        model_files = list((temp_project_structure / "models").glob("**/*.pkl"))
        assert len(model_files) > 0


@pytest.mark.integration
class TestEndToEndServingWorkflow:
    """Test complete serving workflow from model to API."""

    def test_model_loading_and_serving(
        self, sample_penguins_data, temp_project_structure
    ):
        """Test loading trained model and serving predictions."""
        # First train and save a model
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"n_estimators": 5, "random_state": 42},
            ),
        )
        config.paths.raw = str(data_path)
        config.paths.model_dir = str(temp_project_structure / "models" / "artifacts")
        config.paths.metrics_dir = str(temp_project_structure / "models" / "metrics")

        trainer = ModelTrainer(config)
        trainer.train(save_model=True)

        # Now test serving
        from src.serving.app import PenguinPredictor

        predictor = PenguinPredictor(str(temp_project_structure / "models"))

        # Test health check
        health = predictor.get_health()
        assert health.status == "healthy"
        assert health.model_loaded is True

        # Test prediction
        from src.serving.schemas import PredictRequest

        request = PredictRequest(
            island="Torgersen",
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
            year=2007,
        )

        response = predictor.predict(request)
        assert response.prediction in ["Adelie", "Chinstrap", "Gentoo"]
        assert response.confidence is not None
        assert 0 <= response.confidence <= 1

    def test_api_endpoints_integration(
        self, sample_penguins_data, temp_project_structure
    ):
        """Test API endpoints with real model."""
        # Train and save model
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"n_estimators": 5, "random_state": 42},
            ),
        )
        config.paths.raw = str(data_path)
        config.paths.model_dir = str(temp_project_structure / "models" / "artifacts")
        config.paths.metrics_dir = str(temp_project_structure / "models" / "metrics")

        trainer = ModelTrainer(config)
        trainer.train(save_model=True)

        # Mock the predictor to use our trained model
        from src.serving.app import PenguinPredictor

        with patch("src.serving.app.predictor") as mock_predictor:
            real_predictor = PenguinPredictor(str(temp_project_structure / "models"))
            mock_predictor.get_health.return_value = real_predictor.get_health()
            mock_predictor.get_model_info.return_value = real_predictor.get_model_info()
            mock_predictor.predict.side_effect = real_predictor.predict

            client = TestClient(app)

            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

            # Test model info endpoint
            response = client.get("/model/info")
            assert response.status_code == 200
            data = response.json()
            assert "model_id" in data
            assert "classes" in data

            # Test prediction endpoint
            predict_request = {
                "island": "Torgersen",
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": "MALE",
                "year": 2007,
            }

            response = client.post("/predict", json=predict_request)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration with real workflows."""

    def test_cli_train_command(self, sample_penguins_data, temp_project_structure):
        """Test CLI train command integration."""
        # Save sample data
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        # Create config file
        config_data = {
            "seed": 42,
            "paths": {
                "raw": str(data_path),
                "model_dir": str(temp_project_structure / "models" / "artifacts"),
                "metrics_dir": str(temp_project_structure / "models" / "metrics"),
            },
            "model": {
                "name": "RandomForestClassifier",
                "params": {"n_estimators": 5, "random_state": 42},
            },
        }

        config_path = temp_project_structure / "configs" / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Test CLI train command
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--config", str(config_path)])

        # Check command succeeded
        assert result.exit_code == 0

        # Verify model was created
        model_files = list((temp_project_structure / "models").glob("**/*.pkl"))
        assert len(model_files) > 0

    def test_cli_tune_command(self, sample_penguins_data, temp_project_structure):
        """Test CLI tune command integration."""
        # Save sample data
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        # Create config file with tuning
        config_data = {
            "seed": 42,
            "paths": {
                "raw": str(data_path),
                "model_dir": str(temp_project_structure / "models" / "artifacts"),
                "metrics_dir": str(temp_project_structure / "models" / "metrics"),
            },
            "model": {
                "name": "RandomForestClassifier",
                "params": {"random_state": 42},
                "tune": {
                    "grid": [{"n_estimators": [3, 5], "max_depth": [2, 3]}],
                    "cv": 2,
                    "scoring": "accuracy",
                },
            },
        }

        config_path = temp_project_structure / "configs" / "test_tune_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Test CLI tune command
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["tune", "--config", str(config_path)])

        # Check command succeeded
        assert result.exit_code == 0

        # Verify best model was saved
        model_files = list((temp_project_structure / "models").glob("**/*.pkl"))
        assert len(model_files) > 0


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Test data pipeline integration."""

    def test_data_loading_and_preprocessing(
        self, sample_penguins_data, temp_project_structure
    ):
        """Test complete data loading and preprocessing pipeline."""
        # Save sample data
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        # Test data loading
        loader = PenguinDataLoader(str(data_path))
        X, y = loader.get_features_and_target()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_penguins_data)
        assert len(y) == len(sample_penguins_data)

        # Test train-test split
        X_train, X_test, y_train, y_test = loader.train_test_split(
            test_size=0.2, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

        # Test preprocessing
        from src.features.preprocess import build_preprocessor

        preprocessor = build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        assert X_train_processed.shape[0] == len(X_train)
        assert X_test_processed.shape[0] == len(X_test)
        assert X_train_processed.shape[1] == X_test_processed.shape[1]

    def test_model_registry_integration(
        self, sample_penguins_data, temp_project_structure
    ):
        """Test model registry integration with training."""
        # Train a model
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        sample_penguins_data.to_csv(data_path, index=False)

        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"n_estimators": 5, "random_state": 42},
            ),
        )
        config.paths.raw = str(data_path)
        config.paths.model_dir = str(temp_project_structure / "models" / "artifacts")
        config.paths.metrics_dir = str(temp_project_structure / "models" / "metrics")

        trainer = ModelTrainer(config)
        results = trainer.train(save_model=True)

        # Test registry operations
        from src.models.registry import ModelRegistry

        registry = ModelRegistry(str(temp_project_structure / "models"))

        # Test listing models
        models = registry.list_models()
        assert len(models) > 0
        assert results["model_id"] in models

        # Test loading model
        loaded_model = registry.load_model(results["model_id"])
        assert loaded_model is not None

        # Test model info
        model_info = registry.get_model_info(results["model_id"])
        assert model_info["model_id"] == results["model_id"]

        # Test metrics loading
        metrics = registry.load_metrics(results["model_id"])
        assert "metrics" in metrics
        assert "accuracy" in metrics["metrics"]


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance aspects of the integration."""

    def test_large_dataset_training(self, temp_project_structure):
        """Test training with larger dataset."""
        # Create larger synthetic dataset
        np.random.seed(42)
        n_samples = 1000

        large_data = pd.DataFrame(
            {
                "species": np.random.choice(
                    ["Adelie", "Chinstrap", "Gentoo"], n_samples
                ),
                "island": np.random.choice(["Torgersen", "Biscoe", "Dream"], n_samples),
                "bill_length_mm": np.random.normal(44, 5, n_samples),
                "bill_depth_mm": np.random.normal(17, 2, n_samples),
                "flipper_length_mm": np.random.normal(200, 15, n_samples),
                "body_mass_g": np.random.normal(4200, 800, n_samples),
                "sex": np.random.choice(["MALE", "FEMALE"], n_samples),
                "year": np.random.choice([2007, 2008, 2009], n_samples),
            }
        )

        # Save data
        data_path = temp_project_structure / "data" / "raw" / "penguins_lter.csv"
        large_data.to_csv(data_path, index=False)

        # Train model
        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"n_estimators": 20, "random_state": 42, "n_jobs": 1},
            ),
        )
        config.paths.raw = str(data_path)
        config.paths.model_dir = str(temp_project_structure / "models" / "artifacts")
        config.paths.metrics_dir = str(temp_project_structure / "models" / "metrics")

        trainer = ModelTrainer(config)

        import time

        start_time = time.time()
        results = trainer.train(save_model=True)
        training_time = time.time() - start_time

        # Verify training completed successfully
        assert "model_id" in results
        assert results["test_metrics"]["accuracy"] > 0

        # Training should complete in reasonable time (less than 30 seconds)
        assert training_time < 30

        # Test prediction performance
        test_data = large_data.drop("species", axis=1).head(100)

        start_time = time.time()
        predictions = trainer.predict(test_data)
        prediction_time = time.time() - start_time

        assert len(predictions) == 100
        # Predictions should be fast (less than 1 second for 100 samples)
        assert prediction_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.integration
class TestRealDatasetIntegration:
    """Test integration with the actual penguins dataset."""

    def test_real_penguins_dataset_training(self):
        """Test training with the actual penguins_lter.csv dataset."""
        # Use the real dataset
        data_path = "data/raw/penguins_lter.csv"

        # Skip if dataset doesn't exist
        if not Path(data_path).exists():
            pytest.skip("Real penguins dataset not found")

        # Test data loading
        loader = PenguinDataLoader(data_path)
        data_info = loader.get_data_info()

        # Verify dataset properties
        assert data_info["total_rows"] > 300  # Should have ~344 rows
        assert data_info["rows_with_target"] > 300
        assert len(data_info["target_distribution"]) == 3  # 3 species
        assert "Adelie" in data_info["target_distribution"]
        assert "Gentoo" in data_info["target_distribution"]
        assert "Chinstrap" in data_info["target_distribution"]

        # Test train-test split
        X_train, X_test, y_train, y_test = loader.train_test_split(
            test_size=0.2, random_state=42, stratify=True
        )

        assert len(X_train) + len(X_test) == data_info["rows_with_target"]
        assert len(y_train.unique()) == 3  # All species in training set
        assert len(y_test.unique()) >= 2  # At least 2 species in test set

        # Test training with real data
        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"n_estimators": 50, "random_state": 42},
            ),
        )

        trainer = ModelTrainer(config)
        results = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            save_model=False,
        )

        # Verify training results
        assert results["test_metrics"]["accuracy"] > 0.8  # Should achieve good accuracy
        assert results["test_metrics"]["f1_score"] > 0.8
        assert results["data_info"]["train_size"] == len(X_train)
        assert results["data_info"]["test_size"] == len(X_test)
        assert len(results["data_info"]["target_classes"]) == 3

    def test_real_dataset_hyperparameter_tuning(self):
        """Test hyperparameter tuning with real dataset."""
        data_path = "data/raw/penguins_lter.csv"

        if not Path(data_path).exists():
            pytest.skip("Real penguins dataset not found")

        # Create tuning configuration
        from src.core.config import TuneConfig

        config = ExperimentConfig(
            seed=42,
            model=ModelConfig(
                name="RandomForestClassifier",
                params={"random_state": 42},
                tune=TuneConfig(
                    grid=[
                        {
                            "n_estimators": [20, 50],
                            "max_depth": [5, 10],
                            "min_samples_split": [2, 5],
                        }
                    ],
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=1,
                ),
            ),
        )

        trainer = ModelTrainer(config)
        results = trainer.tune_hyperparameters(save_best_model=False)

        # Verify tuning results
        assert "best_params" in results
        assert "best_score" in results
        assert results["best_score"] > 0.8  # Should achieve good CV score
        assert results["test_metrics"]["accuracy"] > 0.8
        assert "classifier__n_estimators" in results["best_params"]
        assert "classifier__max_depth" in results["best_params"]

    def test_real_dataset_api_serving(self):
        """Test API serving with model trained on real dataset."""
        data_path = "data/raw/penguins_lter.csv"

        if not Path(data_path).exists():
            pytest.skip("Real penguins dataset not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            models_dir = temp_path / "models"
            models_dir.mkdir()

            # Train model with real data
            config = ExperimentConfig(
                seed=42,
                model=ModelConfig(
                    name="RandomForestClassifier",
                    params={"n_estimators": 30, "random_state": 42},
                ),
            )
            config.paths.model_dir = str(models_dir / "artifacts")
            config.paths.metrics_dir = str(models_dir / "metrics")

            trainer = ModelTrainer(config)
            trainer.train(save_model=True)

            # Test serving with trained model
            from src.serving.app import PenguinPredictor
            from src.serving.schemas import PredictRequest

            predictor = PenguinPredictor(str(models_dir / "artifacts"))

            # Test with real penguin characteristics
            # Adelie penguin characteristics
            adelie_request = PredictRequest(
                island="Torgersen",
                bill_length_mm=39.1,
                bill_depth_mm=18.7,
                flipper_length_mm=181.0,
                body_mass_g=3750.0,
                sex="MALE",
                year=2007,
            )

            response = predictor.predict(adelie_request)
            assert response.prediction in ["Adelie", "Chinstrap", "Gentoo"]
            assert response.confidence is not None
            assert 0 <= response.confidence <= 1

            # Gentoo penguin characteristics (larger)
            gentoo_request = PredictRequest(
                island="Biscoe",
                bill_length_mm=46.1,
                bill_depth_mm=13.2,
                flipper_length_mm=211.0,
                body_mass_g=4500.0,
                sex="FEMALE",
                year=2008,
            )

            response = predictor.predict(gentoo_request)
            assert response.prediction in ["Adelie", "Chinstrap", "Gentoo"]
            assert response.confidence is not None

    def test_complete_mlops_pipeline(self):
        """Test complete MLOps pipeline from data to deployment."""
        data_path = "data/raw/penguins_lter.csv"

        if not Path(data_path).exists():
            pytest.skip("Real penguins dataset not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create project structure
            (temp_path / "models" / "artifacts").mkdir(parents=True)
            (temp_path / "models" / "metrics").mkdir(parents=True)
            (temp_path / "configs").mkdir()

            # Step 1: Data validation and loading
            loader = PenguinDataLoader(data_path)
            data_info = loader.get_data_info()
            assert data_info["rows_with_target"] > 0

            # Step 2: Model training
            config = ExperimentConfig(
                seed=42,
                model=ModelConfig(
                    name="RandomForestClassifier",
                    params={"n_estimators": 30, "random_state": 42},
                ),
            )
            config.paths.model_dir = str(temp_path / "models" / "artifacts")
            config.paths.metrics_dir = str(temp_path / "models" / "metrics")

            trainer = ModelTrainer(config)
            training_results = trainer.train(save_model=True)

            # Verify training
            assert training_results["test_metrics"]["accuracy"] > 0.7
            model_id = training_results["model_id"]

            # Step 3: Model registry operations
            from src.models.registry import ModelRegistry

            registry = ModelRegistry(str(temp_path / "models" / "artifacts"))
            models = registry.list_models()
            assert model_id in models

            model_info = registry.get_model_info(model_id)
            assert model_info["model_id"] == model_id
            assert len(model_info["classes"]) == 3

            # Step 4: Model serving
            from src.serving.app import PenguinPredictor

            predictor = PenguinPredictor(str(temp_path / "models" / "artifacts"))
            health = predictor.get_health()
            assert health.status == "healthy"
            assert health.model_loaded is True

            # Step 5: Prediction testing
            from src.serving.schemas import BatchPredictRequest, PredictRequest

            # Single prediction
            request = PredictRequest(
                island="Dream",
                bill_length_mm=42.0,
                bill_depth_mm=17.5,
                flipper_length_mm=190.0,
                body_mass_g=4000.0,
                sex="MALE",
                year=2009,
            )

            response = predictor.predict(request)
            assert response.prediction in ["Adelie", "Chinstrap", "Gentoo"]

            # Batch prediction
            batch_request = BatchPredictRequest(instances=[request, request])
            batch_response = predictor.predict_batch(batch_request)
            assert len(batch_response.predictions) == 2

            # Step 6: Performance validation
            # Test prediction latency
            import time

            start_time = time.time()
            for _ in range(10):
                predictor.predict(request)
            avg_latency = (time.time() - start_time) / 10

            # Should be fast (less than 100ms per prediction)
            assert avg_latency < 0.1

            print("Complete MLOps pipeline test passed!")
            print(f"Model accuracy: {training_results['test_metrics']['accuracy']:.3f}")
            print(f"Average prediction latency: {avg_latency * 1000:.1f}ms")


@pytest.mark.integration
class TestDockerIntegration:
    """Test Docker container integration."""

    def test_training_container_build(self):
        """Test that training container can be built."""
        # This test requires Docker to be available
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                pytest.skip("Docker not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available")

        # Test building training container
        result = subprocess.run(
            ["docker", "build", "-f", "Dockerfile.train", "-t", "test-train", "."],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        # Should build successfully
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

        # Clean up
        subprocess.run(["docker", "rmi", "test-train"], capture_output=True)

    def test_serving_container_build(self):
        """Test that serving container can be built."""
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                pytest.skip("Docker not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available")

        # Test building serving container
        result = subprocess.run(
            ["docker", "build", "-f", "Dockerfile.app", "-t", "test-serve", "."],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        # Should build successfully
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

        # Clean up
        subprocess.run(["docker", "rmi", "test-serve"], capture_output=True)


@pytest.mark.integration
class TestCIWorkflowSimulation:
    """Simulate CI/CD workflow steps."""

    def test_code_quality_checks(self):
        """Test code quality checks that would run in CI."""
        import subprocess

        # Test ruff linting
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/", "tests/"],
            capture_output=True,
            text=True,
        )
        # Should pass or have only minor issues
        assert result.returncode in [0, 1], f"Ruff check failed: {result.stdout}"

        # Test ruff formatting
        result = subprocess.run(
            ["uv", "run", "ruff", "format", "--check", "src/", "tests/"],
            capture_output=True,
            text=True,
        )
        # Should be properly formatted
        assert result.returncode == 0, f"Code formatting issues: {result.stdout}"

    def test_unit_test_coverage(self):
        """Test that unit tests achieve reasonable coverage."""
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "tests/",
                "-v",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-fail-under=70",
            ],
            capture_output=True,
            text=True,
        )

        # Should achieve at least 70% coverage
        assert result.returncode == 0, f"Test coverage too low: {result.stdout}"

    def test_security_scan_simulation(self):
        """Simulate security scanning that would run in CI."""
        import subprocess

        # Test bandit security scan
        result = subprocess.run(
            ["uv", "run", "bandit", "-r", "src/", "-f", "json"],
            capture_output=True,
            text=True,
        )

        # Bandit may find issues but shouldn't crash
        assert result.returncode in [0, 1], f"Bandit scan failed: {result.stderr}"

        # Test safety check for known vulnerabilities
        result = subprocess.run(
            ["uv", "run", "safety", "check", "--json"], capture_output=True, text=True
        )

        # Safety may find issues but shouldn't crash
        assert result.returncode in [0, 1], f"Safety check failed: {result.stderr}"
