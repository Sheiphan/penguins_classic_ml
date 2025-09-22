"""Unit tests for model trainer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.core.config import ExperimentConfig, ModelConfig, TuneConfig
from src.models.trainer import ModelTrainer, train_model, tune_model


@pytest.fixture
def sample_config():
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        seed=42,
        model=ModelConfig(
            name="RandomForestClassifier",
            params={"n_estimators": 10, "random_state": 42}
        )
    )


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    
    # Create synthetic data similar to penguins
    n_samples = 100
    
    X = pd.DataFrame({
        "bill_length_mm": np.random.normal(45, 5, n_samples),
        "bill_depth_mm": np.random.normal(18, 2, n_samples),
        "flipper_length_mm": np.random.normal(200, 15, n_samples),
        "body_mass_g": np.random.normal(4000, 500, n_samples),
        "year": np.random.choice([2007, 2008, 2009], n_samples),
        "island": np.random.choice(["Torgersen", "Biscoe", "Dream"], n_samples),
        "sex": np.random.choice(["MALE", "FEMALE"], n_samples)
    })
    
    y = pd.Series(np.random.choice(["Adelie", "Chinstrap", "Gentoo"], n_samples))
    
    return X, y


@pytest.fixture
def trainer_with_temp_dir(sample_config):
    """Create trainer with temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update config to use temp directory
        sample_config.paths.model_dir = temp_dir
        sample_config.paths.metrics_dir = f"{temp_dir}/metrics"
        
        trainer = ModelTrainer(sample_config)
        yield trainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_init(self, sample_config):
        """Test trainer initialization."""
        trainer = ModelTrainer(sample_config)
        
        assert trainer.config == sample_config
        assert trainer.data_loader is not None
        assert trainer.preprocessor is not None
        assert trainer.model_pipeline is None
        assert trainer.metrics_calculator is not None
        assert trainer.registry is not None
    
    def test_get_model_class(self, trainer_with_temp_dir):
        """Test getting model class by name."""
        trainer = trainer_with_temp_dir
        
        # Test valid model names
        rf_class = trainer._get_model_class("RandomForestClassifier")
        assert rf_class == RandomForestClassifier
        
        lr_class = trainer._get_model_class("LogisticRegression")
        assert lr_class == LogisticRegression
        
        # Test invalid model name
        with pytest.raises(ValueError, match="Model InvalidModel not supported"):
            trainer._get_model_class("InvalidModel")
    
    def test_create_model(self, trainer_with_temp_dir):
        """Test model creation from configuration."""
        trainer = trainer_with_temp_dir
        
        model_config = ModelConfig(
            name="RandomForestClassifier",
            params={"n_estimators": 5, "max_depth": 3}
        )
        
        model = trainer._create_model(model_config)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 5
        assert model.max_depth == 3
        assert hasattr(model, 'random_state')  # Should be set automatically
    
    def test_create_pipeline(self, trainer_with_temp_dir):
        """Test pipeline creation."""
        trainer = trainer_with_temp_dir
        
        pipeline = trainer.create_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "preprocessor"
        assert pipeline.steps[1][0] == "classifier"
        assert isinstance(pipeline.steps[1][1], RandomForestClassifier)
    
    @patch('src.models.trainer.PenguinDataLoader')
    def test_train_with_provided_data(self, mock_loader, trainer_with_temp_dir, sample_data):
        """Test training with provided data."""
        trainer = trainer_with_temp_dir
        X, y = sample_data
        
        # Split data for training
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = trainer.train(X_train, y_train, X_test, y_test, save_model=False)
        
        # Check results structure
        assert "model_name" in results
        assert "train_metrics" in results
        assert "test_metrics" in results
        assert "data_info" in results
        
        # Check metrics
        assert "accuracy" in results["train_metrics"]
        assert "accuracy" in results["test_metrics"]
        assert 0 <= results["test_metrics"]["accuracy"] <= 1
        
        # Check that model was trained
        assert trainer.model_pipeline is not None
        assert isinstance(trainer.model_pipeline, Pipeline)
    
    def test_train_without_data(self, trainer_with_temp_dir, sample_data):
        """Test training without provided data (loads from dataset)."""
        trainer = trainer_with_temp_dir
        X, y = sample_data
        
        # Mock the data loader instance that's already created
        trainer.data_loader.train_test_split = Mock(return_value=(
            X[:80], X[80:], y[:80], y[80:]
        ))
        
        results = trainer.train(save_model=False)
        
        # Verify data loader was called
        trainer.data_loader.train_test_split.assert_called_once()
        
        # Check results
        assert "model_name" in results
        assert "train_metrics" in results
        assert "test_metrics" in results
    
    def test_tune_hyperparameters_no_config(self, trainer_with_temp_dir):
        """Test hyperparameter tuning without tune configuration."""
        trainer = trainer_with_temp_dir
        
        with pytest.raises(ValueError, match="Tuning configuration not provided"):
            trainer.tune_hyperparameters()
    
    def test_tune_hyperparameters_with_config(self, sample_config, sample_data):
        """Test hyperparameter tuning with configuration."""
        X, y = sample_data
        
        # Add tuning configuration
        sample_config.model.tune = TuneConfig(
            grid=[
                {"n_estimators": [5, 10], "max_depth": [2, 3]}
            ],
            cv=3,
            scoring="accuracy"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.paths.model_dir = temp_dir
            sample_config.paths.metrics_dir = f"{temp_dir}/metrics"
            
            trainer = ModelTrainer(sample_config)
            
            # Mock the data loader instance that's already created
            trainer.data_loader.train_test_split = Mock(return_value=(
                X[:80], X[80:], y[:80], y[80:]
            ))
            
            results = trainer.tune_hyperparameters(save_best_model=False)
            
            # Check results structure
            assert "best_params" in results
            assert "best_score" in results
            assert "cv_results" in results
            assert "train_metrics" in results
            assert "test_metrics" in results
            
            # Check that best model was set
            assert trainer.model_pipeline is not None
    
    def test_prepare_param_grid(self, trainer_with_temp_dir):
        """Test parameter grid preparation for GridSearchCV."""
        trainer = trainer_with_temp_dir
        
        grid_config = [
            {"n_estimators": [5, 10], "max_depth": [2, 3]},
            {"classifier__min_samples_split": [2, 5]}
        ]
        
        param_grid = trainer._prepare_param_grid(grid_config)
        
        expected = [
            {"classifier__n_estimators": [5, 10], "classifier__max_depth": [2, 3]},
            {"classifier__min_samples_split": [2, 5]}
        ]
        
        assert param_grid == expected
    
    def test_predict_without_model(self, trainer_with_temp_dir, sample_data):
        """Test prediction without trained model."""
        trainer = trainer_with_temp_dir
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="No model loaded"):
            trainer.predict(X[:5])
    
    def test_predict_with_model(self, trainer_with_temp_dir, sample_data):
        """Test prediction with trained model."""
        trainer = trainer_with_temp_dir
        X, y = sample_data
        
        # Mock the data loader instance that's already created
        trainer.data_loader.train_test_split = Mock(return_value=(
            X[:80], X[80:], y[:80], y[80:]
        ))
        
        # Train model
        trainer.train(save_model=False)
        
        # Make predictions
        predictions = trainer.predict(X[:5])
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
        assert all(pred in ["Adelie", "Chinstrap", "Gentoo"] for pred in predictions)
    
    def test_predict_proba_without_support(self, trainer_with_temp_dir, sample_data):
        """Test probability prediction with model that doesn't support it."""
        trainer = trainer_with_temp_dir
        X, _ = sample_data
        
        # Create a mock model without predict_proba
        mock_model = Mock()
        del mock_model.predict_proba  # Remove the attribute entirely
        
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": mock_model}
        
        trainer.model_pipeline = mock_pipeline
        
        with pytest.raises(ValueError, match="Model does not support probability predictions"):
            trainer.predict_proba(X[:5])


class TestTrainerConvenienceFunctions:
    """Test convenience functions for training."""
    
    @patch('src.core.config.load_config')
    @patch('src.models.trainer.ModelTrainer')
    def test_train_model_function(self, mock_trainer_class, mock_load_config):
        """Test train_model convenience function."""
        # Mock configuration loading
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {"accuracy": 0.95}
        mock_trainer_class.return_value = mock_trainer
        
        result = train_model("test_config.yaml", save_model=True)
        
        # Verify calls
        mock_load_config.assert_called_once()
        mock_trainer_class.assert_called_once_with(mock_config)
        mock_trainer.train.assert_called_once_with(save_model=True)
        
        assert result == {"accuracy": 0.95}
    
    @patch('src.core.config.load_config')
    @patch('src.models.trainer.ModelTrainer')
    def test_tune_model_function(self, mock_trainer_class, mock_load_config):
        """Test tune_model convenience function."""
        # Mock configuration loading
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.tune_hyperparameters.return_value = {"best_score": 0.97}
        mock_trainer_class.return_value = mock_trainer
        
        result = tune_model("test_config.yaml", save_best_model=True)
        
        # Verify calls
        mock_load_config.assert_called_once()
        mock_trainer_class.assert_called_once_with(mock_config)
        mock_trainer.tune_hyperparameters.assert_called_once_with(save_best_model=True)
        
        assert result == {"best_score": 0.97}


class TestTrainerIntegration:
    """Integration tests for trainer with real components."""
    
    def test_full_training_pipeline(self, sample_data):
        """Test complete training pipeline with real data."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                seed=42,
                model=ModelConfig(
                    name="RandomForestClassifier",
                    params={"n_estimators": 5, "random_state": 42}
                )
            )
            config.paths.model_dir = temp_dir
            config.paths.metrics_dir = f"{temp_dir}/metrics"
            
            trainer = ModelTrainer(config)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            results = trainer.train(X_train, y_train, X_test, y_test, save_model=True)
            
            # Verify results
            assert results["model_name"] == "RandomForestClassifier"
            assert "model_id" in results
            assert 0 <= results["test_metrics"]["accuracy"] <= 1
            
            # Verify model can make predictions
            predictions = trainer.predict(X_test[:5])
            assert len(predictions) == 5
            
            # Verify model was saved
            model_files = list(Path(temp_dir).glob("**/*.pkl"))
            assert len(model_files) > 0
            
            # Verify metrics were saved
            metrics_files = list(Path(temp_dir).glob("**/*_metrics*.json"))
            assert len(metrics_files) > 0