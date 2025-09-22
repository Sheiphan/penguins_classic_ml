"""Unit tests for model metrics calculation and storage."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.models.metrics import (
    ModelMetrics,
    MetricsStorage,
    calculate_model_metrics,
    load_model_metrics,
    save_model_metrics
)


@pytest.fixture
def sample_classification_data():
    """Create sample classification data for testing."""
    np.random.seed(42)
    
    # Create balanced dataset
    n_samples = 150
    y_true = np.array(["Adelie"] * 50 + ["Chinstrap"] * 50 + ["Gentoo"] * 50)
    
    # Create predictions with some errors
    y_pred = y_true.copy()
    # Introduce some misclassifications
    error_indices = np.random.choice(n_samples, size=15, replace=False)
    for idx in error_indices:
        current_class = y_pred[idx]
        other_classes = [c for c in ["Adelie", "Chinstrap", "Gentoo"] if c != current_class]
        y_pred[idx] = np.random.choice(other_classes)
    
    # Create probability predictions
    n_classes = 3
    y_proba = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    
    # Make probabilities more realistic (higher for correct class)
    for i, true_class in enumerate(y_true):
        class_idx = ["Adelie", "Chinstrap", "Gentoo"].index(true_class)
        y_proba[i] = np.random.dirichlet([1, 1, 1])
        y_proba[i][class_idx] = max(y_proba[i][class_idx], 0.4)  # Ensure some confidence
        y_proba[i] = y_proba[i] / y_proba[i].sum()  # Normalize
    
    return y_true, y_pred, y_proba


@pytest.fixture
def sample_binary_data():
    """Create sample binary classification data."""
    np.random.seed(42)
    
    n_samples = 100
    y_true = np.array([0] * 50 + [1] * 50)
    
    # Create predictions with some errors
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    # Create probability predictions
    y_proba = np.random.rand(n_samples, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    return y_true, y_pred, y_proba


class TestModelMetrics:
    """Test cases for ModelMetrics class."""
    
    def test_init(self):
        """Test metrics calculator initialization."""
        calculator = ModelMetrics()
        assert calculator is not None
    
    def test_calculate_metrics_multiclass(self, sample_classification_data):
        """Test metrics calculation for multi-class classification."""
        y_true, y_pred, y_proba = sample_classification_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Check basic metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # Verify accuracy calculation
        expected_accuracy = accuracy_score(y_true, y_pred)
        assert abs(metrics["accuracy"] - expected_accuracy) < 1e-6
        
        # Check per-class metrics
        assert "per_class" in metrics
        assert len(metrics["per_class"]) == 3
        for class_name in ["Adelie", "Chinstrap", "Gentoo"]:
            assert class_name in metrics["per_class"]
            assert "precision" in metrics["per_class"][class_name]
            assert "recall" in metrics["per_class"][class_name]
            assert "f1_score" in metrics["per_class"][class_name]
        
        # Check confusion matrix
        assert "confusion_matrix" in metrics
        assert "matrix" in metrics["confusion_matrix"]
        assert "labels" in metrics["confusion_matrix"]
        assert len(metrics["confusion_matrix"]["matrix"]) == 3
        assert len(metrics["confusion_matrix"]["labels"]) == 3
        
        # Check classification report
        assert "classification_report" in metrics
        
        # Check ROC AUC (should be calculated for multi-class)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] is not None
        
        # Check support and metadata
        assert "support" in metrics
        assert "total_samples" in metrics
        assert "num_classes" in metrics
        assert "classes" in metrics
        
        assert metrics["total_samples"] == len(y_true)
        assert metrics["num_classes"] == 3
    
    def test_calculate_metrics_binary(self, sample_binary_data):
        """Test metrics calculation for binary classification."""
        y_true, y_pred, y_proba = sample_binary_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Check basic structure
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert metrics["num_classes"] == 2
        
        # Verify ROC AUC for binary case
        assert metrics["roc_auc"] is not None
        assert 0 <= metrics["roc_auc"] <= 1
    
    def test_calculate_metrics_without_probabilities(self, sample_classification_data):
        """Test metrics calculation without probability predictions."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred)
        
        # Should still calculate basic metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # ROC AUC should be None without probabilities
        assert "roc_auc" not in metrics or metrics["roc_auc"] is None
    
    def test_calculate_metrics_with_pandas_input(self, sample_classification_data):
        """Test metrics calculation with pandas Series input."""
        y_true, y_pred, y_proba = sample_classification_data
        
        # Convert to pandas
        y_true_series = pd.Series(y_true)
        y_pred_series = pd.Series(y_pred)
        y_proba_df = pd.DataFrame(y_proba, columns=["Adelie", "Chinstrap", "Gentoo"])
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true_series, y_pred_series, y_proba_df)
        
        # Should work the same as numpy arrays
        assert "accuracy" in metrics
        assert metrics["total_samples"] == len(y_true)
    
    def test_compare_metrics(self):
        """Test metrics comparison functionality."""
        calculator = ModelMetrics()
        
        # Create sample metrics
        metrics1 = {"accuracy": 0.85, "f1_score": 0.83, "precision": 0.86, "recall": 0.80}
        metrics2 = {"accuracy": 0.90, "f1_score": 0.88, "precision": 0.89, "recall": 0.87}
        metrics3 = {"accuracy": 0.82, "f1_score": 0.79, "precision": 0.84, "recall": 0.75}
        
        comparison_df = calculator.compare_metrics(
            [metrics1, metrics2, metrics3],
            ["Model_A", "Model_B", "Model_C"]
        )
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert "model" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "f1_score" in comparison_df.columns
        
        # Check values
        assert comparison_df.loc[0, "model"] == "Model_A"
        assert comparison_df.loc[0, "accuracy"] == 0.85
    
    def test_get_best_model(self):
        """Test best model selection functionality."""
        calculator = ModelMetrics()
        
        # Create sample metrics
        metrics1 = {"accuracy": 0.85, "f1_score": 0.83}
        metrics2 = {"accuracy": 0.90, "f1_score": 0.88}
        metrics3 = {"accuracy": 0.82, "f1_score": 0.79}
        
        best_info = calculator.get_best_model(
            [metrics1, metrics2, metrics3],
            ["Model_A", "Model_B", "Model_C"],
            primary_metric="f1_score"
        )
        
        assert best_info["best_model_name"] == "Model_B"
        assert best_info["best_model_index"] == 1
        assert best_info["best_score"] == 0.88
        assert best_info["primary_metric"] == "f1_score"
    
    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation."""
        calculator = ModelMetrics()
        
        # Create sample regression data
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "total_samples" in metrics
        
        assert metrics["total_samples"] == 100
        assert metrics["r2"] <= 1.0  # R² should be <= 1


class TestMetricsStorage:
    """Test cases for MetricsStorage class."""
    
    def test_init(self):
        """Test metrics storage initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            assert storage.storage_dir == Path(temp_dir)
            assert storage.storage_dir.exists()
    
    def test_save_and_load_metrics(self, sample_classification_data):
        """Test saving and loading metrics."""
        y_true, y_pred, y_proba = sample_classification_data
        
        # Calculate metrics
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            # Save metrics
            saved_path = storage.save_metrics(metrics, "test_model")
            assert saved_path.exists()
            assert saved_path.name == "test_model_metrics.json"
            
            # Load metrics
            loaded_data = storage.load_metrics("test_model")
            
            assert "model_id" in loaded_data
            assert "timestamp" in loaded_data
            assert "metrics" in loaded_data
            
            assert loaded_data["model_id"] == "test_model"
            assert loaded_data["metrics"]["accuracy"] == metrics["accuracy"]
    
    def test_save_metrics_with_suffix(self, sample_classification_data):
        """Test saving metrics with suffix."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            saved_path = storage.save_metrics(metrics, "test_model", suffix="tuned")
            assert saved_path.name == "test_model_metrics_tuned.json"
    
    def test_load_nonexistent_metrics(self):
        """Test loading metrics that don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                storage.load_metrics("nonexistent_model")
    
    def test_list_metrics_files(self, sample_classification_data):
        """Test listing metrics files."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            # Initially empty
            files = storage.list_metrics_files()
            assert len(files) == 0
            
            # Save some metrics
            storage.save_metrics(metrics, "model1")
            storage.save_metrics(metrics, "model2")
            
            files = storage.list_metrics_files()
            assert len(files) == 2
            assert "model1_metrics" in files
            assert "model2_metrics" in files
    
    def test_load_all_metrics(self, sample_classification_data):
        """Test loading all metrics."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            # Save multiple metrics
            storage.save_metrics(metrics, "model1")
            storage.save_metrics(metrics, "model2")
            
            all_metrics = storage.load_all_metrics()
            
            assert len(all_metrics) == 2
            assert "model1" in all_metrics
            assert "model2" in all_metrics
    
    def test_compare_all_models(self, sample_classification_data):
        """Test comparing all stored models."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            # Create different metrics for different models
            metrics1 = calculator.calculate_metrics(y_true, y_pred)
            metrics2 = calculator.calculate_metrics(y_true, y_pred)
            metrics2["accuracy"] = 0.95  # Make it better
            
            # Save with nested structure (like from trainer)
            nested_metrics1 = {"test_metrics": metrics1, "model_name": "RF"}
            nested_metrics2 = {"test_metrics": metrics2, "model_name": "LR"}
            
            storage.save_metrics(nested_metrics1, "rf_model")
            storage.save_metrics(nested_metrics2, "lr_model")
            
            comparison_df = storage.compare_all_models("accuracy")
            
            assert isinstance(comparison_df, pd.DataFrame)
            assert len(comparison_df) == 2
            assert "model_id" in comparison_df.columns
            assert "accuracy" in comparison_df.columns
            
            # Should be sorted by accuracy (descending)
            assert comparison_df.iloc[0]["accuracy"] >= comparison_df.iloc[1]["accuracy"]
    
    def test_get_best_model_id(self, sample_classification_data):
        """Test getting best model ID."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            # Create metrics with different performance
            metrics1 = calculator.calculate_metrics(y_true, y_pred)
            metrics2 = calculator.calculate_metrics(y_true, y_pred)
            metrics2["f1_score"] = 0.95  # Make it better
            
            nested_metrics1 = {"test_metrics": metrics1}
            nested_metrics2 = {"test_metrics": metrics2}
            
            storage.save_metrics(nested_metrics1, "model1")
            storage.save_metrics(nested_metrics2, "model2")
            
            best_id = storage.get_best_model_id("f1_score")
            assert best_id == "model2"
    
    def test_cleanup_old_metrics(self, sample_classification_data):
        """Test cleaning up old metrics files."""
        y_true, y_pred, _ = sample_classification_data
        
        calculator = ModelMetrics()
        metrics = calculator.calculate_metrics(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetricsStorage(temp_dir)
            
            # Save multiple metrics files
            for i in range(5):
                storage.save_metrics(metrics, f"model_{i}")
            
            # Verify all files exist
            files_before = storage.list_metrics_files()
            assert len(files_before) == 5
            
            # Cleanup, keeping only 3
            storage.cleanup_old_metrics(keep_latest=3)
            
            files_after = storage.list_metrics_files()
            assert len(files_after) == 3


class TestConvenienceFunctions:
    """Test convenience functions for metrics."""
    
    def test_calculate_model_metrics(self, sample_classification_data):
        """Test calculate_model_metrics convenience function."""
        y_true, y_pred, y_proba = sample_classification_data
        
        metrics = calculate_model_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert metrics["total_samples"] == len(y_true)
    
    def test_save_and_load_model_metrics(self, sample_classification_data):
        """Test save_model_metrics and load_model_metrics convenience functions."""
        y_true, y_pred, _ = sample_classification_data
        
        metrics = calculate_model_metrics(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save metrics
            saved_path = save_model_metrics(metrics, "test_model", temp_dir)
            assert saved_path.exists()
            
            # Load metrics
            loaded_data = load_model_metrics("test_model", temp_dir)
            assert loaded_data["metrics"]["accuracy"] == metrics["accuracy"]


class TestMetricsIntegration:
    """Integration tests for metrics system."""
    
    def test_full_metrics_workflow(self, sample_classification_data):
        """Test complete metrics workflow."""
        y_true, y_pred, y_proba = sample_classification_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Calculate metrics
            calculator = ModelMetrics()
            metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
            
            # Save metrics
            storage = MetricsStorage(temp_dir)
            storage.save_metrics(metrics, "final_model")
            
            # Load and verify
            loaded_data = storage.load_metrics("final_model")
            loaded_metrics = loaded_data["metrics"]
            
            # Compare key metrics
            assert abs(loaded_metrics["accuracy"] - metrics["accuracy"]) < 1e-6
            assert loaded_metrics["num_classes"] == metrics["num_classes"]
            
            # Test comparison functionality
            comparison_df = storage.compare_all_models()
            assert len(comparison_df) == 1
            assert comparison_df.iloc[0]["model_id"] == "final_model"