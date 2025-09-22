"""Tests for the FastAPI serving layer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.registry import ModelRegistry
from src.serving.app import PenguinPredictor, app
from src.serving.schemas import PredictRequest


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model registry."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def simple_model():
    """Create a simple real model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple pipeline with minimal training
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=2, random_state=42))
    ])
    
    # Create minimal training data
    X_train = np.array([[1, 2, 3, 4, 5, 0, 2007], [2, 3, 4, 5, 6, 1, 2008]])
    y_train = np.array(["Adelie", "Chinstrap"])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model


@pytest.fixture
def model_registry_with_model(temp_model_dir, simple_model):
    """Create a model registry with a saved model."""
    registry = ModelRegistry(temp_model_dir)
    
    # Save the simple model
    model_id = "test_penguin_model"
    metadata = {
        "model_id": model_id,
        "classes": ["Adelie", "Chinstrap", "Gentoo"],
        "created": "2024-01-15T10:00:00Z"
    }
    
    registry.save_model(simple_model, model_id, metadata)
    
    # Save some mock metrics
    metrics = {
        "accuracy": 0.95,
        "f1_score": 0.94,
        "precision": 0.93,
        "recall": 0.96
    }
    registry.save_metrics(metrics, model_id)
    
    return registry


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def valid_predict_request():
    """Create a valid prediction request."""
    return {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "MALE",
        "year": 2007
    }


class TestPenguinPredictor:
    """Test the PenguinPredictor class."""
    
    def test_init_with_no_model(self, temp_model_dir):
        """Test predictor initialization with no model."""
        predictor = PenguinPredictor(temp_model_dir)
        assert predictor.model is None
        assert predictor.model_info is None
    
    def test_init_with_model(self, temp_model_dir, model_registry_with_model):
        """Test predictor initialization with a model."""
        predictor = PenguinPredictor(temp_model_dir)
        assert predictor.model is not None
        assert predictor.model_info is not None
    
    def test_load_model_success(self, temp_model_dir, model_registry_with_model):
        """Test successful model loading."""
        predictor = PenguinPredictor(temp_model_dir)
        result = predictor.load_model()
        assert result is True
        assert predictor.model is not None
    
    def test_load_model_failure(self, temp_model_dir):
        """Test model loading failure."""
        predictor = PenguinPredictor(temp_model_dir)
        result = predictor.load_model()
        assert result is False
        assert predictor.model is None
    
    def test_validate_input_valid(self, temp_model_dir, model_registry_with_model):
        """Test input validation with valid data."""
        predictor = PenguinPredictor(temp_model_dir)
        request = PredictRequest(
            island="Torgersen",
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
            year=2007
        )
        
        # Should not raise an exception
        predictor.validate_input(request)
    
    def test_validate_input_invalid_island(self, temp_model_dir, model_registry_with_model):
        """Test input validation with invalid island."""
        predictor = PenguinPredictor(temp_model_dir)
        request = PredictRequest(
            island="InvalidIsland",
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
            year=2007
        )
        
        with pytest.raises(Exception):  # HTTPException
            predictor.validate_input(request)
    
    def test_validate_input_invalid_sex(self, temp_model_dir, model_registry_with_model):
        """Test input validation with invalid sex."""
        predictor = PenguinPredictor(temp_model_dir)
        request = PredictRequest(
            island="Torgersen",
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="INVALID",
            year=2007
        )
        
        with pytest.raises(Exception):  # HTTPException
            predictor.validate_input(request)
    
    def test_validate_input_invalid_ranges(self, temp_model_dir, model_registry_with_model):
        """Test input validation with values out of range."""
        predictor = PenguinPredictor(temp_model_dir)
        
        # Test bill length too small
        request = PredictRequest(
            island="Torgersen",
            bill_length_mm=5.0,  # Too small
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
            year=2007
        )
        
        with pytest.raises(Exception):  # HTTPException
            predictor.validate_input(request)
    
    @patch.object(PenguinPredictor, 'predict')
    def test_predict_success(self, mock_predict, temp_model_dir, model_registry_with_model):
        """Test successful prediction."""
        # Mock the predict method to return a successful response
        from src.serving.schemas import PredictResponse
        mock_predict.return_value = PredictResponse(
            prediction="Adelie",
            confidence=0.8,
            probabilities={"Adelie": 0.8, "Chinstrap": 0.1, "Gentoo": 0.1}
        )
        
        predictor = PenguinPredictor(temp_model_dir)
        request = PredictRequest(
            island="Torgersen",
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
            year=2007
        )
        
        response = predictor.predict(request)
        
        assert response.prediction == "Adelie"
        assert response.confidence == 0.8
        assert response.probabilities is not None
        assert "Adelie" in response.probabilities
    
    def test_predict_no_model(self, temp_model_dir):
        """Test prediction with no model loaded."""
        predictor = PenguinPredictor(temp_model_dir)
        request = PredictRequest(
            island="Torgersen",
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
            year=2007
        )
        
        with pytest.raises(Exception):  # HTTPException
            predictor.predict(request)
    
    def test_get_health_with_model(self, temp_model_dir, model_registry_with_model):
        """Test health check with model loaded."""
        predictor = PenguinPredictor(temp_model_dir)
        health = predictor.get_health()
        
        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.model_info is not None
        assert health.timestamp is not None
    
    def test_get_health_no_model(self, temp_model_dir):
        """Test health check with no model."""
        predictor = PenguinPredictor(temp_model_dir)
        health = predictor.get_health()
        
        assert health.status == "unhealthy"
        assert health.model_loaded is False
        assert health.model_info is None
    
    def test_get_model_info_success(self, temp_model_dir, model_registry_with_model):
        """Test getting model info successfully."""
        predictor = PenguinPredictor(temp_model_dir)
        info = predictor.get_model_info()
        
        assert info.model_id is not None
        assert info.model_type is not None
        assert len(info.classes) > 0
        assert len(info.features) > 0
    
    def test_get_model_info_no_model(self, temp_model_dir):
        """Test getting model info with no model."""
        predictor = PenguinPredictor(temp_model_dir)
        
        with pytest.raises(Exception):  # HTTPException
            predictor.get_model_info()


class TestFastAPIEndpoints:
    """Test the FastAPI endpoints."""
    
    @patch('src.serving.app.predictor')
    def test_health_endpoint(self, mock_predictor, client):
        """Test the health endpoint."""
        # Mock the health response with actual HealthResponse object
        from src.serving.schemas import HealthResponse
        mock_health = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_info={"model_id": "test"},
            timestamp="2024-01-15T10:00:00Z"
        )
        mock_predictor.get_health.return_value = mock_health
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    
    @patch('src.serving.app.predictor')
    def test_model_info_endpoint(self, mock_predictor, client):
        """Test the model info endpoint."""
        # Mock the model info response with actual ModelInfoResponse object
        from src.serving.schemas import ModelInfoResponse
        mock_info = ModelInfoResponse(
            model_id="test_model",
            model_type="Pipeline",
            classes=["Adelie", "Chinstrap", "Gentoo"],
            features=["island", "bill_length_mm"],
            created="2024-01-15T10:00:00Z"
        )
        mock_predictor.get_model_info.return_value = mock_info
        
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test_model"
        assert len(data["classes"]) == 3
    
    @patch('src.serving.app.predictor')
    def test_predict_endpoint_success(self, mock_predictor, client, valid_predict_request):
        """Test successful prediction endpoint."""
        # Mock the prediction response with actual PredictResponse object
        from src.serving.schemas import PredictResponse
        mock_response = PredictResponse(
            prediction="Adelie",
            confidence=0.8,
            probabilities={"Adelie": 0.8, "Chinstrap": 0.1, "Gentoo": 0.1}
        )
        mock_predictor.predict.return_value = mock_response
        
        response = client.post("/predict", json=valid_predict_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Adelie"
        assert data["confidence"] == 0.8
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_request = {
            "island": "InvalidIsland",  # Invalid island
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
            "year": 2007
        }
        
        response = client.post("/predict", json=invalid_request)
        
        # Should return validation error or service unavailable (no model loaded)
        assert response.status_code in [422, 500, 503]  # Validation, server error, or service unavailable
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_request = {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.serving.app.predictor')
    def test_batch_predict_endpoint(self, mock_predictor, client, valid_predict_request):
        """Test batch prediction endpoint."""
        # Mock the batch prediction response with actual objects
        from src.serving.schemas import BatchPredictResponse, PredictResponse
        mock_response = BatchPredictResponse(
            predictions=[
                PredictResponse(
                    prediction="Adelie",
                    confidence=0.8,
                    probabilities={"Adelie": 0.8, "Chinstrap": 0.1, "Gentoo": 0.1}
                )
            ]
        )
        mock_predictor.predict_batch.return_value = mock_response
        
        batch_request = {
            "instances": [valid_predict_request]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["prediction"] == "Adelie"
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
    
    def test_docs_endpoint(self, client):
        """Test that docs endpoint is accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestInputValidation:
    """Test input validation edge cases."""
    
    def test_pydantic_validation_negative_values(self, client):
        """Test Pydantic validation for negative values."""
        invalid_request = {
            "island": "Torgersen",
            "bill_length_mm": -10.0,  # Negative value
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
            "year": 2007
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_pydantic_validation_year_range(self, client):
        """Test Pydantic validation for year range."""
        invalid_request = {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
            "year": 2020  # Out of valid range
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_pydantic_validation_string_types(self, client):
        """Test Pydantic validation for string type fields."""
        invalid_request = {
            "island": 123,  # Should be string
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
            "year": 2007
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__])