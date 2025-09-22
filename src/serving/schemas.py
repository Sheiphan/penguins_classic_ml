"""Pydantic schemas for API request/response models."""


from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for penguin species prediction."""

    island: str = Field(..., description="Island where penguin was observed")
    bill_length_mm: float = Field(..., ge=0, description="Bill length in millimeters")
    bill_depth_mm: float = Field(..., ge=0, description="Bill depth in millimeters")
    flipper_length_mm: float = Field(
        ..., ge=0, description="Flipper length in millimeters"
    )
    body_mass_g: float = Field(..., ge=0, description="Body mass in grams")
    sex: str = Field(..., description="Sex of the penguin")
    year: int = Field(..., ge=2007, le=2009, description="Year of observation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "island": "Torgersen",
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": "MALE",
                "year": 2007,
            }
        }
    }


class PredictResponse(BaseModel):
    """Response schema for penguin species prediction."""

    prediction: str = Field(..., description="Predicted penguin species")
    confidence: float | None = Field(
        None, ge=0, le=1, description="Prediction confidence score"
    )
    probabilities: dict | None = Field(None, description="Class probabilities")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Adelie",
                "confidence": 0.95,
                "probabilities": {"Adelie": 0.95, "Chinstrap": 0.03, "Gentoo": 0.02},
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Request schema for batch penguin species prediction."""

    instances: list[PredictRequest] = Field(
        ..., description="List of penguin instances to predict"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "instances": [
                    {
                        "island": "Torgersen",
                        "bill_length_mm": 39.1,
                        "bill_depth_mm": 18.7,
                        "flipper_length_mm": 181.0,
                        "body_mass_g": 3750.0,
                        "sex": "MALE",
                        "year": 2007,
                    }
                ]
            }
        }
    }


class BatchPredictResponse(BaseModel):
    """Response schema for batch penguin species prediction."""

    predictions: list[PredictResponse] = Field(..., description="List of predictions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "prediction": "Adelie",
                        "confidence": 0.95,
                        "probabilities": {
                            "Adelie": 0.95,
                            "Chinstrap": 0.03,
                            "Gentoo": 0.02,
                        },
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    model_info: dict | None = Field(
        None, description="Information about loaded model"
    )
    timestamp: str = Field(..., description="Health check timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {
                    "model_id": "penguin_classifier_v1",
                    "model_type": "Pipeline",
                    "classes": ["Adelie", "Chinstrap", "Gentoo"],
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    }


class ErrorResponse(BaseModel):
    """Response schema for API errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {
                    "field": "bill_length_mm",
                    "issue": "must be greater than 0",
                },
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""

    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Type of model")
    classes: list[str] = Field(..., description="Target classes")
    features: list[str] = Field(..., description="Input features")
    created: str = Field(..., description="Model creation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "penguin_classifier_v1",
                "model_type": "Pipeline",
                "classes": ["Adelie", "Chinstrap", "Gentoo"],
                "features": [
                    "island",
                    "bill_length_mm",
                    "bill_depth_mm",
                    "flipper_length_mm",
                    "body_mass_g",
                    "sex",
                    "year",
                ],
                "created": "2024-01-15T09:00:00Z",
            }
        }
    }
