"""FastAPI application for penguin species prediction."""

import traceback
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ..core.config import ServingConfig, load_config
from ..data.schema import ALL_FEATURES, VALID_ISLANDS, VALID_SEXES
from ..models.registry import ModelRegistry
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


class PenguinPredictor:
    """Penguin species prediction service."""

    def __init__(self, model_registry_dir: str = "models"):
        """Initialize the predictor service.

        Args:
            model_registry_dir: Directory containing model registry
        """
        self.model_registry = ModelRegistry(model_registry_dir)
        self.model: BaseEstimator | Pipeline | None = None
        self.model_info: dict | None = None
        self.load_model()

    def load_model(self) -> bool:
        """Load the latest model from registry.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Try to load the best model first, then fall back to latest
            self.model = self.model_registry.load_best_model("f1_score")

            if self.model is None:
                self.model = self.model_registry.load_latest_model()

            if self.model is not None:
                # Get model info
                models = self.model_registry.list_models()
                if models:
                    # Get the latest model ID for info
                    latest_model_id = max(
                        models,
                        key=lambda x: self.model_registry.get_model_info(x)["created"],
                    )
                    self.model_info = self.model_registry.get_model_info(
                        latest_model_id
                    )
                    logger.info(f"Model loaded successfully: {latest_model_id}")
                    return True

            logger.warning("No model found in registry")
            return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.model_info = None
            return False

    def validate_input(self, request: PredictRequest) -> None:
        """Validate prediction request input.

        Args:
            request: Prediction request

        Raises:
            HTTPException: If validation fails
        """
        # Validate island
        if request.island not in VALID_ISLANDS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid island '{request.island}'. Must be one of: {VALID_ISLANDS}",
            )

        # Validate sex
        if request.sex not in VALID_SEXES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid sex '{request.sex}'. Must be one of: {VALID_SEXES}",
            )

        # Validate numeric ranges (basic sanity checks)
        if request.bill_length_mm < 10 or request.bill_length_mm > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Bill length must be between 10 and 100 mm",
            )

        if request.bill_depth_mm < 5 or request.bill_depth_mm > 50:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Bill depth must be between 5 and 50 mm",
            )

        if request.flipper_length_mm < 100 or request.flipper_length_mm > 300:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Flipper length must be between 100 and 300 mm",
            )

        if request.body_mass_g < 1000 or request.body_mass_g > 10000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Body mass must be between 1000 and 10000 g",
            )

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Make a prediction for a single penguin.

        Args:
            request: Prediction request

        Returns:
            Prediction response

        Raises:
            HTTPException: If prediction fails
        """
        if self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        # Validate input
        self.validate_input(request)

        try:
            # Convert request to DataFrame
            data = {
                "island": [request.island],
                "bill_length_mm": [request.bill_length_mm],
                "bill_depth_mm": [request.bill_depth_mm],
                "flipper_length_mm": [request.flipper_length_mm],
                "body_mass_g": [request.body_mass_g],
                "sex": [request.sex],
                "year": [request.year],
            }

            df = pd.DataFrame(data)

            # Make prediction
            prediction = self.model.predict(df)[0]

            # Get prediction probabilities if available
            probabilities = None
            confidence = None

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(df)[0]
                classes = self.model.classes_

                probabilities = {
                    str(cls): float(prob) for cls, prob in zip(classes, proba, strict=False)
                }
                confidence = float(max(proba))

            return PredictResponse(
                prediction=str(prediction),
                confidence=confidence,
                probabilities=probabilities,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )

    def predict_batch(self, request: BatchPredictRequest) -> BatchPredictResponse:
        """Make predictions for multiple penguins.

        Args:
            request: Batch prediction request

        Returns:
            Batch prediction response
        """
        predictions = []

        for instance in request.instances:
            try:
                prediction = self.predict(instance)
                predictions.append(prediction)
            except HTTPException as e:
                # For batch requests, we can choose to skip invalid instances
                # or fail the entire batch. Here we'll fail the entire batch.
                raise e

        return BatchPredictResponse(predictions=predictions)

    def get_health(self) -> HealthResponse:
        """Get service health status.

        Returns:
            Health response
        """
        model_loaded = self.model is not None

        model_info = None
        if model_loaded and self.model_info:
            model_info = {
                "model_id": self.model_info.get("model_id", "unknown"),
                "model_type": self.model_info.get("model_type", "unknown"),
                "classes": self.model_info.get("classes", []),
            }

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            model_info=model_info,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Get information about the loaded model.

        Returns:
            Model information response

        Raises:
            HTTPException: If no model is loaded
        """
        if self.model is None or self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded",
            )

        return ModelInfoResponse(
            model_id=self.model_info.get("model_id", "unknown"),
            model_type=self.model_info.get("model_type", "unknown"),
            classes=self.model_info.get("classes", []),
            features=ALL_FEATURES,
            created=self.model_info.get("created", "unknown"),
        )


# Global predictor instance
predictor = PenguinPredictor()

# Create FastAPI app
app = FastAPI(
    title="Penguin Species Classifier API",
    description="API for predicting penguin species based on physical characteristics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("Starting Penguin Species Classifier API")
    logger.info(f"Model loaded: {predictor.model is not None}")
    if predictor.model_info:
        logger.info(f"Model info: {predictor.model_info}")

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("Shutting down Penguin Species Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            details={"status_code": exc.status_code},
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions with structured error responses."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message="Internal server error",
            details={"original_error": str(exc)},
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    health_status = predictor.get_health()
    logger.debug(f"Health check result: {health_status.status}")
    return health_status


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    return predictor.get_model_info()


@app.post("/predict", response_model=PredictResponse)
async def predict_species(request: PredictRequest):
    """Predict penguin species for a single instance."""
    logger.info(f"Prediction request received: {request.model_dump()}")
    start_time = datetime.now()
    
    try:
        result = predictor.predict(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prediction completed in {duration:.3f}s: {result.prediction} (confidence: {result.confidence})")
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Prediction failed after {duration:.3f}s: {str(e)}")
        raise


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_species_batch(request: BatchPredictRequest):
    """Predict penguin species for multiple instances."""
    logger.info(f"Batch prediction request received with {len(request.instances)} instances")
    start_time = datetime.now()
    
    try:
        result = predictor.predict_batch(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch prediction completed in {duration:.3f}s for {len(result.predictions)} instances")
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Batch prediction failed after {duration:.3f}s: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Penguin Species Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }


if __name__ == "__main__":
    import uvicorn

    # Load serving configuration
    config = load_config("configs/serving.yaml", ServingConfig)

    # Configure logging
    logger.add(
        config.logging.log_file,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        level=config.logging.level,
        format=config.logging.format,
    )

    # Start the server
    uvicorn.run(
        "src.serving.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
    )
