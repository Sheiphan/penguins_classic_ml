"""Model training logic with sklearn pipelines."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..core.config import ExperimentConfig, ModelConfig, TuneConfig
from ..data.dataset import PenguinDataLoader
from ..features.preprocess import PenguinPreprocessor
from .metrics import ModelMetrics
from .registry import ModelRegistry

# Available models mapping
AVAILABLE_MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "SVC": SVC
}


class ModelTrainer:
    """Model trainer for penguins classification."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the trainer with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.data_loader = PenguinDataLoader()
        self.preprocessor = PenguinPreprocessor()
        self.model_pipeline: Optional[Pipeline] = None
        self.metrics_calculator = ModelMetrics()
        self.registry = ModelRegistry(config.paths.model_dir)
        
        # Set random seed for reproducibility
        self._set_random_seed(config.seed)
    
    def _set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random

        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Set seed for sklearn models that support it
        if hasattr(self.config.model, 'params') and self.config.model.params:
            if 'random_state' not in self.config.model.params:
                self.config.model.params['random_state'] = seed
    
    def _get_model_class(self, model_name: str) -> BaseEstimator:
        """Get model class by name.
        
        Args:
            model_name: Name of the model class
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model name is not supported
        """
        if model_name not in AVAILABLE_MODELS:
            available = list(AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_name} not supported. Available: {available}")
        
        return AVAILABLE_MODELS[model_name]
    
    def _create_model(self, model_config: ModelConfig) -> BaseEstimator:
        """Create model instance from configuration.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Configured model instance
        """
        model_class = self._get_model_class(model_config.name)
        
        # Ensure random_state is set for reproducibility
        params = model_config.params.copy()
        if hasattr(model_class(), 'random_state') and 'random_state' not in params:
            params['random_state'] = self.config.seed
        
        return model_class(**params)
    
    def create_pipeline(self, model_config: Optional[ModelConfig] = None) -> Pipeline:
        """Create sklearn pipeline with preprocessing and model.
        
        Args:
            model_config: Model configuration (uses default if None)
            
        Returns:
            Configured sklearn Pipeline
        """
        if model_config is None:
            model_config = self.config.model
        
        # Create preprocessor
        preprocessor = self.preprocessor.build_preprocessor()
        
        # Create model
        model = self._create_model(model_config)
        
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        return pipeline
    
    def train(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """Train the model and evaluate performance.
        
        Args:
            X_train: Training features (loads from data if None)
            y_train: Training target (loads from data if None)
            X_test: Test features (loads from data if None)
            y_test: Test target (loads from data if None)
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting model training")
        
        # Load data if not provided
        if any(x is None for x in [X_train, y_train, X_test, y_test]):
            logger.info("Loading data from dataset")
            X_train, X_test, y_train, y_test = self.data_loader.train_test_split(
                test_size=self.config.features.test_size,
                random_state=self.config.seed,
                stratify=self.config.features.stratify
            )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Target distribution: {y_train.value_counts().to_dict()}")
        
        # Create and train pipeline
        self.model_pipeline = self.create_pipeline()
        
        logger.info(f"Training {self.config.model.name} model")
        self.model_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model_pipeline.predict(X_train)
        y_test_pred = self.model_pipeline.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.metrics_calculator.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.metrics_calculator.calculate_metrics(y_test, y_test_pred)
        
        # Prepare results
        results = {
            "model_name": self.config.model.name,
            "model_params": self.config.model.params,
            "data_info": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "features": list(X_train.columns),
                "target_classes": sorted(y_train.unique())
            },
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "config": self.config.model_dump()
        }
        
        logger.info(f"Training completed. Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Save model and results
        if save_model:
            model_id = self._save_model_and_results(results)
            results["model_id"] = model_id
        
        return results
    
    def tune_hyperparameters(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        save_best_model: bool = True
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features (loads from data if None)
            y_train: Training target (loads from data if None)
            X_test: Test features (loads from data if None)
            y_test: Test target (loads from data if None)
            save_best_model: Whether to save the best model
            
        Returns:
            Dictionary with tuning results
        """
        if self.config.model.tune is None:
            raise ValueError("Tuning configuration not provided")
        
        logger.info("Starting hyperparameter tuning")
        
        # Load data if not provided
        if any(x is None for x in [X_train, y_train, X_test, y_test]):
            logger.info("Loading data from dataset")
            X_train, X_test, y_train, y_test = self.data_loader.train_test_split(
                test_size=self.config.features.test_size,
                random_state=self.config.seed,
                stratify=self.config.features.stratify
            )
        
        # Create base pipeline
        base_pipeline = self.create_pipeline()
        
        # Prepare parameter grid for grid search
        param_grid = self._prepare_param_grid(self.config.model.tune.grid)
        
        logger.info(f"Grid search with {len(param_grid)} parameter combinations")
        logger.info(f"Cross-validation folds: {self.config.model.tune.cv}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=self.config.model.tune.cv,
            scoring=self.config.model.tune.scoring,
            n_jobs=self.config.model.tune.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_pipeline = grid_search.best_estimator_
        
        # Make predictions with best model
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.metrics_calculator.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.metrics_calculator.calculate_metrics(y_test, y_test_pred)
        
        # Prepare results
        results = {
            "model_name": self.config.model.name,
            "tuning_config": self.config.model.tune.model_dump(),
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": {
                "mean_test_scores": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_scores": grid_search.cv_results_["std_test_score"].tolist(),
                "params": grid_search.cv_results_["params"]
            },
            "data_info": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "features": list(X_train.columns),
                "target_classes": sorted(y_train.unique())
            },
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "config": self.config.model_dump()
        }
        
        logger.info(f"Tuning completed. Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Set best model
        self.model_pipeline = best_pipeline
        
        # Save best model if requested
        if save_best_model:
            model_id = self._save_model_and_results(results, suffix="tuned")
            results["model_id"] = model_id
        
        return results
    
    def _prepare_param_grid(self, grid_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare parameter grid for GridSearchCV.
        
        Args:
            grid_config: Grid configuration from config
            
        Returns:
            Parameter grid for sklearn GridSearchCV
        """
        param_grid = []
        
        for grid_item in grid_config:
            # Add classifier__ prefix to model parameters
            sklearn_grid = {}
            for param, values in grid_item.items():
                if param.startswith("classifier__"):
                    sklearn_grid[param] = values
                else:
                    sklearn_grid[f"classifier__{param}"] = values
            
            param_grid.append(sklearn_grid)
        
        return param_grid
    
    def _save_model_and_results(
        self, 
        results: Dict[str, Any], 
        suffix: str = ""
    ) -> str:
        """Save model and results to registry.
        
        Args:
            results: Training/tuning results
            suffix: Optional suffix for model name
            
        Returns:
            Model ID
        """
        if self.model_pipeline is None:
            raise ValueError("No trained model to save")
        
        # Generate model ID
        model_name = self.config.model.name.lower()
        if suffix:
            model_id = f"{model_name}_{suffix}"
        else:
            model_id = model_name
        
        # Save model
        model_path = self.registry.save_model(self.model_pipeline, model_id)
        
        # Save metrics
        metrics_path = self.registry.save_metrics(results, model_id)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metrics saved to: {metrics_path}")
        
        return model_id
    
    def load_model(self, model_id: str) -> Pipeline:
        """Load a trained model from registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded sklearn Pipeline
        """
        self.model_pipeline = self.registry.load_model(model_id)
        return self.model_pipeline
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model_id: Optional[str] = None
    ) -> pd.Series:
        """Make predictions using trained model.
        
        Args:
            X: Features for prediction
            model_id: Model ID to use (loads if not already loaded)
            
        Returns:
            Predictions
        """
        if model_id is not None:
            self.load_model(model_id)
        
        if self.model_pipeline is None:
            raise ValueError("No model loaded. Train a model or provide model_id")
        
        predictions = self.model_pipeline.predict(X)
        return pd.Series(predictions, index=X.index)
    
    def predict_proba(
        self, 
        X: pd.DataFrame, 
        model_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Make probability predictions using trained model.
        
        Args:
            X: Features for prediction
            model_id: Model ID to use (loads if not already loaded)
            
        Returns:
            Prediction probabilities
        """
        if model_id is not None:
            self.load_model(model_id)
        
        if self.model_pipeline is None:
            raise ValueError("No model loaded. Train a model or provide model_id")
        
        if not hasattr(self.model_pipeline.named_steps["classifier"], "predict_proba"):
            raise ValueError("Model does not support probability predictions")
        
        probabilities = self.model_pipeline.predict_proba(X)
        classes = self.model_pipeline.classes_
        
        return pd.DataFrame(probabilities, columns=classes, index=X.index)


def train_model(
    config_path: str = "configs/experiment_default.yaml",
    save_model: bool = True
) -> Dict[str, Any]:
    """Convenience function to train a model.
    
    Args:
        config_path: Path to experiment configuration
        save_model: Whether to save the trained model
        
    Returns:
        Training results
    """
    from ..core.config import load_config
    
    config = load_config(config_path, ExperimentConfig)
    trainer = ModelTrainer(config)
    
    return trainer.train(save_model=save_model)


def tune_model(
    config_path: str = "configs/experiment_default.yaml",
    save_best_model: bool = True
) -> Dict[str, Any]:
    """Convenience function to tune model hyperparameters.
    
    Args:
        config_path: Path to experiment configuration
        save_best_model: Whether to save the best model
        
    Returns:
        Tuning results
    """
    from ..core.config import load_config
    
    config = load_config(config_path, ExperimentConfig)
    trainer = ModelTrainer(config)
    
    return trainer.tune_hyperparameters(save_best_model=save_best_model)