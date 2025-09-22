"""Command-line interface for the Enterprise ML Classifier."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from .core.config import ExperimentConfig, ServingConfig, load_config
from .models.trainer import train_model, tune_model


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        level: Logging level
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )

    # Also log to file if logs directory exists
    logs_dir = Path("logs")
    if logs_dir.exists():
        logger.add(
            logs_dir / "cli.log",
            level=level,
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Enable quiet mode (errors only)")
def cli(verbose: bool, quiet: bool) -> None:
    """Enterprise ML Classifier CLI.

    A command-line interface for training, tuning, and managing ML models.
    """
    if quiet:
        setup_logging("ERROR")
    elif verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")


@cli.command()
@click.option(
    "--config",
    "-c",
    default="configs/experiment_default.yaml",
    help="Path to experiment configuration file",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    help="Output file for training results (JSON format)",
    type=click.Path(),
)
@click.option("--no-save", is_flag=True, help="Don't save the trained model")
def train(config: str, output: Optional[str], no_save: bool) -> None:
    """Train a machine learning model.

    This command trains a model using the specified configuration file.
    The trained model and metrics will be saved to the model registry
    unless --no-save is specified.

    Examples:
        # Train with default config
        python -m src.cli train

        # Train with custom config
        python -m src.cli train --config configs/my_experiment.yaml

        # Train without saving model
        python -m src.cli train --no-save

        # Save results to file
        python -m src.cli train --output results.json
    """
    logger.info(f"Starting model training with config: {config}")

    try:
        # Train the model
        results = train_model(config_path=config, save_model=not no_save)

        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")

        if not no_save and "model_id" in results:
            logger.info(f"Model saved with ID: {results['model_id']}")

        # Save results to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to: {output_path}")

        # Print summary to stdout
        click.echo("\n" + "=" * 50)
        click.echo("TRAINING SUMMARY")
        click.echo("=" * 50)
        click.echo(f"Model: {results['model_name']}")
        click.echo(f"Training samples: {results['data_info']['train_size']}")
        click.echo(f"Test samples: {results['data_info']['test_size']}")
        click.echo(f"Features: {len(results['data_info']['features'])}")
        click.echo(f"Classes: {', '.join(results['data_info']['target_classes'])}")
        click.echo("\nTest Metrics:")
        for metric, value in results["test_metrics"].items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.4f}")
            else:
                click.echo(f"  {metric}: {value}")

        if not no_save and "model_id" in results:
            click.echo(f"\nModel ID: {results['model_id']}")

        click.echo("=" * 50)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    default="configs/experiment_default.yaml",
    help="Path to experiment configuration file",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    help="Output file for tuning results (JSON format)",
    type=click.Path(),
)
@click.option("--no-save", is_flag=True, help="Don't save the best model")
def tune(config: str, output: Optional[str], no_save: bool) -> None:
    """Perform hyperparameter tuning.

    This command performs grid search hyperparameter tuning using the
    parameter grid specified in the configuration file. The best model
    will be saved to the model registry unless --no-save is specified.

    Examples:
        # Tune with default config
        python -m src.cli tune

        # Tune with custom config
        python -m src.cli tune --config configs/tuning_experiment.yaml

        # Tune without saving best model
        python -m src.cli tune --no-save

        # Save results to file
        python -m src.cli tune --output tuning_results.json
    """
    logger.info(f"Starting hyperparameter tuning with config: {config}")

    try:
        # Load config to check if tuning is configured
        experiment_config = load_config(config, ExperimentConfig)

        if experiment_config.model.tune is None:
            click.echo("Error: No tuning configuration found in config file.", err=True)
            click.echo(
                "Please add a 'tune' section to your model configuration.", err=True
            )
            sys.exit(1)

        # Perform tuning
        results = tune_model(config_path=config, save_best_model=not no_save)

        # Log results
        logger.info("Hyperparameter tuning completed successfully!")
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Best CV Score: {results['best_score']:.4f}")
        logger.info(f"Best Params: {results['best_params']}")
        logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")

        if not no_save and "model_id" in results:
            logger.info(f"Best model saved with ID: {results['model_id']}")

        # Save results to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to: {output_path}")

        # Print summary to stdout
        click.echo("\n" + "=" * 50)
        click.echo("HYPERPARAMETER TUNING SUMMARY")
        click.echo("=" * 50)
        click.echo(f"Model: {results['model_name']}")
        click.echo(f"Training samples: {results['data_info']['train_size']}")
        click.echo(f"Test samples: {results['data_info']['test_size']}")
        click.echo(f"CV Folds: {results['tuning_config']['cv']}")
        click.echo(f"Scoring: {results['tuning_config']['scoring']}")
        click.echo(f"\nBest CV Score: {results['best_score']:.4f}")
        click.echo("Best Parameters:")
        for param, value in results["best_params"].items():
            click.echo(f"  {param}: {value}")

        click.echo("\nTest Metrics (Best Model):")
        for metric, value in results["test_metrics"].items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.4f}")
            else:
                click.echo(f"  {metric}: {value}")

        if not no_save and "model_id" in results:
            click.echo(f"\nModel ID: {results['model_id']}")

        click.echo("=" * 50)

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    default="configs/serving.yaml",
    help="Path to serving configuration file",
    type=click.Path(exists=True),
)
@click.option("--host", default=None, help="Host to bind the server to")
@click.option("--port", "-p", default=None, type=int, help="Port to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--workers", default=None, type=int, help="Number of worker processes")
def serve(
    config: str,
    host: Optional[str],
    port: Optional[int],
    reload: bool,
    workers: Optional[int],
) -> None:
    """Start the API server.

    This command starts the FastAPI server for model serving.
    Configuration can be provided via config file or command-line options.
    Command-line options override config file settings.

    Examples:
        # Start with default config
        python -m src.cli serve

        # Start with custom config
        python -m src.cli serve --config configs/my_serving.yaml

        # Override host and port
        python -m src.cli serve --host 127.0.0.1 --port 8080

        # Start in development mode with auto-reload
        python -m src.cli serve --reload
    """
    logger.info(f"Starting API server with config: {config}")

    try:
        # Load serving configuration
        serving_config = load_config(config, ServingConfig)

        # Override with command-line options
        if host is not None:
            serving_config.api.host = host
        if port is not None:
            serving_config.api.port = port
        if reload:
            serving_config.api.reload = True
        if workers is not None:
            serving_config.api.workers = workers

        # Configure logging
        logger.add(
            serving_config.logging.log_file,
            rotation=serving_config.logging.rotation,
            retention=serving_config.logging.retention,
            level=serving_config.logging.level,
            format=serving_config.logging.format,
        )

        logger.info(
            f"Starting server on {serving_config.api.host}:{serving_config.api.port}"
        )

        # Import and start uvicorn
        import uvicorn

        uvicorn.run(
            "src.serving.app:app",
            host=serving_config.api.host,
            port=serving_config.api.port,
            reload=serving_config.api.reload,
            workers=serving_config.api.workers if not serving_config.api.reload else 1,
        )

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--model-dir",
    default="models",
    help="Path to model registry directory",
    type=click.Path(exists=True),
)
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def list_models(model_dir: str, output_format: str) -> None:
    """List available models in the registry.

    This command lists all models stored in the model registry
    along with their metadata and performance metrics.

    Examples:
        # List models in table format
        python -m src.cli list-models

        # List models in JSON format
        python -m src.cli list-models --format json

        # Use custom model directory
        python -m src.cli list-models --model-dir /path/to/models
    """
    try:
        from .models.registry import ModelRegistry

        registry = ModelRegistry(model_dir)
        models = registry.list_models()

        if output_format == "json":
            # JSON output
            model_info = {}
            for model_id in models:
                try:
                    info = registry.get_model_info(model_id)
                    model_info[model_id] = info
                except Exception as e:
                    logger.warning(f"Could not load info for model {model_id}: {e}")

            click.echo(json.dumps(model_info, indent=2, default=str))
            return

        if not models:
            click.echo("No models found in registry.")
            return

        else:
            # Table output
            click.echo("\n" + "=" * 80)
            click.echo("MODEL REGISTRY")
            click.echo("=" * 80)

            for model_id in models:
                try:
                    info = registry.get_model_info(model_id)

                    click.echo(f"\nModel ID: {model_id}")
                    click.echo(f"Type: {info.get('model_type', 'Unknown')}")
                    click.echo(f"Created: {info.get('created', 'Unknown')}")
                    click.echo(f"Classes: {', '.join(info.get('classes', []))}")

                    # Show metrics if available
                    metrics_path = Path(model_dir) / "metrics" / f"{model_id}.json"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            metrics_data = json.load(f)

                        test_metrics = metrics_data.get("test_metrics", {})
                        if test_metrics:
                            click.echo("Test Metrics:")
                            for metric, value in test_metrics.items():
                                if isinstance(value, float):
                                    click.echo(f"  {metric}: {value:.4f}")
                                else:
                                    click.echo(f"  {metric}: {value}")

                    click.echo("-" * 40)

                except Exception as e:
                    logger.warning(f"Could not load info for model {model_id}: {e}")
                    click.echo(f"\nModel ID: {model_id} (info unavailable)")
                    click.echo("-" * 40)

            click.echo("=" * 80)

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option(
    "--model-dir",
    default="models",
    help="Path to model registry directory",
    type=click.Path(exists=True),
)
def model_info(model_id: str, model_dir: str) -> None:
    """Show detailed information about a specific model.

    This command displays detailed information about a model including
    its configuration, metrics, and training details.

    Examples:
        # Show info for a specific model
        python -m src.cli model-info randomforestclassifier

        # Use custom model directory
        python -m src.cli model-info my_model --model-dir /path/to/models
    """
    try:
        from .models.registry import ModelRegistry

        registry = ModelRegistry(model_dir)

        # Check if model exists
        available_models = registry.list_models()
        if model_id not in available_models:
            click.echo(f"Error: Model '{model_id}' not found in registry.", err=True)
            if available_models:
                click.echo(f"Available models: {', '.join(available_models)}")
            sys.exit(1)

        # Get model info
        info = registry.get_model_info(model_id)

        # Load metrics if available
        metrics_data = None
        metrics_path = Path(model_dir) / "metrics" / f"{model_id}.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics_data = json.load(f)

        # Display information
        click.echo("\n" + "=" * 60)
        click.echo(f"MODEL INFORMATION: {model_id}")
        click.echo("=" * 60)

        click.echo(f"Model ID: {model_id}")
        click.echo(f"Type: {info.get('model_type', 'Unknown')}")
        click.echo(f"Created: {info.get('created', 'Unknown')}")
        click.echo(f"Classes: {', '.join(info.get('classes', []))}")
        click.echo(f"Model Path: {info.get('model_path', 'Unknown')}")

        if metrics_data:
            click.echo(f"\nModel Name: {metrics_data.get('model_name', 'Unknown')}")

            # Model parameters
            model_params = metrics_data.get("model_params", {})
            if model_params:
                click.echo("\nModel Parameters:")
                for param, value in model_params.items():
                    click.echo(f"  {param}: {value}")

            # Data info
            data_info = metrics_data.get("data_info", {})
            if data_info:
                click.echo("\nData Information:")
                click.echo(
                    f"  Training samples: {data_info.get('train_size', 'Unknown')}"
                )
                click.echo(f"  Test samples: {data_info.get('test_size', 'Unknown')}")
                click.echo(f"  Features: {len(data_info.get('features', []))}")
                click.echo(
                    f"  Target classes: {', '.join(data_info.get('target_classes', []))}"
                )

            # Training metrics
            train_metrics = metrics_data.get("train_metrics", {})
            if train_metrics:
                click.echo("\nTraining Metrics:")
                for metric, value in train_metrics.items():
                    if isinstance(value, float):
                        click.echo(f"  {metric}: {value:.4f}")
                    else:
                        click.echo(f"  {metric}: {value}")

            # Test metrics
            test_metrics = metrics_data.get("test_metrics", {})
            if test_metrics:
                click.echo("\nTest Metrics:")
                for metric, value in test_metrics.items():
                    if isinstance(value, float):
                        click.echo(f"  {metric}: {value:.4f}")
                    else:
                        click.echo(f"  {metric}: {value}")

            # Tuning info if available
            if "best_params" in metrics_data:
                click.echo(
                    f"\nBest CV Score: {metrics_data.get('best_score', 'Unknown')}"
                )
                click.echo("Best Parameters:")
                for param, value in metrics_data["best_params"].items():
                    click.echo(f"  {param}: {value}")

        click.echo("=" * 60)

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
