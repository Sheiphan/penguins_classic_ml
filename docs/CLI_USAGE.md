# CLI Usage Guide

The Enterprise ML Classifier provides a comprehensive command-line interface (CLI) for training, tuning, and managing machine learning models. This guide covers all available commands and their usage.

## Quick Start

```bash
# Install dependencies and set up environment
make setup

# Train a model with default configuration
make train

# List available models
make list-models

# Start the API server
make api
```

## CLI Commands

### Global Options

All CLI commands support these global options:

- `-v, --verbose`: Enable verbose logging (DEBUG level)
- `-q, --quiet`: Enable quiet mode (ERROR level only)
- `--help`: Show help message

### Training Commands

#### `train` - Train a Machine Learning Model

Train a model using the specified configuration file.

```bash
# Basic usage
python -m src.cli train

# With custom configuration
python -m src.cli train --config configs/my_experiment.yaml

# Save results to file
python -m src.cli train --output results/training_results.json

# Train without saving model (for testing)
python -m src.cli train --no-save

# Makefile shortcuts
make train                    # Train with default config
make quick-train             # Quick training script
```

**Options:**
- `-c, --config PATH`: Path to experiment configuration file (default: `configs/experiment_default.yaml`)
- `-o, --output PATH`: Output file for training results (JSON format)
- `--no-save`: Don't save the trained model to registry

**Example Output:**
```
==================================================
TRAINING SUMMARY
==================================================
Model: RandomForestClassifier
Training samples: 274
Test samples: 69
Features: 7
Classes: Adelie, Chinstrap, Gentoo

Test Metrics:
  accuracy: 0.9710
  f1_score: 0.9710
  precision: 0.9722
  recall: 0.9710

Model ID: randomforestclassifier
==================================================
```

#### `tune` - Hyperparameter Tuning

Perform grid search hyperparameter tuning using the parameter grid specified in the configuration file.

```bash
# Basic usage
python -m src.cli tune

# With custom configuration
python -m src.cli tune --config configs/tuning_experiment.yaml

# Save results to file
python -m src.cli tune --output results/tuning_results.json

# Tune without saving best model
python -m src.cli tune --no-save

# Makefile shortcut
make tune
```

**Options:**
- `-c, --config PATH`: Path to experiment configuration file
- `-o, --output PATH`: Output file for tuning results (JSON format)
- `--no-save`: Don't save the best model to registry

**Configuration Requirements:**
The configuration file must include a `tune` section under `model`:

```yaml
model:
  name: "RandomForestClassifier"
  params:
    n_estimators: 100
  tune:
    grid:
      - n_estimators: [50, 100, 200]
        max_depth: [3, 5, 10]
    cv: 5
    scoring: "accuracy"
    n_jobs: -1
```

### Model Management Commands

#### `list-models` - List Available Models

List all models stored in the model registry.

```bash
# Table format (default)
python -m src.cli list-models

# JSON format
python -m src.cli list-models --format json

# Custom model directory
python -m src.cli list-models --model-dir /path/to/models

# Makefile shortcut
make list-models
```

**Options:**
- `--model-dir PATH`: Path to model registry directory (default: `models`)
- `--format [table|json]`: Output format (default: `table`)

#### `model-info` - Show Model Information

Display detailed information about a specific model.

```bash
# Show model information
python -m src.cli model-info randomforestclassifier

# Custom model directory
python -m src.cli model-info my_model --model-dir /path/to/models

# Makefile shortcut
make model-info MODEL_ID=randomforestclassifier
```

**Arguments:**
- `MODEL_ID`: The identifier of the model to show information for

**Options:**
- `--model-dir PATH`: Path to model registry directory (default: `models`)

### Serving Commands

#### `serve` - Start API Server

Start the FastAPI server for model serving.

```bash
# Basic usage
python -m src.cli serve

# With custom configuration
python -m src.cli serve --config configs/my_serving.yaml

# Override host and port
python -m src.cli serve --host 127.0.0.1 --port 8080

# Development mode with auto-reload
python -m src.cli serve --reload

# Custom number of workers
python -m src.cli serve --workers 4

# Makefile shortcut
make api
```

**Options:**
- `-c, --config PATH`: Path to serving configuration file (default: `configs/serving.yaml`)
- `--host TEXT`: Host to bind the server to (overrides config)
- `-p, --port INTEGER`: Port to bind the server to (overrides config)
- `--reload`: Enable auto-reload for development (overrides config)
- `--workers INTEGER`: Number of worker processes (overrides config)

## Makefile Commands

The project includes a comprehensive Makefile with shortcuts for common operations:

### Setup & Installation
```bash
make setup          # Full development environment setup
make setup-quick     # Quick setup (directories and env only)
make install         # Install dependencies
make validate        # Validate development setup
```

### Code Quality
```bash
make clean          # Clean up generated files
make lint           # Run linting checks
make format         # Format code
make test           # Run tests
```

### ML Workflows
```bash
make train          # Train model
make tune           # Hyperparameter tuning
make quick-train    # Quick training with default config
make run-experiments # Run multiple experiments
```

### Model Management
```bash
make list-models    # List available models
make model-info MODEL_ID=<id>  # Show model info
```

### API & Serving
```bash
make api            # Start API server
```

### CLI Help
```bash
make cli-help       # Show CLI help
make train-help     # Show training command help
make tune-help      # Show tuning command help
make serve-help     # Show serving command help
```

## Developer Scripts

The `scripts/` directory contains useful developer workflow scripts:

### `scripts/setup_dev.sh`
Complete development environment setup script.

```bash
./scripts/setup_dev.sh
# or
make setup
```

### `scripts/quick_train.sh`
Quick training script for development and testing.

```bash
./scripts/quick_train.sh
# or
make quick-train
```

### `scripts/run_experiments.sh`
Run multiple experiments with different configurations.

```bash
./scripts/run_experiments.sh
# or
make run-experiments
```

### `scripts/validate_setup.sh`
Validate that the development setup is working correctly.

```bash
./scripts/validate_setup.sh
# or
make validate
```

## Configuration Files

### Experiment Configuration (`configs/experiment_default.yaml`)

```yaml
seed: 42

paths:
  raw: "data/raw"
  processed_dir: "data/processed"
  model_dir: "models"
  metrics_dir: "models/metrics"

features:
  numeric_features:
    - "bill_length_mm"
    - "bill_depth_mm"
    - "flipper_length_mm"
    - "body_mass_g"
    - "year"
  categorical_features:
    - "island"
    - "sex"
  target: "species"
  test_size: 0.2
  stratify: true

model:
  name: "RandomForestClassifier"
  params:
    n_estimators: 100
    random_state: 42
  tune:
    grid:
      - n_estimators: [50, 100, 200]
        max_depth: [3, 5, 10, null]
        min_samples_split: [2, 5, 10]
    cv: 5
    scoring: "f1_macro"
    n_jobs: -1
```

### Serving Configuration (`configs/serving.yaml`)

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1

logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
  rotation: "1 day"
  retention: "30 days"
  log_file: "logs/app.log"

model_path: "models/latest"
```

## Output Files

### Training Results

When using the `--output` option with training commands, results are saved in JSON format:

```json
{
  "model_name": "RandomForestClassifier",
  "model_params": {
    "n_estimators": 100,
    "random_state": 42
  },
  "data_info": {
    "train_size": 274,
    "test_size": 69,
    "features": ["bill_length_mm", "bill_depth_mm", ...],
    "target_classes": ["Adelie", "Chinstrap", "Gentoo"]
  },
  "train_metrics": {
    "accuracy": 1.0,
    "f1_score": 1.0,
    "precision": 1.0,
    "recall": 1.0
  },
  "test_metrics": {
    "accuracy": 0.9710,
    "f1_score": 0.9710,
    "precision": 0.9722,
    "recall": 0.9710
  },
  "model_id": "randomforestclassifier"
}
```

### Tuning Results

Hyperparameter tuning results include additional information:

```json
{
  "model_name": "RandomForestClassifier",
  "tuning_config": {
    "cv": 5,
    "scoring": "f1_macro",
    "n_jobs": -1
  },
  "best_params": {
    "classifier__max_depth": 10,
    "classifier__n_estimators": 200
  },
  "best_score": 0.9650,
  "cv_results": {
    "mean_test_scores": [0.9500, 0.9650, 0.9600],
    "std_test_scores": [0.0200, 0.0150, 0.0180],
    "params": [...]
  },
  "test_metrics": {
    "accuracy": 0.9710,
    "f1_score": 0.9710
  },
  "model_id": "randomforestclassifier_tuned"
}
```

## Error Handling

The CLI provides comprehensive error handling and informative error messages:

### Common Errors

1. **Missing Configuration File**
   ```
   Error: Invalid value for '--config': Path 'nonexistent.yaml' does not exist.
   ```

2. **No Tuning Configuration**
   ```
   Error: No tuning configuration found in config file.
   Please add a 'tune' section to your model configuration.
   ```

3. **Model Not Found**
   ```
   Error: Model 'nonexistent_model' not found in registry.
   Available models: randomforestclassifier, logisticregression
   ```

4. **Missing Training Data**
   ```
   Warning: Training data not found at data/raw/penguins_lter.csv
   Please ensure the dataset is available before training.
   ```

## Logging

The CLI uses structured logging with different levels:

- **DEBUG** (`--verbose`): Detailed information for debugging
- **INFO** (default): General information about operations
- **ERROR** (`--quiet`): Only error messages

Logs are written to both console and file (if `logs/` directory exists).

## Integration with Other Tools

### Pre-commit Hooks

The CLI integrates with pre-commit hooks for code quality:

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

### Testing

Run CLI integration tests:

```bash
# All CLI tests
uv run pytest tests/test_cli.py -v

# Specific test
uv run pytest tests/test_cli.py::TestCLI::test_train_help -v
```

### Docker Integration

The CLI commands work within Docker containers:

```bash
# Build training container
make docker-build

# Run training in container
docker run ml-classifier-train python -m src.cli train
```

## Best Practices

1. **Use Configuration Files**: Store experiment parameters in YAML files for reproducibility
2. **Save Results**: Use `--output` to save training/tuning results for analysis
3. **Version Control**: Keep configuration files in version control
4. **Model Registry**: Use meaningful model IDs and leverage the model registry
5. **Logging**: Use appropriate log levels for different environments
6. **Testing**: Validate setup with `make validate` before running experiments
7. **Automation**: Use Makefile commands and scripts for common workflows

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `make validate` to check package installation
2. **Permission Errors**: Ensure scripts are executable: `chmod +x scripts/*.sh`
3. **Missing Data**: Check that training data exists in `data/raw/`
4. **Port Conflicts**: Use `--port` option to specify different port for API server
5. **Memory Issues**: Reduce model complexity or use smaller datasets for testing

### Getting Help

```bash
# General CLI help
python -m src.cli --help

# Command-specific help
python -m src.cli train --help
python -m src.cli tune --help
python -m src.cli serve --help

# Makefile help
make help
```