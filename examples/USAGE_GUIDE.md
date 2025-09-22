# Enterprise ML Classifier Usage Guide

This comprehensive guide covers all aspects of using the Enterprise ML Classifier, from basic setup to advanced customization.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Models](#training-models)
3. [API Usage](#api-usage)
4. [Configuration](#configuration)
5. [Docker Deployment](#docker-deployment)
6. [Development Workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd enterprise-ml-classifier

# Set up development environment
make setup

# Verify installation
make validate
```

### 2. Train Your First Model

```bash
# Train with default configuration
make train

# Check training results
ls models/artifacts/
ls models/metrics/
```

### 3. Start the API Server

```bash
# Start API server
make api

# Test the API (in another terminal)
curl http://localhost:8000/health
```

### 4. Make Your First Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Biscoe",
    "bill_length_mm": 48.6,
    "bill_depth_mm": 16.0,
    "flipper_length_mm": 230.0,
    "body_mass_g": 5800.0,
    "sex": "MALE",
    "year": 2009
  }'
```

## Training Models

### Basic Training

```bash
# Train with default RandomForest
python -m src.cli train

# Train with verbose output
python -m src.cli train --verbose

# Train with custom config
python -m src.cli train --config examples/custom_experiment.yaml
```

### Hyperparameter Tuning

```bash
# Run hyperparameter tuning
python -m src.cli tune

# Tune with custom configuration
python -m src.cli tune --config configs/experiment_default.yaml
```

### Model Comparison

```bash
# Train multiple models for comparison
python -m src.cli train --config examples/custom_experiment.yaml
python -m src.cli train --config configs/logistic_regression.yaml
python -m src.cli train --config configs/svm.yaml

# Compare results
python -c "
import json
from pathlib import Path

metrics_dir = Path('models/metrics')
for metrics_file in metrics_dir.glob('*_metrics.json'):
    with open(metrics_file) as f:
        metrics = json.load(f)
    print(f'{metrics_file.stem}: Accuracy={metrics[\"accuracy\"]:.3f}, F1={metrics[\"f1_score\"]:.3f}')
"
```

### Custom Model Configuration

Create a custom configuration file:

```yaml
# configs/my_experiment.yaml
seed: 42

model:
  name: "RandomForestClassifier"
  params:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 5
    random_state: 42
  
  tune:
    grid:
      - n_estimators: [100, 200, 300]
        max_depth: [10, 15, 20]
    cv: 5
    scoring: "f1_macro"

features:
  test_size: 0.25
  stratify: true
```

Then train with it:

```bash
python -m src.cli train --config configs/my_experiment.yaml
```

## API Usage

### Starting the API Server

```bash
# Development server with auto-reload
python -m src.cli serve --reload

# Production server
python -m src.cli serve --host 0.0.0.0 --port 8000 --workers 4

# Using make command
make api
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model Information
```bash
curl http://localhost:8000/model/info
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "MALE",
    "year": 2007
  }'
```

#### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "MALE",
        "year": 2007
      },
      {
        "island": "Dream",
        "bill_length_mm": 46.5,
        "bill_depth_mm": 17.9,
        "flipper_length_mm": 192.0,
        "body_mass_g": 3500.0,
        "sex": "FEMALE",
        "year": 2008
      }
    ]
  }'
```

### Python Client Usage

```python
import requests

# Create a client session
session = requests.Session()
base_url = "http://localhost:8000"

# Health check
health = session.get(f"{base_url}/health").json()
print(f"API Status: {health['status']}")

# Make prediction
penguin_data = {
    "island": "Biscoe",
    "bill_length_mm": 48.6,
    "bill_depth_mm": 16.0,
    "flipper_length_mm": 230.0,
    "body_mass_g": 5800.0,
    "sex": "MALE",
    "year": 2009
}

response = session.post(f"{base_url}/predict", json=penguin_data)
prediction = response.json()
print(f"Predicted species: {prediction['prediction']}")
```

### API Testing

```bash
# Run comprehensive API tests
python examples/test_api.py

# Run shell-based tests
./examples/test_api.sh

# Test with custom host/port
python examples/test_api.py --host localhost --port 8000

# Include performance tests
python examples/test_api.py --performance --num-requests 50
```

## Configuration

### Environment Variables

Copy and customize the environment file:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

Key environment variables:

- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `MODEL_DIR`: Directory for model artifacts
- `RANDOM_SEED`: Seed for reproducibility

### Experiment Configuration

Experiment configurations are stored in YAML files in the `configs/` directory:

```yaml
# configs/experiment_default.yaml
seed: 42

paths:
  raw: "data/raw/penguins_lter.csv"
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

model:
  name: "RandomForestClassifier"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

### Serving Configuration

API server configuration:

```yaml
# configs/serving.yaml
api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1

logging:
  level: "INFO"
  log_file: "logs/app.log"
  rotation: "1 day"
  retention: "30 days"
```

## Docker Deployment

### Local Development

```bash
# Build and run with docker-compose
make docker-run

# Or manually
docker-compose up --build
```

### Production Deployment

```bash
# Build images
docker build -f Dockerfile.train -t ml-classifier:train .
docker build -f Dockerfile.app -t ml-classifier:serve .

# Train model
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ml-classifier:train

# Run API server
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name ml-classifier-api \
  ml-classifier:serve
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-classifier-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-classifier-api
  template:
    metadata:
      labels:
        app: ml-classifier-api
    spec:
      containers:
      - name: api
        image: ml-classifier:serve
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: API_WORKERS
          value: "1"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Development Workflow

### Setting Up Development Environment

```bash
# Install development dependencies
make setup

# Install pre-commit hooks
pre-commit install

# Verify setup
make validate
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all quality checks
make quality

# Run tests
make test

# Run tests with coverage
make test-coverage
```

### Development Commands

```bash
# Clean generated files
make clean

# Reset environment
make clean-all

# Update dependencies
uv sync --upgrade

# Run security checks
make security
```

### Adding New Features

1. **Add new model types**:
   ```python
   # In src/models/trainer.py
   AVAILABLE_MODELS = {
       "RandomForestClassifier": RandomForestClassifier,
       "LogisticRegression": LogisticRegression,
       "YourNewModel": YourNewModel,  # Add here
   }
   ```

2. **Add new features**:
   ```python
   # In src/data/schema.py
   NUMERIC_FEATURES = [
       "bill_length_mm",
       "your_new_feature",  # Add here
   ]
   ```

3. **Add new API endpoints**:
   ```python
   # In src/serving/app.py
   @app.post("/your-endpoint")
   async def your_endpoint():
       # Implementation
   ```

## Troubleshooting

### Common Issues

#### 1. API Server Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Try different port
python -m src.cli serve --port 8001

# Check logs
tail -f logs/app.log
```

#### 2. Model Training Fails

```bash
# Check data file exists
ls -la data/raw/penguins_lter.csv

# Verify configuration
python -c "from src.core.config import load_config; print(load_config('configs/experiment_default.yaml'))"

# Run with debug logging
python -m src.cli train --verbose
```

#### 3. Predictions Return Errors

```bash
# Check model is loaded
curl http://localhost:8000/model/info

# Validate input data
python -c "
from src.data.schema import validate_feature_values
import pandas as pd
df = pd.DataFrame([{
    'island': 'Biscoe',
    'bill_length_mm': 48.6,
    # ... other features
}])
print(validate_feature_values(df))
"
```

#### 4. Docker Build Issues

```bash
# Clean Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -f Dockerfile.app -t ml-classifier:serve .

# Check Docker logs
docker logs <container-id>
```

### Performance Issues

#### 1. Slow Training

```bash
# Use fewer CV folds
# In config file: cv: 3

# Reduce parameter grid size
# In config file: smaller grid ranges

# Use parallel processing
# In config file: n_jobs: -1
```

#### 2. Slow API Responses

```bash
# Increase workers
python -m src.cli serve --workers 4

# Check model loading time
python -c "
import time
from src.models.registry import ModelRegistry
start = time.time()
registry = ModelRegistry('models')
model = registry.load_latest_model()
print(f'Model loading time: {time.time() - start:.2f}s')
"
```

### Debugging

#### Enable Debug Logging

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env file
echo "LOG_LEVEL=DEBUG" >> .env

# Run with debug
python -m src.cli train --verbose
```

#### Check Model Registry

```python
import json
from pathlib import Path

# Check registry contents
registry_file = Path("models/registry.json")
if registry_file.exists():
    with open(registry_file) as f:
        registry = json.load(f)
    print(json.dumps(registry, indent=2))
else:
    print("No model registry found")
```

## Advanced Usage

### Custom Preprocessing

```python
# Create custom preprocessor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_custom_preprocessor():
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

### Model Ensemble

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Create ensemble model
ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('lr', LogisticRegression()),
    ('svc', SVC(probability=True))
], voting='soft')
```

### Custom Metrics

```python
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score

# Use custom scoring function
custom_scorer = make_scorer(balanced_accuracy_score)

# In config file:
# scoring: "balanced_accuracy"
```

### Automated Retraining

```bash
# Create cron job for automated retraining
# Add to crontab (crontab -e):
# 0 2 * * 0 cd /path/to/project && make train
```

### Model Monitoring

```python
import json
from datetime import datetime
from pathlib import Path

def log_prediction(input_data, prediction, confidence):
    """Log predictions for monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "confidence": confidence
    }
    
    log_file = Path("logs/predictions.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

### Integration with MLflow

```python
import mlflow
import mlflow.sklearn

# Log experiment
with mlflow.start_run():
    mlflow.log_params(model_params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
```

This guide covers the most common use cases and advanced scenarios. For additional help, check the API documentation at `/docs` when the server is running, or refer to the source code documentation.