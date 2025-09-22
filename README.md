# Enterprise ML Classifier

A production-ready machine learning system for penguin species classification, built with MLOps best practices and enterprise-grade architecture.

## 🐧 About

This project implements a complete MLOps pipeline for classifying penguin species using the Palmer Penguins dataset. The system provides:

- **Configuration-driven experiments** with YAML-based parameter management
- **Robust data processing** with sklearn pipelines and feature engineering
- **Automated model training** with hyperparameter tuning and evaluation
- **Production-ready API** with FastAPI and Pydantic validation
- **Containerized deployment** with Docker and docker-compose
- **Comprehensive testing** with pytest and code quality tools
- **CI/CD pipeline** with GitHub Actions

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv (Python package manager)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd enterprise-ml-classifier
```

2. **Set up the development environment**
```bash
make setup
```

3. **Train your first model**
```bash
make train
```

4. **Start the API server**
```bash
make api
```

5. **Make a prediction**
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

## 📊 Dataset

The system uses the Palmer Penguins dataset, which includes measurements for three penguin species:

- **Adelie** (Pygoscelis adeliae)
- **Chinstrap** (Pygoscelis antarcticus)
- **Gentoo** (Pygoscelis papua)

### Features
- `island`: Penguin habitat island (Biscoe, Dream, Torgersen)
- `bill_length_mm`: Bill length in millimeters
- `bill_depth_mm`: Bill depth in millimeters
- `flipper_length_mm`: Flipper length in millimeters
- `body_mass_g`: Body mass in grams
- `sex`: Penguin sex (MALE, FEMALE)
- `year`: Study year (2007-2009)

## 🛠️ Usage

### Command Line Interface

The project provides a comprehensive CLI for all ML operations:

```bash
# Train a model with default configuration
python -m src.cli train

# Train with custom configuration
python -m src.cli train --config configs/custom_experiment.yaml

# Hyperparameter tuning
python -m src.cli tune

# Start API server
python -m src.cli serve

# Get help
python -m src.cli --help
```

### Makefile Commands

For convenience, common operations are available through make commands:

```bash
make setup          # Set up development environment
make train          # Train model with default config
make tune           # Run hyperparameter tuning
make api            # Start API server
make test           # Run test suite
make lint           # Run code quality checks
make format         # Format code with black and isort
make clean          # Clean up generated files
make docker-build   # Build Docker images
make docker-run     # Run with docker-compose
```

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
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

#### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "island": "Biscoe",
        "bill_length_mm": 48.6,
        "bill_depth_mm": 16.0,
        "flipper_length_mm": 230.0,
        "body_mass_g": 5800.0,
        "sex": "MALE",
        "year": 2009
      },
      {
        "island": "Dream",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "MALE",
        "year": 2007
      }
    ]
  }'
```

#### Model Information
```bash
curl http://localhost:8000/model/info
```

## 🐳 Docker Deployment

### Local Development with Docker Compose

```bash
# Build and start services
make docker-run

# Or manually
docker-compose up --build
```

### Production Deployment

```bash
# Build training image
docker build -f Dockerfile.train -t ml-classifier:train .

# Build serving image
docker build -f Dockerfile.app -t ml-classifier:serve .

# Run training
docker run --rm -v $(pwd)/models:/app/models ml-classifier:train

# Run API server
docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-classifier:serve
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key environment variables:
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `EXPERIMENT_CONFIG`: Path to experiment configuration
- `MODEL_DIR`: Directory for model artifacts

### Experiment Configuration

Customize training parameters in `configs/experiment_default.yaml`:

```yaml
seed: 42
paths:
  raw: "data/raw/penguins_lter.csv"
  processed_dir: "data/processed"
  model_dir: "models"
  metrics_dir: "models/metrics"

features:
  numeric:
    - "bill_length_mm"
    - "bill_depth_mm"
    - "flipper_length_mm"
    - "body_mass_g"
    - "year"
  categorical:
    - "island"
    - "sex"
  target: "species"

model:
  name: "RandomForestClassifier"
  params:
    n_estimators: 100
    random_state: 42
    max_depth: 10
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_trainer.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows
- **API Tests**: Test FastAPI endpoints
- **Code Quality Tests**: Linting and formatting checks

## 📁 Project Structure

```
enterprise-ml-classifier/
├── .github/workflows/      # CI/CD pipeline
├── configs/               # Configuration files
│   ├── experiment_default.yaml
│   └── serving.yaml
├── data/                  # Data pipeline
│   ├── raw/              # Raw data files
│   ├── interim/          # Intermediate processed data
│   └── processed/        # Final processed data
├── docs/                  # Documentation
├── models/               # Model artifacts and metrics
│   ├── artifacts/        # Serialized models
│   ├── metrics/          # Evaluation metrics
│   └── registry.json     # Model registry
├── src/                  # Source code
│   ├── cli.py           # Command-line interface
│   ├── core/            # Core utilities
│   │   ├── config.py    # Configuration management
│   │   └── logging.py   # Logging setup
│   ├── data/            # Data processing
│   │   ├── dataset.py   # Data loading utilities
│   │   └── schema.py    # Data schema definitions
│   ├── features/        # Feature engineering
│   │   └── preprocess.py # Preprocessing pipelines
│   ├── models/          # Model training and management
│   │   ├── trainer.py   # Training logic
│   │   ├── metrics.py   # Evaluation metrics
│   │   └── registry.py  # Model registry
│   └── serving/         # API serving
│       ├── app.py       # FastAPI application
│       └── schemas.py   # Pydantic models
├── tests/               # Test suite
├── scripts/             # Developer scripts
├── docker-compose.yaml  # Local development stack
├── Dockerfile.app       # Production API container
├── Dockerfile.train     # Training container
├── Makefile            # Build commands
└── pyproject.toml      # Project configuration
```

## 🔧 Development

### Setup Development Environment

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project
make setup

# Install pre-commit hooks
pre-commit install
```

### Code Quality

The project enforces code quality through:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks

```bash
# Format code
make format

# Run linting
make lint

# Run all quality checks
make quality
```

### Adding New Models

1. Add model class to `AVAILABLE_MODELS` in `src/models/trainer.py`
2. Update configuration schema if needed
3. Add model-specific tests
4. Update documentation

### Adding New Features

1. Update feature lists in `src/data/schema.py`
2. Modify preprocessing pipeline in `src/features/preprocess.py`
3. Update API schemas in `src/serving/schemas.py`
4. Add validation and tests

## 📈 Model Performance

The system tracks comprehensive metrics for model evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged precision
- **Recall**: Per-class and macro-averaged recall
- **F1-Score**: Per-class and macro-averaged F1-score
- **Confusion Matrix**: Detailed classification results

Metrics are automatically saved to `models/metrics/` and can be accessed via the model registry.

## 🚀 Production Deployment

### Kubernetes Deployment

Example Kubernetes manifests:

```yaml
# deployment.yaml
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Monitoring and Observability

- **Health Checks**: `/health` endpoint for liveness/readiness probes
- **Structured Logging**: JSON-formatted logs with Loguru
- **Metrics**: Request/response logging and performance metrics
- **Error Tracking**: Comprehensive error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make test lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Palmer Penguins dataset by Dr. Kristen Gorman and Palmer Station LTER
- FastAPI framework for the API layer
- scikit-learn for machine learning capabilities
- The Python community for excellent tooling and libraries

## 📚 Additional Documentation

- [CLI Usage Guide](docs/CLI_USAGE.md)
- [Docker Deployment Guide](docs/DOCKER.md)
- [API Reference](docs/README.md)
- [Development Setup](docs/README.md)

---

**Happy Penguin Classifying! 🐧**
