# Makefile for Enterprise ML Classifier

.PHONY: help setup setup-quick install clean lint format test train tune api list-models model-info quick-train run-experiments validate cli-help train-help tune-help serve-help docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup        - Full development environment setup"
	@echo "  setup-quick  - Quick setup (directories and env only)"
	@echo "  install      - Install dependencies"
	@echo "  validate     - Validate development setup"
	@echo ""
	@echo "Code Quality:"
	@echo "  clean        - Clean up generated files"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  test         - Run tests"
	@echo ""
	@echo "ML Workflows:"
	@echo "  train        - Train model"
	@echo "  tune         - Hyperparameter tuning"
	@echo "  quick-train  - Quick training with default config"
	@echo "  run-experiments - Run multiple experiments"
	@echo ""
	@echo "Model Management:"
	@echo "  list-models  - List available models"
	@echo "  model-info   - Show model info (requires MODEL_ID=<id>)"
	@echo ""
	@echo "API & Serving:"
	@echo "  api          - Start API server"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run   - Run with Docker Compose"
	@echo ""
	@echo "CLI Help:"
	@echo "  cli-help     - Show CLI help"
	@echo "  train-help   - Show training command help"
	@echo "  tune-help    - Show tuning command help"
	@echo "  serve-help   - Show serving command help"

# Development setup
setup: install
	@echo "Setting up development environment..."
	@chmod +x scripts/*.sh
	@./scripts/setup_dev.sh

setup-quick:
	@echo "Quick setup (dependencies only)..."
	@mkdir -p logs data/raw data/interim data/processed models/metrics
	@cp .env.example .env || true
	@echo "Quick setup complete!"

install:
	@echo "Installing dependencies..."
	@uv sync
	@uv run pre-commit install

# Cleanup
clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .pytest_cache
	@rm -rf .coverage htmlcov/

# Code quality
lint:
	@echo "Running linting checks..."
	@uv run ruff check src/ tests/
	@uv run black --check src/ tests/
	@uv run isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	@uv run black src/ tests/
	@uv run isort src/ tests/
	@uv run ruff check --fix src/ tests/

# Testing
test:
	@echo "Running tests..."
	@uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# ML workflows
train:
	@echo "Training model..."
	@uv run python -m src.cli train --config configs/experiment_default.yaml

tune:
	@echo "Running hyperparameter tuning..."
	@uv run python -m src.cli tune --config configs/experiment_default.yaml

# API serving
api:
	@echo "Starting API server..."
	@uv run python -m src.cli serve --reload

# Model management
list-models:
	@echo "Listing available models..."
	@uv run python -m src.cli list-models

model-info:
	@echo "Usage: make model-info MODEL_ID=<model_id>"
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "Error: MODEL_ID is required"; \
		echo "Example: make model-info MODEL_ID=randomforestclassifier"; \
		exit 1; \
	fi
	@uv run python -m src.cli model-info $(MODEL_ID)

# Developer workflows
quick-train:
	@echo "Running quick training..."
	@chmod +x scripts/quick_train.sh
	@./scripts/quick_train.sh

run-experiments:
	@echo "Running multiple experiments..."
	@chmod +x scripts/run_experiments.sh
	@./scripts/run_experiments.sh

validate:
	@echo "Validating development setup..."
	@chmod +x scripts/validate_setup.sh
	@./scripts/validate_setup.sh

# CLI shortcuts
cli-help:
	@echo "CLI help and available commands:"
	@uv run python -m src.cli --help

train-help:
	@echo "Training command help:"
	@uv run python -m src.cli train --help

tune-help:
	@echo "Tuning command help:"
	@uv run python -m src.cli tune --help

serve-help:
	@echo "Serving command help:"
	@uv run python -m src.cli serve --help

# Docker commands (to be implemented in later tasks)
docker-build:
	@echo "Building Docker images..."
	@docker build -f docker/Dockerfile.train -t ml-classifier-train .
	@docker build -f docker/Dockerfile.app -t ml-classifier-app .

docker-run:
	@echo "Running with Docker Compose..."
	@docker-compose up