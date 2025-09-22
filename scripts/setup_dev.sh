#!/bin/bash
# Development environment setup script

set -e

echo "Setting up development environment for Enterprise ML Classifier..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment and install dependencies
echo "Installing dependencies with uv..."
uv sync

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs
mkdir -p data/raw data/interim data/processed
mkdir -p models/artifacts models/metrics
mkdir -p notebooks

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env file from .env.example..."
        cp .env.example .env
    else
        echo "Creating default .env file..."
        cat > .env << EOF
# Environment configuration
PYTHONPATH=.
LOG_LEVEL=INFO
MODEL_REGISTRY_DIR=models
EOF
    fi
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

# Run initial code quality checks
echo "Running initial code quality checks..."
uv run ruff check src/ tests/ --fix || true
uv run black src/ tests/
uv run isort src/ tests/

echo "Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  make help     - Show all available commands"
echo "  make test     - Run tests"
echo "  make train    - Train a model"
echo "  make api      - Start API server"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
