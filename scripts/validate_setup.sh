#!/bin/bash
# Validate that the development setup is working correctly

set -e

echo "Validating development setup..."

# Check Python version
echo "Checking Python version..."
python_version=$(uv run python --version)
echo "Python version: $python_version"

# Check if all required packages are installed
echo "Checking package installation..."
uv run python -c "
import sys
required_packages = [
    'pandas', 'numpy', 'sklearn', 'fastapi', 'uvicorn',
    'pydantic', 'loguru', 'click', 'pytest', 'black', 'isort', 'ruff'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package}')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required packages are installed!')
"

# Check project structure
echo "Checking project structure..."
required_dirs=(
    "src/core"
    "src/data"
    "src/features"
    "src/models"
    "src/serving"
    "tests"
    "configs"
    "data/raw"
    "data/interim"
    "data/processed"
    "models"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
    else
        echo "✗ $dir (missing)"
    fi
done

# Check configuration files
echo "Checking configuration files..."
config_files=(
    "configs/experiment_default.yaml"
    "configs/serving.yaml"
    "pyproject.toml"
    "Makefile"
)

for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
    fi
done

# Test CLI commands
echo "Testing CLI commands..."
uv run python -m src.cli --help > /dev/null && echo "✓ CLI help works"

# Test imports
echo "Testing core imports..."
uv run python -c "
from src.core.config import ExperimentConfig, ServingConfig
from src.data.dataset import PenguinDataLoader
from src.models.trainer import ModelTrainer
from src.serving.app import app
print('✓ All core imports successful')
"

# Run basic tests if they exist
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    echo "Running basic tests..."
    uv run pytest tests/ -v --tb=short -x
    echo "✓ Basic tests passed"
else
    echo "⚠ No tests found to run"
fi

echo ""
echo "Setup validation completed!"
echo ""
echo "Next steps:"
echo "1. Add training data to data/raw/penguins_lter.csv"
echo "2. Run 'make train' to train your first model"
echo "3. Run 'make api' to start the API server"
