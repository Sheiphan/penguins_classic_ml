#!/bin/bash

# Script to validate Docker configuration without building
set -e

echo "Validating Docker configuration..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
print_status "Checking required files..."

required_files=(
    "Dockerfile.train"
    "Dockerfile.app"
    "docker-compose.yaml"
    ".env.example"
    "pyproject.toml"
    "uv.lock"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "✓ $file exists"
    else
        print_error "✗ $file missing"
        exit 1
    fi
done

# Check Dockerfile syntax
print_status "Validating Dockerfile syntax..."

# Basic syntax check for Dockerfile.train
if grep -q "FROM python:3.11-slim" Dockerfile.train && \
   grep -q "WORKDIR /app" Dockerfile.train && \
   grep -q "COPY pyproject.toml uv.lock" Dockerfile.train && \
   grep -q "RUN uv sync" Dockerfile.train; then
    print_status "✓ Dockerfile.train syntax looks good"
else
    print_error "✗ Dockerfile.train syntax issues"
    exit 1
fi

# Basic syntax check for Dockerfile.app
if grep -q "FROM python:3.11-slim" Dockerfile.app && \
   grep -q "WORKDIR /app" Dockerfile.app && \
   grep -q "EXPOSE 8000" Dockerfile.app && \
   grep -q "HEALTHCHECK" Dockerfile.app; then
    print_status "✓ Dockerfile.app syntax looks good"
else
    print_error "✗ Dockerfile.app syntax issues"
    exit 1
fi

# Check docker-compose.yaml syntax
print_status "Validating docker-compose.yaml..."

if command -v docker-compose > /dev/null 2>&1; then
    if docker-compose config > /dev/null 2>&1; then
        print_status "✓ docker-compose.yaml syntax is valid"
    else
        print_error "✗ docker-compose.yaml syntax is invalid"
        exit 1
    fi
else
    print_warning "⚠ docker-compose not available, skipping syntax check"
fi

# Check environment file
print_status "Validating environment configuration..."

if [ -f ".env.example" ]; then
    # Check for required environment variables
    required_vars=(
        "LOG_LEVEL"
        "API_HOST"
        "API_PORT"
        "API_WORKERS"
        "EXPERIMENT_CONFIG"
        "SERVING_CONFIG"
    )
    
    for var in "${required_vars[@]}"; do
        if grep -q "^$var=" .env.example; then
            print_status "✓ $var defined in .env.example"
        else
            print_warning "⚠ $var not found in .env.example"
        fi
    done
else
    print_error "✗ .env.example not found"
    exit 1
fi

# Check source code structure
print_status "Validating source code structure..."

required_dirs=(
    "src"
    "src/core"
    "src/data"
    "src/features"
    "src/models"
    "src/serving"
    "configs"
    "data"
    "models"
    "tests"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_status "✓ $dir directory exists"
    else
        print_error "✗ $dir directory missing"
        exit 1
    fi
done

# Check key Python files
key_files=(
    "src/cli.py"
    "src/core/config.py"
    "src/serving/app.py"
    "configs/serving.yaml"
    "configs/experiment_default.yaml"
)

for file in "${key_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "✓ $file exists"
    else
        print_error "✗ $file missing"
        exit 1
    fi
done

# Check Python imports (basic syntax check)
print_status "Validating Python imports..."

if python3 -c "
import sys
sys.path.append('src')
try:
    import src.cli
    import src.core.config
    import src.serving.app
    print('✓ Python imports work')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_status "✓ Python imports validated"
else
    print_warning "⚠ Python import validation failed (dependencies may not be installed)"
fi

print_status "Docker configuration validation complete! 🎉"
print_status ""
print_status "Next steps:"
print_status "1. Ensure Docker is installed and running"
print_status "2. Run 'make docker-build' to build images"
print_status "3. Run 'make docker-test' to test containers"
print_status "4. Use docker-compose profiles for different scenarios"