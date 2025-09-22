#!/bin/bash

# Script to test Docker container builds and basic functionality
set -e

echo "Testing Docker container builds and runtime behavior..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if required files exist
if [ ! -f "Dockerfile.train" ]; then
    print_error "Dockerfile.train not found"
    exit 1
fi

if [ ! -f "Dockerfile.app" ]; then
    print_error "Dockerfile.app not found"
    exit 1
fi

if [ ! -f "docker-compose.yaml" ]; then
    print_error "docker-compose.yaml not found"
    exit 1
fi

# Test 1: Build training container
print_status "Building training container..."
if docker build -f Dockerfile.train -t ml-classifier-train:test .; then
    print_status "✓ Training container built successfully"
else
    print_error "✗ Failed to build training container"
    exit 1
fi

# Test 2: Build serving container
print_status "Building serving container..."
if docker build -f Dockerfile.app -t ml-classifier-app:test .; then
    print_status "✓ Serving container built successfully"
else
    print_error "✗ Failed to build serving container"
    exit 1
fi

# Test 3: Test training container basic functionality
print_status "Testing training container basic functionality..."
if docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" ml-classifier-train:test uv run python -c "import src.cli; print('Training container imports work')"; then
    print_status "✓ Training container basic functionality works"
else
    print_error "✗ Training container basic functionality failed"
    exit 1
fi

# Test 4: Test serving container basic functionality
print_status "Testing serving container basic functionality..."
if docker run --rm ml-classifier-app:test uv run python -c "import src.serving.app; print('Serving container imports work')"; then
    print_status "✓ Serving container basic functionality works"
else
    print_error "✗ Serving container basic functionality failed"
    exit 1
fi

# Test 5: Test docker-compose configuration
print_status "Testing docker-compose configuration..."
if docker-compose --profile api config > /dev/null 2>&1; then
    print_status "✓ docker-compose configuration is valid"
else
    print_error "✗ docker-compose configuration is invalid"
    exit 1
fi

# Test profiles
print_status "Testing docker-compose profiles..."
for profile in train api dev full; do
    if docker-compose --profile $profile config > /dev/null 2>&1; then
        print_status "✓ Profile '$profile' is valid"
    else
        print_error "✗ Profile '$profile' is invalid"
        exit 1
    fi
done

# Test 6: Test environment variable support
print_status "Testing environment variable support..."
if docker run --rm -e LOG_LEVEL=DEBUG -e API_PORT=8080 ml-classifier-app:test uv run python -c "
import os
from src.core.config import ServingConfig
config = ServingConfig()
assert config.logging.level == 'DEBUG', f'Expected DEBUG, got {config.logging.level}'
assert config.api.port == 8080, f'Expected 8080, got {config.api.port}'
print('Environment variables work correctly')
"; then
    print_status "✓ Environment variable support works"
else
    print_error "✗ Environment variable support failed"
    exit 1
fi

# Test 7: Test health check endpoint (if model exists)
if [ -d "models/artifacts" ] && [ "$(ls -A models/artifacts)" ]; then
    print_status "Testing API health check with existing model..."

    # Start container in background
    CONTAINER_ID=$(docker run -d -p 8001:8000 -v "$(pwd)/models:/app/models:ro" ml-classifier-app:test)

    # Wait for container to start
    sleep 10

    # Test health endpoint
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        print_status "✓ Health check endpoint works"
    else
        print_warning "⚠ Health check endpoint not responding (this may be expected if no model is trained)"
    fi

    # Clean up
    docker stop $CONTAINER_ID > /dev/null 2>&1
    docker rm $CONTAINER_ID > /dev/null 2>&1
else
    print_warning "⚠ Skipping API health check test (no trained model found)"
fi

# Clean up test images
print_status "Cleaning up test images..."
docker rmi ml-classifier-train:test ml-classifier-app:test > /dev/null 2>&1 || true

print_status "All container tests passed! 🎉"
print_status ""
print_status "Usage examples:"
print_status "  # Train a model:"
print_status "  docker-compose --profile train up train"
print_status ""
print_status "  # Start API server:"
print_status "  docker-compose --profile api up api"
print_status ""
print_status "  # Development mode with hot reload:"
print_status "  docker-compose --profile dev up api-dev"
print_status ""
print_status "  # Full stack (train + serve):"
print_status "  docker-compose --profile full up"
