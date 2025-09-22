# Docker Deployment Guide

This guide covers containerized deployment of the Enterprise ML Classifier using Docker and Docker Compose.

## Overview

The project provides two optimized Docker images:
- **Training Container**: For model training and hyperparameter tuning
- **Serving Container**: For production API serving

## Quick Start

### Prerequisites
- Docker installed and running
- Docker Compose (included with Docker Desktop)

### 1. Build Images
```bash
make docker-build
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Run Services

#### Train a Model
```bash
make docker-train
```

#### Start API Server
```bash
make docker-api
```

#### Development Mode
```bash
make docker-dev
```

## Container Images

### Training Container (`Dockerfile.train`)
- **Base**: `python:3.11-slim`
- **Purpose**: CPU-optimized for ML training workloads
- **Features**:
  - Complete ML dependencies
  - Volume mounting for data persistence
  - Environment variable configuration
  - Batch processing optimization

### Serving Container (`Dockerfile.app`)
- **Base**: `python:3.11-slim`
- **Purpose**: Lightweight inference serving
- **Features**:
  - Minimal runtime dependencies
  - Built-in health checks
  - Horizontal scaling support
  - Production logging

## Docker Compose Profiles

The system uses profiles to organize different deployment scenarios:

| Profile | Services | Use Case |
|---------|----------|----------|
| `train` | Training only | Model training |
| `tune` | Hyperparameter tuning | Parameter optimization |
| `api` | API server | Production serving |
| `dev` | API with hot reload | Development |
| `full` | Train + API | Complete pipeline |

### Usage Examples

```bash
# Train a model
docker-compose --profile train up train

# Start production API
docker-compose --profile api up api

# Development with hot reload
docker-compose --profile dev up api-dev

# Full pipeline
docker-compose --profile full up

# Run in detached mode
docker-compose --profile api up -d api
```

## Environment Configuration

### Core Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging
LOG_LEVEL=INFO

# Configuration Files
EXPERIMENT_CONFIG=configs/experiment_default.yaml
SERVING_CONFIG=configs/serving.yaml
```

### Advanced Configuration
```bash
# Resource Limits
MEMORY_LIMIT=2g
CPU_LIMIT=1.0

# Training Parameters
RANDOM_SEED=42
TEST_SIZE=0.2
CV_FOLDS=5
```

## Volume Mounts

### Training Workloads
```yaml
volumes:
  - ./data:/app/data              # Input/output data
  - ./models:/app/models          # Model artifacts
  - ./results:/app/results        # Training results
  - ./logs:/app/logs              # Application logs
  - ./configs:/app/configs        # Configuration files
```

### Serving Workloads
```yaml
volumes:
  - ./models:/app/models:ro       # Model artifacts (read-only)
  - ./logs:/app/logs              # Application logs
  - ./configs:/app/configs:ro     # Configuration files (read-only)
```

## Production Deployment

### Resource Limits
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Scaling
```bash
# Scale API service
docker-compose up --scale api=3

# Load balancer configuration needed for multiple instances
```

### Health Monitoring
```bash
# Check container health
docker ps

# API health endpoint
curl http://localhost:8000/health

# Container logs
docker-compose logs api
```

## Development Workflow

### Local Development
```bash
# Start development environment
make docker-dev

# API available at http://localhost:8000
# Hot reload enabled for code changes
```

### Testing
```bash
# Validate Docker configuration
make docker-validate

# Test container builds and functionality
make docker-test

# Run tests inside container
docker-compose run --rm train uv run pytest
```

### Debugging
```bash
# Execute commands in running container
docker-compose exec api bash

# View logs
docker-compose logs -f api

# Test container without starting services
docker run --rm -it ml-classifier-app bash
```

## Common Workflows

### Complete ML Pipeline
```bash
# 1. Train model
docker-compose --profile train up train

# 2. Start API server
docker-compose --profile api up -d api

# 3. Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Adelie",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "Male",
    "year": 2007
  }'
```

### Hyperparameter Tuning
```bash
# Configure tuning in experiment config
# Then run tuning
docker-compose --profile tune up tune
```

### Model Management
```bash
# List available models
docker-compose run --rm train uv run python -m src.cli list-models

# Get model information
docker-compose run --rm train uv run python -m src.cli model-info <model_id>
```

## Troubleshooting

### Common Issues

#### Model Not Found
```bash
# Ensure model directory is mounted and contains trained models
ls -la models/artifacts/

# Train a model first
make docker-train
```

#### Port Conflicts
```bash
# Change port in .env file
echo "API_PORT=8080" >> .env

# Or override in docker-compose
docker-compose up -e API_PORT=8080
```

#### Permission Errors
```bash
# Ensure directories are writable
chmod -R 755 logs models results

# Check volume mounts
docker-compose config
```

#### Memory Issues
```bash
# Increase Docker memory limits
# Or reduce model complexity in configs
```

### Debugging Commands

```bash
# Check container status
docker ps -a

# View detailed logs
docker-compose logs --tail=100 api

# Inspect container
docker inspect ml-classifier-api

# Test network connectivity
docker-compose exec api curl http://localhost:8000/health
```

## Security Considerations

### Production Security
- No hardcoded secrets (use environment variables)
- Read-only mounts for serving containers
- Network isolation with Docker networks
- Resource limits to prevent DoS
- Regular security updates for base images

### Environment Variables
```bash
# Sensitive configuration
export MODEL_ENCRYPTION_KEY="your-key-here"
export API_SECRET_KEY="your-secret-here"

# Use Docker secrets in production
docker secret create model_key model.key
```

## Performance Optimization

### Image Optimization
- Multi-stage builds minimize image size
- Layer caching optimizes build times
- Dependency pinning ensures reproducibility

### Runtime Optimization
- Health checks for proper startup
- Graceful shutdown handling
- Resource limits prevent resource exhaustion
- Horizontal scaling for high availability

### Monitoring
```bash
# Container resource usage
docker stats

# Application metrics
curl http://localhost:8000/model/info

# Log aggregation
docker-compose logs | grep ERROR
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Docker Build and Test
on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build images
        run: make docker-build
        
      - name: Test containers
        run: make docker-test
        
      - name: Push to registry
        if: github.ref == 'refs/heads/main'
        run: |
          docker tag ml-classifier-app:latest registry.com/ml-classifier-app:latest
          docker push registry.com/ml-classifier-app:latest
```

## Advanced Topics

### Custom Base Images
```dockerfile
# Use custom base with pre-installed dependencies
FROM your-registry.com/python-ml:3.11

# Or multi-architecture builds
FROM --platform=$BUILDPLATFORM python:3.11-slim
```

### Orchestration
- Kubernetes deployment manifests
- Docker Swarm configuration
- Service mesh integration

### Monitoring and Observability
- Prometheus metrics collection
- Grafana dashboards
- Distributed tracing
- Log aggregation with ELK stack

## Support

For Docker-related issues:
1. Run `make docker-validate` to check configuration
2. Use `make docker-test` to verify functionality
3. Check logs with `docker-compose logs`
4. Refer to [Docker documentation](https://docs.docker.com/)