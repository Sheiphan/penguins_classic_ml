# Docker Deployment Guide

This directory contains Docker configurations for the Enterprise ML Classifier project.

## Container Images

### Training Container (`Dockerfile.train`)
- **Purpose**: CPU-optimized container for model training and hyperparameter tuning
- **Base Image**: `python:3.11-slim`
- **Key Features**:
  - Includes all ML dependencies and training code
  - Supports volume mounting for data and model persistence
  - Configurable through environment variables
  - Optimized for batch processing workloads

### Serving Container (`Dockerfile.app`)
- **Purpose**: Lightweight container for model inference via FastAPI
- **Base Image**: `python:3.11-slim`
- **Key Features**:
  - Minimal runtime dependencies for serving
  - Built-in health checks
  - Supports horizontal scaling
  - Production-ready with proper logging

## Quick Start

### 1. Build Images
```bash
# Build both images
make docker-build

# Or build individually
docker build -f Dockerfile.train -t ml-classifier-train .
docker build -f Dockerfile.app -t ml-classifier-app .
```

### 2. Environment Configuration
```bash
# Copy and customize environment file
cp .env.example .env

# Edit configuration as needed
vim .env
```

### 3. Run Services

#### Training
```bash
# Run training with docker-compose
make docker-train

# Or run directly
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  ml-classifier-train
```

#### API Serving
```bash
# Run API server
make docker-api

# Or run directly
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  ml-classifier-app
```

#### Development Mode
```bash
# Run with hot reload
make docker-dev
```

## Docker Compose Profiles

The `docker-compose.yaml` file uses profiles to organize different deployment scenarios:

### Available Profiles

- **`train`**: Training workloads only
- **`tune`**: Hyperparameter tuning only
- **`api`**: Production API serving
- **`dev`**: Development API with hot reload
- **`full`**: Complete stack (train + serve)

### Usage Examples

```bash
# Train a model
docker-compose --profile train up train

# Start API server
docker-compose --profile api up api

# Development mode
docker-compose --profile dev up api-dev

# Full stack
docker-compose --profile full up
```

## Environment Variables

### API Configuration
- `API_HOST`: Host to bind API server (default: 0.0.0.0)
- `API_PORT`: Port for API server (default: 8000)
- `API_WORKERS`: Number of worker processes (default: 1)

### Logging Configuration
- `LOG_LEVEL`: Logging level (default: INFO)

### File Paths
- `EXPERIMENT_CONFIG`: Path to experiment config (default: configs/experiment_default.yaml)
- `SERVING_CONFIG`: Path to serving config (default: configs/serving.yaml)

### Training Configuration
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)

## Volume Mounts

### Training Container
- `./data:/app/data` - Data directory (input/output)
- `./models:/app/models` - Model artifacts and registry
- `./results:/app/results` - Training results and metrics
- `./logs:/app/logs` - Application logs
- `./configs:/app/configs` - Configuration files

### Serving Container
- `./models:/app/models:ro` - Model artifacts (read-only)
- `./logs:/app/logs` - Application logs
- `./configs:/app/configs:ro` - Configuration files (read-only)

## Health Checks

The serving container includes built-in health checks:

```bash
# Check container health
docker ps

# Manual health check
curl http://localhost:8000/health
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
```

### Monitoring
- Health check endpoint: `/health`
- Metrics endpoint: `/model/info`
- Structured logging to stdout/files

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure models directory is properly mounted
   - Check that a trained model exists in `models/artifacts/`

2. **Permission errors**
   - Ensure proper file permissions on mounted volumes
   - Check that log directories are writable

3. **Port conflicts**
   - Change `API_PORT` environment variable
   - Update port mapping in docker-compose.yaml

### Debugging

```bash
# View container logs
docker-compose logs api

# Execute commands in running container
docker-compose exec api bash

# Test container without starting services
docker run --rm -it ml-classifier-app bash
```

### Testing

```bash
# Run comprehensive container tests
make docker-test

# Test individual components
docker run --rm ml-classifier-train python -c "import src.cli; print('OK')"
docker run --rm ml-classifier-app python -c "import src.serving.app; print('OK')"
```

## Security Considerations

1. **No hardcoded secrets**: All sensitive configuration via environment variables
2. **Read-only mounts**: Serving containers use read-only model mounts
3. **Non-root user**: Containers run as non-root user (future enhancement)
4. **Network isolation**: Services communicate through Docker networks
5. **Resource limits**: Proper CPU/memory limits in production

## Performance Optimization

1. **Multi-stage builds**: Minimize image size
2. **Layer caching**: Optimize Dockerfile layer order
3. **Dependency pinning**: Use uv.lock for reproducible builds
4. **Health checks**: Proper startup and liveness probes
5. **Graceful shutdown**: Handle SIGTERM signals properly
