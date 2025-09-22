# Enterprise ML Classifier Documentation

This directory contains comprehensive documentation for the Enterprise ML Classifier project.

## Documentation Files

- **[CLI_USAGE.md](CLI_USAGE.md)** - Complete guide to using the command-line interface
- **[DOCKER.md](DOCKER.md)** - Docker deployment and containerization guide
- **[API_REFERENCE.md](API_REFERENCE.md)** - FastAPI endpoints and usage (coming soon)
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration file reference (coming soon)
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development setup and contribution guide (coming soon)

## Quick Links

### Getting Started
1. [Setup Development Environment](CLI_USAGE.md#setup--installation)
2. [Train Your First Model](CLI_USAGE.md#train---train-a-machine-learning-model)
3. [Start the API Server](CLI_USAGE.md#serve---start-api-server)
4. [Docker Deployment](DOCKER.md#quick-start)

### Common Tasks
- [Training Models](CLI_USAGE.md#training-commands)
- [Hyperparameter Tuning](CLI_USAGE.md#tune---hyperparameter-tuning)
- [Managing Models](CLI_USAGE.md#model-management-commands)
- [Using Makefile Commands](CLI_USAGE.md#makefile-commands)

### Advanced Usage
- [Configuration Files](CLI_USAGE.md#configuration-files)
- [Developer Scripts](CLI_USAGE.md#developer-scripts)
- [Docker Containers](DOCKER.md#container-images)
- [Production Deployment](DOCKER.md#production-deployment)
- [Integration Testing](CLI_USAGE.md#integration-with-other-tools)

## Project Structure

```
enterprise-ml-classifier/
├── src/                    # Source code
│   ├── cli.py             # Command-line interface
│   ├── core/              # Core configuration and utilities
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and registry
│   └── serving/           # API serving layer
├── tests/                 # Test suite
├── configs/               # Configuration files
├── scripts/               # Developer workflow scripts
├── docs/                  # Documentation
└── Makefile              # Build and workflow commands
```

## Support

For questions or issues:
1. Check the [CLI Usage Guide](CLI_USAGE.md)
2. Run `make validate` to check your setup
3. Use `make help` to see available commands
4. Check the test suite with `make test`