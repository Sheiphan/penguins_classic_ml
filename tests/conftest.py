"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_penguins_csv(test_data_dir):
    """Create a sample penguins CSV file for testing."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "species": np.random.choice(["Adelie", "Chinstrap", "Gentoo"], n_samples),
            "island": np.random.choice(["Torgersen", "Biscoe", "Dream"], n_samples),
            "bill_length_mm": np.random.normal(44, 5, n_samples),
            "bill_depth_mm": np.random.normal(17, 2, n_samples),
            "flipper_length_mm": np.random.normal(200, 15, n_samples),
            "body_mass_g": np.random.normal(4200, 800, n_samples),
            "sex": np.random.choice(["MALE", "FEMALE"], n_samples),
            "year": np.random.choice([2007, 2008, 2009], n_samples),
        }
    )

    csv_path = test_data_dir / "penguins_test.csv"
    data.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def temp_model_registry():
    """Create a temporary model registry directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = Path(temp_dir)
        (registry_path / "artifacts").mkdir()
        (registry_path / "metrics").mkdir()
        yield registry_path


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    # Set test environment variables
    test_env = {
        "PYTHONPATH": "src",
        "LOG_LEVEL": "DEBUG",
        "MODEL_DIR": "/tmp/test_models",
        "DATA_DIR": "/tmp/test_data",
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    import logging

    # Suppress verbose logging during tests
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings

    # Suppress sklearn warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    # Suppress pandas warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    yield

    # Reset warnings
    warnings.resetwarnings()


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark tests that require external resources
        if "external" in item.nodeid.lower() or "network" in item.nodeid.lower():
            item.add_marker(pytest.mark.external)


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_runtest_setup(item):
    """Skip tests based on command line options."""
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")

    if "integration" in item.keywords and not item.config.getoption(
        "--run-integration"
    ):
        pytest.skip("need --run-integration option to run")
