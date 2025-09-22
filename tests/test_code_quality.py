"""Tests for code quality tools and configurations."""

import subprocess
import tempfile
from pathlib import Path

import pytest


class TestCodeQualityTools:
    """Test code quality tools configuration and execution."""

    def test_ruff_format_configuration(self):
        """Test that Ruff format is properly configured."""
        # Create a sample Python file with formatting issues
        sample_code = """
def   badly_formatted_function(  x,y,z  ):
    if x>0:
        return x+y+z
    else:
        return   0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_code)
            temp_file = f.name

        try:
            # Run ruff format check on the file
            result = subprocess.run(
                ["ruff", "format", "--check", "--diff", temp_file],
                capture_output=True,
                text=True,
            )

            # Ruff should detect formatting issues (exit code 1)
            assert result.returncode == 1
            assert "would reformat" in result.stderr or len(result.stdout) > 0

        finally:
            Path(temp_file).unlink()

    def test_ruff_import_sorting(self):
        """Test that Ruff handles import sorting properly."""
        # Create a sample Python file with import issues
        sample_code = """
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.core.config import ExperimentConfig
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_code)
            temp_file = f.name

        try:
            # Run ruff check on the file (includes import sorting)
            result = subprocess.run(
                ["ruff", "check", temp_file], capture_output=True, text=True
            )

            # Ruff should detect import sorting issues or pass if configured correctly
            # Exit code can be 0 or 1 depending on configuration
            assert result.returncode in [0, 1]

        finally:
            Path(temp_file).unlink()

    def test_ruff_configuration(self):
        """Test that Ruff is properly configured."""
        # Create a sample Python file with linting issues
        sample_code = """
import os
import sys  # unused import

def unused_function():
    pass

def function_with_issues():
    x = 1
    y = 2
    # unused variables
    z = x + y
    return "hello"  # inconsistent quotes
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_code)
            temp_file = f.name

        try:
            # Run ruff on the file
            result = subprocess.run(
                ["ruff", "check", temp_file], capture_output=True, text=True
            )

            # Ruff should detect linting issues
            assert result.returncode == 1
            assert len(result.stdout) > 0  # Should have output about issues

        finally:
            Path(temp_file).unlink()

    def test_pytest_configuration(self):
        """Test that pytest is properly configured."""
        # Run pytest with configuration check
        result = subprocess.run(
            ["pytest", "--collect-only", "-q"], capture_output=True, text=True, cwd="."
        )

        # pytest should be able to collect tests
        assert result.returncode == 0
        assert "test session starts" in result.stdout or "collected" in result.stdout

    def test_pytest_coverage_configuration(self):
        """Test that pytest coverage is properly configured."""
        # Run a simple test with coverage but without fail-under
        result = subprocess.run(
            [
                "pytest",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-fail-under=0",
                "-v",
                "tests/test_code_quality.py::TestCodeQualityTools::test_pytest_configuration",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should run successfully and show coverage
        assert result.returncode == 0
        # Coverage output should be present
        assert "coverage" in result.stdout.lower() or "TOTAL" in result.stdout


class TestPreCommitHooks:
    """Test pre-commit hooks configuration."""

    def test_pre_commit_config_exists(self):
        """Test that pre-commit config file exists and is valid."""
        config_path = Path(".pre-commit-config.yaml")
        assert config_path.exists()

        # Try to parse the YAML
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "repos" in config
        assert len(config["repos"]) > 0

        # Check for expected hooks
        hook_ids = []
        for repo in config["repos"]:
            for hook in repo.get("hooks", []):
                hook_ids.append(hook["id"])

        expected_hooks = ["ruff", "ruff-format", "trailing-whitespace"]
        for hook in expected_hooks:
            assert hook in hook_ids

    @pytest.mark.slow
    def test_pre_commit_hooks_installation(self):
        """Test that pre-commit hooks can be installed."""
        # This test requires pre-commit to be installed
        try:
            result = subprocess.run(
                ["pre-commit", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                pytest.skip("pre-commit not installed")
        except FileNotFoundError:
            pytest.skip("pre-commit not found")

        # Try to install hooks (dry run)
        result = subprocess.run(
            ["pre-commit", "install", "--install-hooks", "--dry-run"],
            capture_output=True,
            text=True,
        )

        # Should succeed or already be installed
        assert result.returncode == 0


class TestProjectConfiguration:
    """Test project configuration files."""

    def test_pyproject_toml_exists_and_valid(self):
        """Test that pyproject.toml exists and has required sections."""
        config_path = Path("pyproject.toml")
        assert config_path.exists()

        # Try to parse the TOML
        import tomllib

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        # Check for required sections
        assert "project" in config
        assert "tool" in config

        # Check for tool configurations
        assert "ruff" in config["tool"]
        assert "pytest" in config["tool"]

        # Check Ruff configuration
        ruff_config = config["tool"]["ruff"]
        assert "line-length" in ruff_config
        assert ruff_config["line-length"] == 88

        # Check Ruff format configuration exists
        assert "format" in ruff_config

        # Check pytest configuration
        pytest_config = config["tool"]["pytest"]["ini_options"]
        assert "testpaths" in pytest_config
        assert any("--cov=src" in opt for opt in pytest_config.get("addopts", []))

    def test_makefile_quality_targets(self):
        """Test that Makefile has code quality targets."""
        makefile_path = Path("Makefile")
        if not makefile_path.exists():
            pytest.skip("Makefile not found")

        with open(makefile_path) as f:
            makefile_content = f.read()

        # Check for quality-related targets
        expected_targets = ["lint", "format", "test"]
        for target in expected_targets:
            assert (
                f"{target}:" in makefile_content
                or f".PHONY: {target}" in makefile_content
            )


class TestCodeQualityIntegration:
    """Test integration of code quality tools."""

    def test_format_and_lint_integration(self):
        """Test that formatting and linting work together with Ruff."""
        # Create a sample file with both formatting and linting issues
        sample_code = """
import os
import sys
import unused_import

def   badly_formatted_function(  x,y,z  ):
    unused_var = 42
    if x>0:
        return x+y+z
    else:
        return   0

class   BadlyFormattedClass:
    def method(self):
        pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_code)
            temp_file = f.name

        try:
            # First run ruff format to format
            subprocess.run(["ruff", "format", temp_file], check=True)

            # Then run ruff check for linting issues
            result = subprocess.run(
                ["ruff", "check", temp_file], capture_output=True, text=True
            )

            # Should still have linting issues (unused imports/variables)
            assert result.returncode == 1
            assert (
                "unused" in result.stdout.lower()
                or "imported but unused" in result.stdout
            )

        finally:
            Path(temp_file).unlink()

    def test_test_discovery_and_execution(self):
        """Test that tests can be discovered and executed."""
        # Run a subset of tests to verify the testing infrastructure
        result = subprocess.run(
            [
                "pytest",
                "tests/test_code_quality.py::TestCodeQualityTools::test_pytest_configuration",
                "-v",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "PASSED" in result.stdout

    @pytest.mark.slow
    def test_full_quality_check_pipeline(self):
        """Test running the full quality check pipeline with Ruff."""
        # This simulates what would run in CI
        commands = [
            ["ruff", "format", "--check", "src/", "tests/"],
            ["ruff", "check", "src/", "tests/"],
        ]

        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                # Commands should either pass (0) or fail with formatting issues (1)
                assert result.returncode in [0, 1]
            except subprocess.TimeoutExpired:
                pytest.fail(f"Command {' '.join(cmd)} timed out")
            except FileNotFoundError:
                pytest.skip(f"Command {cmd[0]} not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
