"""Integration tests for CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli import cli


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Enterprise ML Classifier CLI" in result.output
        assert "train" in result.output
        assert "tune" in result.output
        assert "serve" in result.output

    def test_train_help(self):
        """Test train command help."""
        result = self.runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train a machine learning model" in result.output
        assert "--config" in result.output
        assert "--output" in result.output
        assert "--no-save" in result.output

    def test_tune_help(self):
        """Test tune command help."""
        result = self.runner.invoke(cli, ["tune", "--help"])
        assert result.exit_code == 0
        assert "Perform hyperparameter tuning" in result.output
        assert "--config" in result.output
        assert "--output" in result.output
        assert "--no-save" in result.output

    def test_serve_help(self):
        """Test serve command help."""
        result = self.runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the API server" in result.output
        assert "--config" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output

    def test_list_models_help(self):
        """Test list-models command help."""
        result = self.runner.invoke(cli, ["list-models", "--help"])
        assert result.exit_code == 0
        assert "List available models" in result.output
        assert "--model-dir" in result.output
        assert "--format" in result.output

    def test_model_info_help(self):
        """Test model-info command help."""
        result = self.runner.invoke(cli, ["model-info", "--help"])
        assert result.exit_code == 0
        assert "Show detailed information" in result.output
        assert "--model-dir" in result.output

    def test_list_models_empty_registry(self):
        """Test list-models with empty registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(cli, ["list-models", "--model-dir", temp_dir])
            assert result.exit_code == 0
            assert "No models found in registry" in result.output

    def test_list_models_json_format(self):
        """Test list-models with JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(
                cli, ["list-models", "--model-dir", temp_dir, "--format", "json"]
            )
            assert result.exit_code == 0
            # Should be valid JSON (empty object for empty registry)
            output = result.output.strip()
            if output:
                json.loads(output)

    def test_model_info_nonexistent(self):
        """Test model-info with nonexistent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(
                cli, ["model-info", "nonexistent_model", "--model-dir", temp_dir]
            )
            assert result.exit_code == 1
            assert "not found in registry" in result.output

    def test_train_missing_config(self):
        """Test train command with missing config file."""
        result = self.runner.invoke(
            cli, ["train", "--config", "nonexistent_config.yaml"]
        )
        assert result.exit_code == 2  # Click error for missing file

    def test_tune_missing_config(self):
        """Test tune command with missing config file."""
        result = self.runner.invoke(
            cli, ["tune", "--config", "nonexistent_config.yaml"]
        )
        assert result.exit_code == 2  # Click error for missing file

    def test_serve_missing_config(self):
        """Test serve command with missing config file."""
        result = self.runner.invoke(
            cli, ["serve", "--config", "nonexistent_config.yaml"]
        )
        assert result.exit_code == 2  # Click error for missing file

    @patch("src.cli.train_model")
    def test_train_with_output_file(self, mock_train):
        """Test train command with output file."""
        # Mock training results
        mock_results = {
            "model_name": "TestModel",
            "test_metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "data_info": {
                "train_size": 100,
                "test_size": 25,
                "features": ["feature1", "feature2"],
                "target_classes": ["class1", "class2"],
            },
        }
        mock_train.return_value = mock_results

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text("seed: 42\n")

            output_file = Path(temp_dir) / "results.json"

            result = self.runner.invoke(
                cli,
                [
                    "train",
                    "--config",
                    str(config_file),
                    "--output",
                    str(output_file),
                    "--no-save",
                ],
            )

            assert result.exit_code == 0
            assert "Training completed successfully" in result.output
            assert "Test Accuracy: 0.8500" in result.output

            # Check output file was created
            assert output_file.exists()
            with open(output_file) as f:
                saved_results = json.load(f)
            assert saved_results["model_name"] == "TestModel"

    @patch("src.cli.tune_model")
    def test_tune_with_output_file(self, mock_tune):
        """Test tune command with output file."""
        # Mock tuning results
        mock_results = {
            "model_name": "TestModel",
            "best_score": 0.88,
            "best_params": {"param1": "value1"},
            "test_metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "tuning_config": {"cv": 5, "scoring": "accuracy"},
            "data_info": {
                "train_size": 100,
                "test_size": 25,
                "features": ["feature1", "feature2"],
                "target_classes": ["class1", "class2"],
            },
        }
        mock_tune.return_value = mock_results

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_content = """
seed: 42
model:
  tune:
    grid:
      - param1: [value1, value2]
    cv: 5
    scoring: accuracy
"""
            config_file.write_text(config_content)

            output_file = Path(temp_dir) / "tune_results.json"

            result = self.runner.invoke(
                cli,
                [
                    "tune",
                    "--config",
                    str(config_file),
                    "--output",
                    str(output_file),
                    "--no-save",
                ],
            )

            assert result.exit_code == 0
            assert "Hyperparameter tuning completed successfully" in result.output
            assert "Best CV Score: 0.8800" in result.output

            # Check output file was created
            assert output_file.exists()
            with open(output_file) as f:
                saved_results = json.load(f)
            assert saved_results["model_name"] == "TestModel"

    def test_tune_without_tuning_config(self):
        """Test tune command without tuning configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text("seed: 42\n")  # No tuning config

            result = self.runner.invoke(cli, ["tune", "--config", str(config_file)])

            assert result.exit_code == 1
            assert "No tuning configuration found" in result.output

    @patch("uvicorn.run")
    def test_serve_command(self, mock_uvicorn):
        """Test serve command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "serving_config.yaml"
            config_content = """
api:
  host: "127.0.0.1"
  port: 8000
  reload: false
  workers: 1
logging:
  level: "INFO"
  log_file: "logs/app.log"
"""
            config_file.write_text(config_content)

            result = self.runner.invoke(
                cli,
                [
                    "serve",
                    "--config",
                    str(config_file),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8080",
                    "--reload",
                ],
            )

            # Should not exit with error (uvicorn.run is mocked)
            assert result.exit_code == 0

            # Check that uvicorn.run was called with correct parameters
            mock_uvicorn.assert_called_once()
            call_args = mock_uvicorn.call_args
            assert call_args[1]["host"] == "0.0.0.0"  # Overridden by CLI
            assert call_args[1]["port"] == 8080  # Overridden by CLI
            assert call_args[1]["reload"] is True  # Overridden by CLI

    def test_verbose_and_quiet_flags(self):
        """Test verbose and quiet logging flags."""
        # Test verbose flag
        result = self.runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

        # Test quiet flag
        result = self.runner.invoke(cli, ["--quiet", "--help"])
        assert result.exit_code == 0

        # Test that both flags work (quiet should take precedence)
        result = self.runner.invoke(cli, ["--verbose", "--quiet", "--help"])
        assert result.exit_code == 0


class TestCLIIntegration:
    """Integration tests that require actual files and data."""

    def test_train_with_real_config(self):
        """Test training with real configuration file."""
        config_path = "configs/experiment_default.yaml"

        if not Path(config_path).exists():
            pytest.skip(f"Config file {config_path} not found")

        runner = CliRunner()

        # Test with --no-save to avoid creating model files
        result = runner.invoke(cli, ["train", "--config", config_path, "--no-save"])

        # Should fail gracefully if no data is available
        # or succeed if data is present
        assert result.exit_code in [0, 1]

        if result.exit_code == 1:
            # Should have a meaningful error message
            assert "Error:" in result.output

    def test_serve_with_real_config(self):
        """Test serve command with real configuration."""
        config_path = "configs/serving.yaml"

        if not Path(config_path).exists():
            pytest.skip(f"Config file {config_path} not found")

        runner = CliRunner()

        # Just test that the command parses correctly
        # We can't actually start the server in tests
        with patch("uvicorn.run") as mock_uvicorn:
            result = runner.invoke(cli, ["serve", "--config", config_path])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()
