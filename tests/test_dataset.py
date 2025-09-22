"""Tests for dataset loading utilities."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.dataset import (PenguinDataLoader, get_sample_data,
                              load_penguins_data)
from src.data.schema import ALL_FEATURES, TARGET


class TestPenguinDataLoader:
    """Test PenguinDataLoader functionality."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        return pd.DataFrame(
            {
                "Culmen Length (mm)": [39.1, 39.5, 40.3, None, 36.7],
                "Culmen Depth (mm)": [18.7, 17.4, 18.0, None, 19.3],
                "Flipper Length (mm)": [181.0, 186.0, 195.0, None, 193.0],
                "Body Mass (g)": [3750.0, 3800.0, 3250.0, None, 3450.0],
                "Island": [
                    "Torgersen",
                    "Torgersen",
                    "Torgersen",
                    "Torgersen",
                    "Torgersen",
                ],
                "Sex": ["MALE", "FEMALE", "FEMALE", None, "FEMALE"],
                "Species": [
                    "Adelie Penguin (Pygoscelis adeliae)",
                    "Adelie Penguin (Pygoscelis adeliae)",
                    "Adelie Penguin (Pygoscelis adeliae)",
                    "Adelie Penguin (Pygoscelis adeliae)",
                    "Adelie Penguin (Pygoscelis adeliae)",
                ],
                "Date Egg": [
                    "11/11/07",
                    "11/11/07",
                    "11/16/07",
                    "11/16/07",
                    "11/16/07",
                ],
            }
        )

    @patch("src.data.dataset.pd.read_csv")
    @patch("pathlib.Path.exists")
    def test_load_raw_data(self, mock_exists, mock_read_csv, sample_raw_data):
        """Test loading raw data from CSV."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data

        loader = PenguinDataLoader("test_path.csv")
        raw_data = loader.load_raw_data()

        assert len(raw_data) == 5
        assert "Culmen Length (mm)" in raw_data.columns
        mock_read_csv.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_load_raw_data_file_not_found(self, mock_exists):
        """Test error when data file doesn't exist."""
        mock_exists.return_value = False

        loader = PenguinDataLoader("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load_raw_data()

    @patch("src.data.dataset.pd.read_csv")
    @patch("pathlib.Path.exists")
    def test_load_clean_data(self, mock_exists, mock_read_csv, sample_raw_data):
        """Test loading and cleaning data."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data

        loader = PenguinDataLoader("test_path.csv")
        clean_data = loader.load_clean_data()

        # Should have mapped column names
        assert "bill_length_mm" in clean_data.columns
        assert "species" in clean_data.columns
        assert "year" in clean_data.columns

        # Should have extracted year from date
        assert clean_data["year"].iloc[0] == 2007

        # Should have cleaned species names
        assert clean_data["species"].iloc[0] == "Adelie"

    @patch("src.data.dataset.pd.read_csv")
    @patch("pathlib.Path.exists")
    def test_get_features_and_target(self, mock_exists, mock_read_csv, sample_raw_data):
        """Test getting features and target."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data

        loader = PenguinDataLoader("test_path.csv")
        X, y = loader.get_features_and_target()

        # Should have all available features
        expected_features = [f for f in ALL_FEATURES if f in X.columns]
        assert list(X.columns) == expected_features

        # Target should be species
        assert y.name == TARGET
        assert len(X) == len(y)  # Same number of rows

    @patch("src.data.dataset.pd.read_csv")
    @patch("pathlib.Path.exists")
    def test_train_test_split(self, mock_exists, mock_read_csv, sample_raw_data):
        """Test train/test split functionality."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data

        loader = PenguinDataLoader("test_path.csv")
        X_train, X_test, y_train, y_test = loader.train_test_split(
            test_size=0.4, random_state=42
        )

        # Check split proportions (approximately)
        total_samples = len(sample_raw_data)
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) + len(y_test) == total_samples

        # Check that we have the right columns
        expected_features = [f for f in ALL_FEATURES if f in X_train.columns]
        assert list(X_train.columns) == expected_features
        assert list(X_test.columns) == expected_features

    @patch("src.data.dataset.pd.read_csv")
    @patch("pathlib.Path.exists")
    def test_get_data_info(self, mock_exists, mock_read_csv, sample_raw_data):
        """Test getting data information."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data

        loader = PenguinDataLoader("test_path.csv")
        info = loader.get_data_info()

        assert "total_rows" in info
        assert "rows_with_target" in info
        assert "features" in info
        assert "target_distribution" in info
        assert "missing_values" in info

        assert info["total_rows"] == 5
        assert info["features"]["total"] > 0

    @patch("src.data.dataset.pd.read_csv")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_save_processed_data(
        self, mock_mkdir, mock_exists, mock_read_csv, sample_raw_data
    ):
        """Test saving processed data."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data

        loader = PenguinDataLoader("test_path.csv")

        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
            loader.save_processed_data("output/processed.csv")
            mock_to_csv.assert_called_once()


class TestLoadPenguinsData:
    """Test convenience function for loading data."""

    @patch("src.data.dataset.PenguinDataLoader")
    def test_load_penguins_data_no_split(self, mock_loader_class):
        """Test loading data without split."""
        mock_loader = MagicMock()
        mock_loader.get_features_and_target.return_value = ("X", "y")
        mock_loader_class.return_value = mock_loader

        X, y = load_penguins_data(return_split=False)

        assert X == "X"
        assert y == "y"
        mock_loader.get_features_and_target.assert_called_once()

    @patch("src.data.dataset.PenguinDataLoader")
    def test_load_penguins_data_with_split(self, mock_loader_class):
        """Test loading data with train/test split."""
        mock_loader = MagicMock()
        mock_loader.train_test_split.return_value = (
            "X_train",
            "X_test",
            "y_train",
            "y_test",
        )
        mock_loader_class.return_value = mock_loader

        X_train, X_test, y_train, y_test = load_penguins_data(
            return_split=True, test_size=0.3, random_state=123
        )

        assert X_train == "X_train"
        assert X_test == "X_test"
        assert y_train == "y_train"
        assert y_test == "y_test"

        mock_loader.train_test_split.assert_called_once_with(
            test_size=0.3, random_state=123, stratify=True
        )


class TestGetSampleData:
    """Test sample data generation."""

    @patch("src.data.dataset.PenguinDataLoader")
    def test_get_sample_data(self, mock_loader_class):
        """Test getting sample data."""
        # Create mock data with different species
        sample_df = pd.DataFrame(
            {
                "bill_length_mm": [39.1, 39.5, 40.3, 41.0, 42.0],
                "species": ["Adelie", "Adelie", "Chinstrap", "Chinstrap", "Gentoo"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.load_clean_data.return_value = sample_df
        mock_loader_class.return_value = mock_loader

        result = get_sample_data(n_samples=3)

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3
        mock_loader.load_clean_data.assert_called_once()

    @patch("src.data.dataset.PenguinDataLoader")
    def test_get_sample_data_with_missing_target(self, mock_loader_class):
        """Test getting sample data when some rows have missing target."""
        # Create mock data with missing species
        sample_df = pd.DataFrame(
            {
                "bill_length_mm": [39.1, 39.5, 40.3],
                "species": ["Adelie", None, "Chinstrap"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.load_clean_data.return_value = sample_df
        mock_loader_class.return_value = mock_loader

        result = get_sample_data(n_samples=5)

        # Should only include rows with non-null species
        assert len(result) <= 2  # Only 2 rows have valid species
        assert result["species"].notna().all()  # All species should be non-null
