"""Tests for data schema validation."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    VALID_ISLANDS,
    VALID_SEXES,
    VALID_SPECIES,
    PenguinRecord,
    PenguinDataset,
    validate_dataframe_schema,
    get_feature_info
)


class TestPenguinRecord:
    """Test PenguinRecord validation."""
    
    def test_valid_record(self):
        """Test creating a valid penguin record."""
        record = PenguinRecord(
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            year=2007,
            island="Torgersen",
            sex="MALE",
            species="Adelie"
        )
        
        assert record.bill_length_mm == 39.1
        assert record.island == "Torgersen"
        assert record.species == "Adelie"
    
    def test_record_with_none_values(self):
        """Test record with None values (missing data)."""
        record = PenguinRecord(
            bill_length_mm=None,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            year=2007,
            island="Torgersen",
            sex="MALE",
            species="Adelie"
        )
        
        assert record.bill_length_mm is None
        assert record.bill_depth_mm == 18.7
    
    def test_invalid_island(self):
        """Test validation of invalid island."""
        with pytest.raises(ValidationError):
            PenguinRecord(
                island="InvalidIsland",
                species="Adelie"
            )
    
    def test_invalid_sex(self):
        """Test validation of invalid sex."""
        with pytest.raises(ValidationError):
            PenguinRecord(
                sex="UNKNOWN",
                species="Adelie"
            )
    
    def test_invalid_species(self):
        """Test validation of invalid species."""
        with pytest.raises(ValidationError):
            PenguinRecord(
                species="InvalidSpecies"
            )
    
    def test_species_name_parsing(self):
        """Test parsing of full species names."""
        record = PenguinRecord(
            species="Adelie Penguin (Pygoscelis adeliae)"
        )
        assert record.species == "Adelie"
        
        record = PenguinRecord(
            species="Chinstrap penguin (Pygoscelis antarctica)"
        )
        assert record.species == "Chinstrap"
    
    def test_negative_values(self):
        """Test validation of negative values."""
        with pytest.raises(ValidationError):
            PenguinRecord(
                bill_length_mm=-10.0
            )
        
        with pytest.raises(ValidationError):
            PenguinRecord(
                body_mass_g=-100.0
            )
    
    def test_invalid_year(self):
        """Test validation of invalid year."""
        with pytest.raises(ValidationError):
            PenguinRecord(
                year=2020  # Outside valid range
            )
        
        with pytest.raises(ValidationError):
            PenguinRecord(
                year=2000  # Outside valid range
            )


class TestPenguinDataset:
    """Test PenguinDataset functionality."""
    
    def test_from_dataframe(self):
        """Test creating dataset from DataFrame."""
        df = pd.DataFrame({
            "bill_length_mm": [39.1, 39.5, None],
            "bill_depth_mm": [18.7, 17.4, 19.0],
            "flipper_length_mm": [181.0, 186.0, 195.0],
            "body_mass_g": [3750.0, 3800.0, 3250.0],
            "year": [2007, 2007, 2007],
            "island": ["Torgersen", "Torgersen", "Torgersen"],
            "sex": ["MALE", "FEMALE", "FEMALE"],
            "species": ["Adelie", "Adelie", "Adelie"]
        })
        
        dataset = PenguinDataset.from_dataframe(df)
        
        assert len(dataset.records) == 3
        assert dataset.records[0].bill_length_mm == 39.1
        assert dataset.records[2].bill_length_mm is None
    
    def test_to_dataframe(self):
        """Test converting dataset to DataFrame."""
        records = [
            PenguinRecord(
                bill_length_mm=39.1,
                bill_depth_mm=18.7,
                island="Torgersen",
                sex="MALE",
                species="Adelie"
            ),
            PenguinRecord(
                bill_length_mm=39.5,
                bill_depth_mm=17.4,
                island="Torgersen",
                sex="FEMALE",
                species="Adelie"
            )
        ]
        
        dataset = PenguinDataset(records=records)
        df = dataset.to_dataframe()
        
        assert len(df) == 2
        assert df.iloc[0]["bill_length_mm"] == 39.1
        assert df.iloc[1]["sex"] == "FEMALE"


class TestValidateDataframeSchema:
    """Test DataFrame schema validation."""
    
    def test_validate_clean_dataframe(self):
        """Test validation of already clean DataFrame."""
        df = pd.DataFrame({
            "bill_length_mm": [39.1, 39.5],
            "bill_depth_mm": [18.7, 17.4],
            "flipper_length_mm": [181.0, 186.0],
            "body_mass_g": [3750.0, 3800.0],
            "year": [2007, 2007],
            "island": ["Torgersen", "Torgersen"],
            "sex": ["MALE", "FEMALE"],
            "species": ["Adelie", "Adelie"]
        })
        
        validated_df = validate_dataframe_schema(df)
        
        assert len(validated_df) == 2
        assert list(validated_df.columns) == ALL_FEATURES + [TARGET]
    
    def test_validate_with_column_mapping(self):
        """Test validation with column name mapping."""
        df = pd.DataFrame({
            "Culmen Length (mm)": [39.1, 39.5],
            "Culmen Depth (mm)": [18.7, 17.4],
            "Flipper Length (mm)": [181.0, 186.0],
            "Body Mass (g)": [3750.0, 3800.0],
            "Island": ["Torgersen", "Torgersen"],
            "Sex": ["MALE", "FEMALE"],
            "Species": ["Adelie Penguin (Pygoscelis adeliae)", "Adelie Penguin (Pygoscelis adeliae)"],
            "Date Egg": ["11/11/07", "11/11/07"]
        })
        
        validated_df = validate_dataframe_schema(df)
        
        assert "bill_length_mm" in validated_df.columns
        assert "species" in validated_df.columns
        assert validated_df["species"].iloc[0] == "Adelie"
        assert validated_df["year"].iloc[0] == 2007
    
    def test_validate_with_missing_columns(self):
        """Test validation with some missing columns."""
        df = pd.DataFrame({
            "bill_length_mm": [39.1, 39.5],
            "bill_depth_mm": [18.7, 17.4],
            "species": ["Adelie", "Adelie"]
        })
        
        validated_df = validate_dataframe_schema(df)
        
        # Should only include available columns
        expected_columns = ["bill_length_mm", "bill_depth_mm", "species"]
        assert list(validated_df.columns) == expected_columns


class TestGetFeatureInfo:
    """Test feature information function."""
    
    def test_get_feature_info(self):
        """Test getting feature information."""
        info = get_feature_info()
        
        assert "numeric_features" in info
        assert "categorical_features" in info
        assert "target" in info
        assert "all_features" in info
        
        assert info["numeric_features"] == NUMERIC_FEATURES
        assert info["categorical_features"] == CATEGORICAL_FEATURES
        assert info["target"] == TARGET
        assert info["all_features"] == ALL_FEATURES
        
        assert info["valid_islands"] == VALID_ISLANDS
        assert info["valid_sexes"] == VALID_SEXES
        assert info["valid_species"] == VALID_SPECIES