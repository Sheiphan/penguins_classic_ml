"""Data schema definitions for the penguins dataset."""

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Feature definitions based on penguins_lter.csv dataset
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "year",
]

CATEGORICAL_FEATURES = ["island", "sex"]

TARGET = "species"

# All features combined
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Column mapping from raw dataset to our schema
COLUMN_MAPPING = {
    "Culmen Length (mm)": "bill_length_mm",
    "Culmen Depth (mm)": "bill_depth_mm",
    "Flipper Length (mm)": "flipper_length_mm",
    "Body Mass (g)": "body_mass_g",
    "Island": "island",
    "Sex": "sex",
    "Species": "species",
    "Date Egg": "date_egg",
}

# Valid values for categorical features
VALID_ISLANDS = ["Torgersen", "Biscoe", "Dream"]
VALID_SEXES = ["MALE", "FEMALE"]
VALID_SPECIES = ["Adelie", "Chinstrap", "Gentoo"]


class PenguinRecord(BaseModel):
    """Schema for a single penguin record."""

    bill_length_mm: float | None = Field(
        None, ge=0, description="Bill length in millimeters"
    )
    bill_depth_mm: float | None = Field(
        None, ge=0, description="Bill depth in millimeters"
    )
    flipper_length_mm: float | None = Field(
        None, ge=0, description="Flipper length in millimeters"
    )
    body_mass_g: float | None = Field(None, ge=0, description="Body mass in grams")
    year: int | None = Field(None, ge=2007, le=2009, description="Year of observation")
    island: str | None = Field(None, description="Island where penguin was observed")
    sex: str | None = Field(None, description="Sex of the penguin")
    species: str | None = Field(None, description="Species of the penguin")

    @field_validator("island")
    @classmethod
    def validate_island(cls, v):
        """Validate island values."""
        if v is not None and v not in VALID_ISLANDS:
            raise ValueError(f"Island must be one of {VALID_ISLANDS}, got {v}")
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        """Validate sex values."""
        if v is not None and v.strip() != "" and v != ".":
            if v not in VALID_SEXES:
                raise ValueError(f"Sex must be one of {VALID_SEXES}, got {v}")
        elif v == "." or (v is not None and v.strip() == ""):
            return None  # Convert empty strings and "." to None
        return v

    @field_validator("species")
    @classmethod
    def validate_species(cls, v):
        """Validate species values."""
        if v is not None:
            # Handle full species names from raw data
            if "Adelie" in v:
                return "Adelie"
            elif "Chinstrap" in v:
                return "Chinstrap"
            elif "Gentoo" in v:
                return "Gentoo"
            elif v not in VALID_SPECIES:
                raise ValueError(f"Species must be one of {VALID_SPECIES}, got {v}")
        return v


class PenguinDataset(BaseModel):
    """Schema for the complete penguins dataset."""

    records: list[PenguinRecord]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "PenguinDataset":
        """Create PenguinDataset from pandas DataFrame."""
        records = []
        for _, row in df.iterrows():
            record_data = {}
            for feature in ALL_FEATURES + [TARGET]:
                if feature in row.index:
                    value = row[feature]
                    # Handle NaN values
                    if pd.isna(value):
                        record_data[feature] = None
                    else:
                        record_data[feature] = value

            records.append(PenguinRecord(**record_data))

        return cls(records=records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert PenguinDataset to pandas DataFrame."""
        data = []
        for record in self.records:
            data.append(record.model_dump())

        return pd.DataFrame(data)


def validate_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a DataFrame according to the penguin schema."""
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Apply column mapping if needed
    df_clean = df_clean.rename(columns=COLUMN_MAPPING)

    # Extract year from date if needed
    if "date_egg" in df_clean.columns and "year" not in df_clean.columns:
        df_clean["year"] = pd.to_datetime(df_clean["date_egg"], errors="coerce").dt.year

    # Select only the features we need
    available_features = [f for f in ALL_FEATURES + [TARGET] if f in df_clean.columns]
    df_clean = df_clean[available_features]

    # Validate through Pydantic schema, but only return available columns
    dataset = PenguinDataset.from_dataframe(df_clean)
    validated_df = dataset.to_dataframe()

    # Return only columns that were in the original cleaned data
    return validated_df[available_features]


def get_feature_info() -> dict:
    """Get information about features in the dataset."""
    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "all_features": ALL_FEATURES,
        "valid_islands": VALID_ISLANDS,
        "valid_sexes": VALID_SEXES,
        "valid_species": VALID_SPECIES,
    }
