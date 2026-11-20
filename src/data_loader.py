
"""
Data Loader
Hotel Booking Cancellation Prediction

Loads the raw hotel booking dataset and provides
an initial overview of the data.
"""

import pandas as pd
from config import RAW_DATA_PATH


def load_raw_data():
    """Load the original hotel booking dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("-" * 50)

    return df


def dataset_overview(df):
    """Display basic dataset information."""

    print("\nDATASET OVERVIEW")
    print("-" * 50)

    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nCOLUMN NAMES")
    print("-" * 50)

    for column in df.columns:
        print(column)

    print("\nDATA TYPES")
    print("-" * 50)

    print(df.dtypes)

    print("\nFIRST FIVE RECORDS")
    print("-" * 50)

    print(df.head())

    print("\nDATASET INFORMATION")
    print("-" * 50)

    df.info()

    return df


def missing_values_report(df):
    """Calculate missing values for every column."""

    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().mean() * 100
        )
    })

    missing = missing.sort_values(
        by="Missing Percentage",
        ascending=False
    )

    print("\nMISSING VALUES REPORT")
    print("-" * 50)

    print(missing.to_string())

    return missing


def duplicate_report(df):
    """Identify duplicate records."""

    duplicate_count = df.duplicated().sum()

    print("\nDUPLICATE RECORDS")
    print("-" * 50)

    print(f"Duplicate rows: {duplicate_count:,}")

    return duplicate_count
