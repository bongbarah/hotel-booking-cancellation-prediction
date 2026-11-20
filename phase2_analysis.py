
"""
Phase 2: Initial Dataset Analysis
Hotel Booking Cancellation Prediction
"""

from src.data_loader import (
    load_raw_data,
    dataset_overview,
    missing_values_report,
    duplicate_report
)

from src.data_cleaning import (
    inspect_missing_values,
    inspect_duplicates,
    inspect_target,
    inspect_numeric_columns,
    inspect_categorical_columns
)


def main():

    # Load original dataset
    df = load_raw_data()

    # General overview
    dataset_overview(df)

    # Missing values
    missing_values_report(df)
    inspect_missing_values(df)

    # Duplicates
    duplicate_report(df)
    inspect_duplicates(df)

    # Target distribution
    inspect_target(df)

    # Numerical analysis
    inspect_numeric_columns(df)

    # Categorical analysis
    inspect_categorical_columns(df)


if __name__ == "__main__":
    main()
