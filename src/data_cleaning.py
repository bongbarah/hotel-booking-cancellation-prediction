
"""
Data Cleaning Diagnostics
Hotel Booking Cancellation Prediction
"""

import pandas as pd


def inspect_missing_values(df):

    print("\nMISSING VALUE ANALYSIS")
    print("=" * 55)

    for column in df.columns:

        missing_count = df[column].isnull().sum()

        if missing_count > 0:

            percentage = (
                missing_count / len(df)
            ) * 100

            print(
                f"{column}: "
                f"{missing_count:,} missing "
                f"({percentage:.2f}%)"
            )


def inspect_duplicates(df):

    print("\nDUPLICATE ANALYSIS")
    print("=" * 55)

    print(
        "Exact duplicate records:",
        df.duplicated().sum()
    )


def inspect_target(df):

    target = "is_canceled"

    if target not in df.columns:
        raise KeyError(f"Target column {target} not found.")

    print("\nTARGET VARIABLE ANALYSIS")
    print("=" * 55)

    counts = df[target].value_counts()

    percentages = df[target].value_counts(
        normalize=True
    ) * 100

    results = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(results)

    print("\nInterpretation:")
    print("0 = Booking not cancelled")
    print("1 = Booking cancelled")


def inspect_numeric_columns(df):

    print("\nNUMERIC DESCRIPTIVE STATISTICS")
    print("=" * 55)

    print(
        df.describe().T.to_string()
    )


def inspect_categorical_columns(df):

    print("\nCATEGORICAL COLUMN ANALYSIS")
    print("=" * 55)

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        print(f"\n{column}")

        print(
            df[column]
            .value_counts(dropna=False)
            .head(10)
        )

