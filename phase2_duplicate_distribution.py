
"""
Phase 2: Duplicate Distribution Analysis

Identifies where exact duplicate copies occur.
The original dataset is never modified.
"""

import pandas as pd
from config import RAW_DATA_PATH


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 70)
    print("DUPLICATE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Identify groups of completely identical records
    exact_groups = (
        df.groupby(
            df.columns.tolist(),
            dropna=False
        )
        .size()
        .reset_index(name="frequency")
    )

    duplicate_groups = exact_groups[
        exact_groups["frequency"] > 1
    ].copy()

    duplicate_groups["extra_copies"] = (
        duplicate_groups["frequency"] - 1
    )

    # 1. Duplicate distribution by hotel and target
    print("\n1. DUPLICATE COPIES BY HOTEL AND CANCELLATION")
    print("-" * 70)

    distribution = (
        duplicate_groups
        .groupby(["hotel", "is_canceled"])
        .agg(
            duplicate_groups=("frequency", "count"),
            rows_in_groups=("frequency", "sum"),
            additional_copies=("extra_copies", "sum")
        )
        .reset_index()
    )

    print(distribution.to_string(index=False))

    # 2. Duplicate distribution by hotel
    print("\n2. DUPLICATE COPIES BY HOTEL")
    print("-" * 70)

    hotel_summary = (
        duplicate_groups
        .groupby("hotel")
        .agg(
            duplicate_groups=("frequency", "count"),
            additional_copies=("extra_copies", "sum")
        )
    )

    print(hotel_summary)

    # 3. Duplicate frequency
    print("\n3. FREQUENCY OF EXACT DUPLICATE RECORDS")
    print("-" * 70)

    print(
        duplicate_groups["frequency"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nMaximum repetition of one identical record:",
        duplicate_groups["frequency"].max()
    )

    # 4. Save reports
    distribution.to_csv(
        "duplicate_distribution_by_hotel.csv",
        index=False
    )

    hotel_summary.to_csv(
        "duplicate_distribution_summary.csv"
    )

    print("\n4. INTEGRITY CHECK")
    print("-" * 70)

    print("Original records:", len(df))
    print(
        "Additional duplicate copies:",
        int(duplicate_groups["extra_copies"].sum())
    )

    print(
        "Original dataset remains unchanged."
    )

    print("\nReports saved successfully.")

    print("\n" + "=" * 70)
    print("DUPLICATE DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()


