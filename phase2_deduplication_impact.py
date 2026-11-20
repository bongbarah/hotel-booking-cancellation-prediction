
"""
Phase 2: Deduplication Impact Analysis

Compares the original dataset with a version
containing one copy of each exact record.

The original dataset is never modified.
"""

import pandas as pd
from config import RAW_DATA_PATH


def target_summary(df, dataset_name):

    counts = df["is_canceled"].value_counts().sort_index()

    summary = {
        "Dataset": dataset_name,
        "Total records": len(df),
        "Not cancelled": counts.get(0, 0),
        "Cancelled": counts.get(1, 0),
        "Cancellation rate (%)": (
            df["is_canceled"].mean() * 100
        )
    }

    return summary


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 70)
    print("DEDUPLICATION IMPACT ANALYSIS")
    print("=" * 70)

    # Original dataset
    original = df.copy()

    # Remove only completely identical rows
    deduplicated = df.drop_duplicates().copy()

    print("\n1. RECORD COUNTS")
    print("-" * 70)
    print("Original records:", len(original))
    print("After exact deduplication:", len(deduplicated))
    print("Records removed:", len(original) - len(deduplicated))

    # Compare cancellation rates
    print("\n2. CANCELLATION DISTRIBUTION")
    print("-" * 70)

    results = pd.DataFrame([
        target_summary(original, "Original"),
        target_summary(deduplicated, "Deduplicated")
    ])

    print(results.to_string(index=False))

    # Compare categorical distributions
    print("\n3. HOTEL TYPE DISTRIBUTION")
    print("-" * 70)

    for name, dataset in [
        ("Original", original),
        ("Deduplicated", deduplicated)
    ]:

        print(f"\n{name}:")
        print(
            dataset["hotel"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    # Check conflicting target groups after deduplication
    excluded_columns = [
        "is_canceled",
        "reservation_status",
        "reservation_status_date"
    ]

    predictor_columns = [
        col for col in deduplicated.columns
        if col not in excluded_columns
    ]

    repeated = deduplicated.duplicated(
        subset=predictor_columns,
        keep=False
    )

    repeated_df = deduplicated.loc[repeated]

    if not repeated_df.empty:

        conflicts = (
            repeated_df.groupby(
                predictor_columns,
                dropna=False,
                sort=False
            )["is_canceled"]
            .nunique()
        )

        print("\n4. CONFLICTING GROUPS AFTER DEDUPLICATION")
        print("-" * 70)
        print(
            "Groups with different target labels:",
            (conflicts > 1).sum()
        )

    print("\nAnalysis completed.")
    print("No files have been changed or deleted.")


if __name__ == "__main__":
    main()
