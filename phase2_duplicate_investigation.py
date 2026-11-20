
"""
Phase 2: Duplicate Record Investigation
Hotel Booking Cancellation Prediction

Investigates duplicate records and conflicting target labels.
Does not modify the original dataset.
"""

import pandas as pd
from config import RAW_DATA_PATH


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    excluded_columns = [
        "is_canceled",
        "reservation_status",
        "reservation_status_date"
    ]

    predictor_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    print("=" * 70)
    print("DUPLICATE RECORD INVESTIGATION")
    print("=" * 70)

    # 1. Exact duplicates
    exact_duplicate_mask = df.duplicated(keep=False)

    print("\n1. EXACT DUPLICATE SUMMARY")
    print("-" * 70)
    print("Total records:", len(df))
    print("Rows belonging to duplicate groups:",
          exact_duplicate_mask.sum())
    print("Additional duplicate copies:",
          df.duplicated().sum())
    print("Rows after removing exact duplicates:",
          len(df.drop_duplicates()))

    print("\nExact duplicate rows by cancellation label:")
    print(
        df.loc[exact_duplicate_mask, "is_canceled"]
        .value_counts()
        .sort_index()
    )

    # 2. Find identical predictors with different target labels
    repeated_mask = df.duplicated(
        subset=predictor_columns,
        keep=False
    )

    repeated_df = df.loc[repeated_mask].copy()

    grouped = repeated_df.groupby(
        predictor_columns,
        dropna=False,
        sort=False
    )

    group_summary = grouped["is_canceled"].agg(
        group_size="size",
        different_labels="nunique"
    )

    conflicting_groups = group_summary[
        group_summary["different_labels"] > 1
    ]

    print("\n2. CONFLICTING TARGET GROUPS")
    print("-" * 70)
    print("Number of conflicting groups:",
          len(conflicting_groups))
    print("Total rows in conflicting groups:",
          conflicting_groups["group_size"].sum())

    print("\nConflicting group sizes:")
    print(conflicting_groups["group_size"].value_counts().sort_index())

    # 3. Display examples of conflicting records
    if not conflicting_groups.empty:

        print("\n3. EXAMPLES OF CONFLICTING RECORDS")
        print("-" * 70)

        grouped_rows = repeated_df.groupby(
            predictor_columns,
            dropna=False,
            sort=False
        )

        # Display up to five conflicting groups
        shown = 0

        for _, group in grouped_rows:

            if group["is_canceled"].nunique() > 1:

                print(f"\nConflicting group {shown + 1}")
                print("Number of rows:", len(group))

                display_columns = [
                    "hotel",
                    "lead_time",
                    "arrival_date_year",
                    "arrival_date_month",
                    "arrival_date_day_of_month",
                    "adults",
                    "children",
                    "babies",
                    "adr",
                    "is_canceled",
                    "reservation_status",
                    "reservation_status_date"
                ]

                print(group[display_columns].to_string(index=True))

                shown += 1

                if shown == 5:
                    break

    print("\nInvestigation completed.")
    print("No records have been deleted or modified.")


if __name__ == "__main__":
    main()

