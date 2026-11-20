
"""
Phase 2: High-Frequency Duplicate Inspection

Inspects records appearing 20 or more times.
Does not modify the original dataset.
"""

import pandas as pd
from config import RAW_DATA_PATH


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 70)
    print("HIGH-FREQUENCY DUPLICATE INVESTIGATION")
    print("=" * 70)

    # Group completely identical records
    duplicate_groups = (
        df.groupby(
            df.columns.tolist(),
            dropna=False
        )
        .size()
        .reset_index(name="frequency")
    )

    # Keep records repeated at least 20 times
    high_frequency = (
        duplicate_groups[
            duplicate_groups["frequency"] >= 20
        ]
        .sort_values(
            "frequency",
            ascending=False
        )
        .copy()
    )

    print("\n1. HIGH-FREQUENCY SUMMARY")
    print("-" * 70)

    print(
        "Groups appearing 20 or more times:",
        len(high_frequency)
    )

    print(
        "Rows represented by these groups:",
        int(high_frequency["frequency"].sum())
    )

    print(
        "Additional duplicate copies:",
        int((high_frequency["frequency"] - 1).sum())
    )

    # 2. Display selected characteristics
    columns_to_display = [
        "hotel",
        "lead_time",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
        "adults",
        "children",
        "babies",
        "meal",
        "market_segment",
        "distribution_channel",
        "customer_type",
        "reserved_room_type",
        "assigned_room_type",
        "adr",
        "is_canceled",
        "reservation_status",
        "frequency"
    ]

    available_columns = [
        col for col in columns_to_display
        if col in high_frequency.columns
    ]

    print("\n2. TOP 20 MOST REPEATED RECORDS")
    print("-" * 70)

    print(
        high_frequency[
            available_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # 3. Distribution by hotel and cancellation
    print("\n3. HIGH-FREQUENCY GROUPS BY HOTEL")
    print("-" * 70)

    summary = (
        high_frequency
        .groupby(
            ["hotel", "is_canceled"]
        )
        .agg(
            groups=("frequency", "count"),
            total_rows=("frequency", "sum"),
            additional_copies=(
                "frequency",
                lambda x: (x - 1).sum()
            )
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # 4. Frequency statistics
    print("\n4. REPETITION FREQUENCY STATISTICS")
    print("-" * 70)

    print(
        high_frequency["frequency"].describe()
    )

    # 5. Save investigation results
    high_frequency.to_csv(
        "high_frequency_duplicate_records.csv",
        index=False
    )

    summary.to_csv(
        "high_frequency_duplicate_summary.csv",
        index=False
    )

    print("\n5. DATA INTEGRITY")
    print("-" * 70)

    print("Original dataset records:", len(df))
    print("Original dataset unchanged.")

    print(
        "\nReports saved successfully."
    )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()


