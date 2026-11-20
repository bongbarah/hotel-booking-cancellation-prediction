
"""
Phase 2: Duplicate Analysis by Booking Segment

Measures duplicate concentration across:
- Market segment
- Distribution channel
- Customer type
- Hotel type

Original dataset remains unchanged.
"""

import pandas as pd
from config import RAW_DATA_PATH


def analyse_segment(df, columns, name):

    # Count original records by segment
    original = (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="original_records")
    )

    # Identify exact duplicate groups
    grouped = (
        df.groupby(
            df.columns.tolist(),
            dropna=False
        )
        .size()
        .reset_index(name="frequency")
    )

    duplicates = grouped[grouped["frequency"] > 1].copy()

    duplicates["additional_copies"] = (
        duplicates["frequency"] - 1
    )

    # Aggregate duplicate copies by segment
    duplicate_summary = (
        duplicates.groupby(columns, dropna=False)
        .agg(
            duplicate_groups=("frequency", "count"),
            additional_copies=("additional_copies", "sum")
        )
        .reset_index()
    )

    result = original.merge(
        duplicate_summary,
        on=columns,
        how="left"
    )

    result[["duplicate_groups", "additional_copies"]] = (
        result[["duplicate_groups", "additional_copies"]]
        .fillna(0)
        .astype(int)
    )

    result["duplicate_copy_rate_pct"] = (
        result["additional_copies"]
        / result["original_records"]
        * 100
    )

    result = result.sort_values(
        "additional_copies",
        ascending=False
    )

    print(f"\n{name}")
    print("-" * 70)
    print(result.to_string(index=False))

    return result


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 70)
    print("DUPLICATE ANALYSIS BY BOOKING SEGMENT")
    print("=" * 70)

    # 1. Market segment
    market = analyse_segment(
        df,
        ["market_segment"],
        "1. MARKET SEGMENT"
    )

    # 2. Distribution channel
    channel = analyse_segment(
        df,
        ["distribution_channel"],
        "2. DISTRIBUTION CHANNEL"
    )

    # 3. Customer type
    customer = analyse_segment(
        df,
        ["customer_type"],
        "3. CUSTOMER TYPE"
    )

    # 4. Hotel and market segment
    hotel_market = analyse_segment(
        df,
        ["hotel", "market_segment"],
        "4. HOTEL TYPE AND MARKET SEGMENT"
    )

    # Save reports
    market.to_csv(
        "duplicate_by_market_segment.csv",
        index=False
    )

    channel.to_csv(
        "duplicate_by_distribution_channel.csv",
        index=False
    )

    customer.to_csv(
        "duplicate_by_customer_type.csv",
        index=False
    )

    hotel_market.to_csv(
        "duplicate_by_hotel_market_segment.csv",
        index=False
    )

    print("\n5. DATA INTEGRITY")
    print("-" * 70)

    print("Original records:", len(df))
    print(
        "Original dataset unchanged."
    )

    print("\nReports saved successfully.")

    print("\n" + "=" * 70)
    print("SEGMENT ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
