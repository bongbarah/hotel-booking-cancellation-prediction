
"""
Phase 2: Segment-Level Deduplication Impact

Compares the original and exact-deduplicated datasets.

The raw dataset is never modified.
"""

import pandas as pd
from config import RAW_DATA_PATH


def create_summary(df, columns, dataset_name):

    summary = (
        df.groupby(columns, dropna=False)
        .agg(
            records=("is_canceled", "size"),
            cancellations=("is_canceled", "sum")
        )
        .reset_index()
    )

    summary["cancellation_rate_pct"] = (
        summary["cancellations"] / summary["records"] * 100
    )

    summary["dataset_share_pct"] = (
        summary["records"] / len(df) * 100
    )

    return summary


def compare_datasets(original, deduplicated, columns, title):

    original_summary = create_summary(
        original, columns, "original"
    )

    deduplicated_summary = create_summary(
        deduplicated, columns, "deduplicated"
    )

    comparison = original_summary.merge(
        deduplicated_summary,
        on=columns,
        how="outer",
        suffixes=("_original", "_deduplicated")
    )

    numeric_columns = [
        "records_original",
        "records_deduplicated",
        "cancellations_original",
        "cancellations_deduplicated",
        "cancellation_rate_pct_original",
        "cancellation_rate_pct_deduplicated",
        "dataset_share_pct_original",
        "dataset_share_pct_deduplicated"
    ]

    comparison[numeric_columns] = (
        comparison[numeric_columns].fillna(0)
    )

    comparison["records_removed"] = (
        comparison["records_original"]
        - comparison["records_deduplicated"]
    )

    comparison["cancellation_rate_change_pp"] = (
        comparison["cancellation_rate_pct_deduplicated"]
        - comparison["cancellation_rate_pct_original"]
    )

    comparison = comparison.sort_values(
        "records_removed",
        ascending=False
    )

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    print(comparison.to_string(index=False))

    return comparison


def main():

    original = pd.read_csv(RAW_DATA_PATH)

    # Remove exact duplicate rows only in memory.
    deduplicated = original.drop_duplicates().copy()

    print("=" * 75)
    print("SEGMENT-LEVEL DEDUPLICATION IMPACT")
    print("=" * 75)

    print("Original records:", len(original))
    print("Deduplicated records:", len(deduplicated))
    print("Records removed:", len(original) - len(deduplicated))

    # 1. Market segment
    market = compare_datasets(
        original,
        deduplicated,
        ["market_segment"],
        "1. MARKET SEGMENT"
    )

    # 2. Hotel and market segment
    hotel_market = compare_datasets(
        original,
        deduplicated,
        ["hotel", "market_segment"],
        "2. HOTEL AND MARKET SEGMENT"
    )

    # 3. Customer type
    customer = compare_datasets(
        original,
        deduplicated,
        ["customer_type"],
        "3. CUSTOMER TYPE"
    )

    # Save reports
    market.to_csv(
        "deduplication_impact_market_segment.csv",
        index=False
    )

    hotel_market.to_csv(
        "deduplication_impact_hotel_market.csv",
        index=False
    )

    customer.to_csv(
        "deduplication_impact_customer_type.csv",
        index=False
    )

    print("\nReports saved successfully.")
    print("Original dataset remains unchanged.")

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
