
"""
Phase 2: Data Quality Diagnostics
Hotel Booking Cancellation Prediction

This script investigates unusual records without
deleting or modifying the original dataset.
"""

import pandas as pd
from config import RAW_DATA_PATH


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 65)
    print("HOTEL BOOKING DATA QUALITY INVESTIGATION")
    print("=" * 65)

    # -------------------------------------------------
    # 1. DUPLICATE INVESTIGATION
    # -------------------------------------------------

    print("\n1. EXACT DUPLICATES")
    print("-" * 65)

    duplicate_mask = df.duplicated(keep=False)

    print("Rows belonging to duplicate groups:",
          duplicate_mask.sum())

    print("Unique records after dropping exact duplicates:",
          len(df.drop_duplicates()))

    # Check whether identical predictor records
    # have conflicting cancellation outcomes.

    excluded_columns = [
        "is_canceled",
        "reservation_status",
        "reservation_status_date"
    ]

    predictor_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    repeated_predictors = df.duplicated(
        subset=predictor_columns,
        keep=False
    )

    repeated_df = df.loc[repeated_predictors].copy()

    if not repeated_df.empty:

        conflicting_groups = (
            repeated_df.groupby(predictor_columns, dropna=False)
            ["is_canceled"]
            .nunique()
        )

        print(
            "Predictor groups with conflicting target labels:",
            (conflicting_groups > 1).sum()
        )

    # -------------------------------------------------
    # 2. ADR INVESTIGATION
    # -------------------------------------------------

    print("\n2. AVERAGE DAILY RATE (ADR)")
    print("-" * 65)

    print(df["adr"].describe(
        percentiles=[0.90, 0.95, 0.99, 0.999]
    ))

    print("\nNegative ADR records:")

    print(
        df.loc[
            df["adr"] < 0,
            ["hotel", "adr", "is_canceled"]
        ].head(20)
    )

    print("\nADR above 1,000:")

    print(
        df.loc[
            df["adr"] > 1000,
            ["hotel", "adr", "is_canceled"]
        ].head(20)
    )

    # -------------------------------------------------
    # 3. GUEST COUNT INVESTIGATION
    # -------------------------------------------------

    print("\n3. GUEST COUNT")
    print("-" * 65)

    df["total_guests"] = (
        df["adults"].fillna(0)
        + df["children"].fillna(0)
        + df["babies"].fillna(0)
    )

    print("Zero-guest bookings:",
          (df["total_guests"] == 0).sum())

    print("Bookings with more than 10 guests:",
          (df["total_guests"] > 10).sum())

    print("\nLargest guest groups:")

    print(
        df.nlargest(
            10, "total_guests"
        )[
            ["hotel", "adults", "children",
             "babies", "total_guests", "is_canceled"]
        ]
    )

    # -------------------------------------------------
    # 4. LENGTH OF STAY
    # -------------------------------------------------

    print("\n4. LENGTH OF STAY")
    print("-" * 65)

    df["total_nights"] = (
        df["stays_in_weekend_nights"]
        + df["stays_in_week_nights"]
    )

    print("Zero-night bookings:",
          (df["total_nights"] == 0).sum())

    print("Bookings longer than 30 nights:",
          (df["total_nights"] > 30).sum())

    print("\nLongest stays:")

    print(
        df.nlargest(
            10, "total_nights"
        )[
            ["hotel", "stays_in_weekend_nights",
             "stays_in_week_nights",
             "total_nights", "is_canceled"]
        ]
    )

    # -------------------------------------------------
    # 5. TARGET AND RESERVATION STATUS
    # -------------------------------------------------

    print("\n5. TARGET AND RESERVATION STATUS")
    print("-" * 65)

    print(
        pd.crosstab(
            df["is_canceled"],
            df["reservation_status"],
            margins=True
        )
    )

    # -------------------------------------------------
    # 6. CATEGORICAL VALUE INVESTIGATION
    # -------------------------------------------------

    print("\n6. UNDEFINED CATEGORICAL VALUES")
    print("-" * 65)

    categorical_columns = [
        "meal",
        "market_segment",
        "distribution_channel"
    ]

    for column in categorical_columns:

        print(f"\n{column}:")

        print(
            df[column].value_counts(
                dropna=False
            )
        )

    # -------------------------------------------------
    # 7. ROOM TYPE INVESTIGATION
    # -------------------------------------------------

    print("\n7. ROOM TYPE COMPARISON")
    print("-" * 65)

    print(
        pd.crosstab(
            df["reserved_room_type"],
            df["assigned_room_type"]
        )
    )

    # -------------------------------------------------
    # 8. ARRIVAL MONTHS
    # -------------------------------------------------

    print("\n8. ARRIVAL MONTHS")
    print("-" * 65)

    print(
        df["arrival_date_month"].value_counts()
    )

    print("\nDiagnostics completed.")
    print("Original dataset has not been modified.")


if __name__ == "__main__":
    main()
