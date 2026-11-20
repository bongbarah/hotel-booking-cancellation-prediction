
import pandas as pd
from config import TARGET


def clean_data(source):

    df = source.copy()
    rows_before = len(df)

    # 1. Remove only completely identical records.
    exact_duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # 2. Handle missing values.
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)

    for column in ["agent", "country"]:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
                .replace("nan", "Unknown")
            )

    # 3. Exclude company because of its very high missing rate.
    if "company" in df.columns:
        df = df.drop(columns=["company"])

    # 4. Replace undefined categories.
    for column in ["meal", "market_segment", "distribution_channel"]:
        if column in df.columns:
            df[column] = df[column].replace(
                "Undefined", "Unknown"
            )

    # 5. Investigate and correct negative ADR values.
    if "adr" in df.columns:

        df["adr_was_negative"] = (df["adr"] < 0).astype(int)

        if "hotel" in df.columns:
            for hotel in df["hotel"].dropna().unique():

                invalid_mask = (
                    (df["hotel"] == hotel) &
                    (df["adr"] < 0)
                )

                if invalid_mask.any():

                    median_adr = df.loc[
                        (df["hotel"] == hotel) &
                        (df["adr"] >= 0),
                        "adr"
                    ].median()

                    df.loc[invalid_mask, "adr"] = median_adr

        # Cap extreme values at the 99.9th percentile.
        adr_cap = float(df["adr"].quantile(0.999))

        df["adr_was_capped"] = (
            df["adr"] > adr_cap
        ).astype(int)

        df["adr"] = df["adr"].clip(upper=adr_cap)

    else:
        adr_cap = None

    # 6. Create derived features when source columns exist.
    guest_columns = ["adults", "children", "babies"]

    if all(column in df.columns for column in guest_columns):

        df["total_guests"] = df[guest_columns].sum(axis=1)

        zero_guest_rows_removed = int(
            (df["total_guests"] == 0).sum()
        )

        df = df.loc[df["total_guests"] > 0].copy()

    else:
        zero_guest_rows_removed = 0

    if {
        "stays_in_weekend_nights",
        "stays_in_week_nights"
    }.issubset(df.columns):

        df["total_nights"] = (
            df["stays_in_weekend_nights"] +
            df["stays_in_week_nights"]
        )

    # 7. Validate target only when it exists.
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)

    # 8. Record cleaning decisions.
    audit = {
        "rows_before": rows_before,
        "exact_duplicates_removed": exact_duplicates_removed,
        "zero_guest_rows_removed": zero_guest_rows_removed,
        "rows_after": len(df),
        "adr_upper_cap_99_9_percentile": adr_cap,
        "conflicting_predictor_groups": "Retained"
    }

    return df, audit
