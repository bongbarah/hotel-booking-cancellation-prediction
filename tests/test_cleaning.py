import pandas as pd
from src.cleaning import clean_data

def test_exact_duplicates_removed_but_conflicting_labels_retained():
    frame = pd.DataFrame({
        "hotel": ["City Hotel", "City Hotel", "City Hotel"],
        "is_canceled": [0, 1, 1],
        "adr": [100.0, 100.0, 100.0],
        "adults": [2, 2, 2],
        "children": [0.0, 0.0, 0.0],
        "babies": [0, 0, 0],
        "stays_in_weekend_nights": [1, 1, 1],
        "stays_in_week_nights": [2, 2, 2],
        "agent": [None, None, None],
        "country": [None, None, None],
        "meal": ["BB", "BB", "BB"],
        "market_segment": ["Direct", "Direct", "Direct"],
        "distribution_channel": ["Direct", "Direct", "Direct"],
        "company": [None, None, None]
    })
    cleaned, audit = clean_data(frame)
    assert len(cleaned) == 2
    assert set(cleaned["is_canceled"]) == {0, 1}
    assert audit["exact_duplicates_removed"] == 1
