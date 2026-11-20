from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
RESULTS = REPORTS / "model_results"
RAW_FILE = RAW / "hotel_bookings.csv"
CLEAN_FILE = PROCESSED / "hotel_bookings_clean.csv"
MODEL_FILE = MODELS / "best_model.joblib"
METRICS_FILE = RESULTS / "model_comparison.csv"
TARGET = "is_canceled"
SEED = 42
NUMERIC = ["lead_time","arrival_date_year","arrival_date_week_number",
"arrival_date_day_of_month","stays_in_weekend_nights","stays_in_week_nights",
"adults","children","babies","is_repeated_guest","previous_cancellations",
"previous_bookings_not_canceled","adr","total_guests","total_nights"]
CATEGORICAL = ["hotel","arrival_date_month","meal","country","market_segment",
"distribution_channel","reserved_room_type","deposit_type","agent","customer_type"]
FEATURES = NUMERIC + CATEGORICAL
for p in [RAW,PROCESSED,REPORTS,FIGURES/"eda",FIGURES/"models",
          FIGURES/"explainability",MODELS,RESULTS]:
    p.mkdir(parents=True, exist_ok=True)
